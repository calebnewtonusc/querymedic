"""
explain_analyzer.py — Parse PostgreSQL EXPLAIN JSON output and produce structured
diagnosis objects suitable for training pair generation.

Unlike explain_plan_parser.py (which handles text EXPLAIN output), this module:
  1. Parses the native EXPLAIN (FORMAT JSON) output — a nested plan tree
  2. Extracts plan nodes recursively with full cost/timing/buffer data
  3. Identifies performance anti-patterns with specific evidence
  4. Generates LLM-ready diagnosis prompts from structured plan analysis

Usage:
    from synthesis.explain_analyzer import ExplainJSONAnalyzer

    analyzer = ExplainJSONAnalyzer()
    result = analyzer.analyze(json_plan_text)
    print(result.bottleneck_summary)
    print(result.llm_diagnosis_prompt)

    # With schema context for richer diagnosis
    result = analyzer.analyze(json_plan_text, schema_ddl="CREATE TABLE ...")
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional


# ─── Node Type Classification ──────────────────────────────────────────────────

SEQ_SCAN_NODES = {"Seq Scan", "Parallel Seq Scan"}
INDEX_NODES = {"Index Scan", "Index Only Scan", "Bitmap Index Scan"}
HEAP_NODES = {"Bitmap Heap Scan"}
JOIN_NODES = {"Hash Join", "Merge Join", "Nested Loop"}
SORT_NODES = {"Sort", "Incremental Sort"}
AGG_NODES = {"HashAggregate", "GroupAggregate", "Aggregate", "MixedAggregate"}
GATHER_NODES = {"Gather", "Gather Merge"}
OTHER_NODES = {
    "Hash",
    "Materialize",
    "Memoize",
    "Limit",
    "Unique",
    "Append",
    "Subquery Scan",
    "CTE Scan",
    "Result",
    "SetOp",
    "WindowAgg",
}

# Row estimation error threshold (factor)
ESTIMATION_ERROR_THRESHOLD = 10.0
# Minimum rows_removed_by_filter to flag a seq scan
MIN_ROWS_REMOVED = 10_000
# Minimum actual time (ms) to flag a sort as expensive
SLOW_SORT_MS = 500.0
# Minimum heap fetches to flag index scan as heap-fetch-heavy
MIN_HEAP_FETCHES = 1_000
# Minimum actual rows in nested loop to flag as large outer
NESTED_LOOP_LARGE_OUTER = 5_000


@dataclass
class PlanNodeAnalysis:
    """Analysis of a single EXPLAIN plan node."""

    node_type: str
    relation: Optional[str]
    index_name: Optional[str]
    startup_cost: float
    total_cost: float
    plan_rows: int
    actual_rows: int
    actual_loops: int
    actual_startup_ms: float
    actual_total_ms: float
    rows_removed_by_filter: int
    rows_removed_by_join_filter: int
    heap_fetches: int
    shared_hit_blocks: int
    shared_read_blocks: int
    shared_written_blocks: int
    sort_method: Optional[str]  # e.g. "quicksort", "external merge"
    sort_space_used_kb: Optional[int]
    depth: int
    children: list["PlanNodeAnalysis"] = field(default_factory=list)

    @property
    def total_actual_ms(self) -> float:
        return self.actual_total_ms * self.actual_loops

    @property
    def row_estimation_ratio(self) -> Optional[float]:
        if self.plan_rows == 0:
            return None
        return self.actual_rows / self.plan_rows

    @property
    def is_bad_row_estimate(self) -> bool:
        ratio = self.row_estimation_ratio
        return ratio is not None and (
            ratio > ESTIMATION_ERROR_THRESHOLD
            or ratio < 1.0 / ESTIMATION_ERROR_THRESHOLD
        )

    @property
    def cache_hit_rate(self) -> Optional[float]:
        total = self.shared_hit_blocks + self.shared_read_blocks
        if total == 0:
            return None
        return self.shared_hit_blocks / total

    @property
    def is_disk_heavy(self) -> bool:
        """True if this node is doing significant disk I/O."""
        return self.shared_read_blocks > 1000


@dataclass
class ExplainAnalysisResult:
    """Complete analysis of an EXPLAIN (FORMAT JSON) plan."""

    planning_time_ms: float
    execution_time_ms: float
    root_node: Optional[PlanNodeAnalysis]
    all_nodes: list[PlanNodeAnalysis]

    # Detected anti-patterns
    seq_scans: list[PlanNodeAnalysis]
    bad_row_estimates: list[PlanNodeAnalysis]
    expensive_sorts: list[PlanNodeAnalysis]
    heap_fetch_heavy: list[PlanNodeAnalysis]
    large_nested_loops: list[PlanNodeAnalysis]
    disk_heavy_nodes: list[PlanNodeAnalysis]

    # Summary
    bottleneck_type: str
    bottleneck_summary: str

    # LLM prompt
    llm_diagnosis_prompt: str

    # Raw plan for reference
    raw_plan: dict = field(default_factory=dict)

    @property
    def has_seq_scan(self) -> bool:
        return bool(self.seq_scans)

    @property
    def total_seq_scan_rows_removed(self) -> int:
        return sum(n.rows_removed_by_filter for n in self.seq_scans)

    @property
    def worst_row_estimate_ratio(self) -> Optional[float]:
        ratios = [
            n.row_estimation_ratio
            for n in self.bad_row_estimates
            if n.row_estimation_ratio
        ]
        return max(ratios, default=None)


def _parse_node(node_dict: dict, depth: int = 0) -> PlanNodeAnalysis:
    """Recursively parse a JSON plan node into a PlanNodeAnalysis."""
    node_type = node_dict.get("Node Type", "Unknown")
    relation = node_dict.get("Relation Name") or node_dict.get("Alias")
    index_name = node_dict.get("Index Name")

    # Costs and row estimates
    startup_cost = float(node_dict.get("Startup Cost", 0.0))
    total_cost = float(node_dict.get("Total Cost", 0.0))
    plan_rows = int(node_dict.get("Plan Rows", 0))

    # Actual timing (only present in EXPLAIN ANALYZE)
    actual_rows = int(node_dict.get("Actual Rows", 0))
    actual_loops = int(node_dict.get("Actual Loops", 1))
    actual_startup_ms = float(node_dict.get("Actual Startup Time", 0.0))
    actual_total_ms = float(node_dict.get("Actual Total Time", 0.0))

    # Filter rows removed
    rows_removed_by_filter = int(node_dict.get("Rows Removed by Filter", 0))
    rows_removed_by_join_filter = int(node_dict.get("Rows Removed by Join Filter", 0))
    heap_fetches = int(node_dict.get("Heap Fetches", 0))

    # Buffer usage
    shared_hit_blocks = int(node_dict.get("Shared Hit Blocks", 0))
    shared_read_blocks = int(node_dict.get("Shared Read Blocks", 0))
    shared_written_blocks = int(node_dict.get("Shared Written Blocks", 0))

    # Sort info
    sort_method = node_dict.get("Sort Method")
    sort_space_used_kb = node_dict.get("Sort Space Used")
    if sort_space_used_kb is not None:
        sort_space_used_kb = int(sort_space_used_kb)

    analysis = PlanNodeAnalysis(
        node_type=node_type,
        relation=relation,
        index_name=index_name,
        startup_cost=startup_cost,
        total_cost=total_cost,
        plan_rows=plan_rows,
        actual_rows=actual_rows,
        actual_loops=actual_loops,
        actual_startup_ms=actual_startup_ms,
        actual_total_ms=actual_total_ms,
        rows_removed_by_filter=rows_removed_by_filter,
        rows_removed_by_join_filter=rows_removed_by_join_filter,
        heap_fetches=heap_fetches,
        shared_hit_blocks=shared_hit_blocks,
        shared_read_blocks=shared_read_blocks,
        shared_written_blocks=shared_written_blocks,
        sort_method=sort_method,
        sort_space_used_kb=sort_space_used_kb,
        depth=depth,
    )

    # Parse children (Plans array)
    for child_dict in node_dict.get("Plans", []):
        child = _parse_node(child_dict, depth + 1)
        analysis.children.append(child)

    return analysis


def _collect_all_nodes(root: PlanNodeAnalysis) -> list[PlanNodeAnalysis]:
    # QM-22: Return nodes sorted by actual_time_ms descending so callers that
    # pick result[0] (or iterate from the front) always encounter the most
    # expensive node first.  The previous DFS stack-pop order was effectively
    # reverse-BFS — arbitrary with respect to cost.
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node)
        stack.extend(node.children)
    result.sort(key=lambda n: n.actual_time_ms, reverse=True)
    return result


def _classify_bottleneck(
    seq_scans: list[PlanNodeAnalysis],
    bad_estimates: list[PlanNodeAnalysis],
    expensive_sorts: list[PlanNodeAnalysis],
    heap_heavy: list[PlanNodeAnalysis],
    large_nested: list[PlanNodeAnalysis],
    execution_ms: float,
) -> tuple[str, str]:
    """Classify the primary bottleneck and generate a human-readable summary."""

    # Priority order: seq scans > bad estimates > nested loops > sorts > heap fetches
    if seq_scans:
        worst = max(seq_scans, key=lambda n: n.rows_removed_by_filter)
        rows_removed = worst.rows_removed_by_filter
        relation = worst.relation or "unknown table"
        bottleneck_type = "seq_scan_missing_index"
        summary = (
            f"Sequential scan on {relation} removes {rows_removed:,} rows "
            f"(actual_rows={worst.actual_rows:,}, "
            f"time={worst.total_actual_ms:.1f}ms). "
            f"Missing index on filter column(s)."
        )

        if bad_estimates and any(n.relation == worst.relation for n in bad_estimates):
            bottleneck_type = "seq_scan_stale_stats"
            ratio = worst.row_estimation_ratio
            ratio_str = f"{ratio:.1f}x" if ratio else "unknown"
            summary += f" Row estimation error: planner expected {worst.plan_rows:,}, got {worst.actual_rows:,} ({ratio_str} off) → run ANALYZE."

        return bottleneck_type, summary

    if bad_estimates and execution_ms > 500:
        worst = max(bad_estimates, key=lambda n: abs(n.row_estimation_ratio or 1.0))
        ratio = worst.row_estimation_ratio
        ratio_str = f"{ratio:.1f}x" if ratio else "N/A"
        return "stale_statistics", (
            f"Row estimation error on {worst.relation or 'subquery'}: "
            f"planner expected {worst.plan_rows:,} rows, "
            f"got {worst.actual_rows:,} ({ratio_str} off). "
            f"Run ANALYZE on affected tables."
        )

    if large_nested:
        worst = max(large_nested, key=lambda n: n.actual_rows)
        return "nested_loop_large_outer", (
            f"Nested Loop with {worst.actual_rows:,} outer rows × inner lookups. "
            f"Consider Hash Join (enable_nestloop=off to test) or add index on inner table."
        )

    if expensive_sorts:
        worst = max(expensive_sorts, key=lambda n: n.total_actual_ms)
        spill = (
            " (spills to disk)"
            if worst.sort_method and "external" in worst.sort_method.lower()
            else ""
        )
        return "sort_overhead", (
            f"Sort node takes {worst.total_actual_ms:.0f}ms{spill}. "
            f"Sort space used: {worst.sort_space_used_kb or 'N/A'}kB. "
            f"Consider adding index with matching sort order, or increasing work_mem."
        )

    if heap_heavy:
        worst = max(heap_heavy, key=lambda n: n.heap_fetches)
        return "heap_fetch_overhead", (
            f"Bitmap Heap Scan on {worst.relation or 'table'} fetches {worst.heap_fetches:,} heap blocks. "
            f"Covering index (INCLUDE) would enable Index Only Scan and eliminate heap fetches."
        )

    return "general_performance", (
        f"Query took {execution_ms:.0f}ms. Review plan nodes for cost discrepancies."
    )


def _build_diagnosis_prompt(
    result: "ExplainAnalysisResult",
    schema_ddl: Optional[str] = None,
) -> str:
    """Generate an LLM-ready diagnosis prompt from the plan analysis."""
    parts = [
        "Analyze this PostgreSQL EXPLAIN ANALYZE output and provide a complete optimization prescription.\n"
    ]

    if schema_ddl:
        parts.append(f"**Schema:**\n```sql\n{schema_ddl[:3000]}\n```\n")

    # Execution summary
    parts.append(
        f"**Execution time:** {result.execution_time_ms:.1f}ms "
        f"(planning: {result.planning_time_ms:.1f}ms)\n"
    )

    # Render plan tree (top-level nodes)
    plan_lines = _format_plan_tree(result.root_node)
    if plan_lines:
        parts.append(f"**Plan:**\n```\n{plan_lines}\n```\n")

    # Key findings
    findings = []
    if result.seq_scans:
        for node in result.seq_scans[:3]:
            findings.append(
                f"- Seq Scan on {node.relation or 'table'}: "
                f"rows_removed={node.rows_removed_by_filter:,}, "
                f"actual_rows={node.actual_rows:,}, "
                f"time={node.total_actual_ms:.1f}ms"
            )
    if result.bad_row_estimates:
        for node in result.bad_row_estimates[:3]:
            ratio = node.row_estimation_ratio
            if ratio:
                findings.append(
                    f"- Row estimate error on {node.relation or 'node'}: "
                    f"planned {node.plan_rows:,} vs actual {node.actual_rows:,} "
                    f"({ratio:.1f}x off)"
                )
    if result.expensive_sorts:
        for node in result.expensive_sorts[:2]:
            spill = f" [{node.sort_method}]" if node.sort_method else ""
            findings.append(
                f"- Sort{spill}: {node.total_actual_ms:.0f}ms, "
                f"space={node.sort_space_used_kb or 'N/A'}kB"
            )
    if result.heap_fetch_heavy:
        for node in result.heap_fetch_heavy[:2]:
            findings.append(
                f"- Bitmap Heap Scan heap_fetches={node.heap_fetches:,} "
                f"(covering index would eliminate)"
            )
    if result.disk_heavy_nodes:
        for node in result.disk_heavy_nodes[:2]:
            findings.append(
                f"- {node.node_type} on {node.relation or 'node'}: "
                f"shared_read_blocks={node.shared_read_blocks:,} "
                f"(cache_hit_rate={node.cache_hit_rate:.1%})"
                if node.cache_hit_rate is not None
                else f"- {node.node_type}: {node.shared_read_blocks:,} disk block reads"
            )

    if findings:
        parts.append("**Key findings:**\n" + "\n".join(findings) + "\n")

    parts.append(
        f"**Primary bottleneck ({result.bottleneck_type}):** {result.bottleneck_summary}"
    )

    return "\n".join(parts)


def _format_plan_tree(node: Optional[PlanNodeAnalysis], indent: int = 0) -> str:
    """Format plan tree nodes as indented text for prompt inclusion."""
    if node is None:
        return ""
    lines = []
    prefix = "  " * indent + ("-> " if indent > 0 else "")
    node_line = f"{prefix}{node.node_type}"
    if node.relation:
        node_line += f" on {node.relation}"
    if node.index_name:
        node_line += f" using {node.index_name}"
    node_line += (
        f"  (cost={node.startup_cost:.2f}..{node.total_cost:.2f} rows={node.plan_rows})"
    )
    if node.actual_total_ms > 0:
        node_line += (
            f" (actual time={node.actual_startup_ms:.3f}..{node.actual_total_ms:.3f}"
            f" rows={node.actual_rows} loops={node.actual_loops})"
        )
    lines.append(node_line)
    if node.rows_removed_by_filter > 0:
        lines.append(
            f"{'  ' * (indent + 1)}Rows Removed by Filter: {node.rows_removed_by_filter:,}"
        )
    if node.heap_fetches > 0:
        lines.append(f"{'  ' * (indent + 1)}Heap Fetches: {node.heap_fetches:,}")

    for child in node.children:
        lines.append(_format_plan_tree(child, indent + 1))

    return "\n".join(lines)


class ExplainJSONAnalyzer:
    """
    Analyzes PostgreSQL EXPLAIN (FORMAT JSON) output.

    Handles two input forms:
      1. JSON array (standard EXPLAIN FORMAT JSON output): [{"Plan": {...}, "Planning Time": ..., "Execution Time": ...}]
      2. JSON object (single plan): {"Plan": {...}}
      3. Text EXPLAIN ANALYZE with embedded JSON (strips leading/trailing text)

    Example:
        analyzer = ExplainJSONAnalyzer()
        result = analyzer.analyze(json_text, schema_ddl="CREATE TABLE orders (...)")
        print(result.bottleneck_type)
        print(result.llm_diagnosis_prompt)
    """

    def analyze(
        self,
        plan_text: str,
        schema_ddl: Optional[str] = None,
    ) -> Optional[ExplainAnalysisResult]:
        """Parse and analyze an EXPLAIN (FORMAT JSON) output."""
        plan_dict = self._parse_json(plan_text)
        if plan_dict is None:
            return None
        return self._analyze_plan(plan_dict, schema_ddl)

    def analyze_dict(
        self,
        plan_dict: dict,
        schema_ddl: Optional[str] = None,
    ) -> ExplainAnalysisResult:
        """Analyze an already-parsed EXPLAIN plan dict."""
        return self._analyze_plan(plan_dict, schema_ddl)

    def _parse_json(self, text: str) -> Optional[dict]:
        """Extract and parse JSON from EXPLAIN output text."""
        text = text.strip()

        # Try direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and parsed:
                return parsed[0]
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from mixed text (e.g., psql output with leading whitespace)
        m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                if isinstance(parsed, list) and parsed:
                    return parsed[0]
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

    def _analyze_plan(
        self,
        plan_dict: dict,
        schema_ddl: Optional[str] = None,
    ) -> ExplainAnalysisResult:
        """Core analysis logic."""
        planning_ms = float(plan_dict.get("Planning Time", 0.0))
        execution_ms = float(plan_dict.get("Execution Time", 0.0))

        root_plan = plan_dict.get("Plan", plan_dict)
        root_node = _parse_node(root_plan, depth=0)
        all_nodes = _collect_all_nodes(root_node)

        # Categorize nodes
        seq_scans = [
            n
            for n in all_nodes
            if n.node_type in SEQ_SCAN_NODES
            and n.rows_removed_by_filter >= MIN_ROWS_REMOVED
        ]
        bad_row_estimates = [
            n for n in all_nodes if n.is_bad_row_estimate and n.plan_rows > 0
        ]
        expensive_sorts = [
            n
            for n in all_nodes
            if n.node_type in SORT_NODES and n.total_actual_ms >= SLOW_SORT_MS
        ]
        heap_fetch_heavy = [
            n
            for n in all_nodes
            if n.node_type == "Bitmap Heap Scan" and n.heap_fetches >= MIN_HEAP_FETCHES
        ]
        large_nested_loops = [
            n
            for n in all_nodes
            if n.node_type == "Nested Loop" and n.actual_rows >= NESTED_LOOP_LARGE_OUTER
        ]
        disk_heavy_nodes = [n for n in all_nodes if n.is_disk_heavy]

        bottleneck_type, bottleneck_summary = _classify_bottleneck(
            seq_scans=seq_scans,
            bad_estimates=bad_row_estimates,
            expensive_sorts=expensive_sorts,
            heap_heavy=heap_fetch_heavy,
            large_nested=large_nested_loops,
            execution_ms=execution_ms,
        )

        result = ExplainAnalysisResult(
            planning_time_ms=planning_ms,
            execution_time_ms=execution_ms,
            root_node=root_node,
            all_nodes=all_nodes,
            seq_scans=seq_scans,
            bad_row_estimates=bad_row_estimates,
            expensive_sorts=expensive_sorts,
            heap_fetch_heavy=heap_fetch_heavy,
            large_nested_loops=large_nested_loops,
            disk_heavy_nodes=disk_heavy_nodes,
            bottleneck_type=bottleneck_type,
            bottleneck_summary=bottleneck_summary,
            llm_diagnosis_prompt="",  # Filled in below
            raw_plan=plan_dict,
        )

        result.llm_diagnosis_prompt = _build_diagnosis_prompt(result, schema_ddl)
        return result

    def batch_analyze(
        self,
        plan_records: list[dict],
        schema_field: str = "schema_ddl",
        plan_field: str = "explain_json",
    ) -> list[ExplainAnalysisResult]:
        """
        Analyze a batch of records containing JSON EXPLAIN plans.

        Args:
            plan_records: List of dicts, each with a JSON plan field.
            schema_field: Key in each dict containing schema DDL.
            plan_field: Key in each dict containing the EXPLAIN JSON text.

        Returns:
            List of ExplainAnalysisResult objects (None entries filtered out).
        """
        results = []
        for record in plan_records:
            plan_text = record.get(plan_field, "")
            schema_ddl = record.get(schema_field, "")
            if not plan_text:
                continue
            result = self.analyze(plan_text, schema_ddl=schema_ddl or None)
            if result is not None:
                results.append(result)
        return results


def analyze_from_file(
    plan_file: str,
    schema_ddl: Optional[str] = None,
    verbose: bool = False,
) -> Optional[ExplainAnalysisResult]:
    """Convenience function: analyze a EXPLAIN JSON file."""
    with open(plan_file) as f:
        plan_text = f.read()
    analyzer = ExplainJSONAnalyzer()
    result = analyzer.analyze(plan_text, schema_ddl=schema_ddl)
    if result and verbose:
        print(f"Execution time:   {result.execution_time_ms:.1f}ms")
        print(f"Bottleneck type:  {result.bottleneck_type}")
        print(f"Summary:          {result.bottleneck_summary}")
        print(f"Seq scans:        {len(result.seq_scans)}")
        print(f"Bad estimates:    {len(result.bad_row_estimates)}")
        print(f"Expensive sorts:  {len(result.expensive_sorts)}")
        print(f"Heap fetch heavy: {len(result.heap_fetch_heavy)}")
        print(f"Disk heavy:       {len(result.disk_heavy_nodes)}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze PostgreSQL EXPLAIN JSON output"
    )
    parser.add_argument("plan_file", help="Path to EXPLAIN (FORMAT JSON) output file")
    parser.add_argument("--schema", default=None, help="Path to schema DDL file")
    parser.add_argument(
        "--prompt", action="store_true", help="Print LLM diagnosis prompt"
    )
    args = parser.parse_args()

    schema_ddl = None
    if args.schema:
        with open(args.schema) as f:
            schema_ddl = f.read()

    result = analyze_from_file(args.plan_file, schema_ddl=schema_ddl, verbose=True)
    if result and args.prompt:
        print("\n" + "─" * 60)
        print("LLM DIAGNOSIS PROMPT:")
        print("─" * 60)
        print(result.llm_diagnosis_prompt)
