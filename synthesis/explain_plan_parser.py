"""
explain_plan_parser.py - Parse PostgreSQL and MySQL EXPLAIN ANALYZE outputs.

Extracts structured information from EXPLAIN plans for training pair generation:
  - Plan nodes and their costs
  - Row estimation errors
  - Actual vs estimated timing
  - Buffer hit ratios
  - Sort, Hash, and Aggregate overhead
  - Sequential scan patterns

Usage:
    from synthesis.explain_plan_parser import ExplainPlanParser
    parser = ExplainPlanParser()
    plan = parser.parse(explain_text, engine="postgresql")
    print(plan.bottleneck_type)
    print(plan.row_estimation_errors)
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class PlanNodeType(Enum):
    SEQ_SCAN = "Seq Scan"
    INDEX_SCAN = "Index Scan"
    INDEX_ONLY_SCAN = "Index Only Scan"
    BITMAP_HEAP_SCAN = "Bitmap Heap Scan"
    BITMAP_INDEX_SCAN = "Bitmap Index Scan"
    HASH_JOIN = "Hash Join"
    MERGE_JOIN = "Merge Join"
    NESTED_LOOP = "Nested Loop"
    SORT = "Sort"
    HASH = "Hash"
    HASH_AGGREGATE = "HashAggregate"
    GROUP_AGGREGATE = "GroupAggregate"
    AGGREGATE = "Aggregate"
    LIMIT = "Limit"
    CTE_SCAN = "CTE Scan"


@dataclass
class PlanNode:
    """A single node in an EXPLAIN plan."""
    node_type: str
    relation: str | None = None
    index_name: str | None = None
    estimated_cost: float = 0.0
    estimated_rows: int = 0
    actual_time_ms: float = 0.0
    actual_rows: int = 0
    loops: int = 1
    rows_removed_by_filter: int = 0
    heap_fetches: int = 0
    buffers_hit: int = 0
    buffers_read: int = 0
    children: list["PlanNode"] = field(default_factory=list)
    depth: int = 0

    @property
    def row_estimation_ratio(self) -> float | None:
        """actual_rows / estimated_rows — large deviation indicates stale stats."""
        if self.estimated_rows == 0:
            return None
        return self.actual_rows / self.estimated_rows

    @property
    def is_estimation_error(self) -> bool:
        """True if row count estimation is off by more than 10x."""
        ratio = self.row_estimation_ratio
        return ratio is not None and (ratio > 10 or ratio < 0.1)

    @property
    def total_actual_time_ms(self) -> float:
        return self.actual_time_ms * self.loops

    @property
    def buffer_cache_hit_rate(self) -> float | None:
        total = self.buffers_hit + self.buffers_read
        if total == 0:
            return None
        return self.buffers_hit / total


@dataclass
class ExplainPlan:
    """Structured representation of an EXPLAIN ANALYZE plan."""
    engine: str
    root_node: PlanNode | None = None
    all_nodes: list[PlanNode] = field(default_factory=list)
    planning_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    raw_text: str = ""

    @property
    def bottleneck_type(self) -> str:
        """Identify the primary performance bottleneck."""
        nodes_by_type: dict[str, list[PlanNode]] = {}
        for node in self.all_nodes:
            nodes_by_type.setdefault(node.node_type, []).append(node)

        # Sequential scan on large table
        for node in self.all_nodes:
            if "Seq Scan" in node.node_type and node.rows_removed_by_filter > 100_000:
                return "seq_scan_large_filter"
            if "Seq Scan" in node.node_type and node.is_estimation_error:
                return "seq_scan_stale_stats"

        # Bad join ordering
        for node in self.all_nodes:
            if "Nested Loop" in node.node_type and node.actual_rows > 10_000:
                return "nested_loop_large_outer"

        # Sort overhead
        sort_nodes = [n for n in self.all_nodes if "Sort" in n.node_type]
        if sort_nodes and sort_nodes[0].actual_time_ms > 1000:
            return "sort_overhead"

        # Bitmap heap scan with heap fetches
        for node in self.all_nodes:
            if node.node_type == "Bitmap Heap Scan" and node.heap_fetches > 1000:
                return "heap_fetch_overhead"

        return "general_performance"

    @property
    def row_estimation_errors(self) -> list[tuple[str, int, int, float]]:
        """List of (relation, estimated, actual, ratio) for significant estimation errors."""
        errors = []
        for node in self.all_nodes:
            if node.is_estimation_error and node.relation:
                ratio = node.row_estimation_ratio or 0.0
                errors.append((node.relation, node.estimated_rows, node.actual_rows, ratio))
        return errors

    @property
    def seq_scans(self) -> list[PlanNode]:
        return [n for n in self.all_nodes if "Seq Scan" in n.node_type]

    @property
    def index_scans(self) -> list[PlanNode]:
        return [n for n in self.all_nodes if "Index" in n.node_type]

    def format_diagnosis(self) -> str:
        """Generate a human-readable diagnosis."""
        parts = []
        parts.append(f"Execution time: {self.execution_time_ms:.1f}ms")

        for seq in self.seq_scans:
            if seq.rows_removed_by_filter > 0:
                parts.append(
                    f"Sequential scan on {seq.relation}: "
                    f"filtered {seq.rows_removed_by_filter:,} rows to find {seq.actual_rows:,}"
                )

        for rel, est, actual, ratio in self.row_estimation_errors:
            parts.append(
                f"Row estimation error on {rel}: "
                f"planner expected {est:,} rows, got {actual:,} ({ratio:.1f}x off)"
            )

        return "\n".join(parts)


class ExplainPlanParser:
    """Parses PostgreSQL and MySQL EXPLAIN ANALYZE output."""

    def parse(self, text: str, engine: str = "postgresql") -> ExplainPlan:
        """Parse an EXPLAIN ANALYZE output into a structured plan."""
        text = text.strip()
        plan = ExplainPlan(engine=engine, raw_text=text)

        if engine == "postgresql":
            self._parse_postgresql(text, plan)
        elif engine in ("mysql", "mariadb"):
            self._parse_mysql(text, plan)

        return plan

    def _parse_postgresql(self, text: str, plan: ExplainPlan) -> None:
        """Parse PostgreSQL EXPLAIN ANALYZE text output."""
        # Planning/execution time
        plan_m = re.search(r"Planning Time:\s*([\d.]+)\s*ms", text)
        exec_m = re.search(r"Execution Time:\s*([\d.]+)\s*ms", text)
        if plan_m:
            plan.planning_time_ms = float(plan_m.group(1))
        if exec_m:
            plan.execution_time_ms = float(exec_m.group(1))

        # Parse plan nodes line by line
        node_pattern = re.compile(
            r"^(\s*->\s*|\s*)([\w ]+?)\s+(?:on\s+(\w+)\s+)?(?:using\s+(\w+)\s+)?(?:on\s+\w+\s+)?"
            r"\(cost=([\d.]+)\.\.([\d.]+)\s+rows=(\d+)\s+width=(\d+)\)"
            r"(?:\s+\(actual time=([\d.]+)\.\.([\d.]+)\s+rows=(\d+)\s+loops=(\d+)\))?",
        )
        rows_removed_pattern = re.compile(r"Rows Removed by Filter:\s*(\d+)")
        heap_fetch_pattern = re.compile(r"Heap Fetches:\s*(\d+)")
        buffers_pattern = re.compile(r"Buffers:\s*shared\s+hit=(\d+)(?:\s+read=(\d+))?")

        current_node: PlanNode | None = None
        for line in text.splitlines():
            m = node_pattern.match(line)
            if m:
                indent = len(line) - len(line.lstrip())
                node = PlanNode(
                    node_type=m.group(2).strip(),
                    relation=m.group(3),
                    index_name=m.group(4),
                    estimated_cost=float(m.group(6)) if m.group(6) else 0.0,
                    estimated_rows=int(m.group(7)) if m.group(7) else 0,
                    actual_time_ms=float(m.group(10)) if m.group(10) else 0.0,
                    actual_rows=int(m.group(11)) if m.group(11) else 0,
                    loops=int(m.group(12)) if m.group(12) else 1,
                    depth=indent // 2,
                )
                plan.all_nodes.append(node)
                current_node = node
                if plan.root_node is None:
                    plan.root_node = node

            elif current_node:
                rm = rows_removed_pattern.search(line)
                if rm:
                    current_node.rows_removed_by_filter = int(rm.group(1))
                hm = heap_fetch_pattern.search(line)
                if hm:
                    current_node.heap_fetches = int(hm.group(1))
                bm = buffers_pattern.search(line)
                if bm:
                    current_node.buffers_hit = int(bm.group(1))
                    current_node.buffers_read = int(bm.group(2)) if bm.group(2) else 0

    def _parse_mysql(self, text: str, plan: ExplainPlan) -> None:
        """Parse MySQL EXPLAIN tabular output.

        QM-17: Use the header row to locate column indices rather than hard-coding
        fixed positions. MySQL EXPLAIN column order can differ between MySQL 5.7
        and 8.0, and between EXPLAIN and EXPLAIN FORMAT=TRADITIONAL.
        """
        header_cols: list[str] = []
        for line in text.splitlines():
            if "|" not in line:
                continue
            cols = [c.strip() for c in line.split("|") if c.strip()]
            if not cols:
                continue
            if cols[0].lower() == "id":
                # This is the header row — capture column names
                header_cols = [c.lower() for c in cols]
                continue
            if not cols[0].isdigit():
                continue  # Separator line or non-data row
            if not header_cols:
                continue  # No header seen yet

            def _col(name: str, default: str = "") -> str:
                try:
                    return cols[header_cols.index(name)]
                except (ValueError, IndexError):
                    return default

            node_type = _col("type", "ALL")
            relation = _col("table", "")
            rows_val = _col("rows", "0")
            rows_est = int(rows_val) if rows_val.isdigit() else 0
            node = PlanNode(
                node_type=f"MySQL {node_type}",
                relation=relation,
                estimated_rows=rows_est,
            )
            plan.all_nodes.append(node)
            if plan.root_node is None:
                plan.root_node = node


def detect_engine(text: str) -> str:
    """Auto-detect database engine from EXPLAIN output."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["planning time:", "seq scan", "index only scan", "bitmap"]):
        return "postgresql"
    if any(kw in text_lower for kw in ["using filesort", "using temporary", "using index condition"]):
        return "mysql"
    if "sqlite_stat" in text_lower or "search table" in text_lower:
        return "sqlite"
    return "postgresql"
