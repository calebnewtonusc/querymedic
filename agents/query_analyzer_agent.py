"""
query_analyzer_agent.py - Reads EXPLAIN plans and diagnoses bottlenecks.

The Query Analyzer Agent:
1. Parses the EXPLAIN ANALYZE output
2. Identifies the dominant bottleneck (seq scan, bad join, stale stats, etc.)
3. Reads the query and schema context
4. Produces a structured diagnosis for the Index and Rewrite Agents

Usage:
    from agents.query_analyzer_agent import QueryAnalyzerAgent
    agent = QueryAnalyzerAgent()
    diagnosis = agent.analyze(query, explain_output, schema_ddl)
"""

import os
from dataclasses import dataclass

import anthropic
from loguru import logger

from synthesis.explain_plan_parser import ExplainPlanParser, ExplainPlan


@dataclass
class QueryDiagnosis:
    """Structured diagnosis from query plan analysis."""

    bottleneck_type: str
    dominant_node: str
    root_cause: str
    row_estimation_error: dict | None  # {"estimated": X, "actual": Y, "ratio": Z}
    plan_summary: str
    engine: str
    table_stats: dict | None
    needs_analyze: bool
    needs_index: bool
    needs_rewrite: bool


ANALYSIS_SYSTEM = """You are QueryMedic's Query Analyzer — a database internals expert.

Given a slow query, its EXPLAIN ANALYZE output, and schema context, produce a structured diagnosis.

Focus on:
1. The SPECIFIC plan node causing the most time (not "it's slow")
2. Row estimation accuracy (planner's estimate vs actual — large discrepancy = stale stats)
3. Whether a sequential scan is justified or a missing index
4. Join strategy issues (wrong join algorithm for this data volume)
5. Whether the query structure itself is causing a bad plan

Be precise: cite cost numbers, actual row counts, and specific plan nodes."""


class QueryAnalyzerAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.parser = ExplainPlanParser()

    def analyze(
        self,
        query: str,
        explain_output: str,
        schema_ddl: str = "",
        table_stats: dict | None = None,
        engine: str = "postgresql",
    ) -> QueryDiagnosis:
        plan = self.parser.parse(explain_output, engine=engine)
        prompt = self._build_prompt(
            query, explain_output, schema_ddl, table_stats, plan
        )

        logger.info(
            f"Analyzing query plan ({plan.execution_time_ms:.0f}ms, {len(plan.all_nodes)} nodes)"
        )

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=ANALYSIS_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text

        return QueryDiagnosis(
            bottleneck_type=plan.bottleneck_type,
            # QM-10: all_nodes[0] is the root node (parse order), not the
            # slowest node. Use the node with the highest actual_time_ms so
            # that the diagnosis correctly names the dominant bottleneck.
            dominant_node=(
                max(plan.all_nodes, key=lambda n: n.actual_time_ms).node_type
                if plan.all_nodes
                else "unknown"
            ),
            root_cause=self._extract_section(text, "root cause") or text[:200],
            row_estimation_error={
                "errors": [
                    {"relation": r, "estimated": e, "actual": a, "ratio": ratio}
                    for r, e, a, ratio in plan.row_estimation_errors
                ]
            }
            if plan.row_estimation_errors
            else None,
            plan_summary=plan.format_diagnosis(),
            engine=engine,
            table_stats=table_stats,
            needs_analyze=bool(plan.row_estimation_errors),
            needs_index=bool(plan.seq_scans),
            needs_rewrite="rewrite" in text.lower() or "join order" in text.lower(),
        )

    def _build_prompt(
        self, query, explain_output, schema_ddl, table_stats, plan: ExplainPlan
    ) -> str:
        parts = [f"## Query\n```sql\n{query[:2000]}\n```"]
        if schema_ddl:
            parts.append(f"## Schema\n```sql\n{schema_ddl[:3000]}\n```")
        if table_stats:
            parts.append(f"## Table Stats\n{table_stats}")
        parts.append(f"## EXPLAIN ANALYZE\n```\n{explain_output[:5000]}\n```")
        parts.append(f"## Parsed Summary\n{plan.format_diagnosis()}")
        parts.append(
            "Diagnose the performance bottleneck. Be specific about plan nodes and costs."
        )
        return "\n\n".join(parts)

    def _extract_section(self, text: str, section: str) -> str | None:
        import re

        m = re.search(
            rf"(?:#{1, 3}\s*)?{re.escape(section)}[:\s]*(.*?)(?=\n#{1, 3}|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        return m.group(1).strip()[:500] if m else None
