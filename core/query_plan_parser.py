"""
query_plan_parser.py - Unified query plan parser (delegates to engine-specific parsers).

Convenience module that auto-detects engine and returns a structured plan object.
"""

from synthesis.explain_plan_parser import ExplainPlan, ExplainPlanParser, detect_engine


def parse_any_plan(text: str, engine: str | None = None) -> ExplainPlan:
    """
    Parse an EXPLAIN ANALYZE output from any supported engine.

    Args:
        text: Raw EXPLAIN output text
        engine: "postgresql", "mysql", or "sqlite" (auto-detected if None)

    Returns:
        ExplainPlan with structured node data
    """
    if engine is None:
        engine = detect_engine(text)

    parser = ExplainPlanParser()
    return parser.parse(text, engine=engine)


def plan_to_training_context(plan: ExplainPlan) -> dict:
    """Convert a parsed plan into training context for synthesis."""
    return {
        "engine": plan.engine,
        "execution_time_ms": plan.execution_time_ms,
        "bottleneck_type": plan.bottleneck_type,
        "seq_scans": [
            {"relation": n.relation, "rows_filtered": n.rows_removed_by_filter, "actual_rows": n.actual_rows}
            for n in plan.seq_scans
        ],
        "index_scans": [
            {"relation": n.relation, "index": n.index_name, "actual_rows": n.actual_rows}
            for n in plan.index_scans
        ],
        "row_estimation_errors": [
            {"relation": r, "estimated": est, "actual": act, "ratio": ratio}
            for r, est, act, ratio in plan.row_estimation_errors
        ],
        "diagnosis": plan.format_diagnosis(),
    }
