"""
rewrite_agent.py - Rewrites SQL queries for better execution plans.

The Rewrite Agent:
1. Receives QueryDiagnosis and the original query
2. Identifies structural issues causing bad plans (NOT IN, OR, correlated subqueries)
3. Rewrites the query using semantically equivalent but plan-friendly SQL
4. Documents why the rewrite produces a better plan

Rewrite strategies:
- NOT IN → NOT EXISTS or LEFT JOIN ... IS NULL
- OR on different columns → UNION ALL
- Correlated subquery → JOIN
- LIKE '%val' → Full-text search rewrite hint
- Non-sargable predicates → Sargable equivalents
- Suboptimal join order hints

Usage:
    from agents.rewrite_agent import RewriteAgent
    agent = RewriteAgent()
    result = agent.rewrite(query, diagnosis, schema_ddl)
"""

import os
import re
from dataclasses import dataclass

import anthropic
from anthropic.types import TextBlock
from loguru import logger

from agents.query_analyzer_agent import QueryDiagnosis


@dataclass
class RewriteResult:
    """Result of a query rewrite."""

    original_query: str
    rewritten_query: str
    rewrite_type: str  # "not_exists", "union", "join", "sargable", "cte", "none"
    plan_improvement: str  # Expected plan change
    semantics_preserved: bool  # Whether we're confident semantics match
    rationale: str
    confidence: str  # "high" | "medium" | "low"
    applied: bool  # Whether a rewrite was actually applied


REWRITE_SYSTEM = """You are QueryMedic's Query Rewriter — a SQL optimization expert.

Given a slow query and its performance diagnosis, rewrite the query to produce a better execution plan.

Rewrite strategies to apply:
1. NOT IN → NOT EXISTS (avoids anti-join issues with NULLs and large sets)
   BEFORE: WHERE id NOT IN (SELECT id FROM t WHERE ...)
   AFTER:  WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.id = main.id AND ...)

2. OR on indexed columns → UNION ALL
   BEFORE: WHERE col1 = $1 OR col2 = $2
   AFTER:  SELECT ... WHERE col1 = $1
           UNION ALL
           SELECT ... WHERE col2 = $2 AND col1 != $1

3. Correlated subquery → JOIN
   BEFORE: SELECT *, (SELECT COUNT(*) FROM t2 WHERE t2.fk = t1.id) AS cnt
   AFTER:  SELECT t1.*, COALESCE(sub.cnt, 0) FROM t1
           LEFT JOIN (SELECT fk, COUNT(*) AS cnt FROM t2 GROUP BY fk) sub ON sub.fk = t1.id

4. Non-sargable → sargable
   BEFORE: WHERE DATE(created_at) = '2024-01-01'
   AFTER:  WHERE created_at >= '2024-01-01' AND created_at < '2024-01-02'

5. OFFSET pagination → keyset pagination
   BEFORE: SELECT ... ORDER BY id LIMIT 20 OFFSET 10000
   AFTER:  SELECT ... WHERE id > $last_id ORDER BY id LIMIT 20

CRITICAL: Preserve exact query semantics. If you can't guarantee equivalence, say so.

Format:
## Rewrite Type
[Which strategy was applied]

## Rewritten Query
```sql
[The rewritten query]
```

## Why This Produces a Better Plan
[Specific plan improvement expected]

## Semantic Equivalence Note
[Confirm or caveat about semantic preservation]"""


class RewriteAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )

    def rewrite(
        self,
        query: str,
        diagnosis: QueryDiagnosis,
        schema_ddl: str = "",
    ) -> RewriteResult:
        """Rewrite query for better execution plan."""

        # Fast-path: if no rewrite needed per diagnosis, return original
        if not diagnosis.needs_rewrite:
            rewrite_type = self._detect_structural_issue(query)
            if not rewrite_type:
                logger.info("No rewrite needed — query structure is sound")
                return RewriteResult(
                    original_query=query,
                    rewritten_query=query,
                    rewrite_type="none",
                    plan_improvement="No structural issues detected",
                    semantics_preserved=True,
                    rationale="Query structure is already optimal for the planner.",
                    confidence="high",
                    applied=False,
                )
        else:
            rewrite_type = self._detect_structural_issue(query)

        logger.info(f"Rewriting query (detected issue: {rewrite_type or 'unknown'})")

        prompt = self._build_prompt(query, diagnosis, schema_ddl, rewrite_type)

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            system=REWRITE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        first_block = resp.content[0]
        text = first_block.text if isinstance(first_block, TextBlock) else ""

        rewritten = self._extract_sql(text) or query
        applied = rewritten.strip() != query.strip()

        return RewriteResult(
            original_query=query,
            rewritten_query=rewritten,
            rewrite_type=self._extract_section(text, "rewrite type")
            or rewrite_type
            or "unknown",
            plan_improvement=self._extract_section(
                text, "why this produces a better plan"
            )
            or "",
            semantics_preserved="equivalent" in text.lower()
            or "preserves" in text.lower(),
            rationale=text,
            confidence=self._assess_confidence(rewrite_type, diagnosis),
            applied=applied,
        )

    def _build_prompt(
        self,
        query: str,
        diagnosis: QueryDiagnosis,
        schema_ddl: str,
        rewrite_type: str | None,
    ) -> str:
        parts = [f"## Query\n```sql\n{query[:3000]}\n```"]

        if schema_ddl:
            parts.append(f"## Schema\n```sql\n{schema_ddl[:2000]}\n```")

        parts.append(
            f"## Diagnosis\n"
            f"- Bottleneck: {diagnosis.bottleneck_type}\n"
            f"- Root cause: {diagnosis.root_cause[:400]}\n"
            f"- Needs rewrite: {diagnosis.needs_rewrite}\n"
            f"- Engine: {diagnosis.engine}"
        )

        if rewrite_type:
            parts.append(
                f"## Detected Issue\nStructural issue detected: `{rewrite_type}`"
            )

        if diagnosis.row_estimation_error:
            parts.append(f"## Row Estimation Errors\n{diagnosis.row_estimation_error}")

        parts.append(
            "Rewrite this query to produce a better execution plan. "
            "If no structural rewrite is needed, say so explicitly and suggest query-level hints instead."
        )
        return "\n\n".join(parts)

    def _detect_structural_issue(self, query: str) -> str | None:
        """Detect structural query issues that benefit from rewriting."""
        q = query.lower()

        if re.search(r"not\s+in\s*\(\s*select", q):
            return "not_in_subquery"

        if re.search(r"\boffset\s+\d{4,}\b", q):
            return "deep_offset_pagination"

        # QM-18: The original condition required either a second WHERE clause
        # (which never appears in a single statement) OR at least two " or "
        # occurrences. Both conditions missed the common single-OR case such as
        # "WHERE col1 = $1 OR col2 = $2". A single OR across different indexed
        # columns is already a candidate for a UNION ALL rewrite.
        if "where" in q and re.search(r"\w+\s*=\s*\S+.*?\bor\b.*?\w+\s*=", q):
            return "or_multiple_columns"

        # Correlated subquery in SELECT
        if re.search(r"select\s+[^(]*\(\s*select\s+", q):
            return "correlated_subquery_in_select"

        # Non-sargable date function
        if re.search(r"(date|date_trunc|extract)\s*\(\s*", q):
            return "non_sargable_date"

        # Leading wildcard
        if re.search(r"like\s+'%[^%']+[^%]'", q):
            return "leading_wildcard"

        return None

    def _extract_sql(self, text: str) -> str | None:
        """Extract SQL from response."""
        # Look for the rewritten query block specifically
        m = re.search(
            r"(?:rewritten query|after)[^\n]*\n```sql\n(.*?)```",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(1).strip()

        # Fallback: any SQL block
        blocks = re.findall(r"```sql\n(.*?)```", text, re.DOTALL)
        if blocks:
            return blocks[-1].strip()  # Last block is usually the rewritten version

        return None

    def _assess_confidence(
        self, rewrite_type: str | None, diagnosis: QueryDiagnosis
    ) -> str:
        """Assess confidence in the rewrite."""
        high_confidence_rewrites = {
            "not_in_subquery",
            "deep_offset_pagination",
            "non_sargable_date",
        }
        medium_confidence_rewrites = {
            "or_multiple_columns",
            "correlated_subquery_in_select",
        }

        if rewrite_type in high_confidence_rewrites:
            return "high"
        if rewrite_type in medium_confidence_rewrites:
            return "medium"
        if diagnosis.needs_rewrite:
            return "medium"
        return "low"

    def _extract_section(self, text: str, section: str) -> str | None:
        m = re.search(
            rf"(?:#{1, 3}\s*)?{re.escape(section)}[:\s]*(.*?)(?=\n#{1, 3}|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        return m.group(1).strip()[:500] if m else None
