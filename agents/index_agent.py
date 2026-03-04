"""
index_agent.py - Proposes optimal index strategy from query diagnosis.

The Index Agent:
1. Receives QueryDiagnosis from the Query Analyzer
2. Selects the correct index type (B-tree, GIN, GiST, BRIN, Hash, partial, covering)
3. Determines optimal column order for multi-column indexes
4. Estimates write amplification and storage cost
5. Produces DDL with full rationale

Usage:
    from agents.index_agent import IndexAgent
    agent = IndexAgent()
    proposal = agent.propose(query, schema_ddl, diagnosis)
"""

import os
import re
from dataclasses import dataclass, field

import anthropic
from loguru import logger

from agents.query_analyzer_agent import QueryDiagnosis
from core.postgres_internals import (
    IndexRecommendation,
    IndexType,
    ANTIPATTERNS,
    PlannerKnowledge,
    INDEX_PROFILES,
)
from core.mysql_internals import estimate_index_size_mb


@dataclass
class IndexProposal:
    """Complete index proposal with DDL and rationale."""
    recommendations: list[IndexRecommendation]
    primary_ddl: str           # The most important index DDL
    all_ddl: list[str]         # All recommended DDL statements
    rationale: str             # LLM-generated explanation
    write_amplification: str   # Impact on writes
    storage_estimate_mb: float
    antipattern_detected: str | None
    confidence: str            # "high" | "medium" | "low"
    engine: str


INDEX_SYSTEM = """You are QueryMedic's Index Strategist — a database indexing expert.

Given a slow query, its schema, and a performance diagnosis, recommend the optimal index strategy.

Rules:
1. ALWAYS specify the correct index type (B-tree, GIN, GiST, BRIN, Hash, partial, covering)
2. For multi-column B-tree indexes: equality predicates first, then range, then ORDER BY
3. For JSONB containment (@>) or array overlap (&&): GIN is mandatory
4. For very large append-only tables with monotonic columns: consider BRIN (tiny size)
5. For partial indexes: only when WHERE clause filters >50% of rows
6. For covering indexes (INCLUDE): add SELECT columns to avoid heap lookups
7. ALWAYS estimate write amplification — reads are free, writes pay

Format your response:
## Index Type Rationale
[Why this specific index type]

## Column Order Rationale
[Why this column order]

## DDL
```sql
[CREATE INDEX statement]
```

## Write Amplification
[Impact on INSERT/UPDATE/DELETE throughput]

## Storage Estimate
[Rough MB estimate with reasoning]"""


class IndexAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    def propose(
        self,
        query: str,
        schema_ddl: str,
        diagnosis: QueryDiagnosis,
        table_stats: dict | None = None,
    ) -> IndexProposal:
        """Propose optimal index strategy based on diagnosis."""

        # Detect antipatterns that indexes can't fix
        antipattern = self._detect_antipattern(query)
        if antipattern:
            logger.warning(f"Antipattern detected: {antipattern}")

        prompt = self._build_prompt(query, schema_ddl, diagnosis, table_stats, antipattern)

        logger.info(f"Proposing index strategy (engine={diagnosis.engine}, bottleneck={diagnosis.bottleneck_type})")

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=INDEX_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text

        # Extract DDL from response
        ddl_blocks = re.findall(r"```sql\n(.*?)```", text, re.DOTALL)
        primary_ddl = ddl_blocks[0].strip() if ddl_blocks else ""
        all_ddl = [b.strip() for b in ddl_blocks]

        # Build structured recommendations from the parsed DDL
        recs = self._parse_recommendations(all_ddl, diagnosis.engine)

        # Estimate storage
        storage_mb = self._estimate_storage(recs, table_stats)

        # Extract write amplification section
        wa_text = self._extract_section(text, "write amplification") or "See rationale."

        return IndexProposal(
            recommendations=recs,
            primary_ddl=primary_ddl,
            all_ddl=all_ddl,
            rationale=text,
            write_amplification=wa_text,
            storage_estimate_mb=storage_mb,
            antipattern_detected=antipattern,
            confidence=self._assess_confidence(diagnosis),
            engine=diagnosis.engine,
        )

    def _build_prompt(
        self,
        query: str,
        schema_ddl: str,
        diagnosis: QueryDiagnosis,
        table_stats: dict | None,
        antipattern: str | None,
    ) -> str:
        parts = [f"## Query\n```sql\n{query[:2000]}\n```"]

        if schema_ddl:
            parts.append(f"## Schema\n```sql\n{schema_ddl[:3000]}\n```")

        parts.append(f"## Diagnosis\n{diagnosis.plan_summary}")
        parts.append(f"- Bottleneck: {diagnosis.bottleneck_type}")
        parts.append(f"- Root cause: {diagnosis.root_cause[:500]}")
        parts.append(f"- Needs index: {diagnosis.needs_index}")
        parts.append(f"- Engine: {diagnosis.engine}")

        if diagnosis.row_estimation_error:
            parts.append(f"- Row estimation errors: {diagnosis.row_estimation_error}")

        if table_stats:
            parts.append(f"## Table Stats\n{table_stats}")

        if antipattern:
            ap_info = ANTIPATTERNS.get(antipattern, {})
            parts.append(
                f"## Antipattern Warning\n"
                f"Pattern: {ap_info.get('pattern', antipattern)}\n"
                f"Diagnosis: {ap_info.get('diagnosis', '')}\n"
                f"Fix: {ap_info.get('fix', '')}"
            )

        # Inject PostgreSQL index type knowledge if relevant
        if diagnosis.engine == "postgresql":
            parts.append(self._build_index_type_guide(query, schema_ddl))

        parts.append(
            "Recommend the optimal index strategy. "
            "If an antipattern is detected, recommend the structural fix instead of (or in addition to) a new index."
        )
        return "\n\n".join(parts)

    def _build_index_type_guide(self, query: str, schema_ddl: str) -> str:
        """Build a targeted index type guide based on query patterns."""
        guide_parts = ["## Relevant Index Type Knowledge"]

        query_lower = query.lower()
        schema_lower = schema_ddl.lower()

        if "@>" in query or "jsonb" in schema_lower or "?" in query:
            p = INDEX_PROFILES[IndexType.GIN]
            guide_parts.append(f"GIN: {p.notes} Write cost: {p.write_cost_factor}x")

        if any(t in schema_lower for t in ["geometry", "geography", "tsrange", "int4range"]):
            p = INDEX_PROFILES[IndexType.GIST]
            guide_parts.append(f"GiST: {p.notes} Write cost: {p.write_cost_factor}x")

        if any(col in query_lower for col in ["created_at", "inserted_at", "timestamp"]):
            p = INDEX_PROFILES[IndexType.BRIN]
            guide_parts.append(f"BRIN: {p.notes} Write cost: {p.write_cost_factor}x")

        if len(guide_parts) == 1:
            p = INDEX_PROFILES[IndexType.BTREE]
            guide_parts.append(f"B-tree: {p.notes}")

        return "\n".join(guide_parts)

    def _detect_antipattern(self, query: str) -> str | None:
        """Detect SQL antipatterns that indexes can't fix."""
        q = query.lower()

        # QM-15: Regex semantics documented below.
        # r"like\s+'%[^%]" — matches LIKE '%<non-percent>' (leading wildcard).
        #   [^%] ensures we don't match LIKE '%%' (escaped percent literal).
        if re.search(r"like\s+'%[^%]", q):
            return "leading_wildcard"

        # r"(lower|upper|date|extract|to_char)\s*\(" — function call wrapping
        #   an indexed column makes the predicate non-sargable.
        if re.search(r"(lower|upper|date|extract|to_char)\s*\(", q):
            return "function_on_indexed_column"

        # r"not\s+in\s*\(\s*select" — NOT IN (subquery) — anti-join that may
        #   materialise a large intermediate result set.
        if re.search(r"not\s+in\s*\(\s*select", q):
            return "not_in_large_list"

        # OR on potentially different indexed columns — PostgreSQL cannot use
        # a single B-tree index for OR across different columns; UNION ALL is
        # usually a better plan.
        if " or " in q and re.search(r"\w+\s*=.*\s+or\s+\w+\s*=", q):
            return "or_on_indexed_columns"

        return None

    def _parse_recommendations(self, ddl_blocks: list[str], engine: str) -> list[IndexRecommendation]:
        """Parse DDL blocks into structured IndexRecommendation objects."""
        recs = []
        for ddl in ddl_blocks:
            rec = self._parse_single_ddl(ddl, engine)
            if rec:
                recs.append(rec)
        return recs

    def _parse_single_ddl(self, ddl: str, engine: str) -> IndexRecommendation | None:
        """Parse a single CREATE INDEX statement."""
        # Match: CREATE [UNIQUE] INDEX [CONCURRENTLY] name ON table USING method(cols) [INCLUDE (inc)] [WHERE pred]
        m = re.match(
            r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(\w+)\s+ON\s+(\w+)\s+USING\s+(\w+)\s*\(([^)]+)\)"
            r"(?:\s+INCLUDE\s*\(([^)]+)\))?"
            r"(?:\s+WHERE\s+(.+?))?;?$",
            ddl.strip(),
            re.IGNORECASE | re.DOTALL,
        )
        if not m:
            # Simpler pattern without USING
            m2 = re.match(
                r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(\w+)\s+ON\s+(\w+)\s*\(([^)]+)\)",
                ddl.strip(),
                re.IGNORECASE,
            )
            if not m2:
                return None
            table = m2.group(2)
            cols = [c.strip() for c in m2.group(3).split(",")]
            return IndexRecommendation(
                index_type=IndexType.BTREE,
                table=table,
                columns=cols,
                ddl=ddl,
            )

        table = m.group(2)
        method = m.group(3).upper()
        cols = [c.strip() for c in m.group(4).split(",")]
        include_cols = [c.strip() for c in m.group(5).split(",")] if m.group(5) else []
        partial_pred = m.group(6).strip() if m.group(6) else None

        idx_type_map = {
            "BTREE": IndexType.BTREE, "GIN": IndexType.GIN,
            "GIST": IndexType.GIST, "BRIN": IndexType.BRIN,
            "HASH": IndexType.HASH, "SPGIST": IndexType.SPGIST,
        }
        idx_type = idx_type_map.get(method, IndexType.BTREE)

        rec = IndexRecommendation(
            index_type=idx_type,
            table=table,
            columns=cols,
            include_columns=include_cols,
            partial_predicate=partial_pred,
            concurrent=True,
            ddl=ddl,
        )
        return rec

    def _estimate_storage(self, recs: list[IndexRecommendation], table_stats: dict | None) -> float:
        """Rough storage estimate across all recommended indexes."""
        if not table_stats or not recs:
            return 0.0

        row_count = table_stats.get("row_count", 100_000)
        total_mb = 0.0

        for rec in recs:
            profile = INDEX_PROFILES.get(rec.index_type, INDEX_PROFILES[IndexType.BTREE])
            # Rough: 50 bytes per row for key columns * storage_factor
            key_bytes = len(rec.columns) * 12  # ~12 bytes per column key
            base_mb = estimate_index_size_mb(row_count, key_bytes)
            total_mb += base_mb * profile.storage_factor

        return round(total_mb, 1)

    def _assess_confidence(self, diagnosis: QueryDiagnosis) -> str:
        """Assess confidence level of the index recommendation."""
        if diagnosis.bottleneck_type in ("seq_scan", "missing_index"):
            return "high"
        if diagnosis.bottleneck_type in ("stale_stats", "bad_join"):
            return "medium"
        return "low"

    def _extract_section(self, text: str, section: str) -> str | None:
        m = re.search(
            rf"(?:#{1,3}\s*)?{re.escape(section)}[:\s]*(.*?)(?=\n#{1,3}|\Z)",
            text, re.IGNORECASE | re.DOTALL
        )
        return m.group(1).strip()[:500] if m else None
