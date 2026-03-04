"""
validation_agent.py - Validates index and rewrite recommendations via live EXPLAIN ANALYZE.

The Validation Agent:
1. Runs EXPLAIN ANALYZE on the original query
2. Applies the recommended index (CREATE INDEX CONCURRENTLY)
3. Runs ANALYZE to update planner statistics
4. Runs EXPLAIN ANALYZE on the optimized query (original or rewritten)
5. Compares timing — requires ≥20% improvement to APPROVE

This is the ground-truth verification layer. The model's recommendations
only count if the database clock confirms improvement.

Usage:
    from agents.validation_agent import ValidationAgent
    agent = ValidationAgent()
    report = agent.validate(original_query, optimized_query, index_ddl, db_url)
"""

import os
import re
import time
from dataclasses import dataclass, field

from loguru import logger

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.sql

    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    import pymysql

    HAS_PYMYSQL = True
except ImportError:
    HAS_PYMYSQL = False


@dataclass
class TimingResult:
    """Raw timing data from EXPLAIN ANALYZE runs."""

    execution_time_ms: float
    planning_time_ms: float
    total_time_ms: float
    plan_text: str
    rows_returned: int = 0
    buffers_hit: int = 0
    buffers_read: int = 0


@dataclass
class ValidationReport:
    """Complete validation report comparing before/after performance."""

    original_query: str
    optimized_query: str
    index_ddl_applied: list[str]

    before_timing: TimingResult | None
    after_timing: TimingResult | None

    improvement_factor: float  # before / after execution time (higher = better)
    improvement_pct: float  # percentage improvement
    verdict: str  # "APPROVED" | "REJECTED" | "INCONCLUSIVE" | "ERROR"
    verdict_reason: str

    before_plan_summary: str = ""
    after_plan_summary: str = ""
    index_scan_appeared: bool = False  # Did the new index appear in the after-plan?

    timing_samples_before: list[float] = field(default_factory=list)
    timing_samples_after: list[float] = field(default_factory=list)

    engine: str = "postgresql"
    error: str | None = None


MIN_IMPROVEMENT_PCT = float(
    os.environ.get("MIN_TIMING_IMPROVEMENT", "20")
)  # 20% default
WARMUP_RUNS = 2
MEASUREMENT_RUNS = 5


def _safe_table_name(table: str) -> str:
    """Validate a table name before interpolating into SQL to prevent injection."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", table):
        raise ValueError(f"Unsafe table name: {table!r}")
    return table


class ValidationAgent:
    def __init__(self, db_url: str | None = None, engine: str = "postgresql"):
        self.db_url = (
            db_url or os.environ.get("POSTGRES_URL") or os.environ.get("MYSQL_URL")
        )
        self.engine = engine

    def validate(
        self,
        original_query: str,
        optimized_query: str,
        index_ddl: list[str] | None = None,
        db_url: str | None = None,
        engine: str | None = None,
    ) -> ValidationReport:
        """
        Run before/after validation of optimization recommendations.

        Args:
            original_query: The original slow query
            optimized_query: The rewritten query (may be same as original if only index added)
            index_ddl: List of CREATE INDEX statements to apply
            db_url: Database URL (overrides constructor)
            engine: "postgresql" | "mysql" (overrides constructor)

        Returns:
            ValidationReport with timing comparison and APPROVED/REJECTED verdict
        """
        url = db_url or self.db_url
        eng = engine or self.engine
        index_ddl = index_ddl or []

        if not url:
            logger.warning("No database URL — returning INCONCLUSIVE (dry run)")
            return self._dry_run_report(original_query, optimized_query, index_ddl)

        if eng == "postgresql":
            return self._validate_postgres(
                original_query, optimized_query, index_ddl, url
            )
        elif eng == "mysql":
            return self._validate_mysql(original_query, optimized_query, index_ddl, url)
        else:
            return self._error_report(
                original_query, optimized_query, index_ddl, f"Unsupported engine: {eng}"
            )

    # ─────────────────────────────────────────────────────────
    # PostgreSQL validation
    # ─────────────────────────────────────────────────────────

    def _validate_postgres(
        self,
        original_query: str,
        optimized_query: str,
        index_ddl: list[str],
        url: str,
    ) -> ValidationReport:
        if not HAS_PSYCOPG2:
            return self._error_report(
                original_query, optimized_query, index_ddl, "psycopg2 not installed"
            )

        try:
            conn = psycopg2.connect(url)
            conn.autocommit = True
            cur = conn.cursor()

            # Step 1: Measure baseline
            logger.info("Measuring baseline performance...")
            before = self._pg_time_query(
                cur, original_query, runs=MEASUREMENT_RUNS, warmup=WARMUP_RUNS
            )

            # Step 2: Apply indexes
            applied_ddl = []
            for ddl in index_ddl:
                try:
                    logger.info(f"Applying: {ddl[:80]}...")
                    cur.execute(ddl)
                    applied_ddl.append(ddl)
                except Exception as e:
                    logger.warning(f"Index creation failed: {e}")

            # Step 3: Update planner statistics
            tables = self._extract_tables_from_query(optimized_query)
            for table in tables:
                try:
                    # QM-11: Use psycopg2.sql.Identifier to safely quote the
                    # table name. An unquoted f-string is vulnerable to SQL
                    # injection via crafted table names extracted from the query.
                    analyze_stmt = psycopg2.sql.SQL("ANALYZE {}").format(
                        psycopg2.sql.Identifier(table)
                    )
                    cur.execute(analyze_stmt)
                    logger.debug(f"ANALYZEd {table}")
                    # QM-12: Warn if the table is too small for ANALYZE to be
                    # meaningful — statistics are only collected when there are
                    # enough rows to build reliable histograms.
                    cur.execute(
                        psycopg2.sql.SQL(
                            "SELECT reltuples::bigint FROM pg_class WHERE relname = %s"
                        ),
                        (table,),
                    )
                    row = cur.fetchone()
                    if row and row[0] < 100:
                        logger.warning(
                            f"Table {table!r} has ~{row[0]} rows — ANALYZE may produce "
                            "unreliable statistics. Populate the table before benchmarking."
                        )
                except Exception:
                    pass

            # Step 4: Measure after
            logger.info("Measuring post-optimization performance...")
            after = self._pg_time_query(
                cur, optimized_query, runs=MEASUREMENT_RUNS, warmup=WARMUP_RUNS
            )

            cur.close()
            conn.close()

            return self._build_report(
                original_query,
                optimized_query,
                applied_ddl,
                before,
                after,
                "postgresql",
            )

        except Exception as e:
            logger.error(f"PostgreSQL validation error: {e}")
            return self._error_report(
                original_query, optimized_query, index_ddl, str(e)
            )

    def _pg_time_query(
        self, cur, query: str, runs: int = 5, warmup: int = 2
    ) -> TimingResult:
        """Run EXPLAIN (ANALYZE, BUFFERS) and collect timing across multiple runs."""
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {query}"
        times = []
        plan_text = ""

        # Warmup runs (not measured)
        for _ in range(warmup):
            try:
                cur.execute(explain_query)
                cur.fetchall()
            except Exception:
                pass

        # Measurement runs
        for _ in range(runs):
            try:
                cur.execute(explain_query)
                rows = cur.fetchall()
                plan_text = "\n".join(r[0] for r in rows)

                exec_time = self._parse_pg_execution_time(plan_text)
                if exec_time > 0:
                    times.append(exec_time)
            except Exception as e:
                logger.debug(f"Query run failed: {e}")

        if not times:
            return TimingResult(
                execution_time_ms=9999.0,
                planning_time_ms=0.0,
                total_time_ms=9999.0,
                plan_text="Query failed to execute",
            )

        median_time = sorted(times)[len(times) // 2]
        planning_time = self._parse_pg_planning_time(plan_text)

        return TimingResult(
            execution_time_ms=median_time,
            planning_time_ms=planning_time,
            total_time_ms=median_time + planning_time,
            plan_text=plan_text,
            buffers_hit=self._parse_pg_buffers(plan_text, "hit"),
            buffers_read=self._parse_pg_buffers(plan_text, "read"),
        )

    def _parse_pg_execution_time(self, plan_text: str) -> float:
        m = re.search(r"Execution\s+Time:\s+([\d.]+)\s+ms", plan_text, re.IGNORECASE)
        return float(m.group(1)) if m else 0.0

    def _parse_pg_planning_time(self, plan_text: str) -> float:
        m = re.search(r"Planning\s+Time:\s+([\d.]+)\s+ms", plan_text, re.IGNORECASE)
        return float(m.group(1)) if m else 0.0

    def _parse_pg_buffers(self, plan_text: str, buf_type: str) -> int:
        m = re.search(rf"Buffers:.*{buf_type}=(\d+)", plan_text, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    # ─────────────────────────────────────────────────────────
    # MySQL validation
    # ─────────────────────────────────────────────────────────

    def _validate_mysql(
        self,
        original_query: str,
        optimized_query: str,
        index_ddl: list[str],
        url: str,
    ) -> ValidationReport:
        if not HAS_PYMYSQL:
            return self._error_report(
                original_query, optimized_query, index_ddl, "pymysql not installed"
            )

        try:
            conn = pymysql.connect(
                host=self._parse_mysql_host(url),
                user=self._parse_mysql_user(url),
                password=self._parse_mysql_password(url),
                database=self._parse_mysql_db(url),
                autocommit=True,
            )
            cur = conn.cursor()

            before = self._mysql_time_query(cur, original_query)

            for ddl in index_ddl:
                try:
                    cur.execute(ddl)
                except Exception as e:
                    logger.warning(f"MySQL index creation failed: {e}")

            tables = self._extract_tables_from_query(optimized_query)
            for table in tables:
                try:
                    cur.execute(f"ANALYZE TABLE `{_safe_table_name(table)}`")
                except Exception:
                    pass

            after = self._mysql_time_query(cur, optimized_query)

            cur.close()
            conn.close()

            return self._build_report(
                original_query, optimized_query, index_ddl, before, after, "mysql"
            )

        except Exception as e:
            return self._error_report(
                original_query, optimized_query, index_ddl, str(e)
            )

    def _mysql_time_query(self, cur, query: str) -> TimingResult:
        """Time a MySQL query using EXPLAIN ANALYZE (MySQL 8.0+)."""
        times = []
        plan_text = ""

        for _ in range(WARMUP_RUNS):
            try:
                cur.execute(f"EXPLAIN ANALYZE {query}")
                cur.fetchall()
            except Exception:
                pass

        for _ in range(MEASUREMENT_RUNS):
            try:
                start = time.perf_counter()
                cur.execute(query)
                cur.fetchall()
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            except Exception as e:
                logger.debug(f"MySQL query failed: {e}")

        if not times:
            return TimingResult(9999.0, 0.0, 9999.0, "Failed")

        median_time = sorted(times)[len(times) // 2]
        return TimingResult(median_time, 0.0, median_time, plan_text)

    # ─────────────────────────────────────────────────────────
    # Report building
    # ─────────────────────────────────────────────────────────

    def _build_report(
        self,
        original_query: str,
        optimized_query: str,
        applied_ddl: list[str],
        before: TimingResult,
        after: TimingResult,
        engine: str,
    ) -> ValidationReport:
        """Build the final ValidationReport with verdict."""
        improvement_factor = before.execution_time_ms / max(
            after.execution_time_ms, 0.001
        )
        improvement_pct = (
            (before.execution_time_ms - after.execution_time_ms)
            / max(before.execution_time_ms, 0.001)
        ) * 100

        # Check if new index appeared in plan
        index_scan_appeared = False
        for ddl in applied_ddl:
            # Extract index name from DDL
            m = re.search(r"INDEX\s+(?:CONCURRENTLY\s+)?(\w+)", ddl, re.IGNORECASE)
            if m:
                idx_name = m.group(1).lower()
                if idx_name in after.plan_text.lower():
                    index_scan_appeared = True
                    break

        # Verdict logic
        if improvement_pct >= MIN_IMPROVEMENT_PCT:
            verdict = "APPROVED"
            verdict_reason = (
                f"Query is {improvement_pct:.1f}% faster ({before.execution_time_ms:.1f}ms → "
                f"{after.execution_time_ms:.1f}ms, {improvement_factor:.2f}x improvement). "
                f"Exceeds {MIN_IMPROVEMENT_PCT}% threshold."
            )
        elif improvement_pct > 0:
            verdict = "REJECTED"
            verdict_reason = (
                f"Only {improvement_pct:.1f}% improvement ({before.execution_time_ms:.1f}ms → "
                f"{after.execution_time_ms:.1f}ms). Below {MIN_IMPROVEMENT_PCT}% threshold."
            )
        elif improvement_pct < -5:
            verdict = "REJECTED"
            verdict_reason = (
                f"Query is {-improvement_pct:.1f}% SLOWER after optimization. "
                f"Regression: {before.execution_time_ms:.1f}ms → {after.execution_time_ms:.1f}ms."
            )
        else:
            verdict = "INCONCLUSIVE"
            verdict_reason = (
                f"Negligible change: {before.execution_time_ms:.1f}ms → {after.execution_time_ms:.1f}ms. "
                f"May need more data or index rebuild."
            )

        if not index_scan_appeared and applied_ddl:
            verdict_reason += " Warning: new index not visible in execution plan."

        logger.info(f"Validation {verdict}: {verdict_reason}")

        return ValidationReport(
            original_query=original_query,
            optimized_query=optimized_query,
            index_ddl_applied=applied_ddl,
            before_timing=before,
            after_timing=after,
            improvement_factor=improvement_factor,
            improvement_pct=improvement_pct,
            verdict=verdict,
            verdict_reason=verdict_reason,
            before_plan_summary=before.plan_text[:500],
            after_plan_summary=after.plan_text[:500],
            index_scan_appeared=index_scan_appeared,
            engine=engine,
        )

    def _dry_run_report(
        self,
        original_query: str,
        optimized_query: str,
        index_ddl: list[str],
    ) -> ValidationReport:
        """Return INCONCLUSIVE report when no database is available."""
        return ValidationReport(
            original_query=original_query,
            optimized_query=optimized_query,
            index_ddl_applied=index_ddl,
            before_timing=None,
            after_timing=None,
            improvement_factor=0.0,
            improvement_pct=0.0,
            verdict="INCONCLUSIVE",
            verdict_reason="No database URL configured — set POSTGRES_URL or MYSQL_URL to validate",
            engine=self.engine,
        )

    def _error_report(
        self,
        original_query: str,
        optimized_query: str,
        index_ddl: list[str],
        error: str,
    ) -> ValidationReport:
        return ValidationReport(
            original_query=original_query,
            optimized_query=optimized_query,
            index_ddl_applied=index_ddl,
            before_timing=None,
            after_timing=None,
            improvement_factor=0.0,
            improvement_pct=0.0,
            verdict="ERROR",
            verdict_reason=f"Validation failed: {error}",
            error=error,
        )

    def _extract_tables_from_query(self, query: str) -> list[str]:
        """Extract table names from FROM and JOIN clauses."""
        q = re.sub(r"--.*$", "", query, flags=re.MULTILINE)
        tables = re.findall(
            r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_.]*)\b", q, re.IGNORECASE
        )
        # Filter out subquery aliases and SQL keywords
        sql_keywords = {
            "select",
            "where",
            "group",
            "order",
            "having",
            "union",
            "intersect",
            "except",
        }
        return list({t.lower() for t in tables if t.lower() not in sql_keywords})

    # ─────────────────────────────────────────────────────────
    # MySQL URL parsing helpers
    # ─────────────────────────────────────────────────────────

    def _parse_mysql_host(self, url: str) -> str:
        m = re.search(r"@([^:/]+)", url)
        return m.group(1) if m else "localhost"

    def _parse_mysql_user(self, url: str) -> str:
        m = re.search(r"://([^:@]+):", url)
        return m.group(1) if m else "root"

    def _parse_mysql_password(self, url: str) -> str:
        m = re.search(r"://[^:]+:([^@]+)@", url)
        return m.group(1) if m else ""

    def _parse_mysql_db(self, url: str) -> str:
        m = re.search(r"/([^/?]+)(?:\?|$)", url)
        return m.group(1) if m else ""
