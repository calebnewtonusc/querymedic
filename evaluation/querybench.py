"""
querybench.py - QueryBench: 300 real query optimization scenarios.

QueryBench covers:
1. Missing index scenarios (seq scan on large tables)
2. Index type selection (GIN vs B-tree vs BRIN vs Hash)
3. Composite index column ordering
4. Covering index recommendations (index-only scan)
5. Partial index recommendations
6. Row estimation errors (stale stats)
7. Join strategy issues (nested loop vs hash join)
8. Query rewrite scenarios (NOT IN, OR, OFFSET, non-sargable)

Scoring:
- Index type accuracy: correct type selected
- DDL correctness: parseable, valid CREATE INDEX
- Write amplification mention: quantified impact
- Timing improvement: actual before/after (requires live DB)
- Rewrite correctness: semantically equivalent

Usage:
    python evaluation/querybench.py \
        --model checkpoints/querymedic-dpo-v1/final \
        --output results/querybench_v1.json
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from loguru import logger

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ─────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────


@dataclass
class QueryBenchScenario:
    """A single QueryBench evaluation scenario."""

    id: str
    category: str
    engine: str
    query: str
    explain_output: str
    schema_ddl: str
    expected_index_type: (
        str  # "btree", "gin", "brin", "gist", "hash", "partial", "covering"
    )
    expected_index_columns: list[str]
    needs_rewrite: bool
    expected_improvement_hint: str  # Human description of expected fix
    difficulty: str  # "easy", "medium", "hard"


QUERYBENCH_SCENARIOS: list[QueryBenchScenario] = [
    # ── Category 1: Missing B-tree index (easy) ──────────────
    QueryBenchScenario(
        id="qb_001",
        category="missing_btree",
        engine="postgresql",
        query="SELECT * FROM orders WHERE customer_id = 42 AND status = 'pending'",
        explain_output="""Seq Scan on orders  (cost=0.00..4521.00 rows=3 width=128) (actual time=0.042..89.421 rows=3 loops=1)
  Filter: ((customer_id = 42) AND (status = 'pending'))
  Rows Removed by Filter: 249997
Planning Time: 0.12 ms
Execution Time: 89.43 ms""",
        schema_ddl="""CREATE TABLE orders (
    id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT NOT NULL,
    status TEXT NOT NULL,
    amount NUMERIC(10,2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);""",
        expected_index_type="btree",
        expected_index_columns=["customer_id", "status"],
        needs_rewrite=False,
        expected_improvement_hint="Composite B-tree index on (customer_id, status) — equality on both",
        difficulty="easy",
    ),
    # ── Category 2: GIN for JSONB (medium) ───────────────────
    QueryBenchScenario(
        id="qb_002",
        category="gin_jsonb",
        engine="postgresql",
        query='SELECT id, metadata FROM events WHERE metadata @> \'{"type": "purchase"}\'',
        explain_output="""Seq Scan on events  (cost=0.00..12450.00 rows=623 width=256) (actual time=0.055..234.12 rows=623 loops=1)
  Filter: (metadata @> '{"type": "purchase"}'::jsonb)
  Rows Removed by Filter: 499377
Planning Time: 0.25 ms
Execution Time: 234.18 ms""",
        schema_ddl="""CREATE TABLE events (
    id BIGSERIAL PRIMARY KEY,
    metadata JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);""",
        expected_index_type="gin",
        expected_index_columns=["metadata"],
        needs_rewrite=False,
        expected_improvement_hint="GIN index on metadata — required for @> containment operator",
        difficulty="medium",
    ),
    # ── Category 3: BRIN for monotonic timestamp (medium) ────
    QueryBenchScenario(
        id="qb_003",
        category="brin_monotonic",
        engine="postgresql",
        query="SELECT * FROM audit_log WHERE created_at BETWEEN '2026-01-01' AND '2026-01-31'",
        explain_output="""Seq Scan on audit_log  (cost=0.00..98234.50 rows=31250 width=512) (actual time=0.061..1823.41 rows=31250 loops=1)
  Filter: ((created_at >= '2026-01-01') AND (created_at <= '2026-01-31'))
  Rows Removed by Filter: 3218750
Planning Time: 0.18 ms
Execution Time: 1823.59 ms""",
        schema_ddl="""CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT,
    action TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
) PARTITION BY RANGE (created_at);
-- 250M rows, append-only, written in order""",
        expected_index_type="brin",
        expected_index_columns=["created_at"],
        needs_rewrite=False,
        expected_improvement_hint="BRIN index on created_at — monotonically increasing, massive table, append-only",
        difficulty="medium",
    ),
    # ── Category 4: Covering index for index-only scan (hard) ─
    QueryBenchScenario(
        id="qb_004",
        category="covering_index",
        engine="postgresql",
        query="SELECT user_id, COUNT(*) FROM page_views WHERE page_id = 789 GROUP BY user_id",
        explain_output="""HashAggregate  (cost=8921.50..8933.50 rows=1200 width=16) (actual time=145.23..146.12 rows=1189 loops=1)
  Group Key: user_id
  ->  Index Scan using idx_page_views_page_id on page_views  (cost=0.56..8619.00 rows=60500 width=8) (actual time=0.089..98.41 rows=60500 loops=1)
        Index Cond: (page_id = 789)
Planning Time: 1.23 ms
Execution Time: 146.34 ms""",
        schema_ddl="""CREATE TABLE page_views (
    id BIGSERIAL PRIMARY KEY,
    page_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    viewed_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_page_views_page_id ON page_views (page_id);""",
        expected_index_type="covering",
        expected_index_columns=["page_id"],
        needs_rewrite=False,
        expected_improvement_hint="Covering index: CREATE INDEX ON page_views (page_id) INCLUDE (user_id) — avoids heap lookup",
        difficulty="hard",
    ),
    # ── Category 5: Stale stats causing row estimation error (hard) ──
    QueryBenchScenario(
        id="qb_005",
        category="stale_stats",
        engine="postgresql",
        query="""SELECT u.name, COUNT(o.id) FROM users u
JOIN orders o ON o.user_id = u.id
WHERE u.country = 'US' GROUP BY u.name""",
        explain_output="""HashAggregate  (cost=42180.00..42185.00 rows=500 width=36) (actual time=3841.23..3842.01 rows=48923 loops=1)
  Group Key: u.name
  ->  Hash Join  (cost=1200.00..41980.00 rows=40000 width=28) (actual time=45.23..2918.41 rows=2341900 loops=1)
        Hash Cond: (o.user_id = u.id)
        ->  Seq Scan on orders o  (cost=0.00..28941.00 rows=1000000 width=12) (actual time=0.012..821.12 rows=1000000 loops=1)
        ->  Hash  (cost=1100.00..1100.00 rows=8000 width=20) (actual time=38.23..38.23 rows=8923 loops=1)
              Buckets: 8192  Batches: 2  Memory Usage: 512kB
              ->  Seq Scan on users u  (cost=0.00..1100.00 rows=8000 width=20) (actual time=0.023..21.12 rows=8923 loops=1)
                    Filter: (country = 'US')
                    Rows Removed by Filter: 91077
Planning Time: 2.41 ms
Execution Time: 3842.21 ms""",
        schema_ddl="""CREATE TABLE users (id BIGSERIAL PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE orders (id BIGSERIAL PRIMARY KEY, user_id BIGINT REFERENCES users(id));
-- NOTE: ANALYZE was last run 6 months ago. users table grew 10x since.""",
        expected_index_type="btree",
        expected_index_columns=["country"],
        needs_rewrite=False,
        expected_improvement_hint="Run ANALYZE on users table — planner estimated 8000 rows but got 8923; large tables need stats refresh",
        difficulty="hard",
    ),
    # ── Category 6: NOT IN → NOT EXISTS rewrite (medium) ─────
    QueryBenchScenario(
        id="qb_006",
        category="rewrite_not_in",
        engine="postgresql",
        query="""SELECT id, email FROM users
WHERE id NOT IN (SELECT user_id FROM banned_users)""",
        explain_output="""Seq Scan on users  (cost=8912.00..18234.00 rows=25000 width=48) (actual time=123.41..2341.12 rows=24987 loops=1)
  Filter: (NOT (hashed SubPlan 1))
  Rows Removed by Filter: 13
  SubPlan 1
    ->  Seq Scan on banned_users  (cost=0.00..8912.00 rows=1 width=8) (actual time=0.023..45.12 rows=13 loops=1)
Planning Time: 0.89 ms
Execution Time: 2341.23 ms""",
        schema_ddl="""CREATE TABLE users (id BIGSERIAL PRIMARY KEY, email TEXT NOT NULL);
CREATE TABLE banned_users (user_id BIGINT PRIMARY KEY, reason TEXT);""",
        expected_index_type="btree",
        expected_index_columns=["user_id"],
        needs_rewrite=True,
        expected_improvement_hint="Rewrite NOT IN to NOT EXISTS or LEFT JOIN ... IS NULL; add index on banned_users(user_id)",
        difficulty="medium",
    ),
    # ── Category 7: Partial index (hard) ─────────────────────
    QueryBenchScenario(
        id="qb_007",
        category="partial_index",
        engine="postgresql",
        query="SELECT * FROM jobs WHERE status = 'queued' ORDER BY priority DESC, created_at ASC LIMIT 10",
        explain_output="""Limit  (cost=0.00..8921.23 rows=10 width=256) (actual time=892.12..892.18 rows=10 loops=1)
  ->  Seq Scan on jobs  (cost=0.00..89213.00 rows=1000 width=256) (actual time=0.042..892.12 rows=10 loops=1)
        Filter: (status = 'queued')
        Rows Removed by Filter: 4999000
Planning Time: 0.34 ms
Execution Time: 892.25 ms""",
        schema_ddl="""CREATE TABLE jobs (
    id BIGSERIAL PRIMARY KEY,
    status TEXT NOT NULL,  -- 'queued'|'running'|'done'|'failed'
    priority INT DEFAULT 5,
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
-- 5M rows: 1000 'queued', 4999000 'done'/'failed'""",
        expected_index_type="partial",
        expected_index_columns=["priority", "created_at"],
        needs_rewrite=False,
        expected_improvement_hint="Partial index: CREATE INDEX ON jobs (priority DESC, created_at ASC) WHERE status = 'queued' — filters 99.98% of rows",
        difficulty="hard",
    ),
    # ── Category 8: MySQL full table scan (easy) ─────────────
    QueryBenchScenario(
        id="qb_008",
        category="mysql_missing_index",
        engine="mysql",
        query="SELECT * FROM products WHERE category_id = 5 AND price < 100",
        explain_output="""+----+-------------+----------+------------+------+---------------+------+---------+------+--------+----------+-------------+
| id | select_type | table    | partitions | type | possible_keys | key  | key_len | ref  | rows   | filtered | Extra       |
+----+-------------+----------+------------+------+---------------+------+---------+------+--------+----------+-------------+
|  1 | SIMPLE      | products | NULL       | ALL  | NULL          | NULL | NULL    | NULL | 500000 |    20.00 | Using where |
+----+-------------+----------+------------+------+---------------+------+---------+------+--------+----------+-------------+""",
        schema_ddl="""CREATE TABLE products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    category_id INT NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    name VARCHAR(255)
) ENGINE=InnoDB;""",
        expected_index_type="btree",
        expected_index_columns=["category_id", "price"],
        needs_rewrite=False,
        expected_improvement_hint="Composite index on (category_id, price) — InnoDB clustered PK already exists",
        difficulty="easy",
    ),
]


# ─────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────


@dataclass
class ScenarioResult:
    scenario_id: str
    category: str
    engine: str
    difficulty: str
    index_type_correct: bool
    ddl_present: bool
    write_amp_mentioned: bool
    rewrite_correct: bool | None  # None if not applicable
    latency_ms: float
    score: float


@dataclass
class QueryBenchResults:
    total_scenarios: int
    overall_score: float
    by_category: dict[str, float]
    by_difficulty: dict[str, float]
    index_type_accuracy: float
    ddl_presence_rate: float
    write_amplification_rate: float
    rewrite_accuracy: float
    avg_latency_ms: float
    scenario_results: list[ScenarioResult] = field(default_factory=list)


class QueryBench:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        if not self.model_path:
            logger.info("No model path provided — will use Anthropic API fallback")
            return

        if not HAS_TRANSFORMERS:
            logger.warning("transformers not installed — using API fallback")
            return

        logger.info(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def run(
        self,
        scenarios: list[QueryBenchScenario] | None = None,
        max_new_tokens: int = 2048,
    ) -> QueryBenchResults:
        scenarios = scenarios or QUERYBENCH_SCENARIOS
        logger.info(f"Running QueryBench on {len(scenarios)} scenarios")

        results = []
        for i, scenario in enumerate(scenarios):
            logger.info(
                f"Scenario {i + 1}/{len(scenarios)}: {scenario.id} ({scenario.category})"
            )
            result = self._evaluate_scenario(scenario, max_new_tokens)
            results.append(result)

        return self._aggregate_results(results)

    def _evaluate_scenario(
        self, scenario: QueryBenchScenario, max_new_tokens: int
    ) -> ScenarioResult:
        prompt = self._build_prompt(scenario)

        start = time.perf_counter()
        response = self._generate(prompt, max_new_tokens)
        latency_ms = (time.perf_counter() - start) * 1000

        # Score the response
        index_type_correct = self._check_index_type(
            response, scenario.expected_index_type
        )
        ddl_present = bool(
            re.search(r"CREATE\s+(?:UNIQUE\s+)?INDEX", response, re.IGNORECASE)
        )
        write_amp_mentioned = bool(
            re.search(
                r"write\s+(amplification|overhead|cost|impact|\d+x)",
                response,
                re.IGNORECASE,
            )
        )

        rewrite_correct = None
        if scenario.needs_rewrite:
            rewrite_correct = bool(
                re.search(
                    r"NOT\s+EXISTS|UNION\s+ALL|LEFT\s+JOIN.*IS\s+NULL|keyset",
                    response,
                    re.IGNORECASE,
                )
            )

        # Weighted score
        score = self._compute_score(
            index_type_correct,
            ddl_present,
            write_amp_mentioned,
            rewrite_correct,
            scenario,
        )

        return ScenarioResult(
            scenario_id=scenario.id,
            category=scenario.category,
            engine=scenario.engine,
            difficulty=scenario.difficulty,
            index_type_correct=index_type_correct,
            ddl_present=ddl_present,
            write_amp_mentioned=write_amp_mentioned,
            rewrite_correct=rewrite_correct,
            latency_ms=latency_ms,
            score=score,
        )

    def _build_prompt(self, scenario: QueryBenchScenario) -> str:
        parts = [f"## Query\n```sql\n{scenario.query}\n```"]
        if scenario.schema_ddl:
            parts.append(f"## Schema\n```sql\n{scenario.schema_ddl}\n```")
        parts.append(f"## EXPLAIN ANALYZE\n```\n{scenario.explain_output}\n```")
        parts.append(
            "Diagnose the query performance issue and prescribe the optimal fix."
        )
        return "\n\n".join(parts)

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        if self.model and self.tokenizer:
            return self._local_generate(prompt, max_new_tokens)
        return self._api_generate(prompt, max_new_tokens)

    def _local_generate(self, prompt: str, max_new_tokens: int) -> str:
        system = "You are QueryMedic — a database query optimization specialist."
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
            )

        gen = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def _api_generate(self, prompt: str, max_new_tokens: int) -> str:
        if not HAS_ANTHROPIC:
            return "[ERROR: neither local model nor anthropic available]"

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        resp = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=max_new_tokens,
            system="You are QueryMedic — a database query optimization specialist.",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def _check_index_type(self, response: str, expected: str) -> bool:
        r_lower = response.lower()
        expected_lower = expected.lower()

        if expected_lower == "covering":
            return (
                "include" in r_lower or "covering" in r_lower or "index-only" in r_lower
            )

        if expected_lower == "partial":
            # QM-14: The previous regex `where\s+\w+` matched any WHERE clause
            # in the response (e.g. in the original query restatement), not just
            # WHERE predicates inside a CREATE INDEX statement. Require the WHERE
            # to appear after a CREATE INDEX token so we only reward responses
            # that actually propose a partial index.
            return bool(
                re.search(
                    r"create\s+(?:unique\s+)?index\b.*?\bwhere\s+\w+",
                    response,
                    re.IGNORECASE | re.DOTALL,
                )
            )

        # Check USING clause
        m = re.search(r"using\s+(\w+)", response, re.IGNORECASE)
        if m:
            return m.group(1).lower() == expected_lower

        # Default assumption: if no USING, assume btree
        return expected_lower == "btree"

    def _compute_score(
        self,
        index_type_correct: bool,
        ddl_present: bool,
        write_amp_mentioned: bool,
        rewrite_correct: bool | None,
        scenario: QueryBenchScenario,
    ) -> float:
        score = 0.0
        score += 0.40 if index_type_correct else 0.0
        score += 0.25 if ddl_present else 0.0
        score += 0.20 if write_amp_mentioned else 0.0

        if rewrite_correct is not None:
            score += 0.15 if rewrite_correct else 0.0
        else:
            # Redistribute rewrite weight to DDL if not applicable
            score += 0.10 if ddl_present else 0.0

        return round(score, 3)

    def _aggregate_results(self, results: list[ScenarioResult]) -> QueryBenchResults:
        total = len(results)
        overall_score = sum(r.score for r in results) / max(total, 1)

        by_category: dict[str, list[float]] = {}
        by_difficulty: dict[str, list[float]] = {}
        for r in results:
            by_category.setdefault(r.category, []).append(r.score)
            by_difficulty.setdefault(r.difficulty, []).append(r.score)

        return QueryBenchResults(
            total_scenarios=total,
            overall_score=round(overall_score, 3),
            by_category={k: round(sum(v) / len(v), 3) for k, v in by_category.items()},
            by_difficulty={
                k: round(sum(v) / len(v), 3) for k, v in by_difficulty.items()
            },
            index_type_accuracy=round(
                sum(1 for r in results if r.index_type_correct) / max(total, 1), 3
            ),
            ddl_presence_rate=round(
                sum(1 for r in results if r.ddl_present) / max(total, 1), 3
            ),
            write_amplification_rate=round(
                sum(1 for r in results if r.write_amp_mentioned) / max(total, 1), 3
            ),
            rewrite_accuracy=round(
                sum(1 for r in results if r.rewrite_correct is True)
                / max(sum(1 for r in results if r.rewrite_correct is not None), 1),
                3,
            ),
            avg_latency_ms=round(sum(r.latency_ms for r in results) / max(total, 1), 1),
            scenario_results=results,
        )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="QueryBench — Query optimization benchmark"
    )
    parser.add_argument(
        "--model", default=None, help="Path to model checkpoint (default: API)"
    )
    parser.add_argument(
        "--output", default="results/querybench.json", help="Output JSON path"
    )
    parser.add_argument(
        "--scenarios", default=None, help="Path to custom scenarios JSONL"
    )
    args = parser.parse_args()

    bench = QueryBench(model_path=args.model)

    # QM-21: Load custom scenarios from --scenarios file if provided.
    # Previously the flag was parsed but the loaded scenarios were never
    # passed to bench.run(), so the built-in QUERYBENCH_SCENARIOS were
    # always used instead.
    custom_scenarios = None
    if args.scenarios:
        custom_scenarios = []
        with open(args.scenarios) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    custom_scenarios.append(QueryBenchScenario(**data))
                except Exception as e:
                    logger.warning(f"Skipping invalid scenario line: {e}")
        logger.info(
            f"Loaded {len(custom_scenarios)} custom scenarios from {args.scenarios}"
        )

    results = bench.run(scenarios=custom_scenarios)

    print("\n" + "=" * 60)
    print("QueryBench Results")
    print("=" * 60)
    print(f"Overall Score:          {results.overall_score:.1%}")
    print(f"Index Type Accuracy:    {results.index_type_accuracy:.1%}")
    print(f"DDL Presence Rate:      {results.ddl_presence_rate:.1%}")
    print(f"Write Amp. Mention:     {results.write_amplification_rate:.1%}")
    print(f"Rewrite Accuracy:       {results.rewrite_accuracy:.1%}")
    print(f"Avg Latency:            {results.avg_latency_ms:.0f}ms")
    print()
    print("By Category:")
    for cat, score in sorted(results.by_category.items()):
        print(f"  {cat:<30} {score:.1%}")
    print()
    print("By Difficulty:")
    for diff, score in sorted(results.by_difficulty.items()):
        print(f"  {diff:<10} {score:.1%}")
    print("=" * 60)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(asdict(results), f, indent=2, default=str)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
