"""
query_synthesizer.py - LLM-backed synthesis of PostgreSQL query optimization training pairs.

Uses Anthropic Claude (primary) and vLLM (secondary, round-robin) to generate
high-quality (slow query + EXPLAIN context → diagnosis + prescription) pairs
from raw corpus data collected by the discovery pipeline.

Input sources:
  - data/raw/query_corpus/     (Stack Exchange, pg_activity)
  - data/raw/pg_patterns/      (pganalyze blog patterns)
  - data/raw/github_db_prs/    (GitHub PRs with DB migrations)
  - data/raw/schemas/          (real-world schemas)

Output (ShareGPT JSONL):
  data/training/sharegpt_train.jsonl
  data/training/sharegpt_val.jsonl

Usage:
    python synthesis/query_synthesizer.py \\
        --vllm-urls http://localhost:8001/v1 http://localhost:8002/v1 \\
        --workers 24

    python synthesis/query_synthesizer.py \\
        --backend claude \\
        --workers 8
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).parent.parent))

import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Optional

# QM-8: The try/except that set HAS_AIOFILES = False and then immediately
# re-raised was dead code — the False branch was unreachable. Replace with a
# clean import guard that gives a clear install message on missing dependency.
try:
    import aiofiles
except ImportError as exc:
    raise ImportError(
        "aiofiles is required for query_synthesizer. Install with: pip install aiofiles"
    ) from exc
import httpx
from anthropic import AsyncAnthropic
from loguru import logger

RAW_DIR = Path(__file__).parents[1] / "data" / "raw"
TRAINING_DIR = Path(__file__).parents[1] / "data" / "training"

# ─── System Prompt ─────────────────────────────────────────────────────────────

QUERYMEDIC_SYSTEM = """You are QueryMedic — a PostgreSQL query optimization specialist with deep expertise in:
- Query plan analysis (EXPLAIN ANALYZE, plan nodes, cost models)
- Index design (B-tree, GIN, GiST, BRIN, partial, covering, expression indexes)
- Join strategy selection (hash join, merge join, nested loop conditions)
- Statistics and row estimation (pg_stats, autovacuum, ANALYZE)
- Query rewriting (NOT IN→NOT EXISTS, OR→UNION, sargable predicates)
- ORM anti-patterns (N+1 queries, select_related, eager loading)

When given a slow query and context, your response always includes:
1. **Root Cause**: The specific PostgreSQL bottleneck (e.g., "Seq Scan filtering 4.2M rows to find 47")
2. **Diagnosis**: Why this is happening (missing index, stale stats, bad join order, N+1)
3. **Prescription DDL**: Ready-to-execute SQL (CREATE INDEX CONCURRENTLY, ANALYZE, query rewrite)
4. **Index Justification**: Column order choice, partial vs full, covering vs key-only
5. **Write Impact**: Estimated overhead per INSERT/UPDATE/DELETE
6. **Expected Result**: Quantified improvement (e.g., "1847ms → ~12ms, EXPLAIN shows Index Scan")

Always cite specific values from the EXPLAIN output (costs, row counts, execution times).
Never recommend indexes that won't be used. Prefer covering indexes over multiple single-column indexes.
Quantify everything — vague advice is not acceptable."""

# ─── Prompt Templates ──────────────────────────────────────────────────────────


def build_user_prompt_from_corpus(record: dict) -> str:
    """Build user prompt from a Stack Exchange / pg_activity corpus record."""
    parts = []

    slow_query = record.get("slow_query") or record.get("query", "")
    schema = record.get("schema") or record.get("schema_ddl", "")
    explain_before = record.get("explain_before") or record.get("explain_output", "")
    question = record.get("question") or record.get("title", "")

    if question:
        parts.append(f"**Context:** {question[:500]}")
    if slow_query:
        parts.append(f"**Slow query:**\n```sql\n{slow_query[:2000]}\n```")
    if schema:
        parts.append(f"**Schema:**\n```sql\n{schema[:2000]}\n```")
    if explain_before:
        parts.append(f"**EXPLAIN ANALYZE output:**\n```\n{explain_before[:3000]}\n```")

    if not parts:
        return ""

    return (
        "Analyze this PostgreSQL performance problem and provide a complete optimization prescription.\n\n"
        + "\n\n".join(parts)
    )


def build_user_prompt_from_pattern(record: dict) -> str:
    """Build user prompt from a pganalyze blog pattern record."""
    parts = []
    patterns = record.get("patterns", [])
    record.get("content", "")
    title = record.get("title", "")
    sql_snippets = record.get("sql_snippets", [])
    explain_outputs = record.get("explain_outputs", [])

    if title:
        parts.append(f"**Blog post context:** {title}")
    if sql_snippets:
        parts.append(f"**Slow query:**\n```sql\n{sql_snippets[0][:2000]}\n```")
    if explain_outputs:
        parts.append(f"**EXPLAIN output:**\n```\n{explain_outputs[0][:2000]}\n```")
    if patterns:
        p = patterns[0]
        if p.get("before_sql") and not sql_snippets:
            parts.append(f"**Query (before):**\n```sql\n{p['before_sql'][:1000]}\n```")

    if not parts or len(parts) < 2:
        return ""

    return (
        "Analyze this PostgreSQL query optimization scenario from a technical blog post.\n\n"
        + "\n\n".join(parts)
    )


def build_user_prompt_from_pr(record: dict) -> str:
    """Build user prompt from a GitHub DB optimization PR record."""
    parts = []
    title = record.get("title", "")
    body = record.get("body", "")
    migration_sql = record.get("migration_sql", [])
    explain_snippets = record.get("explain_snippets", [])
    query_patterns = record.get("query_patterns", [])

    if title:
        parts.append(f"**PR title:** {title}")
    if explain_snippets:
        parts.append(f"**EXPLAIN evidence:**\n```\n{explain_snippets[0][:2000]}\n```")
    if query_patterns:
        p = query_patterns[0]
        if p.get("before_sql"):
            parts.append(
                f"**Query before fix:**\n```sql\n{p['before_sql'][:1000]}\n```"
            )
        if p.get("after_sql"):
            parts.append(f"**Query after fix:**\n```sql\n{p['after_sql'][:1000]}\n```")
    if migration_sql:
        parts.append(f"**Migration/index DDL:**\n```sql\n{migration_sql[0][:500]}\n```")
    if body and len(parts) < 3:
        parts.append(f"**PR description:**\n{body[:800]}")

    if len(parts) < 2:
        return ""

    return (
        "Analyze this database optimization PR and explain the root cause and prescription.\n\n"
        + "\n\n".join(parts)
    )


def build_user_prompt_from_schema(
    record: dict, scenario_type: str = "missing_index"
) -> str:
    """Build a synthetic slow query prompt using a real-world schema."""
    tables = record.get("tables", [])
    if not tables:
        return ""

    # Pick a random table with columns
    rng = random.Random(hash(record.get("schema_hash", "")) & 0xFFFFFFFF)
    candidate_tables = [t for t in tables if len(t.get("columns", [])) >= 3]
    if not candidate_tables:
        return ""

    table = rng.choice(candidate_tables)
    table_name = table["name"]
    columns = table["columns"]
    col_names = [c["name"] for c in columns]

    # Build a realistic slow query based on scenario type
    scenario_prompts = {
        "missing_index": (
            f"Imagine this table has {rng.randint(500_000, 20_000_000):,} rows. "
            f"A query like:\n\n"
            f"```sql\nSELECT * FROM {table_name}\n"
            f"WHERE {col_names[min(1, len(col_names) - 1)]} = $1\n"
            f"  AND {col_names[min(2, len(col_names) - 1)]} > NOW() - INTERVAL '30 days'\n"
            f"ORDER BY {col_names[min(2, len(col_names) - 1)]} DESC\nLIMIT 100;\n```\n\n"
            f"is performing a sequential scan. The EXPLAIN output shows:\n"
            f"```\nSeq Scan on {table_name}  (cost=0.00..{rng.randint(50000, 500000):.2f} "
            f"rows={rng.randint(1000000, 20000000)} width=120)\n"
            f"  Filter: (col = $1 AND created_at > ...)\n"
            f"  Rows Removed by Filter: {rng.randint(499900, 19999900):,}\n"
            f"Execution Time: {rng.randint(1500, 15000):.3f} ms\n```"
        ),
        "n_plus_1": (
            f"An ORM is issuing N+1 queries against {table_name}. "
            f"For each of {rng.randint(100, 10000):,} parent records, "
            f"a separate query fetches related {table_name} rows:\n\n"
            f"```sql\n-- This runs N times:\nSELECT * FROM {table_name} WHERE parent_id = $1;\n```\n\n"
            f"Total database time: {rng.randint(2000, 30000):.0f}ms for a single page load."
        ),
        "stale_stats": (
            f"The query planner estimates {rng.randint(1, 100)} rows but the actual count is "
            f"{rng.randint(50000, 2000000):,}. The EXPLAIN output shows a catastrophically wrong plan:\n\n"
            f"```\nNested Loop  (cost=0.43..247.91 rows=1 width=80)\n"
            f"  (actual time=0.012..{rng.randint(5000, 60000):.3f} rows={rng.randint(50000, 2000000):,} loops=1)\n```\n\n"
            f"The table {table_name} hasn't been analyzed in {rng.randint(3, 90)} days."
        ),
    }

    prompt_body = scenario_prompts.get(scenario_type, scenario_prompts["missing_index"])

    # Build schema DDL for context
    schema_lines = [f"CREATE TABLE {table_name} ("]
    for col in columns[:15]:
        nullable = "" if col.get("nullable", True) else " NOT NULL"
        default = f" DEFAULT {col['default']}" if col.get("default") else ""
        schema_lines.append(f"  {col['name']} {col['data_type']}{nullable}{default},")
    schema_lines.append(");")

    existing_indexes = table.get("indexes", [])
    if existing_indexes:
        schema_lines.append("")
        schema_lines.extend(existing_indexes[:5])

    schema_ddl = "\n".join(schema_lines)

    return (
        f"Analyze this PostgreSQL performance problem and provide a complete optimization prescription.\n\n"
        f"**Schema:**\n```sql\n{schema_ddl}\n```\n\n"
        f"**Problem:**\n{prompt_body}"
    )


# ─── Quality Scoring ───────────────────────────────────────────────────────────


def score_synthesis_quality(response: str) -> float:
    """Score LLM synthesis quality on a 0.0–1.0 scale."""
    score = 0.0
    text_lower = response.lower()

    # Has root cause analysis
    if any(
        kw in text_lower
        for kw in ["root cause", "seq scan", "sequential scan", "missing index", "n+1"]
    ):
        score += 0.15

    # Has specific numbers/metrics
    if re.search(r"\d+\s*ms", response):
        score += 0.15
    if re.search(r"\d+x\s+(?:faster|improvement|speedup)", response, re.IGNORECASE):
        score += 0.10

    # Has DDL prescription
    if re.search(r"CREATE\s+INDEX", response, re.IGNORECASE):
        score += 0.20
    if re.search(r"CONCURRENTLY", response, re.IGNORECASE):
        score += 0.05

    # Has index justification
    if any(
        kw in text_lower
        for kw in [
            "column order",
            "selectivity",
            "cardinality",
            "partial index",
            "covering",
        ]
    ):
        score += 0.10

    # Has write impact analysis
    if any(
        kw in text_lower
        for kw in [
            "write amplification",
            "write overhead",
            "insert overhead",
            "update overhead",
        ]
    ):
        score += 0.10

    # Has EXPLAIN output reference
    if any(
        kw in text_lower
        for kw in [
            "index scan",
            "bitmap heap scan",
            "index only scan",
            "cost=",
            "actual time=",
        ]
    ):
        score += 0.10

    # Adequate length
    if len(response) > 500:
        score += 0.05

    return round(min(1.0, score), 3)


def _extract_json_safe(text: str) -> Optional[dict]:
    """Try to extract JSON from an LLM response."""
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Code block — extract content between fences, then raw_decode from first '{'
    m = re.search(r"```(?:json)?\s*\n([\s\S]+?)\s*```", text)
    if m:
        block = m.group(1)
        start = block.find("{")
        if start != -1:
            try:
                obj, _ = json.JSONDecoder().raw_decode(block, start)
                return obj
            except json.JSONDecodeError:
                pass
    # raw_decode starting from the first '{' in the full text
    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            return obj
        except json.JSONDecodeError:
            pass
    return None


def format_as_sharegpt(user_prompt: str, assistant_response: str, source: str) -> dict:
    """Format a synthesis pair as ShareGPT conversation."""
    return {
        "conversations": [
            {"from": "system", "value": QUERYMEDIC_SYSTEM},
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": assistant_response},
        ],
        "metadata": {
            "source": source,
            "quality_score": score_synthesis_quality(assistant_response),
        },
    }


# ─── LLM Backends ─────────────────────────────────────────────────────────────


class ClaudeBackend:
    """Anthropic Claude API backend."""

    MODEL = "claude-opus-4-6"
    MAX_TOKENS = 4096
    TEMPERATURE = 0.7

    def __init__(self) -> None:
        self._client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    async def generate(self, user_prompt: str) -> Optional[str]:
        try:
            msg = await self._client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
                system=QUERYMEDIC_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return msg.content[0].text if msg.content else None
        except Exception as e:
            logger.debug(f"Claude API error: {e}")
            return None


class VLLMBackend:
    """vLLM OpenAI-compatible backend with round-robin URL selection."""

    MODEL = "Qwen/Qwen2.5-72B-Instruct"
    MAX_TOKENS = 4096
    TEMPERATURE = 0.7

    def __init__(self, base_urls: list[str], api_key: str = "synthesis") -> None:
        self._urls = base_urls
        self._api_key = api_key
        self._idx = 0

    def _next_url(self) -> str:
        url = self._urls[self._idx % len(self._urls)]
        self._idx += 1
        return url

    async def generate(self, user_prompt: str) -> Optional[str]:
        url = self._next_url()
        payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": QUERYMEDIC_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.MAX_TOKENS,
            "temperature": self.TEMPERATURE,
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.debug(f"vLLM error ({url}): {e}")
            return None


# ─── Main Synthesizer ─────────────────────────────────────────────────────────


class QueryMedicSynthesizer:
    """
    Synthesizes PostgreSQL query optimization training pairs using LLMs.

    Iterates over all raw data sources, builds user prompts, calls LLM backends
    to generate expert diagnoses, scores quality, and writes ShareGPT JSONL.
    """

    MIN_QUALITY = 0.35
    VAL_FRACTION = 0.05

    def __init__(
        self,
        backend: str = "claude",
        vllm_urls: Optional[list[str]] = None,
        workers: int = 12,
        raw_dir: Path = RAW_DIR,
        output_dir: Path = TRAINING_DIR,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self._semaphore = asyncio.Semaphore(workers)
        self._stats = {"processed": 0, "saved": 0, "skipped": 0, "errors": 0}

        if backend == "vllm" and vllm_urls:
            self._llm = VLLMBackend(vllm_urls)
        else:
            self._llm = ClaudeBackend()

    def _iter_records(self):
        """Iterate over all raw data source records."""
        # Stack Exchange / pg_activity corpus
        corpus_dir = self.raw_dir / "query_corpus"
        if corpus_dir.exists():
            for jsonl_file in sorted(corpus_dir.rglob("*.jsonl")):
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            yield "corpus", rec
                        except json.JSONDecodeError:
                            continue

        # pganalyze patterns
        patterns_dir = self.raw_dir / "pg_patterns"
        if patterns_dir.exists():
            for jsonl_file in sorted(patterns_dir.rglob("*.jsonl")):
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if rec.get("relevance_score", 0) >= 0.3:
                                yield "pg_pattern", rec
                        except json.JSONDecodeError:
                            continue

        # GitHub DB PRs
        prs_dir = self.raw_dir / "github_db_prs"
        if prs_dir.exists():
            for jsonl_file in sorted(prs_dir.rglob("*.jsonl")):
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if rec.get("relevance_score", 0) >= 0.3:
                                yield "github_pr", rec
                        except json.JSONDecodeError:
                            continue

        # Real-world schemas (synthetic scenarios)
        schemas_dir = self.raw_dir / "schemas"
        if schemas_dir.exists():
            schema_records = []
            for jsonl_file in sorted(schemas_dir.rglob("*.jsonl")):
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if rec.get("table_count", 0) >= 3:
                                schema_records.append(rec)
                        except json.JSONDecodeError:
                            continue

            # Generate multiple scenario types per schema
            scenario_types = ["missing_index", "n_plus_1", "stale_stats"]
            rng = random.Random(42)
            for rec in schema_records:
                scenario = rng.choice(scenario_types)
                yield "schema_synthetic", {"record": rec, "scenario": scenario}

    def _build_prompt(self, source: str, item) -> Optional[str]:
        """Build user prompt for a given source and item."""
        if source == "corpus":
            return build_user_prompt_from_corpus(item)
        elif source == "pg_pattern":
            return build_user_prompt_from_pattern(item)
        elif source == "github_pr":
            return build_user_prompt_from_pr(item)
        elif source == "schema_synthetic":
            return build_user_prompt_from_schema(item["record"], item["scenario"])
        return None

    async def _synthesize_one(
        self,
        source: str,
        item,
        train_file: Path,
        val_file: Path,
        rng: random.Random,
    ) -> bool:
        """Synthesize one training pair."""
        async with self._semaphore:
            user_prompt = self._build_prompt(source, item)
            if not user_prompt or len(user_prompt) < 100:
                self._stats["skipped"] += 1
                return False

            response = await self._llm.generate(user_prompt)
            self._stats["processed"] += 1

            if not response:
                self._stats["errors"] += 1
                return False

            quality = score_synthesis_quality(response)
            if quality < self.MIN_QUALITY:
                self._stats["skipped"] += 1
                return False

            pair = format_as_sharegpt(user_prompt, response, source)
            output_file = val_file if rng.random() < self.VAL_FRACTION else train_file

            async with aiofiles.open(str(output_file), "a") as f:
                await f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            self._stats["saved"] += 1
            return True

    async def synthesize_all(self, limit: Optional[int] = None) -> int:
        """Synthesize training pairs from all raw data sources."""
        train_file = self.output_dir / "sharegpt_train.jsonl"
        val_file = self.output_dir / "sharegpt_val.jsonl"
        rng = random.Random(42)

        records = list(self._iter_records())
        if limit:
            rng.shuffle(records)
            records = records[:limit]

        logger.info(
            f"Synthesizing {len(records):,} training pairs (workers={self.workers})"
        )

        batch_size = self.workers * 4
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            tasks = [
                self._synthesize_one(source, item, train_file, val_file, rng)
                for source, item in batch
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            if (i // batch_size) % 10 == 0:
                logger.info(
                    f"Progress: {i + len(batch)}/{len(records)} | "
                    f"saved={self._stats['saved']} skipped={self._stats['skipped']} "
                    f"errors={self._stats['errors']}"
                )

        logger.success(
            f"Synthesis complete: {self._stats['saved']} pairs saved "
            f"({self._stats['skipped']} skipped, {self._stats['errors']} errors)"
        )
        return self._stats["saved"]


# ─── Synthetic Scenarios (Fallback — no LLM required) ─────────────────────────

SCENARIO_TEMPLATES = [
    # Missing multi-column index for common OLTP pattern
    {
        "category": "missing_index",
        "engine": "postgresql",
        "table_name": "user_events",
        "row_count": 4_200_000,
        "query": "SELECT event_type, COUNT(*) FROM user_events WHERE user_id = $1 AND created_at > NOW() - INTERVAL '30 days' GROUP BY event_type",
        "schema_ddl": "CREATE TABLE user_events (id bigint PRIMARY KEY, user_id int NOT NULL, event_type text, created_at timestamptz NOT NULL, payload jsonb); -- No index on user_id or created_at",
        "explain_before": "Seq Scan on user_events  (cost=0.00..94721.34 rows=3 width=40) (actual time=0.021..1847.234 rows=47 loops=1)\n  Filter: ((user_id = 42) AND (created_at > (now() - '30 days'::interval)))\n  Rows Removed by Filter: 4199953\nPlanning Time: 0.412 ms\nExecution Time: 1847.891 ms",
        "diagnosis": "Sequential scan filtering 4.2M rows to find 47. No index on (user_id, created_at). Row estimation 3 vs actual 47 (15x error) suggests stale statistics too.",
        "prescription_ddl": "-- Step 1: Update statistics\nANALYZE user_events;\n\n-- Step 2: Create multi-column index\n-- user_id first (equality predicate = high selectivity)\n-- created_at DESC (range predicate, matches ORDER BY patterns)\nCREATE INDEX CONCURRENTLY idx_user_events_uid_cat\n  ON user_events(user_id, created_at DESC)\n  WHERE created_at > '2024-01-01';  -- partial: only recent data",
        "index_type": "B-tree multi-column partial",
        "write_amplification": "Partial index covers ~30% of rows. Write overhead: +1 B-tree update per INSERT matching predicate. Low impact.",
        "expected_improvement": "1847ms → ~12ms (154x). EXPLAIN shows Index Scan using idx_user_events_uid_cat.",
        "explain_after": "Index Scan using idx_user_events_uid_cat on user_events  (cost=0.43..8.47 rows=47 width=40) (actual time=0.021..0.312 rows=47 loops=1)\nPlanning Time: 0.387 ms\nExecution Time: 12.234 ms",
    },
    # GIN index for JSONB query
    {
        "category": "jsonb_gin",
        "engine": "postgresql",
        "table_name": "events",
        "row_count": 8_000_000,
        "query": "SELECT id, data FROM events WHERE data @> '{\"event_type\": \"purchase\"}' AND created_at > NOW() - INTERVAL '7 days'",
        "schema_ddl": "CREATE TABLE events (id bigserial PRIMARY KEY, data jsonb NOT NULL, created_at timestamptz DEFAULT now()); -- No GIN index on data",
        "explain_before": "Seq Scan on events  (cost=0.00..212341.00 rows=800 width=240) (actual time=0.031..3421.891 rows=12400 loops=1)\n  Filter: ((data @> '{\"event_type\": \"purchase\"}'::jsonb) AND (created_at > (now() - '7 days'::interval)))\n  Rows Removed by Filter: 7987600\nExecution Time: 3422.341 ms",
        "diagnosis": "JSONB containment operator @> requires GIN index for efficiency. Sequential scan through 8M rows. B-tree cannot index JSONB containment.",
        "prescription_ddl": "-- GIN index supports @> (containment) and ? (key exists) operators\nCREATE INDEX CONCURRENTLY idx_events_data_gin\n  ON events USING gin(data jsonb_path_ops);  -- jsonb_path_ops: smaller, faster for @>\n\n-- Also add B-tree on created_at for the time filter\nCREATE INDEX CONCURRENTLY idx_events_created_at\n  ON events(created_at DESC)\n  WHERE created_at > '2024-01-01';",
        "index_type": "GIN (jsonb_path_ops)",
        "write_amplification": "GIN indexes have higher per-row write cost than B-tree due to posting list maintenance. At 50k events/day: acceptable overhead.",
        "expected_improvement": "3422ms → ~45ms (76x). Bitmap Index Scan on GIN index eliminates 99.8% of rows before heap fetch.",
        "explain_after": 'Bitmap Heap Scan on events  (cost=42.31..1847.23 rows=800 width=240) (actual time=8.234..45.123 rows=12400 loops=1)\n  Recheck Cond: (data @> \'{"event_type": "purchase"}\'::jsonb)\n  ->  Bitmap Index Scan on idx_events_data_gin  (cost=0.00..42.11 rows=800 width=0) (actual time=8.012..8.012 rows=12400 loops=1)\nExecution Time: 45.891 ms',
    },
    # Covering index for index-only scan
    {
        "category": "covering_index",
        "engine": "postgresql",
        "table_name": "orders",
        "row_count": 12_000_000,
        "query": "SELECT customer_id, SUM(total_amount) FROM orders WHERE status = 'completed' AND created_at BETWEEN $1 AND $2 GROUP BY customer_id ORDER BY SUM(total_amount) DESC LIMIT 100",
        "schema_ddl": "CREATE TABLE orders (id bigserial PRIMARY KEY, customer_id int, status text, total_amount numeric, created_at timestamptz); CREATE INDEX idx_orders_status ON orders(status);",
        "explain_before": "Sort  (cost=284721.34..284841.34 rows=48000 width=40) (actual time=2847.123..2847.234 rows=100 loops=1)\n  ->  HashAggregate\n        ->  Bitmap Heap Scan on orders  (cost=4234.12..272123.12 rows=1660000 width=20)\n              Recheck Cond: (status = 'completed')\n              Filter: ((created_at >= $1) AND (created_at <= $2))\n              Heap Blocks: exact=124000\nExecution Time: 2891.234 ms",
        "diagnosis": "Bitmap Heap Scan fetches 124k heap blocks even though we only need customer_id and total_amount. Heap fetch is the bottleneck. A covering index with INCLUDE would enable index-only scan.",
        "prescription_ddl": "-- Covering index: (status, created_at) for WHERE + range, INCLUDE for SELECT columns\nCREATE INDEX CONCURRENTLY idx_orders_status_cat_covering\n  ON orders(status, created_at)\n  INCLUDE (customer_id, total_amount)\n  WHERE status = 'completed';",
        "index_type": "B-tree covering (INCLUDE) partial",
        "write_amplification": "Partial index (WHERE status = 'completed') limits write overhead to completed orders only. If 40% of orders complete: 40% of INSERT/UPDATE overhead.",
        "expected_improvement": "2891ms → ~180ms (16x). Index Only Scan eliminates 124k heap block fetches. Heap Fetches should drop to near 0.",
        "explain_after": "Sort  (cost=8234.12..8284.12 rows=20000 width=40) (actual time=172.123..172.234 rows=100 loops=1)\n  ->  HashAggregate\n        ->  Index Only Scan using idx_orders_status_cat_covering on orders\n              Index Cond: ((status = 'completed') AND (created_at >= $1) AND (created_at <= $2))\n              Heap Fetches: 0\nExecution Time: 180.234 ms",
    },
]


def generate_static_scenarios(count: int, output_dir: Path) -> int:
    """
    Generate synthetic training pairs from built-in templates (no LLM needed).
    Useful for bootstrapping when no raw data is available yet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "synthetic_queries.jsonl"
    rng = random.Random(0)
    saved = 0

    with out_path.open("w") as f:
        for i in range(count):
            template = SCENARIO_TEMPLATES[i % len(SCENARIO_TEMPLATES)]
            row_count = rng.choice(
                [100_000, 500_000, 2_000_000, 10_000_000, 50_000_000]
            )

            user_msg = (
                f"Analyze this PostgreSQL performance problem.\n\n"
                f"**Slow query:**\n```sql\n{template['query']}\n```\n\n"
                f"**Schema:**\n```sql\n{template['schema_ddl']}\n```\n\n"
                f"**EXPLAIN ANALYZE (table has {row_count:,} rows):**\n"
                f"```\n{template['explain_before']}\n```"
            )

            assistant_msg = (
                f"## Root Cause\n{template['diagnosis']}\n\n"
                f"## Prescription DDL\n```sql\n{template['prescription_ddl']}\n```\n\n"
                f"## Index Type\n{template['index_type']}\n\n"
                f"## Write Impact\n{template['write_amplification']}\n\n"
                f"## Expected Result\n{template['expected_improvement']}\n\n"
                f"## EXPLAIN After Optimization\n```\n{template['explain_after']}\n```"
            )

            pair = {
                "conversations": [
                    {"from": "system", "value": QUERYMEDIC_SYSTEM},
                    {"from": "human", "value": user_msg},
                    {"from": "gpt", "value": assistant_msg},
                ],
                "metadata": {
                    "source": "synthetic_template",
                    "category": template["category"],
                    "row_count": row_count,
                    "quality_score": 0.85,
                },
            }
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            saved += 1

    logger.success(f"Generated {saved} static synthetic scenarios → {out_path}")
    return saved


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="QueryMedic LLM synthesis pipeline")
    parser.add_argument("--backend", choices=["claude", "vllm"], default="claude")
    parser.add_argument(
        "--vllm-urls",
        nargs="+",
        default=None,
        help="vLLM base URLs (e.g., http://localhost:8001/v1)",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--limit", type=int, default=None, help="Max records to process (default: all)"
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Generate static synthetic scenarios only (no LLM)",
    )
    parser.add_argument("--static-count", type=int, default=5000)
    args = parser.parse_args()

    if args.static_only:
        n = generate_static_scenarios(args.static_count, args.output_dir)
        print(f"Generated {n} static scenarios")
    else:
        synthesizer = QueryMedicSynthesizer(
            backend=args.backend,
            vllm_urls=args.vllm_urls,
            workers=args.workers,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
        )
        n = asyncio.run(synthesizer.synthesize_all(limit=args.limit))
        print(f"\nTotal training pairs synthesized: {n:,}")
