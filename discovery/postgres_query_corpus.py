"""
postgres_query_corpus.py — Build a corpus of slow PostgreSQL queries with EXPLAIN plans.

Data sources:
  1. Stack Overflow API: PostgreSQL performance questions with accepted answers
     containing EXPLAIN ANALYZE before/after optimization
  2. DBA Stack Exchange: query optimization questions (higher signal-to-noise)
  3. Brent Ozar blog: sp_BlitzCache-style slow query patterns (PostgreSQL analog)
  4. pg_activity public datasets: real EXPLAIN plans from production systems
  5. PostgreSQL mailing list archives: pgsql-performance list slow query reports

Output: data/raw/query_corpus/<source>_<hash>.jsonl
Each record: {
    source, url, question, slow_query, schema, explain_before,
    explain_after, optimized_query, diagnosis, index_ddl,
    tags, score, has_explain_before, has_explain_after
}

Usage:
    export STACK_EXCHANGE_KEY=your_key   # optional, raises rate limit
    python discovery/postgres_query_corpus.py --all
    python discovery/postgres_query_corpus.py --source stackoverflow --limit 2000
"""

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from loguru import logger
from stackapi import StackAPI

# QM-13: Use an absolute path anchored to this file's location so the
# collector works correctly regardless of the working directory from which
# it is invoked (e.g. python discovery/postgres_query_corpus.py from repo root).
OUTPUT_DIR = Path(__file__).parents[1] / "data" / "raw" / "query_corpus"

# SQL keywords that signal a slow query question
SLOW_QUERY_SIGNALS = [
    r"slow\s+query",
    r"query\s+slow",
    r"performance\s+issue",
    r"taking\s+too\s+long",
    r"seq\s+scan",
    r"sequential\s+scan",
    r"full\s+table\s+scan",
    r"explain\s+analyze",
    r"explain\s+plan",
    r"query\s+plan",
    r"missing\s+index",
    r"add\s+an?\s+index",
    r"create\s+index",
    r"n\+1",
    r"n\s*\+\s*1\s+query",
    r"join\s+is\s+slow",
    r"hash\s+join",
    r"nested\s+loop",
    r"timeout",
    r"1[0-9]{3,}\s*ms",
    r"seconds\s+to\s+execute",
]
SLOW_SIGNAL_PATTERN = re.compile("|".join(SLOW_QUERY_SIGNALS), re.IGNORECASE)

# EXPLAIN plan indicators in text
EXPLAIN_BEFORE_PATTERN = re.compile(
    r"(?:Seq Scan|Index Scan|Bitmap Heap Scan|Hash Join|Nested Loop|Merge Join|Sort|Aggregate)"
    r".*cost=[\d.]+\.\.",
    re.IGNORECASE | re.DOTALL,
)

# SQL detection pattern
SQL_BLOCK_PATTERN = re.compile(
    r"(?:```(?:sql|postgresql|postgres)?\s*(.*?)```|"
    r"SELECT\s+.+?(?:FROM|WHERE|JOIN|GROUP|ORDER|LIMIT).+?;)",
    re.IGNORECASE | re.DOTALL,
)

# DDL patterns (CREATE INDEX, ALTER TABLE)
DDL_PATTERN = re.compile(
    r"(CREATE\s+(?:UNIQUE\s+)?INDEX\s+.+?;|ALTER\s+TABLE\s+.+?;)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class QueryRecord:
    source: str
    url: str
    question: str
    slow_query: str
    schema: str
    explain_before: str
    explain_after: str
    optimized_query: str
    diagnosis: str
    index_ddl: str
    tags: list
    score: int
    has_explain_before: bool
    has_explain_after: bool
    fetched_at: float = 0.0


class StackExchangeCollector:
    """
    Collect PostgreSQL performance Q&A from Stack Overflow and DBA Stack Exchange.

    Uses the official Stack Exchange API with query filtering for
    questions that contain EXPLAIN plans and query optimization content.
    """

    SITE_TAGS = {
        "stackoverflow": [
            "postgresql",
            "postgresql-performance",
            "query-optimization",
            "sql-performance",
        ],
        "dba": ["postgresql", "query-performance", "index", "explain"],
    }

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        api_key: Optional[str] = None,
    ) -> None:
        self.output_dir = output_dir / "stackoverflow"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("STACK_EXCHANGE_KEY", "")

    def _extract_sql_blocks(self, text: str) -> list[str]:
        """Extract SQL code blocks from markdown/HTML text."""
        blocks = []
        # Fenced code blocks
        for match in re.finditer(
            r"```(?:sql|postgresql|postgres|pgsql)?\n(.*?)```",
            text,
            re.DOTALL | re.IGNORECASE,
        ):
            blocks.append(match.group(1).strip())
        # Inline code with SQL keywords
        for match in re.finditer(r"`([^`]{20,})`", text):
            content = match.group(1)
            if re.search(
                r"\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|EXPLAIN)\b",
                content,
                re.IGNORECASE,
            ):
                blocks.append(content.strip())
        return blocks

    def _extract_explain_plan(self, text: str) -> str:
        """Extract EXPLAIN ANALYZE output from text."""
        # Look for cost= pattern that appears in explain output
        match = re.search(
            r"(?:^|\n)((?:Seq Scan|Index Scan|Bitmap|Hash Join|Nested Loop|Sort|Aggregate|"
            r"Gather|Parallel|Result|Limit|Subquery Scan|CTE Scan|Values Scan|"
            r"Hash|Materialize|Memoize|Unique|SetOp|WindowAgg|GroupAggregate|HashAggregate)"
            r".*?(?:Planning Time|Execution Time|\Z))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            plan = match.group(1)
            # Truncate if very long
            return plan[:3000].strip()
        return ""

    def _extract_index_ddl(self, text: str) -> str:
        """Extract CREATE INDEX statements from text."""
        matches = DDL_PATTERN.findall(text)
        return "\n".join(m.strip() for m in matches[:5]) if matches else ""

    def _parse_question_answer(
        self,
        question: dict,
        answers: list[dict],
        site: str,
    ) -> Optional[QueryRecord]:
        """Parse a Stack Exchange question + accepted answer into a QueryRecord."""
        q_title = question.get("title", "")
        q_body = BeautifulSoup(question.get("body", ""), "lxml").get_text()
        tags = question.get("tags", [])
        score = question.get("score", 0)
        url = question.get("link", "")

        # Check if this is a slow-query / performance question
        if not SLOW_SIGNAL_PATTERN.search(f"{q_title} {q_body}"):
            return None

        # Find the accepted answer (or highest-voted)
        accepted = next((a for a in answers if a.get("is_accepted")), None)
        if not accepted:
            answers_sorted = sorted(
                answers, key=lambda a: a.get("score", 0), reverse=True
            )
            accepted = answers_sorted[0] if answers_sorted else None
        if not accepted or accepted.get("score", 0) < 1:
            return None

        a_body = BeautifulSoup(accepted.get("body", ""), "lxml").get_text()

        # Extract components
        q_sql_blocks = self._extract_sql_blocks(q_body)
        a_sql_blocks = self._extract_sql_blocks(a_body)

        slow_query = q_sql_blocks[0] if q_sql_blocks else ""
        optimized_query = a_sql_blocks[0] if a_sql_blocks else ""

        explain_before = self._extract_explain_plan(q_body)
        explain_after = self._extract_explain_plan(a_body)
        index_ddl = self._extract_index_ddl(a_body)

        # Require at least a slow query or an EXPLAIN plan
        if not slow_query and not explain_before:
            return None

        # Diagnosis: first paragraph of accepted answer
        a_paragraphs = [p.strip() for p in a_body.split("\n\n") if p.strip()]
        diagnosis = a_paragraphs[0][:500] if a_paragraphs else ""

        # Schema: look for CREATE TABLE statements in the question
        schema_matches = re.findall(
            r"CREATE\s+TABLE\s+.*?;", q_body, re.DOTALL | re.IGNORECASE
        )
        schema = "\n\n".join(schema_matches[:3])[:2000] if schema_matches else ""

        return QueryRecord(
            source=f"stackexchange_{site}",
            url=url,
            question=f"{q_title}\n\n{q_body[:1000]}",
            slow_query=slow_query[:2000],
            schema=schema,
            explain_before=explain_before,
            explain_after=explain_after,
            optimized_query=optimized_query[:2000],
            diagnosis=diagnosis,
            index_ddl=index_ddl,
            tags=tags,
            score=score,
            has_explain_before=bool(explain_before),
            has_explain_after=bool(explain_after),
            fetched_at=time.time(),
        )

    def collect(self, site: str = "stackoverflow", limit: int = 2000) -> int:
        """Collect slow-query Q&A from a Stack Exchange site."""
        if site not in self.SITE_TAGS:
            raise ValueError(
                f"Unknown site: {site}. Choose from {list(self.SITE_TAGS)}"
            )

        api = StackAPI(site)
        if self.api_key:
            api.key = self.api_key

        saved = 0
        tags_to_query = self.SITE_TAGS[site]

        for tag in tags_to_query:
            if saved >= limit:
                break

            logger.info(f"  Fetching [{tag}] questions from {site}...")
            try:
                # Questions tagged with this tag, sorted by votes
                questions = api.fetch(
                    "questions",
                    tagged=tag,
                    sort="votes",
                    order="desc",
                    filter="withbody",
                    pagesize=100,
                )
                items = questions.get("items", [])
                logger.info(f"    Got {len(items)} questions for tag [{tag}]")

                for q in items:
                    if saved >= limit:
                        break

                    q_id = q.get("question_id")
                    if not q_id:
                        continue

                    # Fetch answers
                    try:
                        answers_resp = api.fetch(
                            f"questions/{q_id}/answers",
                            filter="withbody",
                        )
                        answers = answers_resp.get("items", [])
                    except Exception as e:
                        logger.debug(f"    Failed to fetch answers for {q_id}: {e}")
                        continue

                    record = self._parse_question_answer(q, answers, site)
                    if record is None:
                        continue

                    # Save to file
                    h = hashlib.md5(record.url.encode(), usedforsecurity=False).hexdigest()[:10]
                    out_path = self.output_dir / f"{site}_{tag}_{h}.jsonl"
                    with open(out_path, "w") as f:
                        f.write(json.dumps(asdict(record)) + "\n")
                    saved += 1

                    time.sleep(0.1)  # polite rate limiting

            except Exception as e:
                logger.warning(f"  Error collecting from [{tag}] on {site}: {e}")
                continue

        logger.info(f"  {site} [{','.join(tags_to_query)}]: {saved} records saved")
        return saved


class PgActivityCrawler:
    """
    Collect EXPLAIN plans from pg_activity datasets and related public sources.

    Targets:
      - GitHub repos with collected PostgreSQL EXPLAIN plans
      - PostgreSQL wiki performance case studies
      - pganalyze blog (explain plan analysis articles)
    """

    SOURCES = [
        {
            "name": "pganalyze_blog",
            "base_url": "https://pganalyze.com",
            "index_urls": [
                "https://pganalyze.com/blog",
                "https://pganalyze.com/blog/page/2",
            ],
        },
        {
            "name": "postgres_wiki_perf",
            "base_url": "https://wiki.postgresql.org",
            "index_urls": [
                "https://wiki.postgresql.org/wiki/Performance_Optimization",
                "https://wiki.postgresql.org/wiki/Slow_Query_Questions",
            ],
        },
    ]

    def __init__(self, output_dir: Path = OUTPUT_DIR) -> None:
        self.output_dir = output_dir / "pg_activity"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def _fetch_page(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> Optional[str]:
        try:
            async with session.get(
                url,
                headers={
                    "User-Agent": "QueryMedic-Research/1.0 (github.com/calebnewtonusc/querymedic)"
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    return await resp.text(errors="replace")
        except Exception as e:
            logger.debug(f"Failed to fetch {url}: {e}")
        return None

    def _extract_query_content(
        self, html: str, source_name: str
    ) -> Optional[QueryRecord]:
        """Extract query optimization content from a blog post or wiki page."""
        soup = BeautifulSoup(html, "lxml")

        # Get page title
        title = soup.title.string if soup.title else ""

        # Get main content
        content_el = (
            soup.select_one("article")
            or soup.select_one("div.post-content")
            or soup.select_one("main")
            or soup.select_one("div#content")
        )
        if not content_el:
            return None

        text = content_el.get_text(separator="\n", strip=True)

        # Check relevance
        if not SLOW_SIGNAL_PATTERN.search(f"{title} {text[:1000]}"):
            return None

        # Extract SQL blocks from the page
        sql_blocks = []
        for code_el in content_el.find_all(["code", "pre"]):
            code_text = code_el.get_text(strip=True)
            if re.search(
                r"\b(?:SELECT|EXPLAIN|CREATE\s+INDEX)\b", code_text, re.IGNORECASE
            ):
                sql_blocks.append(code_text[:1500])

        if not sql_blocks:
            return None

        # Classify blocks as EXPLAIN output vs SQL vs DDL
        explain_plans = [b for b in sql_blocks if re.search(r"cost=[\d.]+\.\.", b)]
        sql_queries = [
            b for b in sql_blocks if re.search(r"\bSELECT\b", b, re.IGNORECASE)
        ]
        ddl = [
            b for b in sql_blocks if re.search(r"\bCREATE\s+INDEX\b", b, re.IGNORECASE)
        ]

        if not explain_plans and not sql_queries:
            return None

        # Heuristic: first explain is "before", last is "after" (if multiple)
        explain_before = explain_plans[0] if explain_plans else ""
        explain_after = explain_plans[-1] if len(explain_plans) > 1 else ""
        slow_query = sql_queries[0] if sql_queries else ""
        optimized_query = sql_queries[-1] if len(sql_queries) > 1 else ""

        # Extract diagnosis from first few paragraphs
        paras = [
            p.get_text(strip=True)
            for p in content_el.find_all("p")
            if len(p.get_text(strip=True)) > 50
        ]
        diagnosis = " ".join(paras[:3])[:800] if paras else ""

        return QueryRecord(
            source=source_name,
            url="",  # set by caller
            question=str(title),
            slow_query=slow_query,
            schema="",
            explain_before=explain_before,
            explain_after=explain_after,
            optimized_query=optimized_query,
            diagnosis=diagnosis,
            index_ddl="\n".join(ddl[:3]),
            tags=[],
            score=0,
            has_explain_before=bool(explain_before),
            has_explain_after=bool(explain_after),
            fetched_at=time.time(),
        )

    async def collect_source(
        self,
        session: aiohttp.ClientSession,
        source_config: dict,
    ) -> int:
        """Collect query records from a single source."""
        source_name = source_config["name"]
        base_url = source_config["base_url"]
        saved = 0
        visited: set[str] = set()
        to_crawl = list(source_config["index_urls"])

        from urllib.parse import urljoin, urlparse

        while to_crawl:
            batch, to_crawl = to_crawl[:5], to_crawl[5:]

            for url in batch:
                if url in visited:
                    continue
                visited.add(url)

                await asyncio.sleep(1.0)  # polite crawl delay
                html = await self._fetch_page(session, url)
                if not html:
                    continue

                record = self._extract_query_content(html, source_name)
                if record:
                    record.url = url
                    h = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:10]
                    out_path = self.output_dir / f"{source_name}_{h}.jsonl"
                    async with aiofiles.open(str(out_path), "w") as f:
                        await f.write(json.dumps(asdict(record)) + "\n")
                    saved += 1

                # Discover new links on index pages
                if url in source_config["index_urls"]:
                    soup = BeautifulSoup(html, "lxml")
                    domain = urlparse(base_url).netloc
                    for a in soup.find_all("a", href=True):
                        href = urljoin(base_url, a["href"])
                        parsed = urlparse(href)
                        if parsed.netloc == domain and href not in visited:
                            to_crawl.append(href)

        logger.info(f"  {source_name}: {saved} records saved")
        return saved

    async def collect_all(self) -> int:
        """Collect from all configured sources."""
        total = 0
        async with aiohttp.ClientSession() as session:
            for source in self.SOURCES:
                n = await self.collect_source(session, source)
                total += n
        return total


def stream_all_records(data_dir: Path = OUTPUT_DIR) -> Iterator[QueryRecord]:
    """Iterate over all QueryRecord objects from the corpus directory."""
    for jsonl_file in sorted(data_dir.rglob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def build_training_pairs(
    data_dir: Path = OUTPUT_DIR,
    output_path: Optional[Path] = None,
    min_score: int = 2,
    require_explain: bool = False,
) -> int:
    """
    Post-process raw corpus into clean SFT training pairs.
    Writes to data/training/sft_pairs.jsonl.
    """
    if output_path is None:
        output_path = Path("data/training/sft_pairs.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved = 0
    with open(output_path, "w") as out_f:
        for record in stream_all_records(data_dir):
            # Quality filters
            if record.get("score", 0) < min_score and record.get(
                "source", ""
            ).startswith("stackexchange"):
                continue
            if require_explain and not record.get("has_explain_before"):
                continue
            if not record.get("slow_query") and not record.get("explain_before"):
                continue

            # Format as SFT training example
            user_parts = []
            if record.get("slow_query"):
                user_parts.append(f"Slow query:\n```sql\n{record['slow_query']}\n```")
            if record.get("schema"):
                user_parts.append(f"Schema:\n```sql\n{record['schema']}\n```")
            if record.get("explain_before"):
                user_parts.append(
                    f"EXPLAIN ANALYZE output:\n```\n{record['explain_before']}\n```"
                )

            if not user_parts:
                continue

            assistant_parts = []
            if record.get("diagnosis"):
                assistant_parts.append(f"**Diagnosis:** {record['diagnosis']}")
            if record.get("index_ddl"):
                assistant_parts.append(
                    f"**Recommended index:**\n```sql\n{record['index_ddl']}\n```"
                )
            if record.get("optimized_query"):
                assistant_parts.append(
                    f"**Optimized query:**\n```sql\n{record['optimized_query']}\n```"
                )
            if record.get("explain_after"):
                assistant_parts.append(
                    f"**EXPLAIN after optimization:**\n```\n{record['explain_after']}\n```"
                )

            if not assistant_parts:
                continue

            example = {
                "conversations": [
                    {
                        "role": "system",
                        "content": "You are QueryMedic, a PostgreSQL query optimization specialist.",
                    },
                    {"role": "user", "content": "\n\n".join(user_parts)},
                    {"role": "assistant", "content": "\n\n".join(assistant_parts)},
                ],
                "metadata": {
                    "source": record.get("source", ""),
                    "url": record.get("url", ""),
                    "score": record.get("score", 0),
                    "has_explain_before": record.get("has_explain_before", False),
                    "has_explain_after": record.get("has_explain_after", False),
                    "tags": record.get("tags", []),
                },
            }
            out_f.write(json.dumps(example) + "\n")
            saved += 1

    logger.info(f"Built {saved} SFT training pairs → {output_path}")
    return saved


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Build PostgreSQL query optimization corpus."
    )
    parser.add_argument("--all", action="store_true", help="Collect from all sources")
    parser.add_argument(
        "--source",
        choices=["stackoverflow", "dba", "pg_activity"],
        help="Specific source",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Max records to collect (Stack Exchange)",
    )
    parser.add_argument(
        "--build-pairs",
        action="store_true",
        help="Build SFT training pairs from existing data",
    )
    parser.add_argument(
        "--require-explain",
        action="store_true",
        help="Only include records with EXPLAIN output",
    )
    parser.add_argument("--stats", action="store_true", help="Print corpus statistics")
    args = parser.parse_args()

    if args.stats:
        total = with_explain = with_sql = 0
        for record in stream_all_records():
            total += 1
            if record.get("has_explain_before"):
                with_explain += 1
            if record.get("slow_query"):
                with_sql += 1
        print(f"Total records: {total:,}")
        print(
            f"With EXPLAIN before: {with_explain:,} ({100 * with_explain / max(total, 1):.1f}%)"
        )
        print(
            f"With slow query SQL: {with_sql:,} ({100 * with_sql / max(total, 1):.1f}%)"
        )
        raise SystemExit(0)

    if args.build_pairs:
        build_training_pairs(require_explain=args.require_explain)
        raise SystemExit(0)

    if args.all or args.source == "stackoverflow":
        collector = StackExchangeCollector()
        n = collector.collect("stackoverflow", limit=args.limit)
        print(f"Stack Overflow: {n} records")

    if args.all or args.source == "dba":
        collector = StackExchangeCollector()
        n = collector.collect("dba", limit=args.limit)
        print(f"DBA Stack Exchange: {n} records")

    if args.all or args.source == "pg_activity":
        crawler = PgActivityCrawler()
        n = asyncio.run(crawler.collect_all())
        print(f"pg_activity/blog sources: {n} records")
