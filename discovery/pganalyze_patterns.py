"""
pganalyze_patterns.py — Scrape pganalyze, Citus, and Timescale blog posts
for PostgreSQL query optimization patterns.

These sources are highly authoritative:
  - pganalyze.com/blog: PostgreSQL performance monitoring case studies
  - citusdata.com/blog: Multi-node PostgreSQL scaling patterns
  - timescale.com/blog: Time-series query optimization
  - pganalyze.com/docs: Index advisor documentation
  - dataegret.com/blog: PostgreSQL internals deep dives
  - pgdash.io/blog: PostgreSQL diagnostics

Output format:
  {
    source, url, title, content,
    patterns: [{"pattern", "before_sql", "after_sql", "diagnosis", "index_type"}],
    has_explain_output, has_benchmark_numbers,
    relevance_score
  }

Usage:
    python discovery/pganalyze_patterns.py --all
    python discovery/pganalyze_patterns.py --sources pganalyze citus
"""

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp
from loguru import logger

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "raw" / "pg_patterns"

# Blog sources with known PostgreSQL optimization content
BLOG_SOURCES = {
    "pganalyze": {
        "name": "pganalyze Blog",
        "base_url": "https://pganalyze.com",
        "blog_url": "https://pganalyze.com/blog",
        "link_pattern": re.compile(r'/blog/[\w-]+'),
    },
    "citus": {
        "name": "Citus Data Blog",
        "base_url": "https://www.citusdata.com",
        "blog_url": "https://www.citusdata.com/blog",
        "link_pattern": re.compile(r'/blog/[\w-]+'),
    },
    "timescale": {
        "name": "Timescale Blog",
        "base_url": "https://www.timescale.com",
        "blog_url": "https://www.timescale.com/blog",
        "link_pattern": re.compile(r'/blog/[\w-]+'),
    },
    "pgdash": {
        "name": "PgDash Blog",
        "base_url": "https://pgdash.io",
        "blog_url": "https://pgdash.io/blog",
        "link_pattern": re.compile(r'/blog/[\w-]+'),
    },
    "postgresql_wiki": {
        "name": "PostgreSQL Wiki",
        "base_url": "https://wiki.postgresql.org",
        "blog_url": "https://wiki.postgresql.org/wiki/Category:Performance",
        "link_pattern": re.compile(r'/wiki/[\w:]+'),
    },
}

# Patterns for extracting SQL and EXPLAIN output
SQL_CODE_PATTERN = re.compile(
    r'```(?:sql|SQL|postgresql|plpgsql)?\s*\n(.*?)```',
    re.DOTALL,
)
EXPLAIN_PATTERN = re.compile(
    r'(?:Seq\s+Scan|Index\s+Scan|Index\s+Only\s+Scan|Bitmap\s+Heap\s+Scan|'
    r'Hash\s+Join|Merge\s+Join|Nested\s+Loop|Sort|HashAggregate|'
    r'GroupAggregate|Limit)\s+on\s+\w+',
    re.IGNORECASE,
)
EXECUTION_TIME_PATTERN = re.compile(
    r'(?:Execution\s+Time|Planning\s+Time):\s*([\d.]+)\s*ms',
    re.IGNORECASE,
)
INDEX_DDL_PATTERN = re.compile(
    r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?(\w+)',
    re.IGNORECASE,
)

# PostgreSQL-specific optimization keywords
PG_KEYWORDS = [
    "seq scan", "index scan", "bitmap heap scan",
    "hash join", "nested loop", "merge join",
    "explain analyze", "explain (analyze",
    "execution time", "planning time",
    "rows removed by filter", "actual rows",
    "b-tree index", "gin index", "gist index", "brin index",
    "partial index", "covering index", "index-only scan",
    "create index", "create index concurrently",
    "analyze", "vacuum analyze",
    "pg_stat_statements", "auto_explain",
    "work_mem", "shared_buffers", "effective_cache_size",
    "row estimation", "statistics", "pg_stats",
    "ctid", "heap fetch", "lossy",
    "n+1 query", "n+1",
]


@dataclass
class QueryPattern:
    """A single query optimization pattern."""
    pattern_type: str              # missing_index | seq_scan | n_plus_1 | join_order | etc.
    before_sql: str
    after_sql: str
    diagnosis: str
    index_ddl: str
    execution_time_before_ms: Optional[float]
    execution_time_after_ms: Optional[float]
    improvement_factor: Optional[float]


@dataclass
class PGBlogPost:
    """A PostgreSQL optimization blog post with extracted patterns."""
    post_id: str
    source: str
    source_name: str
    url: str
    title: str
    content: str
    sql_snippets: list[str]
    explain_outputs: list[str]
    patterns: list[dict]
    has_explain_output: bool
    has_benchmark_numbers: bool
    relevance_score: float
    index_types_mentioned: list[str]


def html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    if BS4_AVAILABLE:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    text = re.sub(r'<[^>]+>', ' ', html)
    return re.sub(r'\s+', ' ', text).strip()


def extract_links(html: str, base_url: str, pattern: re.Pattern) -> list[str]:
    """Extract matching links from HTML."""
    links = set()
    if BS4_AVAILABLE:
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("http"):
                href = base_url.rstrip("/") + "/" + href.lstrip("/")
            if pattern.search(href):
                links.add(href)
    else:
        for m in re.finditer(r'href=["\']([^"\']+)["\']', html):
            href = m.group(1)
            if not href.startswith("http"):
                href = base_url.rstrip("/") + "/" + href.lstrip("/")
            if pattern.search(href):
                links.add(href)
    return list(links)


def extract_title(html: str) -> str:
    """Extract page title."""
    if BS4_AVAILABLE:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title:
            return soup.title.get_text().strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text().strip()
    m = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
    return re.sub(r'\s+', ' ', m.group(1)).strip() if m else ""


def score_relevance(text: str) -> tuple[float, bool, bool, list[str]]:
    """Score PostgreSQL relevance."""
    text_lower = text.lower()

    keyword_hits = sum(1 for kw in PG_KEYWORDS if kw in text_lower)
    keyword_score = min(1.0, keyword_hits / 5)

    has_explain = bool(EXPLAIN_PATTERN.search(text))
    has_numbers = bool(EXECUTION_TIME_PATTERN.search(text))

    index_types = []
    for itype in ["b-tree", "gin", "gist", "brin", "hash", "btree", "spgist"]:
        if itype in text_lower:
            index_types.append(itype)

    score = (
        keyword_score * 0.5 +
        (0.25 if has_explain else 0) +
        (0.25 if has_numbers else 0)
    )

    return round(min(1.0, score), 3), has_explain, has_numbers, index_types


def extract_patterns(text: str) -> list[dict]:
    """Extract query optimization patterns from text."""
    patterns = []

    # Look for SQL code blocks
    sql_blocks = SQL_CODE_PATTERN.findall(text)

    # Find execution times
    times = []
    for m in EXECUTION_TIME_PATTERN.finditer(text):
        context = text[max(0, m.start() - 50):m.start()].lower()
        label = "before" if "before" in context or "slow" in context else "after"
        times.append((label, float(m.group(1))))

    # Build patterns from consecutive slow/fast SQL pairs
    for i in range(0, len(sql_blocks) - 1, 2):
        before_sql = sql_blocks[i][:500]
        after_sql = sql_blocks[i + 1][:500] if i + 1 < len(sql_blocks) else ""

        # Detect pattern type
        pattern_type = "query_optimization"
        if EXPLAIN_PATTERN.search(before_sql):
            pattern_type = "explain_based"
        if "CREATE INDEX" in before_sql.upper() or "CREATE INDEX" in after_sql.upper():
            pattern_type = "missing_index"
        if "NOT IN" in before_sql.upper():
            pattern_type = "not_in_rewrite"
        if re.search(r'\bLIKE\s+["\']%', before_sql, re.IGNORECASE):
            pattern_type = "leading_wildcard"

        index_ddl = ""
        for m in INDEX_DDL_PATTERN.finditer(before_sql + "\n" + after_sql):
            index_ddl = m.group(0)
            break

        patterns.append({
            "pattern_type": pattern_type,
            "before_sql": before_sql,
            "after_sql": after_sql,
            "diagnosis": "",
            "index_ddl": index_ddl,
        })

    return patterns[:5]  # Cap at 5 patterns per post


class PGAnalyzePatternHarvester:
    """
    Harvests PostgreSQL query optimization patterns from authoritative blogs.
    """

    REQUEST_DELAY = 0.5

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        sources: Optional[list[str]] = None,
        workers: int = 5,
        min_relevance: float = 0.3,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sources = sources or list(BLOG_SOURCES.keys())
        self.workers = workers
        self.min_relevance = min_relevance
        self._semaphore = asyncio.Semaphore(workers)
        self._stats = {"fetched": 0, "saved": 0, "errors": 0}

    async def _fetch(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch HTML with polite delay."""
        for attempt in range(3):
            await asyncio.sleep(self.REQUEST_DELAY)
            try:
                headers = {"User-Agent": "Mozilla/5.0 (QueryMedic Research)"}
                async with session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=30),
                    allow_redirects=True,
                ) as resp:
                    if resp.status == 200:
                        return await resp.text(errors="replace")
                    elif resp.status == 429:
                        await asyncio.sleep(10 * (attempt + 1))
                    else:
                        return None
            except Exception as e:
                logger.debug(f"Fetch error {url}: {e}")
                await asyncio.sleep(2 ** attempt)
        return None

    async def _scrape_post(
        self,
        session: aiohttp.ClientSession,
        url: str,
        source_key: str,
        output_file: Path,
    ) -> bool:
        """Scrape a single blog post."""
        async with self._semaphore:
            html = await self._fetch(session, url)
            if not html:
                self._stats["errors"] += 1
                return False

            config = BLOG_SOURCES[source_key]
            title = extract_title(html)
            content = html_to_text(html)
            self._stats["fetched"] += 1

            if len(content) < 300:
                return False

            score, has_explain, has_numbers, index_types = score_relevance(content)
            if score < self.min_relevance:
                return False

            sql_snippets = SQL_CODE_PATTERN.findall(content)[:10]
            explain_outputs = [
                m.group(0)[:500]
                for m in re.finditer(
                    r'(?:Seq Scan|Index Scan|Bitmap Heap Scan)[^\n]*\n(?:[^\n]+\n){0,10}',
                    content,
                )
            ][:5]

            patterns = extract_patterns(content)

            post = PGBlogPost(
                post_id=hashlib.md5(url.encode()).hexdigest()[:12],
                source=source_key,
                source_name=config["name"],
                url=url,
                title=title,
                content=content[:6000],
                sql_snippets=[s[:500] for s in sql_snippets],
                explain_outputs=explain_outputs,
                patterns=patterns,
                has_explain_output=has_explain,
                has_benchmark_numbers=has_numbers,
                relevance_score=score,
                index_types_mentioned=index_types,
            )

            async with aiofiles.open(output_file, "a") as f:
                await f.write(json.dumps(asdict(post)) + "\n")

            self._stats["saved"] += 1
            return True

    async def _harvest_source(
        self, session: aiohttp.ClientSession, source_key: str
    ) -> int:
        """Harvest all posts from one source."""
        config = BLOG_SOURCES[source_key]
        output_file = self.output_dir / f"{source_key}.jsonl"
        logger.info(f"Harvesting {config['name']}...")

        html = await self._fetch(session, config["blog_url"])
        if not html:
            logger.warning(f"Could not fetch index for {source_key}")
            return 0

        links = extract_links(html, config["base_url"], config["link_pattern"])
        # Filter to unique absolute URLs
        links = list(set(
            l if l.startswith("http") else config["base_url"] + l
            for l in links
        ))

        logger.info(f"  {source_key}: {len(links)} post links found")

        tasks = [self._scrape_post(session, url, source_key, output_file) for url in links]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        saved = sum(1 for r in results if r is True)
        logger.info(f"  {source_key}: {saved}/{len(links)} posts saved")
        return saved

    async def harvest_all(self) -> int:
        """Harvest from all configured sources."""
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10),
        ) as session:
            total = 0
            for source_key in self.sources:
                if source_key not in BLOG_SOURCES:
                    logger.warning(f"Unknown source: {source_key}")
                    continue
                n = await self._harvest_source(session, source_key)
                total += n

        logger.success(
            f"PG pattern harvest complete: {self._stats['saved']} posts, "
            f"{self._stats['errors']} errors"
        )
        return total


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Harvest PostgreSQL optimization patterns from blogs")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--sources", nargs="+", default=None,
                        choices=list(BLOG_SOURCES.keys()))
    parser.add_argument("--output-dir", default="data/raw/pg_patterns")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    sources = list(BLOG_SOURCES.keys()) if args.all else args.sources

    harvester = PGAnalyzePatternHarvester(
        output_dir=args.output_dir,
        sources=sources,
        workers=args.workers,
    )
    n = asyncio.run(harvester.harvest_all())
    print(f"\nTotal patterns harvested: {n:,}")
