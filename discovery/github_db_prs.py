"""
github_db_prs.py — Harvest GitHub PRs containing DB migrations, ORM query fixes,
and query optimization changes for PostgreSQL training data.

Searches for merged PRs that demonstrate:
  - Slow query → optimized query with EXPLAIN evidence
  - Missing index additions (CREATE INDEX migrations)
  - ORM N+1 fixes (select_related, eager loading, prefetch)
  - Query rewrite patterns (NOT IN→NOT EXISTS, subquery→join, etc.)
  - Schema evolution: adding partial/covering indexes

Output format (JSONL):
  {
    repo, pr_number, pr_url, title, body,
    diff_hunks: [{filename, patch}],
    migration_sql: [str],
    explain_snippets: [str],
    query_patterns: [{before, after, pattern_type}],
    has_explain_evidence, has_migration, has_orm_fix,
    relevance_score, fetched_at
  }

Usage:
    export GITHUB_TOKEN=ghp_xxxx
    python discovery/github_db_prs.py --all --limit 3000
    python discovery/github_db_prs.py --languages python --limit 1000
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp
from loguru import logger

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "raw" / "github_db_prs"

# ─── Search Queries ────────────────────────────────────────────────────────────

# GitHub PR search queries targeting DB optimization patterns
SEARCH_QUERIES = {
    "migration_index": [
        "CREATE INDEX CONCURRENTLY is:pr is:merged language:python",
        "CREATE INDEX CONCURRENTLY is:pr is:merged language:ruby",
        "CREATE INDEX CONCURRENTLY is:pr is:merged language:go",
        "add_index migration slow query is:pr is:merged",
        "database migration add index performance is:pr is:merged",
    ],
    "explain_analyze": [
        "EXPLAIN ANALYZE seq scan fix is:pr is:merged",
        "EXPLAIN ANALYZE index scan is:pr is:merged language:python",
        "explain analyze postgres optimization is:pr is:merged",
        "seq scan index performance fix is:pr is:merged",
        "rows removed by filter optimization is:pr is:merged",
    ],
    "orm_n_plus_1": [
        "select_related prefetch_related N+1 fix is:pr is:merged language:python",
        "N+1 query fix eager loading is:pr is:merged",
        "includes preload eager_load rails n+1 is:pr is:merged language:ruby",
        "n plus 1 query optimization is:pr is:merged",
        "lazy loading fix database performance is:pr is:merged",
    ],
    "query_rewrite": [
        "NOT IN to NOT EXISTS refactor performance is:pr is:merged",
        "subquery to join optimization is:pr is:merged language:python",
        "query optimization postgres index is:pr is:merged language:go",
        "slow query rewrite index performance is:pr is:merged",
        "covering index query optimization is:pr is:merged",
    ],
    "django_optimization": [
        "django queryset optimization select_related is:pr is:merged",
        "django ORM slow query fix is:pr is:merged",
        "django prefetch_related fix is:pr is:merged",
        "django database index migration is:pr is:merged",
    ],
    "sqlalchemy_optimization": [
        "sqlalchemy lazy loading n+1 fix is:pr is:merged",
        "sqlalchemy joinedload optimization is:pr is:merged",
        "sqlalchemy query optimization index is:pr is:merged",
        "sqlalchemy subquery join refactor is:pr is:merged",
    ],
    "activerecord_optimization": [
        "activerecord includes eager loading fix is:pr is:merged language:ruby",
        "rails migration add_index slow query is:pr is:merged language:ruby",
        "activerecord n+1 fix is:pr is:merged language:ruby",
    ],
}

# ─── Detection Patterns ────────────────────────────────────────────────────────

EXPLAIN_PATTERN = re.compile(
    r"(?:Seq\s+Scan|Index\s+Scan|Index\s+Only\s+Scan|Bitmap\s+Heap\s+Scan|"
    r"Hash\s+Join|Merge\s+Join|Nested\s+Loop|Sort\s+on|HashAggregate|"
    r"GroupAggregate|Limit).*?cost=[\d.]+\.\.",
    re.IGNORECASE | re.DOTALL,
)
EXECUTION_TIME_PATTERN = re.compile(
    r"(?:Execution\s+Time|Planning\s+Time):\s*([\d.]+)\s*ms",
    re.IGNORECASE,
)
INDEX_DDL_PATTERN = re.compile(
    r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?[\w.]+\s+"
    r"ON\s+[\w.]+\s*(?:USING\s+\w+\s*)?\([^)]+\)",
    re.IGNORECASE,
)
MIGRATION_PATTERN = re.compile(
    r"(?:add_index|AddIndex|CREATE\s+INDEX|add_column.*index|"
    r"op\.create_index|schema\.Index|migrate\.AddIndex)",
    re.IGNORECASE,
)
ORM_N1_PATTERN = re.compile(
    r"(?:select_related|prefetch_related|includes|eager_load|preload|"
    r"joinedload|subqueryload|contains_eager|load_only)",
    re.IGNORECASE,
)
SQL_BLOCK_PATTERN = re.compile(
    r"```(?:sql|SQL|postgresql|pgsql)?\s*\n(.*?)```",
    re.DOTALL,
)
BEFORE_AFTER_PATTERN = re.compile(
    r"(?:before|slow|original|old|was)[\s\S]{0,200}?"
    r"(SELECT[\s\S]{20,500}?)(?:after|fast|optimized|new|now|->|=>)",
    re.IGNORECASE,
)


@dataclass
class DiffHunk:
    """A single diff hunk from a PR."""

    filename: str
    patch: str


@dataclass
class QueryPattern:
    """An extracted before/after query optimization pattern."""

    pattern_type: (
        str  # missing_index | n_plus_1 | query_rewrite | covering_index | etc.
    )
    before_sql: str
    after_sql: str
    index_ddl: str
    execution_time_before_ms: Optional[float]
    execution_time_after_ms: Optional[float]


@dataclass
class DBPullRequest:
    """A GitHub PR containing database query optimization."""

    repo: str
    pr_number: int
    pr_url: str
    title: str
    body: str
    diff_hunks: list[dict]  # [{filename, patch}]
    migration_sql: list[str]
    explain_snippets: list[str]
    query_patterns: list[dict]
    has_explain_evidence: bool
    has_migration: bool
    has_orm_fix: bool
    relevance_score: float
    fetched_at: float = field(default_factory=time.time)


def _extract_explain_snippets(text: str) -> list[str]:
    """Extract EXPLAIN ANALYZE output snippets from text."""
    snippets = []
    for m in EXPLAIN_PATTERN.finditer(text):
        # Capture context around the plan node
        start = m.start()
        end = min(len(text), m.end() + 500)
        snippet = text[start:end].strip()[:1000]
        if snippet:
            snippets.append(snippet)
    return snippets[:5]


def _extract_migration_sql(diff_patch: str) -> list[str]:
    """Extract SQL DDL statements from a diff patch."""
    statements = []
    for m in INDEX_DDL_PATTERN.finditer(diff_patch):
        statements.append(m.group(0).strip()[:500])
    for m in MIGRATION_PATTERN.finditer(diff_patch):
        # Grab surrounding context for migration functions
        start = max(0, m.start() - 10)
        end = min(len(diff_patch), m.end() + 200)
        stmt = diff_patch[start:end].strip()[:300]
        if stmt not in statements:
            statements.append(stmt)
    return statements[:10]


def _extract_query_patterns(body: str, diff_text: str) -> list[dict]:
    """Extract before/after query patterns from PR body and diff."""
    patterns = []
    all_text = body + "\n" + diff_text

    # SQL code blocks in PR body
    sql_blocks = SQL_BLOCK_PATTERN.findall(all_text)

    # Build consecutive before/after pairs from SQL blocks
    for i in range(0, len(sql_blocks) - 1, 2):
        before_sql = sql_blocks[i][:500].strip()
        after_sql = sql_blocks[i + 1][:500].strip() if i + 1 < len(sql_blocks) else ""

        if not before_sql:
            continue

        # Detect pattern type
        pattern_type = "query_optimization"
        combined = (before_sql + " " + after_sql).upper()
        if "CREATE INDEX" in combined or MIGRATION_PATTERN.search(combined):
            pattern_type = "missing_index"
        elif ORM_N1_PATTERN.search(after_sql) and not ORM_N1_PATTERN.search(before_sql):
            pattern_type = "n_plus_1_fix"
        elif "NOT IN" in before_sql.upper() and "NOT EXISTS" in after_sql.upper():
            pattern_type = "not_in_rewrite"
        elif re.search(r"""\bLIKE\s+['"]%""", before_sql, re.IGNORECASE):
            pattern_type = "leading_wildcard"
        elif "INCLUDE" in combined:
            pattern_type = "covering_index"

        # Extract execution times from surrounding context
        times_before = []
        times_after = []
        for m in EXECUTION_TIME_PATTERN.finditer(all_text):
            context = all_text[max(0, m.start() - 100) : m.start()].lower()
            val = float(m.group(1))
            if any(w in context for w in ["before", "slow", "was", "old"]):
                times_before.append(val)
            else:
                times_after.append(val)

        index_ddl = ""
        for m in INDEX_DDL_PATTERN.finditer(before_sql + "\n" + after_sql):
            index_ddl = m.group(0)
            break

        patterns.append(
            {
                "pattern_type": pattern_type,
                "before_sql": before_sql,
                "after_sql": after_sql,
                "index_ddl": index_ddl,
                "execution_time_before_ms": times_before[0] if times_before else None,
                "execution_time_after_ms": times_after[0] if times_after else None,
            }
        )

    return patterns[:5]


def score_relevance(
    title: str,
    body: str,
    diff_text: str,
    has_explain: bool,
    has_migration: bool,
    has_orm_fix: bool,
) -> float:
    """Score a PR's relevance for DB optimization training data."""
    score = 0.0
    text_lower = (title + " " + body + " " + diff_text).lower()

    # High-signal keywords
    high_signal = [
        "seq scan",
        "index scan",
        "explain analyze",
        "execution time",
        "query plan",
        "n+1",
        "missing index",
        "slow query",
        "query optimization",
        "index concurrently",
        "rows removed",
        "nested loop",
        "hash join",
    ]
    hits = sum(1 for kw in high_signal if kw in text_lower)
    score += min(0.4, hits * 0.05)

    # Structural bonuses
    if has_explain:
        score += 0.3
    if has_migration:
        score += 0.15
    if has_orm_fix:
        score += 0.1
    if EXECUTION_TIME_PATTERN.search(body):
        score += 0.05

    return round(min(1.0, score), 3)


class GitHubDBPRHarvester:
    """
    Harvests GitHub PRs containing database query optimization patterns.

    Two-phase approach:
      1. Search PRs via GitHub Issues Search API (query text matching)
      2. Fetch PR diff and details for each relevant PR
    """

    SEARCH_API = "https://api.github.com/search/issues"
    PR_API = "https://api.github.com/repos/{repo}/pulls/{number}"
    FILES_API = "https://api.github.com/repos/{repo}/pulls/{number}/files"
    REQUEST_DELAY = 1.0  # seconds between requests (search API: 30 req/min)
    MAX_DIFF_BYTES = 100_000

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        tokens: Optional[list[str]] = None,
        workers: int = 5,
        min_relevance: float = 0.3,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_token = os.environ.get("GITHUB_TOKEN", "")
        self.tokens = tokens or (
            [t for t in raw_token.split(",") if t] if raw_token else []
        )
        self.workers = workers
        self.min_relevance = min_relevance
        self._semaphore = asyncio.Semaphore(workers)
        self._token_idx = 0
        self._stats = {"searched": 0, "fetched": 0, "saved": 0, "errors": 0}

    def _auth_headers(self) -> dict:
        if not self.tokens:
            return {"Accept": "application/vnd.github.v3+json"}
        token = self.tokens[self._token_idx % len(self.tokens)]
        self._token_idx += 1
        return {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

    async def _get(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: Optional[dict] = None,
        retries: int = 3,
    ) -> Optional[dict | list]:
        """Make a GET request to GitHub API with rate limit handling."""
        for attempt in range(retries):
            await asyncio.sleep(self.REQUEST_DELAY)
            try:
                async with session.get(
                    url,
                    params=params,
                    headers=self._auth_headers(),
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    # Rate limit check
                    remaining = int(resp.headers.get("X-RateLimit-Remaining", 1))
                    if remaining < 5:
                        reset_at = int(
                            resp.headers.get("X-RateLimit-Reset", time.time() + 60)
                        )
                        wait = max(1, reset_at - time.time() + 2)
                        logger.warning(
                            f"Rate limit low ({remaining}), sleeping {wait:.0f}s"
                        )
                        await asyncio.sleep(wait)

                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 403:
                        await asyncio.sleep(30 * (attempt + 1))
                    elif resp.status == 422:
                        return None  # Unprocessable — bad query
                    elif resp.status in (404, 451):
                        return None
                    else:
                        logger.debug(f"HTTP {resp.status} for {url}")
                        await asyncio.sleep(5 * (attempt + 1))
            except Exception as e:
                logger.debug(f"Request error ({url}): {e}")
                await asyncio.sleep(2**attempt)
        return None

    async def _search_prs(
        self,
        session: aiohttp.ClientSession,
        query: str,
        per_page: int = 100,
        max_pages: int = 10,
    ) -> list[dict]:
        """Search GitHub PRs using the Issues Search API."""
        results = []
        for page in range(1, max_pages + 1):
            data = await self._get(
                session,
                self.SEARCH_API,
                params={
                    "q": query,
                    "per_page": per_page,
                    "page": page,
                    "sort": "created",
                    "order": "desc",
                },
            )
            if not data or not isinstance(data, dict):
                break
            items = data.get("items", [])
            if not items:
                break
            results.extend(items)
            self._stats["searched"] += len(items)
            total = data.get("total_count", 0)
            if len(results) >= min(1000, total):
                break

        return results

    async def _fetch_pr_files(
        self,
        session: aiohttp.ClientSession,
        repo: str,
        pr_number: int,
    ) -> list[dict]:
        """Fetch the list of files changed in a PR."""
        url = self.FILES_API.format(repo=repo, number=pr_number)
        data = await self._get(session, url, params={"per_page": 100})
        if not data or not isinstance(data, list):
            return []
        return data

    def _extract_repo(self, issue_url: str) -> Optional[str]:
        """Extract 'owner/repo' from a GitHub issue/PR URL."""
        m = re.search(r"github\.com/([^/]+/[^/]+)/", issue_url)
        return m.group(1) if m else None

    async def _process_pr(
        self,
        session: aiohttp.ClientSession,
        item: dict,
        output_file: Path,
    ) -> bool:
        """Process a single PR: fetch diff, extract patterns, save."""
        async with self._semaphore:
            pr_url = item.get("html_url", "")
            repo = self._extract_repo(pr_url)
            if not repo:
                return False

            pr_number = item.get("number", 0)
            title = item.get("title", "")
            body = item.get("body", "") or ""

            # Fetch PR files/diff
            files = await self._fetch_pr_files(session, repo, pr_number)
            self._stats["fetched"] += 1

            # Extract diff text from relevant file types
            db_file_extensions = {
                ".py",
                ".rb",
                ".go",
                ".java",
                ".ts",
                ".js",
                ".sql",
                ".migration",
                ".ex",
                ".cs",
                ".php",
            }
            db_filename_patterns = re.compile(
                r"(?:migration|migrate|model|schema|repository|repo|"
                r"database|db_|query|orm|activerecord|sequel)",
                re.IGNORECASE,
            )

            diff_hunks = []
            diff_text = ""
            for f in files:
                filename = f.get("filename", "")
                patch = f.get("patch", "") or ""
                if not patch:
                    continue

                ext = Path(filename).suffix.lower()
                is_db_file = (
                    ext in db_file_extensions
                    or db_filename_patterns.search(filename)
                    or ext == ".sql"
                )
                if is_db_file:
                    diff_hunks.append({"filename": filename, "patch": patch[:2000]})
                    diff_text += f"\n# File: {filename}\n{patch[:3000]}"
                    if len(diff_text) > self.MAX_DIFF_BYTES:
                        break

            # Extract features
            explain_snippets = _extract_explain_snippets(body + "\n" + diff_text)
            migration_sql = _extract_migration_sql(diff_text)
            query_patterns = _extract_query_patterns(body, diff_text)

            has_explain = bool(explain_snippets)
            has_migration = bool(migration_sql)
            has_orm_fix = bool(ORM_N1_PATTERN.search(diff_text))

            score = score_relevance(
                title, body, diff_text, has_explain, has_migration, has_orm_fix
            )
            if score < self.min_relevance:
                return False

            pr_record = DBPullRequest(
                repo=repo,
                pr_number=pr_number,
                pr_url=pr_url,
                title=title,
                body=body[:3000],
                diff_hunks=diff_hunks[:10],
                migration_sql=migration_sql,
                explain_snippets=explain_snippets,
                query_patterns=query_patterns,
                has_explain_evidence=has_explain,
                has_migration=has_migration,
                has_orm_fix=has_orm_fix,
                relevance_score=score,
            )

            async with aiofiles.open(str(output_file), "a") as f:
                await f.write(json.dumps(asdict(pr_record)) + "\n")

            self._stats["saved"] += 1
            return True

    async def _harvest_query_group(
        self,
        session: aiohttp.ClientSession,
        group_name: str,
        queries: list[str],
        limit: int,
    ) -> int:
        """Harvest PRs matching a query group."""
        output_file = self.output_dir / f"{group_name}.jsonl"
        seen_urls: set[str] = set()
        saved = 0

        for query in queries:
            if saved >= limit:
                break
            logger.info(f"  Searching [{group_name}]: {query[:60]}...")
            items = await self._search_prs(session, query)
            logger.info(f"    Found {len(items)} PRs")

            tasks = []
            for item in items:
                url = item.get("html_url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                tasks.append(self._process_pr(session, item, output_file))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                batch_saved = sum(1 for r in results if r is True)
                saved += batch_saved
                logger.info(f"    Saved {batch_saved}/{len(tasks)} PRs from this query")

        return saved

    async def harvest_all(self, limit_per_group: int = 500) -> int:
        """Harvest DB optimization PRs from all query groups."""
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            total = 0
            for group_name, queries in SEARCH_QUERIES.items():
                n = await self._harvest_query_group(
                    session, group_name, queries, limit_per_group
                )
                total += n
                logger.info(f"Group [{group_name}]: {n} PRs saved")

        logger.success(
            f"GitHub DB PR harvest complete: "
            f"{self._stats['saved']} saved, {self._stats['errors']} errors, "
            f"{self._stats['searched']} searched"
        )
        return total


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Harvest GitHub DB optimization PRs")
    parser.add_argument("--all", action="store_true", help="Harvest all query groups")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=list(SEARCH_QUERIES.keys()),
        help="Specific query groups to harvest",
    )
    parser.add_argument(
        "--limit", type=int, default=500, help="Max PRs per query group"
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--min-relevance", type=float, default=0.3)
    args = parser.parse_args()

    if not args.all and not args.groups:
        parser.error("Specify --all or --groups")

    active_queries = (
        SEARCH_QUERIES
        if args.all
        else {k: SEARCH_QUERIES[k] for k in args.groups if k in SEARCH_QUERIES}
    )

    harvester = GitHubDBPRHarvester(
        output_dir=Path(args.output_dir),
        workers=args.workers,
        min_relevance=args.min_relevance,
    )
    # QM-7: When --groups is specified, restrict the search to the selected
    # query groups by patching the module-level SEARCH_QUERIES dict so that
    # harvest_all() only iterates over the requested groups.
    if not args.all:
        import sys

        sys.modules[__name__].SEARCH_QUERIES = active_queries

    n = asyncio.run(harvester.harvest_all(limit_per_group=args.limit))
    print(f"\nTotal DB optimization PRs harvested: {n:,}")
