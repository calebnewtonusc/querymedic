"""
db_blog_crawler.py - Crawl high-scale database engineering blogs.

Target sources:
    - Citus Data Blog (PostgreSQL at scale, partitioning, distributed)
    - 2ndQuadrant Blog (PostgreSQL internals)
    - Planet PostgreSQL (community aggregator)
    - Discord Engineering (PostgreSQL at 10B+ rows)
    - Slack Engineering (PostgreSQL MVCC, vacuuming)
    - Shopify Engineering (MySQL optimization)
    - GitHub Engineering (MySQL at scale)
    - Use The Index, Luke (Markus Winand — indexing deep dives)

Usage:
    python discovery/db_blog_crawler.py --all
    python discovery/db_blog_crawler.py --source citus
"""

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "raw" / "blogs"

DB_KEYWORDS = [
    "index", "query", "explain", "execution plan", "sequential scan", "index scan",
    "performance", "optimization", "slow query", "vacuum", "analyze", "statistics",
    "gin", "gist", "btree", "brin", "covering index", "partial index",
    "write amplification", "write ahead log", "wal", "mvcc", "bloat",
    "connection pool", "pg_stat", "pg_statio", "autovacuum", "toast",
    "partitioning", "sharding", "replication", "foreign key",
    "n+1", "orm", "join", "hash join", "nested loop", "merge join",
]


@dataclass
class DBBlogSource:
    name: str
    base_url: str
    index_urls: list[str]
    content_selectors: list[str]
    engine: str = "postgresql"  # primary engine coverage


DB_BLOG_SOURCES: list[DBBlogSource] = [
    DBBlogSource(
        name="citus",
        base_url="https://www.citusdata.com",
        index_urls=["https://www.citusdata.com/blog/"],
        content_selectors=["div.post-content", "article", "div.entry-content"],
        engine="postgresql",
    ),
    DBBlogSource(
        name="use_the_index_luke",
        base_url="https://use-the-index-luke.com",
        index_urls=["https://use-the-index-luke.com/sql/table-of-contents"],
        content_selectors=["div#content", "article", "main"],
        engine="all",
    ),
    DBBlogSource(
        name="planet_postgresql",
        base_url="https://planet.postgresql.org",
        index_urls=["https://planet.postgresql.org/"],
        content_selectors=["div.post", "article", "div.content"],
        engine="postgresql",
    ),
    DBBlogSource(
        name="2ndquadrant",
        base_url="https://www.2ndquadrant.com",
        index_urls=["https://www.2ndquadrant.com/en/blog/"],
        content_selectors=["div.blog-content", "article", "div.entry-content"],
        engine="postgresql",
    ),
    DBBlogSource(
        name="percona",
        base_url="https://www.percona.com",
        index_urls=["https://www.percona.com/blog/category/mysql/"],
        content_selectors=["div.post-content", "article"],
        engine="mysql",
    ),
    DBBlogSource(
        name="discord_engineering",
        base_url="https://discord.com",
        index_urls=["https://discord.com/blog/engineering"],
        content_selectors=["article", "div.post-content", "div.blog-content"],
        engine="postgresql",
    ),
    DBBlogSource(
        name="shopify_engineering",
        base_url="https://shopify.engineering",
        index_urls=["https://shopify.engineering/search?q=mysql+performance"],
        content_selectors=["div.article-body", "article", "main"],
        engine="mysql",
    ),
]


class DBBlogCrawler:
    """Crawler for database engineering blogs."""

    def __init__(self, output_dir: Path = OUTPUT_DIR, max_concurrent: int = 3):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent

    def _url_to_filename(self, url: str) -> str:
        h = hashlib.md5(url.encode()).hexdigest()[:12]
        slug = re.sub(r"[^\w-]", "_", urlparse(url).path)[:60]
        return f"{slug}_{h}.json"

    def _is_db_relevant(self, text: str) -> bool:
        text_lower = text.lower()
        return sum(1 for kw in DB_KEYWORDS if kw in text_lower) >= 3

    def _extract_content(self, html: str, source: DBBlogSource) -> str | None:
        soup = BeautifulSoup(html, "lxml")
        for sel in source.content_selectors:
            el = soup.select_one(sel)
            if el and len(el.get_text(strip=True)) > 200:
                return el.get_text(separator="\n", strip=True)
        return None

    # QM-19: Extracted the nested async def fetch() closure into this class
    # method. Defining async functions inside a while-loop body re-creates the
    # function object on every iteration and captures loop variables via closure
    # — a well-known source of subtle bugs (late-binding, stale captures).
    async def _fetch_url(
        self,
        url: str,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        visited: set[str],
    ) -> tuple[str, str | None]:
        async with sem:
            if url in visited:
                return url, None
            visited.add(url)
            headers = {"User-Agent": "Mozilla/5.0 (research bot; github.com/calebnewtonusc/querymedic)"}
            await asyncio.sleep(0.8)
            try:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as r:
                    return url, await r.text(errors="replace") if r.status == 200 else None
            except Exception:
                return url, None

    async def _crawl_source(self, source: DBBlogSource, session: aiohttp.ClientSession, sem: asyncio.Semaphore) -> int:
        saved = 0
        visited: set[str] = set()
        to_crawl = list(source.index_urls)

        while to_crawl:
            batch, to_crawl = to_crawl[:5], to_crawl[5:]

            results = await asyncio.gather(*[
                self._fetch_url(u, session, sem, visited) for u in batch
            ])

            for url, html in results:
                if not html:
                    continue
                content = self._extract_content(html, source)
                if not content:
                    continue

                soup = BeautifulSoup(html, "lxml")
                title = soup.title.string if soup.title else url

                if not self._is_db_relevant(f"{title} {content}"):
                    continue

                filename = self._url_to_filename(url)
                out_path = self.output_dir / filename
                if not out_path.exists():
                    out_path.write_text(json.dumps({
                        "source": source.name,
                        "engine": source.engine,
                        "url": url,
                        "title": str(title),
                        "text": content,
                        "word_count": len(content.split()),
                        "crawled_at": time.time(),
                    }, ensure_ascii=False, indent=2))
                    saved += 1

                if url in source.index_urls:
                    domain = urlparse(source.base_url).netloc
                    for a in soup.find_all("a", href=True):
                        href = urljoin(source.base_url, a["href"])
                        parsed = urlparse(href)
                        if parsed.netloc == domain and href not in visited and href not in to_crawl:
                            to_crawl.append(f"{parsed.scheme}://{parsed.netloc}{parsed.path}")

        logger.info(f"  {source.name}: {saved} posts saved")
        return saved

    def crawl_all(self, sources: list[str] | None = None) -> int:
        active = [s for s in DB_BLOG_SOURCES if sources is None or s.name in sources]
        return asyncio.run(self._run_async(active))

    async def _run_async(self, sources: list[DBBlogSource]) -> int:
        sem = asyncio.Semaphore(self.max_concurrent)
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=2)
        async with aiohttp.ClientSession(connector=connector) as session:
            counts = await asyncio.gather(*[self._crawl_source(s, session, sem) for s in sources])
        total = sum(counts)
        logger.success(f"DB blog crawl complete: {total} posts saved")
        return total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--source", nargs="+")
    args = parser.parse_args()
    c = DBBlogCrawler()
    n = c.crawl_all(sources=None if args.all else args.source)
    print(f"Total DB blog posts: {n}")
