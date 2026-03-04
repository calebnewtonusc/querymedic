"""
real_world_schemas.py — Collect real-world PostgreSQL schema DDL from open-source projects.

Harvests schema.sql, migrations, and ERD data from 1000+ OSS repositories
to build a corpus of realistic table structures for training pair generation.

Sources:
  1. GitHub Code Search: search for schema.sql, init.sql, migrations/*.sql
  2. Known OSS project schemas (GitLab, Mastodon, Discourse, Redash, etc.)
  3. Awesome-Postgres curated schema repositories
  4. Django, Rails, Laravel project migration files

Output format (JSONL):
  {
    source, repo, file_path, file_url,
    raw_ddl: str,
    tables: [{name, columns: [{name, type, nullable, default}], indexes: [str], constraints: [str]}],
    table_count, has_indexes, has_foreign_keys, has_partial_indexes,
    database_type, schema_hash, fetched_at
  }

Usage:
    export GITHUB_TOKEN=ghp_xxxx
    python discovery/real_world_schemas.py --all
    python discovery/real_world_schemas.py --sources known_projects github_search
"""

import asyncio
import hashlib
import json
import re
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp
from loguru import logger

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "raw" / "schemas"

# ─── Known OSS Projects with Quality Schemas ──────────────────────────────────

KNOWN_PROJECTS = [
    # GitLab (massive PostgreSQL schema)
    {
        "repo": "gitlabhq/gitlabhq",
        "paths": ["db/structure.sql", "db/schema.rb"],
        "type": "gitlab",
    },
    # Mastodon (Rails/ActiveRecord)
    {
        "repo": "mastodon/mastodon",
        "paths": ["db/schema.rb"],
        "type": "mastodon",
    },
    # Discourse (Ruby/PostgreSQL)
    {
        "repo": "discourse/discourse",
        "paths": ["db/structure.sql"],
        "type": "discourse",
    },
    # Redash (Python/SQLAlchemy)
    {
        "repo": "getredash/redash",
        "paths": ["redash/models/__init__.py", "migrations/versions/"],
        "type": "redash",
    },
    # Metabase (Clojure/SQL)
    {
        "repo": "metabase/metabase",
        "paths": ["resources/migrations/000_migrations.yaml"],
        "type": "metabase",
    },
    # Sentry (Python/Django)
    {
        "repo": "getsentry/sentry",
        "paths": ["src/sentry/migrations/"],
        "type": "sentry",
    },
    # Posthog (Django)
    {
        "repo": "PostHog/posthog",
        "paths": ["posthog/migrations/"],
        "type": "posthog",
    },
    # Cal.com (Next.js/Prisma)
    {
        "repo": "calcom/cal.com",
        "paths": ["packages/prisma/schema.prisma"],
        "type": "prisma",
    },
    # Plane (Django)
    {
        "repo": "makeplane/plane",
        "paths": ["apiserver/plane/db/migrations/"],
        "type": "plane",
    },
    # Mattermost (Go)
    {
        "repo": "mattermost/mattermost-server",
        "paths": ["db/migrations/postgres/", "app/model/"],
        "type": "mattermost",
    },
    # Rocketchat
    {
        "repo": "RocketChat/Rocket.Chat",
        "paths": ["apps/meteor/server/startup/migrations/"],
        "type": "rocketchat",
    },
    # Netdata Cloud (Go)
    {
        "repo": "netdata/netdata",
        "paths": ["database/"],
        "type": "netdata",
    },
    # Budibase (Node.js)
    {
        "repo": "Budibase/budibase",
        "paths": ["packages/server/src/db/"],
        "type": "budibase",
    },
    # Hasura (Haskell)
    {
        "repo": "hasura/graphql-engine",
        "paths": ["server/src-rsr/migrations/"],
        "type": "hasura",
    },
    # Keycloak (Java)
    {
        "repo": "keycloak/keycloak",
        "paths": ["model/jpa/src/main/resources/META-INF/"],
        "type": "keycloak",
    },
    # Supabase
    {
        "repo": "supabase/supabase",
        "paths": ["packages/db-build/"],
        "type": "supabase",
    },
    # Temporal
    {
        "repo": "temporalio/temporal",
        "paths": ["schema/postgresql/v12/"],
        "type": "temporal",
    },
    # Hydra
    {
        "repo": "ory/hydra",
        "paths": ["persistence/sql/migrations/"],
        "type": "ory_hydra",
    },
    # Kratos
    {
        "repo": "ory/kratos",
        "paths": ["persistence/sql/migrations/"],
        "type": "ory_kratos",
    },
    # Open edX
    {
        "repo": "openedx/edx-platform",
        "paths": ["lms/djangoapps/courseware/migrations/", "openedx/core/djangoapps/"],
        "type": "edx",
    },
    # Frappe/ERPNext
    {
        "repo": "frappe/frappe",
        "paths": ["frappe/database/"],
        "type": "frappe",
    },
    # Directus
    {
        "repo": "directus/directus",
        "paths": ["api/src/database/migrations/"],
        "type": "directus",
    },
    # Airbyte
    {
        "repo": "airbytehq/airbyte",
        "paths": ["airbyte-db/db-lib/src/main/resources/configs_database/"],
        "type": "airbyte",
    },
    # Grafana
    {
        "repo": "grafana/grafana",
        "paths": ["pkg/services/sqlstore/migrations/"],
        "type": "grafana",
    },
    # n8n
    {
        "repo": "n8n-io/n8n",
        "paths": ["packages/cli/src/databases/migrations/"],
        "type": "n8n",
    },
    # Open Telemetry Collector
    {
        "repo": "open-telemetry/opentelemetry-collector",
        "paths": [],
        "type": "otel",
    },
    # Pocketbase
    {
        "repo": "pocketbase/pocketbase",
        "paths": ["daos/", "migrations/"],
        "type": "pocketbase",
    },
    # Lago (billing)
    {
        "repo": "getlago/lago",
        "paths": ["db/migrate/", "db/schema.rb"],
        "type": "lago",
    },
]

# GitHub code search queries for finding SQL schema files
GITHUB_SEARCH_QUERIES = [
    "filename:schema.sql CREATE TABLE extension:sql",
    "filename:init.sql CREATE TABLE extension:sql",
    "filename:structure.sql CREATE TABLE extension:sql",
    "filename:001_create_tables.sql CREATE INDEX extension:sql",
    "CREATE INDEX CONCURRENTLY extension:sql CREATE TABLE",
    "filename:schema.sql postgresql CREATE INDEX extension:sql stars:>100",
]

# ─── Parsing ───────────────────────────────────────────────────────────────────

CREATE_TABLE_PATTERN = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:[`\"']?\w+[`\"']?\.)?"
    r"[`\"']?(\w+)[`\"']?\s*\(",
    re.IGNORECASE,
)
COLUMN_PATTERN = re.compile(
    r"^\s+[`\"']?(\w+)[`\"']?\s+"
    r"((?:character\s+varying|timestamp(?:\s+with\s+time\s+zone)?|double\s+precision|"
    r"bigint|integer|int|smallint|text|boolean|bool|jsonb?|uuid|numeric|decimal|"
    r"real|float|serial|bigserial|bytea|date|time|interval|oid|"
    r"(?:varchar|char)\s*\(\d+\)|\w+))"
    r"(.*?)(?:,\s*$|$)",
    re.IGNORECASE,
)
INDEX_IN_TABLE_PATTERN = re.compile(
    r"(?:CREATE\s+(?:UNIQUE\s+)?INDEX|PRIMARY\s+KEY|UNIQUE\s*\(|"
    r"INDEX\s+\w+\s+ON|KEY\s+\w+\s*\()",
    re.IGNORECASE,
)
STANDALONE_INDEX_PATTERN = re.compile(
    r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:\w+)\s+ON\s+(?:\w+\.)?(\w+)\s*(?:USING\s+\w+\s*)?\(([^)]+)\)"
    r"(?:\s+WHERE\s+.+?)?;",
    re.IGNORECASE,
)
FK_PATTERN = re.compile(
    r"(?:FOREIGN\s+KEY|REFERENCES)\s+\w+",
    re.IGNORECASE,
)


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool
    default: Optional[str]


@dataclass
class TableInfo:
    name: str
    columns: list[dict]
    indexes: list[str]
    constraints: list[str]


@dataclass
class SchemaRecord:
    source: str
    repo: str
    file_path: str
    file_url: str
    raw_ddl: str
    tables: list[dict]
    table_count: int
    has_indexes: bool
    has_foreign_keys: bool
    has_partial_indexes: bool
    database_type: str
    schema_hash: str
    fetched_at: float = field(default_factory=time.time)


def detect_database_type(ddl: str) -> str:
    """Detect database type from DDL content."""
    ddl_lower = ddl.lower()
    if any(
        kw in ddl_lower
        for kw in ["serial", "bigserial", "bytea", "timestamptz", "jsonb", "pg_"]
    ):
        return "postgresql"
    if any(
        kw in ddl_lower
        for kw in ["auto_increment", "engine=innodb", "tinyint", "mediumint"]
    ):
        return "mysql"
    if any(kw in ddl_lower for kw in ["autoincrement", "sqlite_sequence"]):
        return "sqlite"
    if "go.sum" in ddl_lower or any(
        kw in ddl_lower for kw in ["nvarchar", "datetime2", "uniqueidentifier"]
    ):
        return "mssql"
    # QM-16: Return "unknown" instead of assuming "postgresql" for DDL that
    # does not match any known dialect. Callers can handle "unknown" explicitly
    # rather than silently treating non-PG schemas as PostgreSQL.
    return "unknown"


def parse_schema_ddl(ddl: str) -> list[dict]:
    """
    Parse CREATE TABLE statements from SQL DDL into structured table info.
    Returns a list of table dicts.
    """
    tables = []
    # Split DDL into statements
    # Find each CREATE TABLE block
    table_blocks = list(CREATE_TABLE_PATTERN.finditer(ddl))

    for i, m in enumerate(table_blocks):
        table_name = m.group(1)
        # Find the matching closing paren
        start = m.end() - 1  # position of opening paren
        depth = 0
        end = start
        for j, ch in enumerate(ddl[start:], start):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break

        table_body = ddl[start:end]
        columns = []
        indexes = []
        constraints = []

        for line in table_body.split("\n"):
            line = line.strip().rstrip(",")
            if not line or line in ("(", ")"):
                continue

            # Detect indexes and constraints
            if INDEX_IN_TABLE_PATTERN.search(line):
                indexes.append(line[:200])
                continue
            if any(
                kw in line.upper()
                for kw in ["CONSTRAINT", "CHECK", "FOREIGN KEY", "REFERENCES"]
            ):
                constraints.append(line[:200])
                continue

            # Parse column
            col_m = COLUMN_PATTERN.match(line)
            if col_m:
                col_name = col_m.group(1)
                col_type = col_m.group(2).strip()
                rest = col_m.group(3).lower()
                nullable = "not null" not in rest
                default_m = re.search(r"default\s+(\S+)", rest, re.IGNORECASE)
                default = default_m.group(1) if default_m else None
                columns.append(
                    {
                        "name": col_name,
                        "data_type": col_type,
                        "nullable": nullable,
                        "default": default,
                    }
                )

        # Collect standalone CREATE INDEX statements referencing this table
        for idx_m in STANDALONE_INDEX_PATTERN.finditer(ddl):
            if idx_m.group(1).lower() == table_name.lower():
                indexes.append(idx_m.group(0)[:300])

        if columns or indexes:  # Only include tables with parseable content
            tables.append(
                {
                    "name": table_name,
                    "columns": columns,
                    "indexes": indexes[:20],
                    "constraints": constraints[:10],
                }
            )

    return tables[:200]  # Cap at 200 tables


class RealWorldSchemaHarvester:
    """
    Harvests real-world PostgreSQL schemas from OSS projects on GitHub.

    Two sources:
      1. Known high-quality projects (GitLab, Mastodon, Discourse, etc.)
      2. GitHub code search for schema.sql files
    """

    RAW_URL = "https://raw.githubusercontent.com/{repo}/HEAD/{path}"
    CONTENTS_API = "https://api.github.com/repos/{repo}/contents/{path}"
    SEARCH_CODE_API = "https://api.github.com/search/code"
    REQUEST_DELAY = 0.5
    MIN_TABLES = 3  # Skip trivial schemas

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        tokens: Optional[list[str]] = None,
        workers: int = 8,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_token = os.environ.get("GITHUB_TOKEN", "")
        self.tokens = tokens or (
            [t for t in raw_token.split(",") if t] if raw_token else []
        )
        self._token_idx = 0
        self._semaphore = asyncio.Semaphore(workers)
        self._stats = {"fetched": 0, "saved": 0, "errors": 0}
        self._seen_hashes: set[str] = set()

    def _auth_headers(self) -> dict:
        base = {"Accept": "application/vnd.github.v3+json"}
        if self.tokens:
            token = self.tokens[self._token_idx % len(self.tokens)]
            self._token_idx += 1
            base["Authorization"] = f"token {token}"
        return base

    async def _fetch_text(
        self,
        session: aiohttp.ClientSession,
        url: str,
        is_api: bool = False,
    ) -> Optional[str]:
        """Fetch raw text content from a URL."""
        for attempt in range(3):
            await asyncio.sleep(self.REQUEST_DELAY)
            try:
                headers = (
                    self._auth_headers()
                    if is_api
                    else {"User-Agent": "QueryMedic-Schema-Harvester/1.0"}
                )
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    remaining = int(resp.headers.get("X-RateLimit-Remaining", 99))
                    if remaining < 5:
                        reset_at = int(
                            resp.headers.get("X-RateLimit-Reset", time.time() + 60)
                        )
                        wait = max(1, reset_at - time.time() + 2)
                        await asyncio.sleep(wait)

                    if resp.status == 200:
                        return await resp.text(errors="replace")
                    elif resp.status in (403, 429):
                        await asyncio.sleep(30 * (attempt + 1))
                    elif resp.status in (404, 451):
                        return None
            except Exception as e:
                logger.debug(f"Fetch error {url}: {e}")
                await asyncio.sleep(2**attempt)
        return None

    async def _save_schema(
        self,
        source: str,
        repo: str,
        file_path: str,
        file_url: str,
        ddl: str,
        output_file: Path,
    ) -> bool:
        """Parse and save a schema DDL file."""
        if len(ddl) < 200:
            return False

        # Deduplicate by content hash
        content_hash = hashlib.md5(ddl.encode(), usedforsecurity=False).hexdigest()
        if content_hash in self._seen_hashes:
            return False
        self._seen_hashes.add(content_hash)

        db_type = detect_database_type(ddl)

        # Only keep PostgreSQL-compatible schemas
        if db_type == "mysql":
            # MySQL schemas are less useful for PG training, keep only if they have indexes
            if not STANDALONE_INDEX_PATTERN.search(ddl):
                return False

        tables = parse_schema_ddl(ddl)
        if len(tables) < self.MIN_TABLES:
            return False

        has_indexes = any(t["indexes"] for t in tables)
        has_fk = bool(FK_PATTERN.search(ddl))
        has_partial = bool(re.search(r"WHERE\s+.+?;", ddl, re.IGNORECASE))

        record = SchemaRecord(
            source=source,
            repo=repo,
            file_path=file_path,
            file_url=file_url,
            raw_ddl=ddl[:50_000],  # Cap DDL size
            tables=tables,
            table_count=len(tables),
            has_indexes=has_indexes,
            has_foreign_keys=has_fk,
            has_partial_indexes=has_partial,
            database_type=db_type,
            schema_hash=content_hash,
        )

        async with aiofiles.open(str(output_file), "a") as f:
            await f.write(json.dumps(asdict(record)) + "\n")

        self._stats["saved"] += 1
        return True

    async def _harvest_known_project(
        self,
        session: aiohttp.ClientSession,
        project: dict,
        output_file: Path,
    ) -> int:
        """Harvest schema files from a known OSS project."""
        async with self._semaphore:
            saved = 0
            repo = project["repo"]
            for path in project["paths"]:
                if path.endswith("/"):
                    # Directory — list files via API
                    api_url = self.CONTENTS_API.format(repo=repo, path=path.rstrip("/"))
                    data = await self._fetch_text(session, api_url, is_api=True)
                    if not data:
                        continue
                    try:
                        items = json.loads(data)
                        if not isinstance(items, list):
                            continue
                        sql_files = [
                            it["path"]
                            for it in items
                            if it.get("type") == "file"
                            and (
                                it["name"].endswith(".sql")
                                or "migration" in it["name"].lower()
                            )
                        ][:20]
                    except (json.JSONDecodeError, KeyError):
                        continue

                    for sql_path in sql_files:
                        raw_url = self.RAW_URL.format(repo=repo, path=sql_path)
                        ddl = await self._fetch_text(session, raw_url)
                        if ddl:
                            self._stats["fetched"] += 1
                            ok = await self._save_schema(
                                source="known_project",
                                repo=repo,
                                file_path=sql_path,
                                file_url=f"https://github.com/{repo}/blob/HEAD/{sql_path}",
                                ddl=ddl,
                                output_file=output_file,
                            )
                            if ok:
                                saved += 1
                else:
                    # Single file
                    raw_url = self.RAW_URL.format(repo=repo, path=path)
                    ddl = await self._fetch_text(session, raw_url)
                    if ddl:
                        self._stats["fetched"] += 1
                        ok = await self._save_schema(
                            source="known_project",
                            repo=repo,
                            file_path=path,
                            file_url=f"https://github.com/{repo}/blob/HEAD/{path}",
                            ddl=ddl,
                            output_file=output_file,
                        )
                        if ok:
                            saved += 1

            return saved

    async def _harvest_github_search(
        self,
        session: aiohttp.ClientSession,
        output_file: Path,
        max_results: int = 1000,
    ) -> int:
        """Harvest schema files found via GitHub code search."""
        saved = 0
        for query in GITHUB_SEARCH_QUERIES:
            if saved >= max_results:
                break

            for page in range(1, 11):
                await asyncio.sleep(self.REQUEST_DELAY * 2)
                try:
                    async with session.get(
                        self.SEARCH_CODE_API,
                        params={"q": query, "per_page": 100, "page": page},
                        headers=self._auth_headers(),
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        remaining = int(resp.headers.get("X-RateLimit-Remaining", 5))
                        if remaining < 3:
                            reset_at = int(
                                resp.headers.get("X-RateLimit-Reset", time.time() + 60)
                            )
                            await asyncio.sleep(max(5, reset_at - time.time() + 2))

                        if resp.status != 200:
                            break
                        data = await resp.json()
                except Exception as e:
                    logger.debug(f"Search error for [{query}]: {e}")
                    break

                items = data.get("items", [])
                if not items:
                    break

                for item in items:
                    raw_url = (
                        item.get("html_url", "")
                        .replace("github.com", "raw.githubusercontent.com")
                        .replace("/blob/", "/")
                    )
                    repo_name = item.get("repository", {}).get("full_name", "unknown")
                    file_path = item.get("path", "")
                    file_url = item.get("html_url", "")

                    ddl = await self._fetch_text(session, raw_url)
                    if ddl:
                        self._stats["fetched"] += 1
                        ok = await self._save_schema(
                            source="github_search",
                            repo=repo_name,
                            file_path=file_path,
                            file_url=file_url,
                            ddl=ddl,
                            output_file=output_file,
                        )
                        if ok:
                            saved += 1
                        if saved >= max_results:
                            break

        return saved

    async def harvest_all(
        self,
        include_known: bool = True,
        include_search: bool = True,
        search_limit: int = 1000,
    ) -> int:
        """Harvest schemas from all sources."""
        output_file = self.output_dir / "schemas.jsonl"
        total = 0

        connector = aiohttp.TCPConnector(limit=20)
        async with aiohttp.ClientSession(connector=connector) as session:
            if include_known:
                logger.info(f"Harvesting {len(KNOWN_PROJECTS)} known OSS projects...")
                tasks = [
                    self._harvest_known_project(session, proj, output_file)
                    for proj in KNOWN_PROJECTS
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                known_saved = sum(r for r in results if isinstance(r, int))
                total += known_saved
                logger.info(f"Known projects: {known_saved} schemas saved")

            if include_search:
                logger.info("Harvesting via GitHub code search...")
                n = await self._harvest_github_search(
                    session, output_file, search_limit
                )
                total += n
                logger.info(f"GitHub search: {n} schemas saved")

        logger.success(
            f"Schema harvest complete: {self._stats['saved']} saved, "
            f"{self._stats['fetched']} fetched, {self._stats['errors']} errors"
        )
        return total


def stream_schemas(data_dir: Path = OUTPUT_DIR):
    """Iterate over all schema records from the output directory."""
    for jsonl_file in sorted(data_dir.rglob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def build_schema_stats(data_dir: Path = OUTPUT_DIR) -> None:
    """Print statistics about the collected schemas."""
    total = table_count = indexed = partial_indexed = 0
    type_counts: dict[str, int] = {}

    for record in stream_schemas(data_dir):
        total += 1
        table_count += record.get("table_count", 0)
        if record.get("has_indexes"):
            indexed += 1
        if record.get("has_partial_indexes"):
            partial_indexed += 1
        db_type = record.get("database_type", "unknown")
        type_counts[db_type] = type_counts.get(db_type, 0) + 1

    print(f"Total schemas: {total:,}")
    print(f"Total tables: {table_count:,}")
    print(f"Schemas with indexes: {indexed:,} ({100 * indexed / max(total, 1):.1f}%)")
    print(
        f"Schemas with partial indexes: {partial_indexed:,} ({100 * partial_indexed / max(total, 1):.1f}%)"
    )
    print(f"Database types: {type_counts}")


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Harvest real-world PostgreSQL schemas"
    )
    parser.add_argument("--all", action="store_true", help="Harvest from all sources")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["known_projects", "github_search"],
        help="Specific sources to harvest",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=1000,
        help="Max schemas from GitHub code search",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stats", action="store_true", help="Print schema statistics")
    args = parser.parse_args()

    if args.stats:
        build_schema_stats(Path(args.output_dir))
        raise SystemExit(0)

    include_known = args.all or (args.sources and "known_projects" in args.sources)
    include_search = args.all or (args.sources and "github_search" in args.sources)

    if not args.all and not args.sources:
        parser.error("Specify --all or --sources")

    harvester = RealWorldSchemaHarvester(
        output_dir=Path(args.output_dir),
        workers=args.workers,
    )
    n = asyncio.run(
        harvester.harvest_all(
            include_known=include_known,
            include_search=include_search,
            search_limit=args.search_limit,
        )
    )
    print(f"\nTotal schemas harvested: {n:,}")
