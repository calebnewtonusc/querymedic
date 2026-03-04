"""
explain_plan_corpus.py - Collect EXPLAIN ANALYZE outputs from public sources.

Primary sources:
    - explain.depesz.com (public PostgreSQL EXPLAIN plan sharing)
    - explain.tensor.ru (Russian PostgreSQL community plan sharing)
    - Stack Overflow/DBA answers with EXPLAIN ANALYZE pastes
    - GitHub issues and PRs with EXPLAIN ANALYZE in comments

Usage:
    python discovery/explain_plan_corpus.py --all
    python discovery/explain_plan_corpus.py --source depesz --limit 1000
"""

import json
import re
import time
from pathlib import Path

import requests
from loguru import logger

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "raw" / "explain_plans"

# Regex patterns for EXPLAIN ANALYZE detection
EXPLAIN_PATTERNS = [
    # PostgreSQL
    re.compile(r"((?:Seq Scan|Index Scan|Index Only Scan|Bitmap Heap Scan|Hash Join|Merge Join|Nested Loop|Sort|Aggregate|Hash|Limit|CTE Scan).*?\(cost=[\d.]+\.\.[\d.]+ rows=\d+ width=\d+\).*?actual time=[\d.]+\.\.[\d.]+ rows=\d+ loops=\d+)", re.DOTALL),
    # MySQL EXPLAIN
    re.compile(r"(\|\s*\d+\s*\|\s*\w+\s*\|\s*\w+\s*\|.*?\|)", re.DOTALL),
]

EXPLAIN_HEADER = re.compile(
    r"(EXPLAIN ANALYZE|EXPLAIN \(ANALYZE|explain analyze|explain \(analyze)",
    re.IGNORECASE,
)


class ExplainPlanCorpus:
    """Collect EXPLAIN ANALYZE corpus from public sources."""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (research bot; github.com/calebnewtonusc/querymedic)"

    def collect_depesz(self, limit: int = 2000) -> int:
        """
        Collect plans from explain.depesz.com.
        The site has a public listing of shared plans with unique IDs.
        """
        saved = 0
        logger.info("Collecting from explain.depesz.com")

        for plan_id in range(1, limit + 1):
            out_path = self.output_dir / f"depesz_{plan_id:06d}.json"
            if out_path.exists():
                continue

            url = f"https://explain.depesz.com/s/{plan_id}"
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code != 200:
                    continue

                html = resp.text
                # Extract the plan text from the page
                plan_match = re.search(r'<pre[^>]*class="[^"]*plan[^"]*"[^>]*>(.*?)</pre>', html, re.DOTALL)
                if not plan_match:
                    continue

                plan_text = plan_match.group(1)
                # Clean HTML entities
                plan_text = plan_text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

                if len(plan_text) < 100:
                    continue

                # Extract query if present
                query_match = re.search(r'<pre[^>]*class="[^"]*query[^"]*"[^>]*>(.*?)</pre>', html, re.DOTALL)
                query = query_match.group(1) if query_match else None

                record = {
                    "source": "depesz",
                    "plan_id": plan_id,
                    "url": url,
                    "plan": plan_text,
                    "query": query,
                    "engine": "postgresql",
                    "collected_at": time.time(),
                }
                out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
                saved += 1
                time.sleep(0.3)

            except Exception as e:
                logger.debug(f"Depesz plan {plan_id}: {e}")

        logger.info(f"  depesz: {saved} plans")
        return saved

    def extract_from_stackoverflow(self, raw_dir: Path) -> int:
        """
        Extract EXPLAIN ANALYZE blocks from existing Stack Overflow/DBA harvested Q&As.
        Called after dba_stackexchange.py has run.
        """
        saved = 0
        so_dir = raw_dir / "stackexchange"
        if not so_dir.exists():
            return 0

        for json_file in so_dir.glob("q_*.json"):
            try:
                data = json.loads(json_file.read_text())
            except json.JSONDecodeError:
                continue

            combined_text = data.get("body", "") + data.get("best_answer", {}).get("body", "")

            # Find EXPLAIN ANALYZE blocks
            explain_blocks = []
            # HTML code blocks
            code_blocks = re.findall(r"<code>(.*?)</code>", combined_text, re.DOTALL)
            for block in code_blocks:
                if EXPLAIN_HEADER.search(block) or "Seq Scan" in block or "Index Scan" in block:
                    explain_blocks.append(block)

            if not explain_blocks:
                continue

            for i, block in enumerate(explain_blocks[:3]):
                out_path = self.output_dir / f"so_extract_{data['question_id']}_{i}.json"
                if not out_path.exists():
                    out_path.write_text(json.dumps({
                        "source": "stackoverflow_extract",
                        "question_id": data["question_id"],
                        "plan": block,
                        "engine": data.get("engine", "postgresql"),
                        "title": data.get("title", ""),
                        "extracted_at": time.time(),
                    }, ensure_ascii=False, indent=2))
                    saved += 1

        logger.info(f"  SO extract: {saved} EXPLAIN blocks extracted")
        return saved

    def collect_all(self, limit_per_source: int = 2000) -> int:
        total = 0
        total += self.collect_depesz(limit=limit_per_source)
        total += self.extract_from_stackoverflow(self.output_dir.parent)
        logger.success(f"EXPLAIN plan corpus: {total} plans collected")
        return total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--source", choices=["depesz", "stackoverflow"])
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()

    corpus = ExplainPlanCorpus()
    n = corpus.collect_all(limit_per_source=args.limit)
    print(f"Total EXPLAIN plans: {n}")
