"""
dba_stackexchange.py - Harvest high-quality database Q&A from DBA Stack Exchange.

DBA Stack Exchange (dba.stackexchange.com) is distinct from StackOverflow — it's the
dedicated community for database administrators, with higher signal/noise ratio on
query optimization, index design, and schema architecture questions.

Usage:
    python discovery/dba_stackexchange.py --min-votes 10
    python discovery/dba_stackexchange.py --tags "postgresql performance" --min-votes 5
"""

import json
import time
from pathlib import Path

import requests
from loguru import logger

OUTPUT_DIR = Path(__file__).parents[1] / "data" / "raw" / "stackexchange"

# DBA Stack Exchange tags to harvest
DBA_TAGS = [
    "query-performance",
    "postgresql",
    "mysql",
    "index",
    "performance",
    "query-optimization",
    "explain",
    "partitioning",
    "sqlite",
    "execution-plan",
    "statistics",
    "vacuum",
    "analyze",
    "covering-index",
    "partial-index",
]

STACKOVERFLOW_API = "https://api.stackexchange.com/2.3"


class DBAStackExchangeHarvester:
    """Harvest Q&A from DBA Stack Exchange."""

    def __init__(self, output_dir: Path = OUTPUT_DIR, api_key: str | None = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.session = requests.Session()
        self.quota_remaining = 10000

    def _api_get(self, endpoint: str, params: dict) -> dict:
        """Make API request to Stack Exchange."""
        import os

        if not self.api_key:
            self.api_key = os.environ.get("STACKOVERFLOW_KEY")
        if self.api_key:
            params["key"] = self.api_key
        params["site"] = "dba"  # DBA Stack Exchange, not stackoverflow

        time.sleep(0.15 if self.api_key else 0.5)

        resp = self.session.get(
            f"{STACKOVERFLOW_API}/{endpoint}", params=params, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        self.quota_remaining = data.get("quota_remaining", self.quota_remaining)
        if data.get("backoff"):
            time.sleep(data["backoff"])
        return data

    def _is_db_optimization(self, text: str) -> bool:
        """Filter for database optimization content."""
        keywords = [
            "explain",
            "seq scan",
            "index scan",
            "query plan",
            "execution plan",
            "create index",
            "partial index",
            "covering index",
            "gin",
            "gist",
            "btree",
            "hash index",
            "brin",
            "statistics",
            "analyze",
            "write amplification",
            "heap fetch",
            "index only scan",
            "query rewrite",
            "cte",
            "lateral",
            "anti-join",
            "row estimate",
            "planner",
            "optimizer",
        ]
        text_lower = text.lower()
        return sum(1 for kw in keywords if kw in text_lower) >= 2

    def fetch_all(self, min_score: int = 10) -> int:
        """Fetch all high-quality database optimization Q&A."""
        total_saved = 0
        seen_ids: set[int] = set()

        tag_combos = list(DBA_TAGS)
        for lang in ["postgresql", "mysql", "sqlite"]:
            for perf_tag in ["performance", "query-performance", "index", "explain"]:
                tag_combos.append(f"{lang};{perf_tag}")

        logger.info(f"Harvesting {len(tag_combos)} DBA tag combinations")

        for i, tag in enumerate(tag_combos, 1):
            data = self._api_get(
                "questions",
                {
                    "tagged": tag,
                    "sort": "votes",
                    "order": "desc",
                    "filter": "withbody",
                    "pagesize": 100,
                    "min": min_score,
                },
            )

            for q in data.get("items", []):
                qid = q["question_id"]
                if qid in seen_ids:
                    continue
                seen_ids.add(qid)

                out_path = self.output_dir / f"q_{qid}.json"
                if out_path.exists():
                    continue

                # Fetch answers
                ans_data = self._api_get(
                    f"questions/{qid}/answers",
                    {
                        "sort": "votes",
                        "order": "desc",
                        "filter": "withbody",
                        "pagesize": 5,
                        "min": 5,
                    },
                )
                answers = ans_data.get("items", [])

                combined = q.get("body", "") + " ".join(
                    a.get("body", "") for a in answers
                )
                if not self._is_db_optimization(combined):
                    continue

                best = next((a for a in answers if a.get("is_accepted")), None)
                if not best and answers:
                    best = max(answers, key=lambda a: a.get("score", 0))
                if not best:
                    continue

                # Detect database engine from tags
                qtags = q.get("tags", [])
                engine = "postgresql"
                if "mysql" in qtags or "mariadb" in qtags:
                    engine = "mysql"
                elif "sqlite" in qtags:
                    engine = "sqlite"

                record = {
                    "question_id": qid,
                    "title": q.get("title", ""),
                    "body": q.get("body", ""),
                    "tags": qtags,
                    "score": q.get("score", 0),
                    "engine": engine,
                    "best_answer": {
                        "answer_id": best["answer_id"],
                        "body": best.get("body", ""),
                        "score": best.get("score", 0),
                        "is_accepted": best.get("is_accepted", False),
                    },
                    "link": q.get("link", ""),
                    "source": "dba_stackexchange",
                    "fetched_at": time.time(),
                }
                out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
                total_saved += 1

            if self.quota_remaining < 10:
                logger.warning("API quota exhausted")
                break

            logger.info(f"  [{i}/{len(tag_combos)}] {tag}: {total_saved} total saved")

        logger.success(f"DBA Stack Exchange: {total_saved} Q&A pairs saved")
        return total_saved


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--min-votes", type=int, default=10)
    args = parser.parse_args()
    h = DBAStackExchangeHarvester(api_key=os.environ.get("STACKOVERFLOW_KEY"))
    n = h.fetch_all(min_score=args.min_votes)
    print(f"Saved: {n} Q&A pairs")
