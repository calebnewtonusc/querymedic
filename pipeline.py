"""
pipeline.py - End-to-end QueryMedic dataset and training pipeline.

Usage:
    python pipeline.py                    # full run
    python pipeline.py --collect-only     # data collection only
    python pipeline.py --synth-only       # synthesis only
    python pipeline.py --eval             # evaluate latest checkpoint
    python pipeline.py --stats            # print statistics
"""

import argparse
import json
from pathlib import Path

from loguru import logger

RAW_DIR = Path(__file__).parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
TRAIN_DIR = Path(__file__).parent / "data" / "train"
MASTER_JSONL = PROCESSED_DIR / "dataset.jsonl"
CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"


def run_collection() -> int:
    total = 0
    logger.info("=== COLLECTION PHASE ===")

    from discovery.dba_stackexchange import DBAStackExchangeHarvester

    harvester = DBAStackExchangeHarvester(output_dir=RAW_DIR / "stackexchange")
    n = harvester.fetch_all(min_score=10)
    logger.info(f"  DBA Stack Exchange: {n} Q&A pairs")
    total += n

    from discovery.db_blog_crawler import DBBlogCrawler

    crawler = DBBlogCrawler(output_dir=RAW_DIR / "blogs")
    n = crawler.crawl_all()
    logger.info(f"  DB engineering blogs: {n} posts")
    total += n

    from discovery.explain_plan_corpus import ExplainPlanCorpus

    corpus = ExplainPlanCorpus(output_dir=RAW_DIR / "explain_plans")
    n = corpus.collect_all()
    logger.info(f"  EXPLAIN plan pairs: {n} pairs")
    total += n

    logger.success(f"Collection complete: {total} total items")
    return total


def run_synthesis(backend: str = "claude", vllm_urls: list[str] | None = None) -> int:
    logger.info("=== SYNTHESIS PHASE ===")
    from synthesis.synthesize_bulk import SynthesisPipeline

    pipeline = SynthesisPipeline(backend=backend, vllm_urls=vllm_urls or [])
    return pipeline.run_all()


def print_stats() -> None:
    logger.info("=== DATASET STATS ===")
    for subdir, label in [
        (RAW_DIR / "stackexchange", "DBA Stack Exchange Q&A"),
        (RAW_DIR / "blogs", "DB blog posts"),
        (RAW_DIR / "explain_plans", "EXPLAIN plan pairs"),
    ]:
        count = len(list(subdir.glob("*.json"))) if subdir.exists() else 0
        logger.info(f"  {label:<30} {count}")

    if MASTER_JSONL.exists():
        lines = [line for line in MASTER_JSONL.read_text().splitlines() if line.strip()]
        logger.info(f"\n  Total training pairs: {len(lines)}")

        engine_counts: dict[str, int] = {}
        for line in lines:
            try:
                pair = json.loads(line)
                engine = pair.get("engine", "unknown")
                engine_counts[engine] = engine_counts.get(engine, 0) + 1
            except json.JSONDecodeError:
                continue

        logger.info("\n  Engine distribution:")
        for eng, count in sorted(engine_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    {eng:<20} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="QueryMedic pipeline")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--synth-only", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--backend", default="claude", choices=["claude", "vllm"])
    parser.add_argument("--vllm-urls", nargs="+", default=[])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.stats:
        print_stats()
    elif args.eval:
        # QM-1: Guard against empty checkpoint list — sorted(...)[-1] raises
        # IndexError when CHECKPOINTS_DIR has no matching subdirectories.
        if args.model:
            model = args.model
        else:
            checkpoints = sorted(CHECKPOINTS_DIR.glob("querymedic-*"))
            if not checkpoints:
                raise FileNotFoundError(
                    f"No checkpoints found in {CHECKPOINTS_DIR}. "
                    "Run training first or pass --model <path>."
                )
            model = str(checkpoints[-1])
        from evaluation.querybench import QueryBench

        bench = QueryBench(model_path=model)
        bench.run()
    elif args.collect_only:
        run_collection()
    elif args.synth_only:
        run_synthesis(args.backend, args.vllm_urls)
    else:
        run_collection()
        run_synthesis(args.backend, args.vllm_urls)


if __name__ == "__main__":
    main()
