"""
synthesize_bulk.py - Async synthesis of QueryMedic training pairs.

Processes DBA Q&A, blog posts, and EXPLAIN plans into structured training pairs.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from loguru import logger

from synthesis.prompts import QUERY_OPTIMIZATION_SYSTEM_PROMPT, EXPLAIN_ANALYSIS_SYSTEM_PROMPT

RAW_DIR = Path(__file__).parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CLAUDE_MODEL = "claude-opus-4-6"
VLLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"


class SynthesisPipeline:
    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        output_dir: Path = PROCESSED_DIR,
        backend: str = "claude",
        vllm_urls: list[str] | None = None,
    ):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.backend = backend
        self.vllm_urls = vllm_urls or []
        self._vllm_idx = 0

    def _next_vllm(self) -> str:
        url = self.vllm_urls[self._vllm_idx % len(self.vllm_urls)]
        self._vllm_idx += 1
        return url

    async def _call(self, system: str, user: str, client: httpx.AsyncClient) -> str | None:
        if self.backend == "vllm" and self.vllm_urls:
            url = self._next_vllm()
            try:
                resp = await client.post(f"{url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'qm')}"},
                    json={"model": VLLM_MODEL, "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ], "max_tokens": 4096, "temperature": 0.3},
                    timeout=120.0)
                return resp.json()["choices"][0]["message"]["content"] if resp.status_code == 200 else None
            except Exception as e:
                logger.debug(f"vLLM error: {e}")
                return None
        else:
            try:
                resp = await client.post("https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                             "anthropic-version": "2023-06-01", "content-type": "application/json"},
                    json={"model": CLAUDE_MODEL, "max_tokens": 4096,
                          "system": system, "messages": [{"role": "user", "content": user}]},
                    timeout=120.0)
                # QM-3: Only attempt JSON parsing on successful responses.
                # resp.json() on a non-200 body (e.g. rate-limit HTML) raises
                # a JSONDecodeError which was silently swallowed before.
                if resp.status_code != 200:
                    return None
                return resp.json()["content"][0]["text"]
            except Exception as e:
                logger.debug(f"Claude API error: {e}")
                return None

    def _build_so_prompt(self, src: Path) -> str:
        data = json.loads(src.read_text())
        return (
            f"Title: {data.get('title', '')}\n"
            f"Engine: {data.get('engine', 'postgresql')}\n\n"
            f"Question:\n{data.get('body', '')[:3000]}\n\n"
            f"Answer:\n{data.get('best_answer', {}).get('body', '')[:4000]}\n\n"
            "Extract query optimization training pairs from this Q&A."
        )

    def _build_explain_prompt(self, src: Path) -> str:
        data = json.loads(src.read_text())
        return (
            f"Engine: {data.get('engine', 'postgresql')}\n\n"
            f"EXPLAIN ANALYZE output:\n```\n{data.get('plan', '')[:5000]}\n```\n\n"
            f"Query: {data.get('query', 'Unknown')}\n\n"
            "Analyze this plan and generate an optimization training pair."
        )

    async def _synthesize_one(self, system: str, user: str, out_path: Path,
                               client: httpx.AsyncClient, sem: asyncio.Semaphore) -> bool:
        async with sem:
            raw = await self._call(system, user, client)
            if not raw:
                return False
            import re
            data = None
            code_m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
            if code_m:
                try:
                    data = json.loads(code_m.group(1))
                except json.JSONDecodeError:
                    pass
            if not data:
                try:
                    data = json.loads(raw.strip())
                except json.JSONDecodeError:
                    pass
            if not data:
                return False
            out_path.write_text(json.dumps(data if isinstance(data, list) else [data], ensure_ascii=False, indent=2))
            return True

    def run_all(self, limit: int | None = None) -> int:
        return asyncio.run(self._run_all_async(limit=limit))

    async def _run_all_async(self, limit: int | None = None) -> int:
        sem = asyncio.Semaphore(5 if self.backend == "claude" else 20)
        total = 0

        async with httpx.AsyncClient() as client:
            # DBA Stack Exchange Q&A
            so_files = list((self.raw_dir / "stackexchange").glob("q_*.json"))
            logger.info(f"Stream 1 (DBA SE): {len(so_files)} files")
            for i, f in enumerate(so_files[:limit]):
                out = self.output_dir / f"se_{i:06d}.json"
                if not out.exists():
                    ok = await self._synthesize_one(QUERY_OPTIMIZATION_SYSTEM_PROMPT, self._build_so_prompt(f), out, client, sem)
                    if ok:
                        total += 1

            # EXPLAIN plans
            explain_files = list((self.raw_dir / "explain_plans").glob("*.json"))
            logger.info(f"Stream 2 (EXPLAIN): {len(explain_files)} files")
            for i, f in enumerate(explain_files[:limit]):
                out = self.output_dir / f"explain_{i:06d}.json"
                if not out.exists():
                    ok = await self._synthesize_one(EXPLAIN_ANALYSIS_SYSTEM_PROMPT, self._build_explain_prompt(f), out, client, sem)
                    if ok:
                        total += 1

        # Merge to JSONL
        master = self.output_dir / "dataset.jsonl"
        count = 0
        merged_files = []
        with master.open("w") as f_out:
            for json_file in sorted(self.output_dir.glob("*.json")):
                try:
                    data = json.loads(json_file.read_text())
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                        count += 1
                    merged_files.append(json_file)
                except Exception:
                    pass

        # QM-5: Delete the intermediate per-source .json files now that they
        # have been merged into the master JSONL. Leaving them would cause them
        # to be re-read and double-counted on the next synthesis run.
        for json_file in merged_files:
            try:
                json_file.unlink()
            except OSError:
                pass

        logger.success(f"QueryMedic synthesis: {total} sources → {count} pairs → {master}")
        return count
