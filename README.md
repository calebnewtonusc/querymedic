# QueryMedic — The Database Query Optimization Specialist Model

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model: Qwen2.5-7B-Coder](https://img.shields.io/badge/base_model-Qwen2.5--7B--Coder-purple.svg)](https://huggingface.co/Qwen)
[![GPUs: 18x A6000](https://img.shields.io/badge/training-18×_A6000_48GB-red.svg)](https://www.nvidia.com)
[![Databases: PostgreSQL MySQL SQLite](https://img.shields.io/badge/databases-PostgreSQL_MySQL_SQLite-blue.svg)]()

> **"Diagnose. Prescribe. Prove faster."**

QueryMedic is the first AI trained to optimize database queries the way a senior DBA does — not just surface-level syntax suggestions, but workload-aware schema design, EXPLAIN ANALYZE plan reasoning, index type selection (GIN vs. GiST vs. partial vs. covering), and write amplification budgeting. Feed it a slow query and its execution plan, and it prescribes the fix with measured proof.

This repository contains the complete dataset pipeline, training infrastructure, and deployment stack for QueryMedic — from raw DBA Stack Exchange posts to a production-ready optimization API.

---

## What Makes QueryMedic Different

| Capability | DBA tools | ORM advisors | ChatGPT | QueryMedic |
|---|---|---|---|---|
| EXPLAIN ANALYZE interpretation | partial | — | generic | **Full: planner stats, row estimation, node costs** |
| Index type selection | limited | — | generic | **GIN/GiST/partial/covering/BRIN — workload-specific** |
| Write amplification awareness | — | — | — | **Tracks write overhead of every index recommendation** |
| Query rewrite for plan improvement | — | — | generic | **Rewrites targeting specific plan nodes** |
| Storage budget constraints | — | — | — | **Index recommendations within storage budget** |
| Multi-engine support | per-engine | — | generic | **PostgreSQL, MySQL, SQLite internals** |
| Timing validation | — | — | — | **EXPLAIN ANALYZE before/after with real timing** |
| Workload pattern reasoning | — | — | — | **OLTP vs. OLAP vs. time-series schema design** |

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
  EXPLAIN Output  ─►│             QueryMedic Model                 │
  Query Workload  ─►│  (Qwen2.5-7B-Coder + LoRA, 3-stage training)│
  Schema DDL      ─►│   SFT → GRPO Reward RL → DPO preference     │
                    └─────────────────┬────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────────┐
                    │           Database Internals Core            │
                    │  PostgreSQL Planner (cost model, stats)      │
                    │  MySQL InnoDB (B+tree, covering, prefix)     │
                    │  SQLite Query Optimizer                      │
                    └──────┬───────────────────────────────────────┘
                           │
           ┌───────────────┼────────────────────────────────┐
           ▼               ▼                                ▼
  ┌──────────────┐  ┌──────────────────┐          ┌──────────────────┐
  │  Query       │  │  Index           │          │  Validation      │
  │  Analyzer    │  │  Agent           │          │  Agent           │
  │  Agent       │  │                  │          │                  │
  │              │  │  GIN vs GiST     │          │  Runs EXPLAIN    │
  │  Reads plan, │  │  Partial vs full │          │  ANALYZE before/ │
  │  diagnoses   │  │  Covering index  │          │  after, checks   │
  │  bottleneck  │  │  BRIN for ranges │          │  timing + rows   │
  └──────┬───────┘  └──────┬───────────┘          └───────┬──────────┘
         │                 │                              │
         │         ┌───────────────────┐                 │
         │         │  Rewrite Agent    │                 │
         │         │  CTE, subquery,   │                 │
         │         │  join order,      │                 │
         │         │  lateral joins    │                 │
         └─────────┴───────────────────┴─────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────────┐
                    │            Verified Prescription             │
                    │  - Index DDL with type rationale            │
                    │  - Query rewrite (if applicable)            │
                    │  - EXPLAIN ANALYZE timing improvement       │
                    │  - Write amplification impact               │
                    │  - Storage cost estimate                    │
                    └──────────────────────────────────────────────┘
```

**Training data sources (4 streams, 300k+ pairs):**
- Stream 1: DBA Stack Exchange (high-vote, accepted answers) → query optimization pairs (30%)
- Stream 2: PostgreSQL/MySQL/Oracle optimization documentation → synthesized scenarios (25%)
- Stream 3: High-scale engineering blogs (Discord, Slack, Dropbox, GitHub) → workload patterns (20%)
- Stream 4: EXPLAIN ANALYZE output + before/after optimization pairs (25%)

---

## Comparison: Generic vs. QueryMedic

**Generic LLM response to "my user_activity query is slow":**
> "Consider adding an index on frequently queried columns. Use EXPLAIN to see the query plan."

**QueryMedic response:**
> "Your EXPLAIN ANALYZE shows `Seq Scan on user_activity (cost=0.00..84721.00 rows=2831432 width=8)` with `actual time=0.021..1847.234 rows=47 loops=1`. Row estimate is off by 60,000x — planner thinks it'll get 2.8M rows but gets 47. This is a stale statistics problem on `user_id` + `event_type`. **Fix 1**: `ANALYZE user_activity;` then re-run. **Fix 2**: Multi-column index: `CREATE INDEX CONCURRENTLY idx_user_activity_uid_etype ON user_activity(user_id, event_type)` — this is a covering index for your SELECT list. With your write:read ratio of ~3:1, consider a partial index: `WHERE event_type = 'purchase'` (15% of rows) to minimize write amplification. Expected: EXPLAIN shows `Index Scan` at `cost=0.43..12.87` — 143x improvement. Storage: ~240MB for full index, ~36MB partial."

---

## Quick Start

**1. Clone**
```bash
git clone https://github.com/calebnewtonusc/querymedic.git
cd querymedic
pip install -r requirements.txt
cp .env.example .env
```

**2. Fill in `.env`** (see `.env.example` for all variables)

**3. Validate**
```bash
bash scripts/check_env.sh
```

**4. Run full pipeline**
```bash
bash scripts/run_all.sh

# Or step by step:
python discovery/dba_stackexchange.py --min-votes 10
python discovery/db_blog_crawler.py --all
python synthesis/synthesize_bulk.py --backend claude
deepspeed --num_gpus=8 training/train.py --deepspeed training/configs/ds_config.json
python evaluation/querybench.py --model checkpoints/querymedic-final --all
```

---

## QueryBench

QueryBench is QueryMedic's task-specific evaluation suite — 300 real query optimization scenarios across PostgreSQL, MySQL, and SQLite. It tests:

- **Plan interpretation**: Does the model correctly read EXPLAIN ANALYZE output?
- **Index prescription**: Does the recommended index type match the query pattern?
- **Timing improvement**: Measured EXPLAIN ANALYZE before/after improvement
- **Write amplification**: Does the model account for write overhead?
- **Storage budget**: Recommendations within stated storage constraints
- **Correctness**: Zero result set change after optimization

```bash
python evaluation/querybench.py --model checkpoints/querymedic-final --all
```

Categories: Index selection (80), Query rewrite (80), Schema design (60), Statistics tuning (40), Partitioning (40)

---

## Hardware Requirements

### Training
| Resource | Specification |
|---|---|
| GPUs | 18x NVIDIA A6000 (48GB each) |
| Total VRAM | 864GB |
| Strategy | DeepSpeed ZeRO-3 + CPU offload |
| SFT time | ~5 hours (300k pairs, 3 epochs) |

### Inference
| Configuration | Latency | Throughput |
|---|---|---|
| 2x A100 (80GB) | < 300ms | 25 req/s |
| 1x RTX 4090 (24GB) | ~600ms | 8 req/s |

---

## License

**Code:** MIT License
**Model weights:** Apache 2.0

*Target: 864GB VRAM, 300,000+ training pairs. EXPLAIN ANALYZE reasoning. Training in progress.*
