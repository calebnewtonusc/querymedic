# QueryMedic — Full System Architecture
## "Diagnose. Prescribe. Prove faster."

---

## THE VISION

DBA pastes a slow query and its EXPLAIN ANALYZE output. QueryMedic reads the plan, identifies the 60,000x row estimate error, prescribes a partial covering index with write amplification analysis, and returns DDL + EXPLAIN ANALYZE proof of improvement.

Not "add an index on that column." Not "run ANALYZE." Real plan-level diagnosis. Measured proof. Write amplification budget.

---

## 4-PHASE PRODUCT VISION

```
Phase 1 (v1):   DIAGNOSE & PRESCRIBE    EXPLAIN → index + rewrite + proof   ← CURRENT BUILD
Phase 2 (v1.5): WORKLOAD ANALYSIS       pg_stat_statements → schema recommendations
Phase 3 (v2):   CONTINUOUS ADVISOR      monitors slow query log, alerts on regressions
Phase 4 (v3):   SCHEMA ARCHITECT        full schema design from access patterns
```

### Phase 1 — v1: DIAGNOSE & PRESCRIBE (Current)

The foundation. Any slow query + EXPLAIN ANALYZE output produces:
- Root cause: which plan node is the bottleneck and why
- Index prescription: type, columns, partial predicate, INCLUDE clause
- Query rewrite: CTE restructuring, join order, lateral, EXISTS vs IN
- Validation: EXPLAIN ANALYZE timing before/after
- Write amplification: write overhead analysis for every index recommendation
- Storage estimate: index size prediction

### Phase 2 — v1.5: WORKLOAD ANALYSIS

QueryMedic ingests `pg_stat_statements` / `performance_schema` (MySQL) output:
- Full workload-level analysis (not single-query myopia)
- Index recommendations that optimize the full workload, not just one query
- Index conflict detection: does this index help Q1 but hurt Q2's write performance?
- Missing index inference from sequential scan patterns

### Phase 3 — v2: CONTINUOUS ADVISOR

QueryMedic monitors the slow query log:
- Real-time alerting on new slow queries
- Regression detection: was this query fast yesterday?
- Correlation: did a recent deploy introduce the slow query?
- Automated PR suggestions for migration scripts

### Phase 4 — v3: SCHEMA ARCHITECT

Given an access pattern specification (read:write ratio, query patterns, data volume, growth rate), QueryMedic designs the schema from scratch:
- Table partitioning strategy (range, hash, list)
- Index strategy for the full access pattern
- Denormalization recommendations (when to break 3NF for performance)
- Connection pooling recommendations (PgBouncer config)

---

## TARGET METRICS

| Version | Task | Success Rate | Latency | Key Benchmark |
|---------|------|-------------|---------|---------------|
| v1 | EXPLAIN interpretation accuracy | >85% | <500ms | QueryBench-Plan |
| v1 | Index type correctly selected | >80% | — | QueryBench-Index |
| v1 | Query rewrite improves plan | >75% | — | QueryBench-Rewrite |
| v1 | Write amplification correctly modeled | >85% | — | QueryBench-Write |
| v1.5 | Workload-level index recommendation | >75% | <2s | QueryBench-Workload |
| v2 | Slow query detection latency | <5min | — | QueryBench-Monitor |
| v3 | Schema design quality vs. expert | TBD | <30s | QueryBench-Schema |

---

## 6 TECHNICAL DIFFERENTIATORS

### 1. EXPLAIN ANALYZE → Optimization Corpus (the data moat)

The EXPLAIN ANALYZE output + optimization pair corpus has never been assembled. QueryMedic is trained on thousands of (slow query + EXPLAIN plan → index DDL + proof of improvement) pairs. Every other optimizer tool generates generic advice. QueryMedic reads the actual plan.

### 2. Index Type Intelligence

Not all indexes are created equal. QueryMedic knows:
- **B-tree**: default, range queries, ORDER BY
- **GIN**: multi-valued data (arrays, JSONB, tsvector full-text)
- **GiST**: geometric data, range types, full-text (slower than GIN but updateable in-place)
- **BRIN**: monotonically increasing data (timestamps, IDs), tiny storage overhead
- **Hash**: equality lookups only, faster than B-tree for pure equality
- **Partial**: WHERE predicate reduces index size, faster writes for inactive rows
- **Covering (INCLUDE)**: avoids heap fetch for index-only scans
- **Multi-column**: column order matters (selectivity first vs. equality first)

### 3. Write Amplification Modeling

Every index recommendation includes write overhead analysis:
- Each index adds ~1 additional B-tree update per INSERT/UPDATE/DELETE
- Partial index write cost: proportional to fraction of rows matching predicate
- GIN index write cost: higher per-update due to posting list maintenance
- For write-heavy workloads (>70% writes): QueryMedic may recommend deferred index builds or no index

### 4. Row Estimation Debugging

The #1 cause of bad query plans is stale statistics causing wrong row estimates. QueryMedic:
- Identifies estimation errors from EXPLAIN ANALYZE (actual rows vs. estimated rows)
- Recommends targeted `ANALYZE` with `default_statistics_target` adjustment
- Identifies correlation statistics gaps (multi-column statistics: `CREATE STATISTICS`)
- Understands when PostgreSQL's planner needs hints (pg_hint_plan)

### 5. Query Rewrite for Plan Steering

Sometimes the schema is fine but the query is wrong. QueryMedic can:
- Convert correlated subqueries to lateral joins
- Restructure EXISTS vs IN vs JOIN for the planner
- Add CTEs to materialize intermediate results (or remove CTEs to let planner inline)
- Restructure anti-joins (NOT EXISTS vs LEFT JOIN ... IS NULL)
- Rewrite DISTINCT with GROUP BY when planner chooses HashAggregate over Sort

### 6. Multi-Engine Internal Knowledge

PostgreSQL, MySQL/InnoDB, and SQLite have different internals:
- **PostgreSQL**: MVCC, HOT updates, index-only scans, parallel query, partial indexes
- **MySQL/InnoDB**: clustered primary key, secondary index includes PK, covering index = PK included
- **SQLite**: B-tree with row IDs, covering indexes, WITHOUT ROWID tables

---

## TRAINING ARCHITECTURE

### Three-Stage Training

```
Stage 1: SFT (Supervised Fine-Tuning)
  Data: 300k query optimization pairs (4 streams)
  Loss: Next-token prediction on assistant turns only
  Duration: ~5 hours on 18x A6000

Stage 2: GRPO RL (Group Relative Policy Optimization)
  Reward: EXPLAIN ANALYZE timing improvement + index size within budget + no result change
  Rollouts: Model generates 8 index/rewrite candidates per slow query
  Duration: ~6 hours on 18x A6000

Stage 3: DPO (Direct Preference Optimization)
  Data: 50k preferred/rejected optimization pairs
  Preferred: Plan-level reasoning, write amplification analysis, measured improvement
  Rejected: Generic index advice, no plan interpretation, ignoring write overhead
  Duration: ~3 hours on 12x A6000
```

### Reward Function Design (GRPO)

```python
def query_optimization_reward(before_plan, after_plan, index_ddl, before_timing, after_timing):
    reward = 0.0

    # Primary: timing improvement (0.5 weight)
    if after_timing < before_timing:
        improvement = before_timing / after_timing
        reward += 0.5 * min(1.0, (improvement - 1) / 10)

    # Secondary: plan improvement (0.2 weight)
    if "Seq Scan" in before_plan and "Index Scan" in after_plan:
        reward += 0.2  # Eliminated sequential scan

    # Constraint: write amplification budget (0.2 weight)
    index_count = count_new_indexes(index_ddl)
    if index_count <= 2:  # Reward lean recommendations
        reward += 0.2

    # Constraint: correctness (0.1 weight, hard penalty)
    if results_match:
        reward += 0.1
    else:
        reward = -5.0  # Hard penalty for result change

    return reward
```
