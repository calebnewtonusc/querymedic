# QueryMedic Roadmap

**"Diagnose. Prescribe. Prove faster."**

---

## Phase 1 — v1: DIAGNOSE & PRESCRIBE (Current Build)

**Status:** In training. Target release: Q2 2026.

**What it does:** Any slow query + EXPLAIN ANALYZE output → root cause + index prescription + query rewrite + timing proof.

**Key capabilities shipping in v1:**
- [ ] EXPLAIN ANALYZE parser (PostgreSQL, MySQL, SQLite)
- [ ] Index type reasoning (B-tree, GIN, GiST, BRIN, Hash, Partial, Covering)
- [ ] Write amplification analysis for every recommendation
- [ ] Storage estimate for proposed indexes
- [ ] Query rewrite for plan improvement (CTE, subquery, join order)
- [ ] Row estimation error detection
- [ ] Validation agent (EXPLAIN ANALYZE before/after timing)
- [ ] QueryBench v1.0 (300 scenarios, 5 categories) — published as community standard
- [ ] REST API + streaming recommendations
- [ ] Docker one-command deploy

**Target metrics (v1):**
| Metric | Target |
|---|---|
| QueryBench overall | >80 |
| EXPLAIN interpretation accuracy | >85% |
| Index type correct selection | >80% |
| Query rewrite improves plan | >75% |
| Write amplification correctly modeled | >85% |
| API latency p50 | <500ms |
| Supported engines | PostgreSQL, MySQL, SQLite |

---

## Phase 2 — v1.5: WORKLOAD ANALYSIS (Q3 2026)

**What it adds:** QueryMedic moves from single-query optimization to workload-level analysis.

### pg_stat_statements Integration

Feed QueryMedic a `pg_stat_statements` snapshot:
- Identifies top-N slow queries by total_time or mean_time
- Cross-query index analysis: finds indexes that benefit multiple queries
- Index conflict detection: flag indexes that hurt high-frequency write queries
- Workload-optimized recommendation set (fewer indexes that do more)

### Schema Workload Profiling

Given the access pattern (OLTP vs. OLAP vs. time-series):
- OLTP: tight covering indexes, no GIN, low-overhead partial indexes
- OLAP: BRIN on date columns, deferred index builds, partial indexes on hot data
- Time-series: range partitioning by date, BRIN dominant, minimal secondary indexes

**Target metrics (v1.5):**
| Metric | Target |
|---|---|
| Workload-level index recommendation quality | >75% |
| Index conflict detection recall | >80% |
| Cross-query optimization improvement | >20% vs. per-query optimization |

---

## Phase 3 — v2: CONTINUOUS ADVISOR (Q4 2026)

**What it adds:** Real-time monitoring with proactive recommendations.

### Slow Query Log Monitoring

- Parses PostgreSQL slow query log, MySQL slow query log, SQLite trace
- Maintains a ring buffer of slow query patterns
- Deduplicates similar queries (normalized by literals)
- Alerts on new patterns not previously optimized

### Regression Detection

- Tracks execution plan for known-good queries
- Alerts if a plan changes (optimizer regression after statistics update, data volume change)
- Correlates with recent schema changes or deploy events
- Generates automated migration scripts for new indexes

**Target metrics (v2):**
| Metric | Target |
|---|---|
| Slow query detection latency | <5 minutes |
| Regression detection recall | >90% |
| False positive rate | <5% |
| Automated migration script correctness | >95% |

---

## Phase 4 — v3: SCHEMA ARCHITECT (Q1 2027)

**What it adds:** Full schema design from access pattern specifications.

### Schema Design from Access Patterns

Input: access pattern specification document or ORM model
Output: DDL with:
- Table partitioning strategy
- Full index set
- Denormalization recommendations
- Materialized view suggestions

### Connection Pool Configuration

Given: max_connections, application workload, hardware
Output: PgBouncer configuration, connection pool sizing, statement timeout settings

**Target metrics (v3):**
| Metric | Target |
|---|---|
| Schema design quality vs. expert DBA | >80% overlap |
| Partition strategy correctness | >85% |
| Connection pool recommendation accuracy | >80% |
