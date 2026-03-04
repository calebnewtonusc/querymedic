# QueryMedic — Data Sources

## Overview

QueryMedic's training data is built from 4 streams, targeting 300,000+ (slow query + context → optimized prescription) pairs. Every pair includes the database internals rationale — not just what to do, but why the planner chose the wrong plan and how the fix addresses it.

---

## Stream 1: DBA Stack Exchange (30% — ~90k pairs)

The world's highest-quality database Q&A. Accepted answers from senior DBAs explaining query plans, index selection, and schema design.

### Filters Applied
- Tags: `performance`, `indexing`, `postgresql`, `mysql`, `sqlite`, `query-optimization`, `explain`, `partitioning`
- Minimum question score: 10
- Minimum answer score: 15 OR accepted answer
- Body must reference: EXPLAIN output, execution plan, timing, or index recommendation

### Target Volume by Tag

| Tag Combination | Estimated Pairs |
|---|---|
| postgresql + performance | 12,000+ |
| postgresql + indexing | 8,000+ |
| mysql + query-optimization | 6,000+ |
| postgresql + explain | 4,000+ |
| postgresql + partitioning | 2,000+ |
| mysql + indexing | 5,000+ |

### Synthesis Process

Each Q&A → structured training pair:
```json
{
  "query": "SELECT * FROM user_events WHERE user_id = $1 AND created_at > NOW() - INTERVAL '30 days'",
  "engine": "postgresql",
  "version": "15",
  "explain_before": "Seq Scan on user_events (cost=0.00..94721.00 rows=3 width=152) ...",
  "schema_context": "CREATE TABLE user_events (id bigint, user_id int, created_at timestamptz, ...)",
  "table_stats": {"row_count": 4200000, "table_size_mb": 3200},
  "diagnosis": "Sequential scan despite user_id equality predicate. No index on (user_id, created_at).",
  "prescription": {
    "index_ddl": "CREATE INDEX CONCURRENTLY idx_user_events_uid_cat ON user_events(user_id, created_at DESC) WHERE created_at > '2024-01-01';",
    "index_type": "B-tree partial",
    "rationale": "Multi-column index with user_id first (equality → high selectivity) then created_at DESC (range, matches ORDER BY). Partial predicate reduces index to ~30% of rows.",
    "write_amplification": "Low: ~30% of INSERT/UPDATE events match partial predicate. Estimated overhead: +15ms/100k writes.",
    "storage_estimate_mb": 280,
    "expected_plan": "Index Scan using idx_user_events_uid_cat (cost=0.43..8.47 rows=3 width=152)"
  }
}
```

---

## Stream 2: Database Internals Documentation + Synthesized Scenarios (25% — ~75k pairs)

### Primary Documents

| Document | Content | Synthesis Target |
|---|---|---|
| PostgreSQL Documentation (15.x) | Planner, executor, index types | 20,000+ pairs |
| MySQL 8.0 Reference Manual | InnoDB internals, optimizer, EXPLAIN FORMAT=TREE | 15,000+ pairs |
| SQLite Documentation | Query planner, ANALYZE, WITHOUT ROWID | 5,000+ pairs |
| "PostgreSQL: Up and Running" (Regina Obe) | Schema design patterns | 5,000+ pairs |
| "High Performance MySQL" (O'Reilly) | InnoDB B+tree, covering indexes | 5,000+ pairs |
| "Use The Index, Luke" (Markus Winand) | Index internals across engines | 5,000+ pairs |
| PostgreSQL Wiki: Performance | Slow query tuning, EXPLAIN tips | 10,000+ pairs |
| pg_stat_statements documentation | Workload analysis | 5,000+ pairs |

### Synthesis Approach

From specification text → scenario:
```
Spec: "A GIN index on a jsonb column supports @> (contains) and ? (exists) operators but
      not equality on extracted values (jsonb->>'key' = 'value')."
→ Scenario: Query uses WHERE data->>'event_type' = 'purchase' with GIN index on data.
→ Diagnosis: GIN index doesn't accelerate ->> equality. Need B-tree on (data->>'event_type').
→ Prescription: CREATE INDEX ON events ((data->>'event_type')) WHERE data ? 'event_type';
```

---

## Stream 3: High-Scale Engineering Blogs (20% — ~60k pairs)

### Primary Sources

| Source | Focus | Relevant Posts |
|---|---|---|
| Discord Engineering | PostgreSQL at 10B+ rows, partitioning | 50+ |
| Slack Engineering | PostgreSQL MVCC, vacuuming at scale | 30+ |
| Dropbox Tech Blog | MySQL optimization, connection pools | 40+ |
| GitHub Engineering | MySQL at scale, read replicas | 30+ |
| Instagram Engineering | PostgreSQL sharding, index design | 25+ |
| Shopify Engineering | MySQL performance, connection pools | 30+ |
| Notion Engineering | PostgreSQL at scale | 20+ |
| Figma Engineering | PostgreSQL optimization | 15+ |
| Citus Data Blog | PostgreSQL distributed, partitioning | 60+ |
| 2ndQuadrant Blog | PostgreSQL internals | 100+ |
| Planet PostgreSQL | Community posts | 500+ |

---

## Stream 4: EXPLAIN ANALYZE Output + Optimization Pairs (25% — ~75k pairs)

The rarest and most valuable data: actual EXPLAIN ANALYZE output paired with the optimization that fixed it.

### Collection Sources

**GitHub Open-Source Optimization PRs:**
- PostgreSQL ecosystem projects (Rails ActiveRecord, Django ORM migration PRs)
- Database migration files with EXPLAIN ANALYZE in PR descriptions
- DBA retrospectives in README files

**Community Collections:**
- explain.depesz.com — public EXPLAIN ANALYZE sharing (thousands of plans)
- explain.tensor.ru — another popular plan sharing service
- Stack Overflow answers with EXPLAIN ANALYZE paste + optimization
- PostgreSQL mailing list archives (pgsql-performance list)

### Format Variety Covered

| Format | Notes |
|---|---|
| PostgreSQL EXPLAIN ANALYZE text | Standard format, most common |
| PostgreSQL EXPLAIN (FORMAT JSON) | Machine-readable, full cost breakdown |
| PostgreSQL EXPLAIN (FORMAT YAML) | Less common |
| MySQL EXPLAIN FORMAT=TRADITIONAL | Classic tabular format |
| MySQL EXPLAIN FORMAT=TREE | Modern tree format |
| MySQL EXPLAIN ANALYZE | Added in MySQL 8.0.18 |
| SQLite EXPLAIN QUERY PLAN | Simplified format |

---

## Data Quality Filters

1. **Engine identified**: PostgreSQL/MySQL/SQLite must be determinable
2. **EXPLAIN present**: Training pairs with actual plan output preferred
3. **Prescription is specific**: Must name index type and columns (not "add an index")
4. **Write amplification mentioned**: Pairs that ignore write cost are down-weighted
5. **Quantitative improvement**: Must include measured or estimated timing improvement
6. **Anti-patterns removed**: "run ANALYZE", "add more RAM", "check your connection pool" answers

## Estimated Final Dataset Size

| Stream | Pairs | % |
|---|---|---|
| DBA Stack Exchange | 90,000 | 30% |
| Database documentation | 75,000 | 25% |
| Engineering blogs | 60,000 | 20% |
| EXPLAIN + optimization pairs | 75,000 | 25% |
| **Total** | **300,000** | **100%** |
