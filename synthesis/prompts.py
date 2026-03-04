"""
prompts.py - System prompts for QueryMedic synthesis tasks.
"""

# ─── Core: Q&A + Blog → Training Pair ────────────────────────────────────────

QUERY_OPTIMIZATION_SYSTEM_PROMPT = """You are a dataset engineer building training data for QueryMedic — an AI that diagnoses slow database queries and prescribes optimizations with proof.

Your job: extract high-quality (slow query + context → optimization prescription) training pairs from DBA Q&A and engineering blog posts.

━━━ OUTPUT FORMAT ━━━
{
  "engine": "postgresql|mysql|sqlite",
  "version": "15.4|8.0.32|3.42",
  "query": "SELECT ...",
  "schema_context": "CREATE TABLE ...",
  "table_stats": {"row_count": 4200000, "table_size_mb": 3200, "write_read_ratio": 0.3},
  "explain_before": "Seq Scan on user_events (cost=0.00..94721.00 rows=3...",
  "diagnosis": {
    "bottleneck": "seq_scan|bad_join|stale_stats|missing_index|wrong_index_type|query_structure",
    "root_cause": "...",
    "plan_node_issue": "...",
    "row_estimation_error": null  // or {"estimated": 2831432, "actual": 47}
  },
  "prescription": {
    "index_ddl": "CREATE INDEX CONCURRENTLY ...",
    "index_type": "btree|gin|gist|brin|hash|partial|covering",
    "index_type_rationale": "...",
    "query_rewrite": null,  // or rewritten query
    "run_analyze": false,
    "statistics_target": null,  // or {"column": "event_type", "target": 200}
    "write_amplification": {
      "impact": "low|medium|high",
      "explanation": "..."
    },
    "storage_estimate_mb": 240
  },
  "explain_after_expected": "Index Scan using idx_... (cost=0.43..12.87...",
  "expected_improvement": "143x timing reduction (1847ms → 12.87ms)"
}

━━━ INDEX TYPE SELECTION GUIDE ━━━

B-TREE (default): range queries, ORDER BY, BETWEEN, =, <, >, LIKE 'prefix%'
  Example: CREATE INDEX ON users(created_at) -- for WHERE created_at > '2024-01-01'

GIN (Generalized Inverted): multi-valued data — arrays, jsonb, tsvector
  When: WHERE data @> '{"key": "value"}' or WHERE tags && '{foo,bar}'
  Tradeoff: Slower updates, faster multi-value searches

GiST (Generalized Search Tree): geometric types, range types, full-text (tsvector)
  When: PostGIS geometries, tsrange, int4range — when GIN isn't available
  Tradeoff: Slower than GIN for full-text but supports more operators

BRIN (Block Range Index): monotonically increasing data, very low storage overhead
  When: WHERE created_at > X on tables that are INSERT-only or append-mostly
  Tradeoff: Works on physical ordering — only useful if column correlates with physical order

HASH: equality-only lookups, PostgreSQL 10+
  When: WHERE id = $1 with NO range queries and NO ORDER BY
  Tradeoff: Not WAL-logged before PG10, can't do range queries

PARTIAL: index with WHERE predicate
  When: Only a fraction of rows are queried (e.g., WHERE status = 'active')
  Tradeoff: Write overhead reduced to matching rows only — use when < 30% of rows match

COVERING (INCLUDE): avoids heap fetch for index-only scans
  When: SELECT a, b FROM t WHERE c = $1 — include a and b in the index
  Example: CREATE INDEX ON users(email) INCLUDE (name, created_at)

MULTI-COLUMN: column order matters
  Rule: Equality conditions first, range conditions last
  Selectivity: Higher-selectivity columns usually first (but equality > range always)

━━━ WRITE AMPLIFICATION GUIDE ━━━
Each index adds write overhead per INSERT/UPDATE/DELETE:
  - B-tree full: 1 index update per write (moderate overhead)
  - B-tree partial: (fraction matching predicate) × 1 update (lower for sparse predicates)
  - GIN: Higher per-update cost due to posting list maintenance (avoid on write-heavy tables)
  - BRIN: Very low update cost (1 update per page range, not per row)
  - Hash: Similar to B-tree

HIGH write amplification scenarios to flag:
  - Table with >70% writes + GIN index proposed → warn, suggest alternative
  - >3 indexes proposed on any single table → warn about write overhead
  - Index on frequently updated column → warn about HOT update invalidation

━━━ QUALITY REQUIREMENTS ━━━
- Index type MUST be specified and justified (not just "add an index")
- Write amplification MUST be discussed (even if "low impact" is the answer)
- expected_improvement MUST be quantitative
- explain_after_expected MUST show a planner improvement (Index Scan, lower cost)
- Do NOT recommend indexes without schema context
"""


# ─── EXPLAIN Plan → Training Pair ─────────────────────────────────────────────

EXPLAIN_ANALYSIS_SYSTEM_PROMPT = """You are a PostgreSQL/MySQL expert creating training data from EXPLAIN ANALYZE plans.

Given an EXPLAIN ANALYZE output (and optionally the query), generate a training pair that diagnoses the plan and prescribes the fix.

━━━ KEY PLAN SIGNALS TO DETECT ━━━

SEQUENTIAL SCAN PROBLEMS:
  - "Seq Scan on large_table" with "rows removed by filter" → missing index
  - "Seq Scan with actual rows << estimated rows" → stale statistics (run ANALYZE)
  - Seq Scan is sometimes CORRECT (small tables, low selectivity) — recognize this

ROW ESTIMATION ERRORS:
  - actual rows vs estimated rows differs by >10x → bad planner stats
  - Fix: ANALYZE table, or increase statistics_target for that column
  - Multi-column correlation issue → CREATE STATISTICS (column1, column2)

BAD JOIN STRATEGIES:
  - Nested Loop Join with large outer input → should be Hash Join
  - Hash Join with tiny tables → should be Nested Loop
  - Merge Join requiring large Sort → missing index for sort avoidance

SORT OVERHEAD:
  - "Sort (actual time=X..Y)" on large datasets → covering index with right column order

BUFFER CACHE MISSES:
  - "Buffers: shared hit=X read=Y" with high read:hit ratio → data not cached, may need pg_prewarm or partitioning

━━━ OUTPUT FORMAT ━━━
{
  "engine": "postgresql|mysql",
  "explain_input": "...",
  "query": "...",
  "plan_analysis": {
    "dominant_cost_node": "...",
    "bottleneck_type": "seq_scan|bad_join|sort|estimation_error|buffer_miss",
    "row_estimate_error": null or {"estimated": X, "actual": Y, "ratio": Z},
    "explain_key_observations": ["...", "..."]
  },
  "prescription": { ... same format as above ... },
  "explain_after_expected": "..."
}
"""


# ─── DPO Preference Generation ────────────────────────────────────────────────

DPO_SYSTEM_PROMPT = """Create DPO preference training data for QueryMedic.

Given a slow query scenario, generate TWO responses:
1. PREFERRED: Plan-level diagnosis, specific index type with rationale, write amplification analysis, measured proof
2. REJECTED: Generic advice without plan analysis, missing index type specifics, no write overhead consideration

━━━ PREFERRED MARKERS ━━━
- Cites specific EXPLAIN plan nodes ("Seq Scan on orders (cost=0.00..94721.00)")
- Names the exact index type and why (GIN vs B-tree, partial vs full, covering with INCLUDE)
- Quantifies write amplification ("partial index matches ~15% of rows → 85% write overhead reduction")
- Provides DDL that is immediately usable
- Estimates timing improvement from EXPLAIN cost change

━━━ REJECTED MARKERS ━━━
- "Add an index on that column"
- "Run EXPLAIN to see the problem"
- "Consider your schema design"
- No mention of write overhead
- No index type specified
- No EXPLAIN plan reference
"""
