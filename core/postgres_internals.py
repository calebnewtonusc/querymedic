"""
postgres_internals.py - PostgreSQL planner, index types, and MVCC knowledge.

This module encodes PostgreSQL internals used by the Query Analyzer and Index Agents
to generate hardware-aware (planner-aware) recommendations.

Reference:
    - PostgreSQL Documentation: Chapter 14 (Performance Tips)
    - PostgreSQL Source: src/backend/optimizer/
    - "Use The Index, Luke" by Markus Winand
"""

from dataclasses import dataclass, field
from enum import Enum


class IndexType(Enum):
    BTREE = "btree"
    GIN = "gin"
    GIST = "gist"
    BRIN = "brin"
    HASH = "hash"
    SPGIST = "spgist"


@dataclass
class IndexTypeProfile:
    """Complete profile of a PostgreSQL index type."""

    name: IndexType
    display_name: str
    supported_operators: list[str]
    best_for: list[str]
    write_cost_factor: float  # relative to B-tree (1.0 = same, >1 = more expensive)
    storage_factor: float  # relative to B-tree (1.0 = same)
    update_strategy: str
    notes: str


INDEX_PROFILES: dict[IndexType, IndexTypeProfile] = {
    IndexType.BTREE: IndexTypeProfile(
        name=IndexType.BTREE,
        display_name="B-tree (default)",
        supported_operators=[
            "=",
            "<",
            ">",
            "<=",
            ">=",
            "BETWEEN",
            "IN",
            "LIKE 'prefix%'",
            "IS NULL",
            "ORDER BY",
        ],
        best_for=["equality lookups", "range queries", "ORDER BY", "BETWEEN"],
        write_cost_factor=1.0,
        storage_factor=1.0,
        update_strategy="Page-level lock during insert/update",
        notes="Default index type. Works for most cases. Column order matters for multi-column: equality predicates first, then range.",
    ),
    IndexType.GIN: IndexTypeProfile(
        name=IndexType.GIN,
        display_name="GIN (Generalized Inverted Index)",
        supported_operators=["@>", "?", "?|", "?&", "<@", "@@", "&&"],
        best_for=[
            "jsonb containment (@>)",
            "array overlap (&&)",
            "full-text search (@@)",
            "tsvector",
        ],
        write_cost_factor=2.5,
        storage_factor=1.5,
        update_strategy="Buffered inserts via pending list (gin_pending_list_limit)",
        notes="Best for multi-valued data. Higher write cost due to posting list maintenance. For write-heavy tables, consider partial index or GiST alternative.",
    ),
    IndexType.GIST: IndexTypeProfile(
        name=IndexType.GIST,
        display_name="GiST (Generalized Search Tree)",
        supported_operators=["&&", "@>", "<@", "~=", ">>=", "<<=", "@@"],
        best_for=[
            "geometric types (PostGIS)",
            "range types (tsrange, int4range)",
            "full-text (when GIN not available)",
        ],
        write_cost_factor=1.8,
        storage_factor=1.3,
        update_strategy="In-place update (no pending list)",
        notes="More flexible than GIN but slower for full-text. Required for PostGIS geometry indexing. Can index ranges natively.",
    ),
    IndexType.BRIN: IndexTypeProfile(
        name=IndexType.BRIN,
        display_name="BRIN (Block Range Index)",
        supported_operators=["=", "<", ">", "<=", ">=", "BETWEEN"],
        best_for=[
            "monotonically increasing columns (timestamps, IDs)",
            "very large tables",
            "append-only tables",
        ],
        write_cost_factor=0.1,
        storage_factor=0.01,
        update_strategy="Summarizes block ranges — updates are cheap",
        notes="Tiny size (a few KB for billion-row tables). ONLY works when column correlates with physical storage order. For created_at on append-only tables: excellent. For user_id on randomly inserted data: useless.",
    ),
    IndexType.HASH: IndexTypeProfile(
        name=IndexType.HASH,
        display_name="Hash",
        supported_operators=["="],
        best_for=[
            "pure equality lookups with no range queries",
            "very high-cardinality columns",
        ],
        write_cost_factor=1.0,
        storage_factor=0.8,
        update_strategy="Hash table page lock",
        notes="Faster than B-tree for pure equality (=) only. Cannot support range queries, ORDER BY, or LIKE. WAL-logged since PostgreSQL 10. Rarely used — B-tree is usually competitive.",
    ),
    IndexType.SPGIST: IndexTypeProfile(
        name=IndexType.SPGIST,
        display_name="SP-GiST (Space-Partitioned GiST)",
        supported_operators=["=", "<", ">", "<=", ">=", "<<", ">>", "~=", "@>", "<@"],
        best_for=[
            "non-balanced data structures (quad-trees, k-d trees, radix trees)",
            "IP/CIDR ranges (inet)",
            "geometric points",
            "text prefix lookups",
        ],
        write_cost_factor=1.5,
        storage_factor=1.1,
        update_strategy="In-place update using partitioned space (no pending list)",
        notes="Space-Partitioned GiST. More efficient than GiST for data with natural partitioning (e.g., IP addresses, points in bounded space). Use for inet/cidr columns or 2D point data where GiST would be too large.",
    ),
}


@dataclass
class IndexRecommendation:
    """A complete index recommendation with rationale."""

    index_type: IndexType
    table: str
    columns: list[str]
    include_columns: list[str] = field(default_factory=list)
    partial_predicate: str | None = None
    concurrent: bool = True
    ddl: str = ""
    rationale: str = ""
    write_amplification: str = ""
    storage_estimate_mb: float = 0.0

    def generate_ddl(self) -> str:
        """Generate the CREATE INDEX DDL statement."""
        concurrently = "CONCURRENTLY " if self.concurrent else ""
        idx_name = f"idx_{self.table}_{'_'.join(self.columns)}"
        if self.partial_predicate:
            idx_name += "_partial"

        method = self.index_type.value
        cols = ", ".join(self.columns)

        include_clause = ""
        if self.include_columns:
            include_clause = f" INCLUDE ({', '.join(self.include_columns)})"

        partial_clause = ""
        if self.partial_predicate:
            partial_clause = f" WHERE {self.partial_predicate}"

        return (
            f"CREATE INDEX {concurrently}{idx_name}\n"
            f"  ON {self.table} USING {method}({cols})"
            f"{include_clause}"
            f"{partial_clause};"
        )

    def estimate_write_amplification(self, write_fraction: float = 1.0) -> str:
        """Estimate write overhead for this index."""
        profile = INDEX_PROFILES[self.index_type]
        base_cost = profile.write_cost_factor

        if self.partial_predicate:
            effective_cost = base_cost * write_fraction
            return (
                f"Partial index matches ~{write_fraction * 100:.0f}% of rows. "
                f"Effective write overhead: {effective_cost:.1f}x base (vs {base_cost:.1f}x for full index). "
                f"Per 100k writes: ~{int(effective_cost * 10)}ms overhead."
            )
        return (
            f"Full index write overhead: {base_cost:.1f}x base. "
            f"Per 100k writes: ~{int(base_cost * 10)}ms overhead."
        )


class PlannerKnowledge:
    """
    PostgreSQL query planner knowledge base.

    Encodes the cost model, statistics, and join strategy selection logic
    that QueryMedic uses to explain WHY the planner made a bad choice.
    """

    # Cost model constants (default PostgreSQL values)
    SEQ_PAGE_COST = 1.0  # Cost to read a sequential page
    RANDOM_PAGE_COST = 4.0  # Cost to read a random page (default, assumes HDD)
    CPU_TUPLE_COST = 0.01  # Cost per row processed
    CPU_INDEX_TUPLE_COST = 0.005  # Cost per index row processed
    CPU_OPERATOR_COST = 0.0025  # Cost per operator evaluation

    @classmethod
    def explain_seq_scan_cost(cls, row_count: int, page_count: int) -> str:
        """Explain why PostgreSQL might choose a sequential scan."""
        seq_cost = page_count * cls.SEQ_PAGE_COST
        return (
            f"Sequential scan cost: {page_count} pages × {cls.SEQ_PAGE_COST} = {seq_cost:.0f}. "
            f"Index becomes preferred when: "
            f"(index pages + {cls.RANDOM_PAGE_COST}×heap pages) < {seq_cost:.0f}. "
            f"At {row_count:,} rows, selectivity threshold for index preference: "
            f"~{min(100, 100 * 3 / max(1, row_count / 8.0)):.1f}% of rows."
        )

    @classmethod
    def recommend_index_type(cls, query_pattern: str, data_type: str) -> IndexType:
        """Recommend the best index type for a query pattern and data type."""
        query_lower = query_pattern.lower()
        type_lower = data_type.lower()

        # JSONB patterns → GIN
        if any(op in query_lower for op in ["@>", "?", "&&"]) or type_lower == "jsonb":
            return IndexType.GIN

        # Full text search → GIN
        if "@@" in query_lower or "tsvector" in type_lower:
            return IndexType.GIN

        # Geometric types → GiST
        if any(
            t in type_lower
            for t in ["geometry", "geography", "point", "box", "polygon"]
        ):
            return IndexType.GIST

        # Range types → GiST or BRIN
        if any(t in type_lower for t in ["range", "tsrange", "int4range", "numrange"]):
            return IndexType.GIST

        # Monotonic append-only patterns → BRIN
        if any(
            col in query_lower for col in ["created_at", "inserted_at", "timestamp"]
        ):
            return IndexType.BRIN

        # Pure equality on hash-able type → Hash (but usually B-tree is fine)
        if "=" in query_lower and ">" not in query_lower and "<" not in query_lower:
            return IndexType.BTREE  # prefer B-tree for portability

        return IndexType.BTREE


# Common anti-patterns and their diagnoses
ANTIPATTERNS = {
    "leading_wildcard": {
        "pattern": "LIKE '%value%' or LIKE '%value'",
        "diagnosis": "Leading wildcard in LIKE cannot use B-tree index. Left-anchored patterns (LIKE 'value%') work. For full-text search, use tsvector + GIN index with @@ operator.",
        "fix": "Use tsvector and GIN index for arbitrary substring matching.",
    },
    "function_on_indexed_column": {
        "pattern": "WHERE LOWER(email) = 'x'  or WHERE DATE(created_at) = '2024-01-01'",
        "diagnosis": "Function applied to indexed column prevents index usage — planner can't use B-tree index on LOWER(email).",
        "fix": "Create expression index: CREATE INDEX ON users (LOWER(email)); or rewrite query: WHERE created_at >= '2024-01-01' AND created_at < '2024-01-02'",
    },
    "implicit_type_cast": {
        "pattern": "WHERE user_id = '42' when user_id is integer",
        "diagnosis": "Implicit cast from text to integer prevents index use in some PostgreSQL versions. Planner may choose Seq Scan.",
        "fix": "Use correctly typed parameter: WHERE user_id = 42 (no quotes).",
    },
    "not_in_large_list": {
        "pattern": "WHERE id NOT IN (SELECT id FROM ...)",
        "diagnosis": "NOT IN with correlated subquery often results in anti-join via Nested Loop. Can be slow with large sets.",
        "fix": "Rewrite as: WHERE NOT EXISTS (SELECT 1 FROM ... WHERE id = t.id) or use LEFT JOIN ... WHERE id IS NULL",
    },
    "or_on_indexed_columns": {
        "pattern": "WHERE col1 = $1 OR col2 = $2",
        "diagnosis": "OR on different indexed columns forces Bitmap OR scan or Seq Scan. Neither column's index is used alone.",
        "fix": "Rewrite as UNION: SELECT ... WHERE col1 = $1 UNION SELECT ... WHERE col2 = $2",
    },
}
