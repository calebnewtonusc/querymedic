"""
mysql_internals.py - MySQL/InnoDB internals for query optimization reasoning.

Reference:
    - MySQL 8.0 Reference Manual: Chapter 8 (Optimization)
    - "High Performance MySQL" (O'Reilly, 4th edition)
    - InnoDB source: storage/innobase/
"""

from dataclasses import dataclass
from enum import Enum


class MySQLAccessType(Enum):
    """MySQL EXPLAIN access type (from best to worst)."""

    SYSTEM = "system"  # Single row (system table)
    CONST = "const"  # Single row via primary key or unique index
    EQ_REF = "eq_ref"  # One row per combination of previous tables
    REF = "ref"  # Non-unique index lookup
    FULLTEXT = "fulltext"  # Full-text index
    REF_OR_NULL = "ref_or_null"
    INDEX_MERGE = "index_merge"  # Union/intersection of indexes
    UNIQUE_SUBQUERY = "unique_subquery"
    INDEX_SUBQUERY = "index_subquery"
    RANGE = "range"  # Range scan using index
    INDEX = "index"  # Full index scan (better than ALL)
    ALL = "all"  # Full table scan (worst)


@dataclass
class InnoDBFacts:
    """
    Key InnoDB internals facts that affect query optimization.
    QueryMedic uses these to generate InnoDB-specific advice.
    """

    # Clustered primary key
    clustered_pk: str = (
        "InnoDB stores rows in B+tree order of the PRIMARY KEY (clustered index). "
        "Every secondary index automatically includes the primary key value. "
        "Range scans on PK are extremely fast — sequential disk I/O. "
        "If no PK is defined, InnoDB creates a hidden 6-byte rowid — use explicit PK always."
    )

    # Secondary index overhead
    secondary_index_pk_inclusion: str = (
        "Every InnoDB secondary index leaf node contains the primary key value. "
        "This means: (1) secondary index size = columns + PK size, "
        "(2) to read a non-indexed column, InnoDB must do a PK lookup (bookmark lookup), "
        "(3) covering indexes in InnoDB include PK automatically — you don't need to INCLUDE it."
    )

    # Covering index for InnoDB
    covering_index: str = (
        "An InnoDB covering index contains all columns needed for the query. "
        "Since secondary indexes already include PK, you only need to add SELECT columns. "
        "A covering index avoids the bookmark lookup (random I/O on clustered index). "
        "EXPLAIN shows 'Using index' in Extra column when covering index is used."
    )

    # Page size and buffer pool
    page_size: str = (
        "InnoDB default page size: 16KB (can be 4KB-64KB). "
        "Buffer pool caches pages — target: working set fits in buffer pool. "
        "Rule of thumb: set innodb_buffer_pool_size to 70-80% of available RAM on dedicated DB server. "
        "With cold buffer pool: full index scans may be slower than expected — prewarm with SELECT COUNT(*)."
    )

    # MVCC (Multi-Version Concurrency Control)
    mvcc: str = (
        "InnoDB uses MVCC for transaction isolation. "
        "REPEATABLE READ (default): reads use a consistent snapshot of the data at transaction start. "
        "Long-running transactions hold old versions in the undo log — can cause 'history list length' growth. "
        "Implication for optimization: a query slow due to a long history list is not fixed by indexes."
    )

    # Index prefix limitations
    index_prefix: str = (
        "InnoDB has index key length limit of 767 bytes (row_format=COMPACT/REDUNDANT) "
        "or 3072 bytes (row_format=DYNAMIC, MySQL 5.7+). "
        "For utf8mb4 (4 bytes/char): VARCHAR(191) is the max for 767-byte limit. "
        "For TEXT/BLOB: must use prefix index — CREATE INDEX ON t(col(100))."
    )


INNODB_FACTS = InnoDBFacts()


def interpret_mysql_explain(explain_output: str) -> dict:
    """
    Parse MySQL EXPLAIN output (tabular format) and return optimization hints.

    Returns:
        dict with keys: access_types, bottlenecks, recommendations
    """

    rows = []
    for line in explain_output.splitlines():
        if "|" not in line:
            continue
        cols = [c.strip() for c in line.split("|") if c.strip()]
        if cols and cols[0].isdigit():
            # QM-20: MySQL EXPLAIN tabular columns (left to right):
            # 0:id  1:select_type  2:table  3:partitions  4:type
            # 5:possible_keys  6:key  7:key_len  8:ref  9:rows  10:filtered  11:Extra
            # The previous code used cols[5] for "key" — that is "possible_keys".
            # cols[6] is the actual key chosen by the optimizer.
            rows.append(
                {
                    "id": cols[0] if len(cols) > 0 else "",
                    "select_type": cols[1] if len(cols) > 1 else "",
                    "table": cols[2] if len(cols) > 2 else "",
                    "type": cols[4] if len(cols) > 4 else "",
                    "key": cols[6] if len(cols) > 6 else "",
                    "rows": int(cols[9]) if len(cols) > 9 and cols[9].isdigit() else 0,
                    "extra": cols[11] if len(cols) > 11 else "",
                }
            )

    bottlenecks = []
    recommendations = []

    for row in rows:
        access = str(row["type"])
        table = str(row["table"])
        extra = str(row["extra"])
        row_count = row["rows"]

        if access == "ALL":
            bottlenecks.append(
                f"Full table scan (ALL) on {table} — {row_count:,} rows examined"
            )
            recommendations.append(
                f"Add index on {table}. Check WHERE clause columns and join conditions."
            )
        elif access == "index":
            bottlenecks.append(
                f"Full index scan on {table} — slower than range but faster than ALL"
            )

        if "Using filesort" in extra:
            bottlenecks.append(f"Filesort on {table} — no index for ORDER BY")
            recommendations.append(
                f"Add index on ORDER BY columns for {table}, or ensure existing index column order matches ORDER BY."
            )

        if "Using temporary" in extra:
            bottlenecks.append(
                f"Temporary table created for {table} — GROUP BY/DISTINCT without index"
            )
            recommendations.append(
                f"Add index with GROUP BY columns first for {table}."
            )

    return {
        "access_types": {row["table"]: row["type"] for row in rows},
        "bottlenecks": bottlenecks,
        "recommendations": recommendations,
    }


def estimate_index_size_mb(
    row_count: int,
    key_bytes: int,
    pk_bytes: int = 8,
    fill_factor: float = 0.85,
) -> float:
    """
    Estimate InnoDB secondary index size in MB.

    InnoDB secondary index leaf = key columns + PK value.
    Page fill factor ~85% by default.
    """
    bytes_per_row = key_bytes + pk_bytes + 12  # 12 bytes overhead per row
    rows_per_page = int((16 * 1024 * fill_factor) / bytes_per_row)
    pages = max(1, row_count / rows_per_page)
    size_mb = (pages * 16 * 1024) / (1024 * 1024)
    return round(size_mb, 1)
