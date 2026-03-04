# Contributing to QueryMedic

The highest-value contributions are real EXPLAIN ANALYZE outputs paired with the optimization that fixed them. Here's how to contribute.

---

## Highest-Value Contributions

### 1. EXPLAIN ANALYZE + Optimization Pairs
```json
{
  "engine": "postgresql",
  "version": "15.4",
  "query": "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE u.created_at > '2024-01-01' GROUP BY u.id",
  "explain_before": "Hash Left Join (cost=89.50..1289.50 rows=...",
  "ddl": "CREATE INDEX CONCURRENTLY idx_orders_user_id ON orders(user_id);",
  "explain_after": "Merge Left Join (cost=...",
  "before_ms": 1840,
  "after_ms": 23,
  "write_amplification_note": "orders table: +1 B-tree update per INSERT. At 10k orders/day: ~10ms overhead/day.",
  "storage_mb": 42
}
```
Submit to: `data/community/explain_pairs/`

### 2. Schema Design Patterns
Write-heavy vs. read-heavy schema examples with access pattern descriptions.

### 3. QueryBench Test Cases
```bash
python evaluation/querybench.py --add-test
```

---

## Code Contributions

```bash
git clone https://github.com/calebnewtonusc/querymedic.git
cd querymedic
pip install -r requirements.txt
cp .env.example .env
```

Standards:
- All code passes `python -m pytest tests/`
- Use `loguru` for logging
- Type hints on all public functions
- Docstrings required (Google style)
- Generated SQL must be tested against a live database before PR

## Reporting Issues

Tag issues with the database engine (`[postgresql]`, `[mysql]`, `[sqlite]`) and include the EXPLAIN output that triggered the wrong recommendation.
