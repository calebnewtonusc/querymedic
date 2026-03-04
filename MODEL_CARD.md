# QueryMedic Model Card

## Model Details

| Field | Value |
|---|---|
| **Model name** | QueryMedic-7B |
| **Base model** | Qwen/Qwen2.5-7B-Coder-Instruct |
| **Model type** | Causal LM, fine-tuned with LoRA |
| **Training stages** | SFT → GRPO RL → DPO |
| **Parameters** | 7.6B (base) + ~50M LoRA |
| **Context length** | 32,768 tokens |
| **Supported engines** | PostgreSQL 14+, MySQL 8.0+, SQLite 3.x |
| **Hardware trained on** | 18x NVIDIA A6000 (48GB each) |
| **Training framework** | HuggingFace Transformers + TRL + DeepSpeed ZeRO-3 |
| **Developed by** | Caleb Newton (USC) |
| **License** | Apache 2.0 |
| **Repository** | github.com/calebnewtonusc/querymedic |

---

## Intended Use

### Primary Use Cases

1. **Query optimization**: Feed slow query + EXPLAIN ANALYZE → receive indexed prescription with timing proof
2. **Index design**: Ask for optimal index type for a specific query pattern and workload
3. **Schema review**: Get write amplification and storage impact of proposed indexes
4. **Plan interpretation**: Explain what an EXPLAIN ANALYZE plan is telling you

### Out-of-Scope Uses

- Replacing human DBA review for critical production changes
- General SQL writing assistance (use Qwen2.5-Coder base)
- NoSQL / document database optimization
- Distributed query optimization (Citus, Redshift, BigQuery)

---

## Training Data

**Volume:** ~300,000 training pairs
**Sources:** DBA Stack Exchange (high-vote), database documentation, engineering blogs, EXPLAIN ANALYZE pairs
**Engines:** PostgreSQL (60%), MySQL (25%), SQLite (15%)

**Data filters:** Index type must be specified, write amplification must be addressed, EXPLAIN plan improvement must be estimable

---

## Evaluation

### QueryBench (300 scenarios)

| Category | N | QueryMedic v1 | GPT-4o | Claude Opus |
|---|---|---|---|---|
| Index selection | 80 | TBD | — | — |
| Query rewrite | 80 | TBD | — | — |
| Schema design | 60 | TBD | — | — |
| Statistics tuning | 40 | TBD | — | — |
| Partitioning | 40 | TBD | — | — |
| **Overall** | **300** | **TBD** | **—** | **—** |

Results to be published post-training.

---

## Limitations

1. **Engine version specificity**: Strongest on PostgreSQL 14-16. Older versions may have different planner behavior.
2. **Workload assumptions**: Index recommendations assume representative workload — if write ratio changes significantly, recommendations may need revision.
3. **Schema knowledge**: QueryMedic is more accurate when given complete DDL including all indexes, foreign keys, and constraints.
4. **Distributed systems**: Not trained on Citus, TimescaleDB, or distributed SQL. Single-node only.
5. **Production safety**: All DDL should be reviewed before execution. Use `CONCURRENTLY` for index creation on live systems.

---

## Ethical Considerations

- Running `CREATE INDEX` on a production table without `CONCURRENTLY` causes a table lock. QueryMedic always recommends `CONCURRENTLY` for live systems.
- Dropping indexes based on QueryMedic recommendations requires workload validation.
- Never run QueryMedic-generated DDL directly in production without review.

---

## Citation

```bibtex
@inproceedings{newton2026querymedic,
  title     = {QueryMedic: Workload-Aware Query Optimization via EXPLAIN-to-Prescription Training},
  author    = {Newton, Caleb},
  booktitle = {Proceedings of the ACM SIGMOD International Conference on Management of Data},
  year      = {2026},
  note      = {In submission}
}
```
