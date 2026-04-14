# SEBI Control Pack

Generated on 2026-04-10 for the 235-row SEBI Orders corpus.

## Included artifacts

- `document_index.csv`: compact per-document index with record key, title, bucket, date, entities, and summary.
- `confusion_list.csv`: curated easy-to-confuse record pairs for retrieval hardening.
- `eval_queries.jsonl`: broad eval set covering direct LLM, exact lookup, hierarchical RAG, memory-scoped RAG, abstain, and explicit comparison.
- `entity_aliases.csv`: canonical entity names with short-name and suffix variants.
- `wrong_answer_examples.jsonl`: current-tool bad outputs useful for regression testing.
- `strict_answer_rule.md`: named-matter grounding rule to adopt explicitly.

## Corpus notes

- Manifest rows: 235
- Downloaded rows: 215
- Ingested rows with current DB versions: 215
- Missing/not-downloaded rows: 20

## Counts

- Confusion examples: 25
- Eval queries: 71
- Alias rows: 297
- Wrong-answer examples: 11
