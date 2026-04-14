# Strict Answer Rule

If the query names a specific matter, entity, or title, answer from that matter only unless the user explicitly asks to compare.

Operational implications:

1. Apply an exact entity/title lock before broader retrieval.
2. Do not merge facts across record keys for named queries unless the query is explicitly comparative.
3. If multiple unrelated record keys appear in a named-query draft answer, treat it as likely contamination and regenerate or abstain.
4. If no grounded support exists inside the locked matter, abstain rather than borrowing thematically similar facts from another matter.
