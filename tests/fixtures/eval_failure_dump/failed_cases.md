# SEBI Eval Failure Dump

- Generated at: `2026-04-13T12:34:55`
- Cases: `35/82` passed
- Failed cases: `47`
- Route accuracy: `0.4507`
- Candidate-list correctness: `0.1176`
- Wrong-example regressions: `5/11` passed

## Buckets

- contamination: 1
- stale expectation: 9
- wrong answer despite correct route: 2
- wrong candidate ranking: 6
- wrong route: 29

## Wrong Route

### 1. What is a settlement order?

- Case kind: `eval_query`
- Expected route: `direct_llm`
- Actual route: `abstain`
- Expected record: `-`
- Actual cited record keys: `-`
- Clarify fired: `False`
- Mixed-record guardrail fired: `True`
- Reasons: `expected route direct_llm got abstain`
- Expected behavior note: General explanatory query; no record lock expected.

### 2. What is an adjudication order under the SEBI Act?

- Case kind: `eval_query`
- Expected route: `direct_llm`
- Actual route: `abstain`
- Expected record: `-`
- Actual cited record keys: `-`
- Clarify fired: `False`
- Mixed-record guardrail fired: `True`
- Reasons: `expected route direct_llm got abstain`
- Expected behavior note: General explanatory query; no record lock expected.

### 3. Explain what a corrigendum does in a SEBI order.

- Case kind: `eval_query`
- Expected route: `direct_llm`
- Actual route: `abstain`
- Expected record: `-`
- Actual cited record keys: `-`
- Clarify fired: `False`
- Mixed-record guardrail fired: `True`
- Reasons: `expected route direct_llm got abstain`
- Expected behavior note: General explanatory query; no record lock expected.

### 4. What is an RTI appellate order?

- Case kind: `eval_query`
- Expected route: `direct_llm`
- Actual route: `abstain`
- Expected record: `-`
- Actual cited record keys: `-`
- Clarify fired: `False`
- Mixed-record guardrail fired: `True`
- Reasons: `expected route direct_llm got abstain`
- Expected behavior note: General explanatory query; no record lock expected.

### 5. What is an exemption order under the takeover regulations?

- Case kind: `eval_query`
- Expected route: `direct_llm`
- Actual route: `abstain`
- Expected record: `-`
- Actual cited record keys: `-`
- Clarify fired: `False`
- Mixed-record guardrail fired: `True`
- Reasons: `expected route direct_llm got abstain`
- Expected behavior note: General explanatory query; no record lock expected.

### 6. What is a SAT order?

- Case kind: `eval_query`
- Expected route: `direct_llm`
- Actual route: `abstain`
- Expected record: `-`
- Actual cited record keys: `-`
- Clarify fired: `False`
- Mixed-record guardrail fired: `True`
- Reasons: `expected route direct_llm got abstain`
- Expected behavior note: General explanatory query; no record lock expected.

### 7. What is an ex-parte interim order?

- Case kind: `eval_query`
- Expected route: `direct_llm`
- Actual route: `abstain`
- Expected record: `-`
- Actual cited record keys: `-`
- Clarify fired: `False`
- Mixed-record guardrail fired: `True`
- Reasons: `expected route direct_llm got abstain`
- Expected behavior note: General explanatory query; no record lock expected.

### 8. Explain orders issued under Regulation 30A.

- Case kind: `eval_query`
- Expected route: `direct_llm`
- Actual route: `general_knowledge`
- Expected record: `-`
- Actual cited record keys: `-`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route direct_llm got general_knowledge`
- Expected behavior note: General explanatory query; no record lock expected.

### 10. What did the appellate authority decide for Rajat Kumar?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:100722`
- Actual cited record keys: `external:100722, external:100722`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:100722 (Appeal No. 6795 of 2026 filed by Rajat Kumar)

### 12. What was SEBI's finding against Trdez Investment Private Limited?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:100846`
- Actual cited record keys: `external:100846, external:100846, external:100846`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:100846 (Adjudication Order in the matter of Trdez Investment Private Limited)

### 14. What did SEBI order for Elitecon International Limited?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:100685`
- Actual cited record keys: `external:100685, external:100685, external:100685, external:100685, external:100685, external:100685`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:100685 (Ex-Parte Interim Order in the matter of Elitecon International Limited)

### 16. What scheme approval or direction was issued for The Cochin Stock Exchange Limited (Demutualisation) Scheme, 2005?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:10784`
- Actual cited record keys: `external:10784`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:10784 (Order in the matter of The Cochin Stock Exchange Limited (Demutualisation) Scheme, 2005)

### 18. What did the court decide in the matter concerning Samir Sudhir Porecha v. Geeta Jaswal and Ors. No. 3803 of 2025)?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:100736`
- Actual cited record keys: `external:100736`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:100736 (Order in the matter of Samir Sudhir Porecha v. Geeta Jaswal and Ors. (Writ Petition (L) No. 3803 of 2025).)

### 20. What action did SEBI take against Basan Financial Services Limited?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:100770`
- Actual cited record keys: `external:100770, external:100770`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:100770 (Final Order in respect of Basan Financial Services Limited)

### 24. What did the Special Court hold concerning Judgment dated 7.2.2026 passed by the Hon’ble SEBI Special Court, Mumbai in SEBI Special Case 244 of 2020 –SEBI?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:99769`
- Actual cited record keys: `external:99769, external:99769, external:99769`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:99769 (Judgment dated 7.2.2026 passed by the Hon’ble SEBI Special Court, Mumbai in SEBI Special Case 244 of 2020 –SEBI vs Wada Arun Asbestos Products Pvt. Ltd. & Ors.)

### 26. What non-compliance did SEBI identify for Wealthmax Solutions Investment Advisor (Proprietor: Piyush Jain)?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:100851`
- Actual cited record keys: `external:100851, external:100851, external:100851`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:100851 (Order in the matter of Wealthmax Solutions Investment Advisor (Proprietor: Piyush Jain))

### 28. What did SEBI finally direct for Mangalam Global Enterprise Limited?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:100669`
- Actual cited record keys: `external:100669, external:100669, external:100669`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:100669 (Settlement Order in the matter of Mangalam Global Enterprise Limited)

### 30. What did SEBI finally direct for JP Morgan Chase Bank N.A?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:100486`
- Actual cited record keys: `external:100486`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:100486 (Settlement Order in the matter of JP Morgan Chase Bank N.A.)

### 32. What did SEBI order for Vishvaraj Environment Limited?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `derived:58a47f2f3ab8169bffd6d73800717916db1fee7c0b8ee6259945c332c2e066a1`
- Actual cited record keys: `derived:58a47f2f3ab8169bffd6d73800717916db1fee7c0b8ee6259945c332c2e066a1, derived:58a47f2f3ab8169bffd6d73800717916db1fee7c0b8ee6259945c332c2e066a1`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: derived:58a47f2f3ab8169bffd6d73800717916db1fee7c0b8ee6259945c332c2e066a1 (Order in the matter of Vishvaraj Environment Limited)

### 38. What action did SEBI take against Chaturvedi Group?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:100648`
- Actual cited record keys: `external:100648, external:100648, external:100648`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:100648 (Order in respect of Front Running of Trades of Big Client by certain entities of Chaturvedi Group)

### 40. What did the Special Court hold concerning Judgment dated 08.10.2024 passed by the Hon’ble SEBI Special Court, Delhi in SEBI?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:88411`
- Actual cited record keys: `external:88411`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:88411 (Judgment dated 08.10.2024 passed by the Hon’ble SEBI Special Court, Delhi in SEBI vs Kisley Plantation Limited & Ors. (CNR No. DLSW01-006824-2023; CC No. 594/2023))

### 42. What did the Special Court hold concerning Judgment dated 08.04.2024 passed by the Hon’ble SEBI Special Court, Delhi in SEBI?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `abstain`
- Expected record: `external:87947`
- Actual cited record keys: `-`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got abstain`
- Expected behavior note: Named substantive query should stay within the named matter.

### 44. What non-compliance did SEBI identify for certain Research Analysts?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:93065`
- Actual cited record keys: `external:93065, external:93065, external:93065`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:93065 (Order in the matter of certain Research Analysts)

### 46. What happened in the SAT matter concerning Prime Broking Company Limited?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:30161`
- Actual cited record keys: `external:30161, external:30161, external:30161, external:30161, external:30161, external:30161, external:30161`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:30161 (In the matter of Prime Broking Company Limited)

### 48. What was SEBI's finding against ITI Securities Broking Limited?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `exact_lookup`
- Expected record: `external:100605`
- Actual cited record keys: `external:100605, external:100605, external:100605`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got exact_lookup`
- Expected behavior note: Named substantive query should stay within the named matter.
- Actual cited records: external:100605 (Adjudication Order in respect of ITI Securities Broking Limited in the matter of TradeTron and other Algo Platforms)

### 52. What did SEBI finally direct?

- Case kind: `eval_query`
- Expected route: `memory_scoped_rag`
- Actual route: `abstain`
- Expected record: `external:100669`
- Actual cited record keys: `-`
- Clarify fired: `False`
- Mixed-record guardrail fired: `True`
- Reasons: `expected route memory_scoped_rag got abstain`
- Expected behavior note: Follow-up should stay anchored to prior record.

### 58. What sentence was imposed?

- Case kind: `eval_query`
- Expected route: `memory_scoped_rag`
- Actual route: `abstain`
- Expected record: `external:88411`
- Actual cited record keys: `-`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route memory_scoped_rag got abstain`
- Expected behavior note: Follow-up should stay anchored to prior record.

### 66. Tell me more about Cochin Stock Exchange Limited.

- Case kind: `eval_query`
- Expected route: `abstain`
- Actual route: `exact_lookup`
- Expected record: `-`
- Actual cited record keys: `external:10784`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route abstain got exact_lookup`
- Expected behavior note: The system should abstain rather than fuse loosely related matters.
- Actual cited records: external:10784 (Order in the matter of The Cochin Stock Exchange Limited (Demutualisation) Scheme, 2005)

### 71. Compare the Chaturvedi Group and Sarvottam Securities front-running orders.

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `abstain`
- Expected record: `external:100648;external:100535`
- Actual cited record keys: `-`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got abstain`
- Expected behavior note: Explicit comparison request permits multi-record grounding.


## Stale Expectation

### 21. Prime Broking Company (India) Limited

- Case kind: `eval_query`
- Expected route: `exact_lookup`
- Actual route: `clarify`
- Expected record: `external:30222`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route exact_lookup got clarify; expected grounded record key but answer cited none; strict single-matter lock did not trigger`
- Expected behavior note: Exact title lookup should lock to one record.

### 22. What happened in the SAT matter concerning Prime Broking Company (India) Limited?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `clarify`
- Expected record: `external:30222`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got clarify; expected grounded record key but answer cited none; strict single-matter lock did not trigger`
- Expected behavior note: Named substantive query should stay within the named matter.

### 34. What did SEBI order for Hardcastle and Waud Manufacturing Ltd?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `clarify`
- Expected record: `derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got clarify; expected grounded record key but answer cited none; strict single-matter lock did not trigger`
- Expected behavior note: Named substantive query should stay within the named matter.

### 36. What did SEBI order for Pacheli Industrial Finance Limited?

- Case kind: `eval_query`
- Expected route: `hierarchical_rag`
- Actual route: `clarify`
- Expected record: `derived:69093d449801258e05a31962b30bba4735b0524616cc6679f7eba25e3480fd46`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route hierarchical_rag got clarify; expected grounded record key but answer cited none; strict single-matter lock did not trigger`
- Expected behavior note: Named substantive query should stay within the named matter.

### 60. Tell me more about Appeal No. 9999 of 2026 filed by Nonexistent Person.

- Case kind: `eval_query`
- Expected route: `abstain`
- Actual route: `clarify`
- Expected record: `-`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route abstain got clarify`
- Expected behavior note: The system should abstain rather than fuse loosely related matters.

### 62. Tell me more about the RTI appeal filed by Prime Broking Company India Limited.

- Case kind: `eval_query`
- Expected route: `abstain`
- Actual route: `clarify`
- Expected record: `-`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route abstain got clarify`
- Expected behavior note: The system should abstain rather than fuse loosely related matters.

### 63. What was the settlement amount in the Vishvaraj Environment Limited matter?

- Case kind: `eval_query`
- Expected route: `abstain`
- Actual route: `clarify`
- Expected record: `-`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route abstain got clarify`
- Expected behavior note: The system should abstain rather than fuse loosely related matters.

### 64. What IPO proceeds were raised in the Pacheli Industrial Finance Limited matter?

- Case kind: `eval_query`
- Expected route: `abstain`
- Actual route: `clarify`
- Expected record: `-`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route abstain got clarify`
- Expected behavior note: The system should abstain rather than fuse loosely related matters.

### 65. Tell me more about Adani Green Energy Limited.

- Case kind: `eval_query`
- Expected route: `abstain`
- Actual route: `clarify`
- Expected record: `-`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected route abstain got clarify`
- Expected behavior note: The system should abstain rather than fuse loosely related matters.


## Contamination

### 50. What was the settlement amount?

- Case kind: `eval_query`
- Expected route: `memory_scoped_rag`
- Actual route: `abstain`
- Expected record: `external:100486`
- Actual cited record keys: `-`
- Clarify fired: `False`
- Mixed-record guardrail fired: `True`
- Reasons: `expected route memory_scoped_rag got abstain; single-matter rule not respected`
- Expected behavior note: Follow-up should stay anchored to prior record.


## Wrong Answer Despite Correct Route

### 54. What exemption was granted?

- Case kind: `eval_query`
- Expected route: `memory_scoped_rag`
- Actual route: `memory_scoped_rag`
- Expected record: `derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e`
- Actual cited record keys: `derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e, derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e, derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `strict single-matter lock did not trigger`
- Expected behavior note: Follow-up should stay anchored to prior record.
- Actual cited records: derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e (Exemption order in the matter of Hardcastle and Waud Manufacturing Ltd.)

### 56. What did the appellate authority decide?

- Case kind: `eval_query`
- Expected route: `memory_scoped_rag`
- Actual route: `memory_scoped_rag`
- Expected record: `external:100722`
- Actual cited record keys: `external:100722, external:100722`
- Clarify fired: `False`
- Mixed-record guardrail fired: `False`
- Reasons: `strict single-matter lock did not trigger`
- Expected behavior note: Follow-up should stay anchored to prior record.
- Actual cited records: external:100722 (Appeal No. 6795 of 2026 filed by Rajat Kumar)


## Wrong Candidate Ranking

### 73. Tell me more about Hardcastle and Waud Manufacturing Ltd.

- Case kind: `wrong_answer_regression`
- Expected route: `-`
- Actual route: `clarify`
- Expected record: `derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected grounded record key but answer cited none; strict single-matter lock did not trigger`

### 74. Tell me more about Prime Broking Company India Limited

- Case kind: `wrong_answer_regression`
- Expected route: `-`
- Actual route: `clarify`
- Expected record: `external:30189`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected grounded record key but answer cited none; strict single-matter lock did not trigger`

### 76. Tell me more about Adani Green Energy Limited by Pranav Adani

- Case kind: `wrong_answer_regression`
- Expected route: `-`
- Actual route: `clarify`
- Expected record: `derived:551259f200f62065e076213d712072bed57ea9c610044f61b542791220f62c09`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected grounded record key but answer cited none; strict single-matter lock did not trigger`

### 78. Tell me more about Kisley Plantation Limited

- Case kind: `wrong_answer_regression`
- Expected route: `-`
- Actual route: `clarify`
- Expected record: `external:88411`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected grounded record key but answer cited none; strict single-matter lock did not trigger`

### 79. Tell me more about Neelgiri Forest Ltd

- Case kind: `wrong_answer_regression`
- Expected route: `-`
- Actual route: `clarify`
- Expected record: `external:87947`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected grounded record key but answer cited none; strict single-matter lock did not trigger`

### 82. What IPO proceeds were raised in the Pacheli Industrial Finance Limited matter?

- Case kind: `wrong_answer_regression`
- Expected route: `-`
- Actual route: `clarify`
- Expected record: `derived:69093d449801258e05a31962b30bba4735b0524616cc6679f7eba25e3480fd46`
- Actual cited record keys: `-`
- Clarify fired: `True`
- Mixed-record guardrail fired: `False`
- Reasons: `expected grounded record key but answer cited none; strict single-matter lock did not trigger`

