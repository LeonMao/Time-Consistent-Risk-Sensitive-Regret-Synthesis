# Robotica claim-to-evidence map

`claim_evidence_manifest.json` is the machine-readable source of truth and
`verify_robotica_claims.py` executes every row below.

| ID | Paper result | Primary evidence | Reproduction driver |
|---|---|---|---|
| C01 | Static terminal CVaR can replan inconsistently; nested risk is time-consistent | `stage3_3_p2_time_consistency_summary.csv` | final Stage 3.5 chain |
| C02 | Explicit and factored solvers agree on 21/21 complete policy maps | `stage3_3_p0_full_policy_exactness.csv` | P1 workload driver |
| C03 | Static replan mismatch in 97/720 randomized cases; maximum improvement 2.803 | `stage3_5_static_nested_mismatch_summary.csv` | final Stage 3.5 chain |
| C04 | Risk aversion lowers tail regret with an expected-regret tradeoff | P0 exact-alpha m8/m10 tables | P0 exact-alpha driver |
| C05 | High-alpha nested risk reaches the minimax endpoint | P1 thresholds and plateau tables | P1 endpoint driver |
| C06 | Absolute-cost risk and hindsight-regret risk can select different behavior | P2.3 disagreement tables | P2.3 disagreement driver |
| C07 | Factoring improves controlled runtime and can reduce workload | controlled timing and P1 workload tables | P1 workload + final chain |
| C08 | Horizon/rank conclusions and Intel stabilization | horizon, rank, and Intel sensitivity tables | final Stage 3.5 chain |
| C09 | Prior-weight and missing-support robustness conclusions | prior robustness table | final Stage 3.5 chain |
| C10 | Native-metric and common-objective baseline comparisons | baseline summary tables | final Stage 3.5 chain |
| C11 | Ten final figures and exact manuscript identity | `01_FIGURES/`, `PAPER_VERSION.json` | Figures 4-10 only |

The paper locators in the JSON manifest are semantic locators because table and
figure numbering can change during typesetting. The locked manuscript hashes
identify the exact numbering used for this release.
