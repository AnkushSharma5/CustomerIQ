# CustomerIQ Run Report

## Verification

- Date: 2026-10-26; branch `main`; HEAD `91d8b7eb17a9eb4d956916036312ecdd406506c1`.
- Interpreter: `C:\CustomerIQ\venv\Scripts\python.exe` (Python 3.14.6).
- Test command: `& '.\venv\Scripts\python.exe' -m pytest -q` — **76 passed in 15.05s** in the final run. The segmentation tests passed: **15 passed in 3.15s**.
- Pipeline command: `& '.\venv\Scripts\python.exe' run_pipeline.py` — completed successfully, exit code 0, in 54.2 seconds. It loaded all seven raw tables. Raw files were not changed.

The seven CSVs present in `data/raw/` were `distribution_centers.csv`, `events.csv`, `inventory_items.csv`, `order_items.csv`, `orders.csv`, `products.csv`, and `users.csv`.

## Output artifacts

`src.artifacts.validate_output_artifacts()` reported all nine configured Parquets as `ready`. DuckDB `read_parquet` counts confirmed they are readable:

| File | Rows | Validation |
| --- | ---: | --- |
| `category_analytics.parquet` | 26 | ready |
| `churn_feature_importance.parquet` | 8 | ready |
| `churn_predictions.parquet` | 54,317 | ready |
| `cohort_retention.parquet` | 61 | ready |
| `customer_clv.parquet` | 64,649 | ready |
| `customer_rfm.parquet` | 64,649 | ready |
| `customer_segments.parquet` | 64,649 | ready |
| `monthly_kpis.parquet` | 61 | ready |
| `recommendations.parquet` | 323,245 | ready |

The check used `validate_output_artifacts()` and `SELECT count(*) FROM read_parquet(?)` for each filename in `src.config.OUTPUT_FILES.values()`. `data/output/run_summary.json` is absent; the pipeline configuration lists nine Parquet outputs and does not generate a JSON summary.

## RFM segmentation

Historical values below are the reference values supplied for comparison; the current rule was selected from the stated business meaning, not to target those counts. Actual counts came from DuckDB queries over `customer_rfm.parquet` and `customer_segments.parquet`: `COUNT(DISTINCT customer_id)`, `GROUP BY segment`, and filters on `frequency`, `r_score`, and `f_score`. Revenue share uses `SUM(total_revenue)` in `customer_segments.parquet`.

| Metric | Historical reference | Current output | Result |
| --- | ---: | ---: | --- |
| Customers analyzed | 64,649 | 64,649 | Match |
| Non-empty segments | 9 | 8 | Mismatch |
| “Other” present | No | No | Match |
| Lost Customers | 15,744 | 13,480 | Mismatch |
| Promising | 15,695 | 10,466 | Mismatch |
| Hibernating | 15,027 | 11,552 | Mismatch |
| New Customers | 9,215 | 9,215 | Count match; not targeted |
| Potential Loyalists | 6,229 | 0 | Mismatch |
| At Risk | 1,481 | 4,610 | Mismatch |
| Champions | 639 | 3,614 | Mismatch |
| Champions / customers | 0.99% | 5.5902% | Mismatch |
| Champions / revenue | 3.02% | 14.3831% | Mismatch |
| Loyal Customers | 548 | 9,031 | Mismatch |
| Cannot Lose Them | 71 | 2,681 | Mismatch |
| Single-order customers | 69.16% | 69.1627% | Match when rounded |
| Every one-order customer has `f_score=1` | True | False: 0 of 44,713 | Mismatch |

Current counts: At Risk 4,610; Cannot Lose Them 2,681; Champions 3,614; Hibernating 11,552; Lost Customers 13,480; Loyal Customers 9,031; New Customers 9,215; Promising 10,466. No customer is currently assigned Potential Loyalists. Champions represent `$1,140,358.55` of `$7,928,479.84` total segment revenue.

### Cause and correction

`src.feature_engineering.build_rfm_features()` defines frequency as the number of distinct completed orders. `src.segmentation.compute_rfm_scores()` converts frequency to a tie-aware average-rank score. In the current data, all 44,713 one-order customers have `f_score=2`, so the former New Customers condition (`f_score=1`) could not match them. The current output before the correction placed 9,215 recent single-order customers in Potential Loyalists.

The corrected business rule is **New Customers = `r_score` 4–5 and exactly one completed order**. The R score threshold preserves the existing “recent” boundary; raw frequency expresses first/only purchase without changing scores. The rule has priority over score-based segments. Potential Loyalists retains its existing R/F-score rule. The new count of 9,215 is the observed result of this rule on current data, not a target used to choose it.

The actual data has zero Potential Loyalists after the correction: the recent one-order customers are now New Customers, and the recent multi-order customers have `f_score=4` under the unchanged rank scoring. The assignment test confirms a recent repeat customer that meets the Potential Loyalists score rule remains in that segment. If the business needs Potential Loyalists populated for this dataset, its frequency definition needs a separate decision; it was not broadened here.

In the current RFM data, 4,545 recent customers have two orders and 1,684 have three orders; both groups have `f_score=4` and are currently Loyal Customers. Reclassifying recent two- or three-order customers as Potential Loyalists is a plausible raw-frequency rule, but it would change that segment's definition. It was not included in this correction.

RFM score computation was not changed. Tests cover one-order customers whose tied score is 2, the recency boundary (`r_score` 3 vs 4 and 5), distinction from repeat Potential Loyalists, and all 125 R/F/M score combinations receiving one recognized label (`Other` remains the existing explicit fallback). The current real output contains 64,649 unique customer IDs, no null segment labels, and no `Other` customers.

## Churn

The successful pipeline output reported:

| Metric | Actual | Evidence |
| --- | ---: | --- |
| Customers scored | 54,317 | Churn feature/model log; prediction Parquet has 54,317 rows |
| Churn-label rate | 91.3% (49,565 churned / 4,752 active) | `train_churn_models` log |
| Best model | Logistic Regression | Pipeline output; `src/churn.py` selects maximum ROC-AUC |
| Logistic Regression ROC-AUC | 0.6639 | Model evaluation log |
| Random Forest ROC-AUC | 0.5959 | Model evaluation log |

These values came from `run_pipeline.py` output, not `run_summary.json`. The output Parquet contains scored predictions, not evaluation labels or both models’ evaluation probabilities.

## Dashboard and Git

- Dashboard entry point: `streamlit run dashboard/app.py` (README and repository entry point). No visual dashboard check was performed in this run.
- The segmentation rules help text now describes New Customers as one completed order with R score 4–5 and notes that it is assigned before Potential Loyalists. Recommendations subtitle source remains “Segment-based popularity”; recommendation logic was not changed.
- `.gitignore` ignores `data/output/*` but explicitly permits the nine Parquets. `git ls-files data/output` lists all nine. Pipeline regeneration changed `customer_rfm.parquet`, `customer_segments.parquet`, and `recommendations.parquet`; the latter reflects the segment-based recommendation inputs.
- Existing staged changes `A  check.py` and `M  sql/rfm.sql` were preserved. `check.py` is a one-off diagnostic script and remains staged for the user's decision. No commit, push, or deployment occurred.

`run_summary.json` is absent, and the modified Recommendations page was not visually checked. `Potential Loyalists` has zero assigned customers under the existing score-based rule after the one-order correction; if a populated cohort is required, agree on its raw-frequency definition before changing that rule.

Current `git status --short`:

```text
A  check.py
 M dashboard/app.py
 M data/output/customer_rfm.parquet
 M data/output/customer_segments.parquet
 M data/output/recommendations.parquet
M  sql/rfm.sql
 M src/segmentation.py
 M tests/test_segmentation.py
?? docs/RUN_REPORT.md
```
