# CustomerIQ Interview Guide

This guide describes the implementation in this repository as it exists today.
It avoids claiming production performance or model quality that has not been
measured and retained.

## 1. 60-second project explanation

CustomerIQ is a batch customer analytics project built around the synthetic
theLook eCommerce tables. A Python pipeline loads seven CSV or Parquet tables,
checks required schemas and references, cleans values, and loads the cleaned
tables into DuckDB. Pandas feature engineering then creates RFM scores and
segments, CLV estimates, churn features and predictions, segment-popularity
recommendations, monthly KPIs, cohort retention, and category analytics. The
pipeline stages and validates nine Parquet artifacts that a seven-section
Streamlit dashboard reads. The churn model uses a historical feature window and
a separate 90-day target window. Current limitations include a customer-level
random holdout instead of a time-based evaluation, predictions saved for the
same historical cohort used in model development, and no persisted evaluation
metrics. I would treat the scores and estimated CLV as a portfolio demonstration,
not as validated production decision systems.

## 2. Why DuckDB?

DuckDB provides an in-process SQL database for loading and querying the cleaned
tables without a separate server. The repository includes SQL examples and a
helper that can regenerate the monthly KPI and category artifacts from DuckDB.
The main pipeline currently computes its customer and dashboard analytics with
Pandas; DuckDB is loaded but is not the main pipeline's query engine. This keeps
local setup simple, while the in-memory DataFrames still determine much of the
pipeline's memory use.

## 3. How are RFM scores and segments calculated?

- **Recency** is elapsed days from the observation timestamp to the customer's
  most recent completed order.
- **Frequency** is the number of distinct completed order IDs.
- **Monetary** is the sum of sale prices on completed line items belonging to a
  completed order.
- Percentile ranks map the measures onto scores from 1 to 5. Low recency gets a
  high R score; high frequency and monetary value get high scores. Equal values
  share a score, which makes results independent of input row order but may
  produce fewer than five distinct scores when values are tied.
- A documented, ordered set of nine business rules maps the scores to segment
  names. The first matching rule wins; unmatched combinations are labelled
  `Other`.

The KMeans function is a source-level helper. The main pipeline does not call it
and the dashboard does not display its clusters.

## 4. How is churn defined?

By default, the cutoff is the latest completed-order timestamp minus the
configured horizon (`CHURN_FUTURE_DAYS`, 90 days by default). Features use
completed orders strictly before the cutoff. The label is 1 when a customer
with historical orders has no completed order from the cutoff through the end
of the horizon; otherwise it is 0. An explicit cutoff is accepted only if the
available completed-order data covers the entire configured horizon.

The code treats `complete`, `shipped`, and `processing` as purchase statuses;
it excludes other statuses such as `cancelled` and `returned`.

The current model features are recency, distinct historical order count,
historical revenue, average historical basket value, number of distinct
products, tenure, a binary gender feature, and age. Customers without a
historical completed order are outside this training cohort.

## 5. How is leakage limited, and what does evaluation mean?

The label window is separate from the historical feature window. The customer
rows are split with a stratified random 80/20 split and a fixed random seed.
`StandardScaler` is fitted only on the training rows and then applied to the
holdout rows. Both Logistic Regression and Random Forest use the same split;
the selected model is the one with the higher holdout ROC-AUC.

This is not a temporal holdout. The published prediction artifact is generated
for the full feature cohort, including rows used to fit the selected model.
Therefore those saved probabilities are not independent predictions for a
future campaign. The cutoff, customer population, and model evaluation also
come from one historical snapshot. A stronger follow-up would train on an
earlier cutoff and evaluate on a later, fully matured cohort.

## 6. How are recommendations generated?

For each RFM segment, the recommender counts completed line-item purchases by
product. It considers a bounded pool of popular products, excludes products
already purchased by each customer, and ranks the remaining choices. A sparse
segment can be backfilled from overall product popularity. The explanation
states whether the product is popular with peers or broadly popular. The
affinity score is a normalized count within a customer's selected rows; it is
not a probability or a measured likelihood to buy. A bounded candidate pool
may return fewer than the configured maximum when a customer already owns the
available candidates.

## 7. How do the pipeline and dashboard interact?

`python run_pipeline.py` loads the raw tables, validates and preprocesses them,
loads DuckDB, computes the outputs, and stages all nine Parquet files. The new
set is checked for non-empty contents and expected columns before replacement.
The dashboard reads those files relative to the repository location, validates
their status and schema, and displays the seven sections. It can attempt one
pipeline recovery in a process when outputs are invalid and all required raw
tables are present; visitors are not asked to run commands. The checked-in
artifacts allow the hosted dashboard to start without private local data files.

The main pipeline calculates with Pandas. The SQL files are examples; the
`scripts/generate_outputs.py` helper updates only monthly KPIs and category
analytics. Run the full pipeline when a consistent set of all nine artifacts
is required.

## 8. Design trade-offs and known failure cases

- Raw data and trained model files are not part of the repository. Reproducing
  outputs requires obtaining the documented source tables and honoring their
  applicable terms.
- The pipeline is batch-oriented and holds several tables in Pandas memory.
  Scale and runtime have not been benchmarked in this audit.
- Orders with missing or invalid timestamps, invalid line-item prices, and
  duplicate IDs are handled during cleaning; missing required tables/columns,
  conflicting references, and a failed model stage stop publication.
- CLV is a transparent historical extrapolation. It uses average order basket
  value and annualized order frequency over observed tenure, with a median
  frequency fallback for very short tenure. It is not a survival or
  probabilistic lifetime-value model.
- Churn metrics are logged by the training run but are not saved as a dashboard
  artifact. Thresholds are configured manually and probabilities are not
  documented as calibrated.
- The dashboard shows category analytics, not per-product performance. A
  per-product aggregation helper exists but is not wired to an output or page.
- Clickstream, inventory, and distribution-center tables are loaded and stored
  in DuckDB but are not currently used by the analytics.
- KMeans is implemented as a helper but is not used in the pipeline.
- External Streamlit Cloud configuration and the live app's rendered pages
  must be checked in that service; local tests cannot establish those states.

## 9. Ten challenging interview questions and answers

### 1. Why is this called batch analytics rather than real time?

The pipeline reads a fixed collection of source files and writes Parquet
snapshots. There is no event stream, scheduled retraining service, or online
prediction endpoint in the repository.

### 2. How do you avoid assigning different RFM scores to equal customers?

Scoring uses average percentile ranks for ties. Equal input values get equal
scores regardless of row order. The trade-off is that tied distributions can
occupy fewer than five score levels.

### 3. How would you explain the churn target to a business stakeholder?

For a chosen cutoff, a customer with historical completed purchases is labelled
churned when there is no completed purchase in the following configured
90-day window. The label depends on the observation date and horizon; it is not
a permanent customer attribute.

### 4. What prevents target leakage in the features?

Only completed orders before the cutoff contribute to recency, frequency,
monetary value, basket value, product count, and tenure. Future orders are used
only to construct the target label. The input data must contain a complete
future horizon before the label is accepted.

### 5. Is the current train/test evaluation enough to estimate future performance?

No. It is a stratified random split by customer from one feature/label snapshot.
It avoids fitting the scaler on test data, but it does not test temporal drift.
An earlier training cutoff and later evaluation cutoff would be more
representative of future deployment.

### 6. Which metrics are computed, and are they available in the dashboard?

The trainer computes accuracy, precision, recall, F1, ROC-AUC, a confusion
matrix, and a classification report for both models. It logs them, but the
pipeline does not persist those metrics or display them in the dashboard.

### 7. What does a churn probability of 0.8 mean here?

It is the model's estimated class probability under this training setup, not a
guarantee that the customer will churn. The code applies configurable risk
bands, but there is no calibration analysis or business threshold optimization
in the repository.

### 8. Are the product recommendations personalized?

They are personalized by excluding each customer's purchased products and
choosing items popular in that customer's segment. They do not learn a latent
individual preference model, and the normalized affinity score is not a
purchase probability.

### 9. Why can the CLV estimate be wrong even after the AOV calculation is fixed?

It annualizes the observed order frequency over a short historical tenure and
projects that rate over a configured horizon. Purchase behavior may change,
recent customers need a fallback, and the model does not account for margin,
discounting, retention curves, or uncertainty.

### 10. How does the app respond to missing or incompatible outputs?

It checks presence, Parquet readability, row count, and required columns. A
page whose required artifact is unavailable shows a maintainer-facing message.
If every required raw input is present, the process can attempt one recovery
run. The app still depends on successful Cloud configuration and a compatible
artifact set, which must be verified in the hosted service.
