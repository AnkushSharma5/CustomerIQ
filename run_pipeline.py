"""
CustomerIQ — End-to-End Pipeline.

Usage:
    python run_pipeline.py

Executes all 14 pipeline steps in order:
    [1/14]  Validating dataset
    [2/14]  Loading raw data
    [3/14]  Cleaning data
    [4/14]  Loading DuckDB
    [5/14]  Calculating RFM
    [6/14]  Creating customer segments
    [7/14]  Estimating CLV
    [8/14]  Building churn features
    [9/14]  Training churn models
    [10/14] Generating recommendations
    [11/14] Running cohort analysis
    [12/14] Calculating monthly KPIs
    [13/14] Running product analytics
    [14/14] Exporting outputs

All outputs are saved to data/output/.
The DuckDB database is at database/customeriq.duckdb.
Trained models are saved to models/.
"""
import sys
import time
import traceback
from pathlib import Path

# Ensure the project root is on the Python path regardless of where the
# script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR, OUTPUT_FILES
from src.utils import print_header, print_step, print_success, print_error, setup_logging, save_parquet
from src.data_loader import load_all_tables
from src.data_validator import validate_tables
from src.preprocessing import preprocess_all
from src.duckdb_manager import load_tables_to_duckdb
from src.feature_engineering import build_rfm_features, build_churn_features, build_customer_aggregates
from src.segmentation import compute_rfm_scores, assign_segments, estimate_clv
from src.churn import train_churn_models, generate_churn_predictions
from src.recommendation import build_recommendations
from src.analytics import (
    compute_monthly_kpis,
    compute_cohort_retention,
    compute_category_analytics,
    compute_product_analytics,
)

TOTAL_STEPS = 14
logger = setup_logging()


def run_pipeline() -> bool:
    """
    Execute the full CustomerIQ analytics pipeline.

    Returns True on success, False on failure.
    """
    start_time = time.perf_counter()
    print_header("CUSTOMERIQ PIPELINE")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # ------------------------------------------------------------------ #
        # Step 1: Validate dataset
        # ------------------------------------------------------------------ #
        print_step(1, TOTAL_STEPS, "Validating dataset")
        raw_tables = load_all_tables()  # exits cleanly if missing
        validation_report = validate_tables(raw_tables)
        if not validation_report["passed"]:
            print_error(
                "Dataset validation found critical errors. "
                "Review the issues above before continuing."
            )
            # Continue anyway with warnings; fatal errors would have raised SystemExit
            logger.warning("Continuing pipeline despite validation warnings.")

        # ------------------------------------------------------------------ #
        # Step 2: Load raw data (already loaded in step 1, log summary)
        # ------------------------------------------------------------------ #
        print_step(2, TOTAL_STEPS, "Loading raw data")
        for name, df in raw_tables.items():
            logger.info("  %-25s %d rows", name, len(df))

        # ------------------------------------------------------------------ #
        # Step 3: Clean data
        # ------------------------------------------------------------------ #
        print_step(3, TOTAL_STEPS, "Cleaning data")
        tables = preprocess_all(raw_tables)
        logger.info("Data cleaning complete.")

        # ------------------------------------------------------------------ #
        # Step 4: Load DuckDB
        # ------------------------------------------------------------------ #
        print_step(4, TOTAL_STEPS, "Loading DuckDB")
        load_tables_to_duckdb(tables)
        print_success("DuckDB tables loaded.")

        # ------------------------------------------------------------------ #
        # Step 5: Calculate RFM
        # ------------------------------------------------------------------ #
        print_step(5, TOTAL_STEPS, "Calculating RFM")
        rfm_df = build_rfm_features(tables["orders"], tables["order_items"])
        rfm_scored = compute_rfm_scores(rfm_df)
        logger.info("RFM computed for %d customers.", len(rfm_scored))

        # ------------------------------------------------------------------ #
        # Step 6: Create customer segments
        # ------------------------------------------------------------------ #
        print_step(6, TOTAL_STEPS, "Creating customer segments")
        rfm_segmented = assign_segments(rfm_scored)
        logger.info("Segments assigned.")

        # ------------------------------------------------------------------ #
        # Step 7: Estimate CLV
        # ------------------------------------------------------------------ #
        print_step(7, TOTAL_STEPS, "Estimating CLV")
        customer_agg = build_customer_aggregates(
            tables["orders"], tables["order_items"], tables["users"]
        )
        customer_agg = estimate_clv(customer_agg)

        # Merge segment into customer aggregates without column collisions
        segment_map = rfm_segmented[["customer_id", "segment", "r_score", "f_score",
                                      "m_score", "rfm_score", "recency", "frequency", "monetary"]].copy()
        # Drop overlapping columns from customer_agg to keep rfm_segmented values
        overlap_cols = [c for c in segment_map.columns if c in customer_agg.columns and c != "customer_id"]
        customer_agg_clean = customer_agg.drop(columns=overlap_cols)
        customer_full = customer_agg_clean.merge(segment_map, on="customer_id", how="left")
        logger.info("CLV estimated.")

        # ------------------------------------------------------------------ #
        # Step 8: Build churn features
        # ------------------------------------------------------------------ #
        print_step(8, TOTAL_STEPS, "Building churn features")
        try:
            churn_features = build_churn_features(
                tables["orders"], tables["order_items"], tables["users"]
            )
            churn_ok = True
        except ValueError as exc:
            print_error(f"Could not build churn features: {exc}")
            churn_ok = False
            churn_features = None

        # ------------------------------------------------------------------ #
        # Step 9: Train churn models
        # ------------------------------------------------------------------ #
        print_step(9, TOTAL_STEPS, "Training churn models")
        churn_results = None
        churn_predictions = None

        if churn_ok and churn_features is not None:
            try:
                churn_results = train_churn_models(churn_features)
                churn_predictions = generate_churn_predictions(
                    churn_features,
                    churn_results["best_model"],
                    churn_results["scaler"],
                    churn_results["feature_cols"],
                )
                print_success(
                    f"Best model: {churn_results['best_model_name']} | "
                    f"ROC-AUC: {churn_results['results'][churn_results['best_model_name']]['roc_auc']:.4f}"
                )
            except ValueError as exc:
                print_error(f"Churn model training failed: {exc}")
                churn_results = None
                churn_predictions = None

        # ------------------------------------------------------------------ #
        # Step 10: Generate recommendations
        # ------------------------------------------------------------------ #
        print_step(10, TOTAL_STEPS, "Generating recommendations")
        segment_for_recs = rfm_segmented[["customer_id", "segment"]].copy()
        recommendations = build_recommendations(
            tables["order_items"],
            tables["products"],
            segment_for_recs,
        )
        logger.info("Recommendations generated for %d customers.", recommendations["customer_id"].nunique())

        # ------------------------------------------------------------------ #
        # Step 11: Run cohort analysis
        # ------------------------------------------------------------------ #
        print_step(11, TOTAL_STEPS, "Running cohort analysis")
        cohort_retention = compute_cohort_retention(tables["orders"])
        logger.info("Cohort matrix: %s", cohort_retention.shape)

        # ------------------------------------------------------------------ #
        # Step 12: Calculate monthly KPIs
        # ------------------------------------------------------------------ #
        print_step(12, TOTAL_STEPS, "Calculating monthly KPIs")
        monthly_kpis = compute_monthly_kpis(
            tables["orders"], tables["order_items"], tables["users"]
        )
        logger.info("Monthly KPIs: %d months.", len(monthly_kpis))

        # ------------------------------------------------------------------ #
        # Step 13: Run product analytics
        # ------------------------------------------------------------------ #
        print_step(13, TOTAL_STEPS, "Running product analytics")
        category_analytics = compute_category_analytics(tables["order_items"], tables["products"])
        product_analytics = compute_product_analytics(tables["order_items"], tables["products"])
        logger.info("Category analytics: %d categories.", len(category_analytics))

        # ------------------------------------------------------------------ #
        # Step 14: Export outputs
        # ------------------------------------------------------------------ #
        print_step(14, TOTAL_STEPS, "Exporting outputs")

        # RFM + segments
        save_parquet(rfm_segmented, OUTPUT_DIR / OUTPUT_FILES["customer_rfm"], "customer_rfm")

        # Customer segments (full profile)
        # Ensure we have segment column
        segments_out = customer_full.copy()
        save_parquet(segments_out, OUTPUT_DIR / OUTPUT_FILES["customer_segments"], "customer_segments")

        # Churn predictions
        if churn_predictions is not None:
            save_parquet(churn_predictions, OUTPUT_DIR / OUTPUT_FILES["churn_predictions"], "churn_predictions")

        # Churn feature importance
        if churn_results is not None:
            save_parquet(
                churn_results["feature_importance"],
                OUTPUT_DIR / OUTPUT_FILES["churn_feature_importance"],
                "churn_feature_importance",
            )

        # Monthly KPIs
        save_parquet(monthly_kpis, OUTPUT_DIR / OUTPUT_FILES["monthly_kpis"], "monthly_kpis")

        # Cohort retention (reset index for parquet)
        cohort_out = cohort_retention.reset_index()
        cohort_out.columns = [str(c) for c in cohort_out.columns]
        save_parquet(cohort_out, OUTPUT_DIR / OUTPUT_FILES["cohort_retention"], "cohort_retention")

        # Category analytics
        save_parquet(category_analytics, OUTPUT_DIR / OUTPUT_FILES["category_analytics"], "category_analytics")

        # Recommendations
        save_parquet(recommendations, OUTPUT_DIR / OUTPUT_FILES["recommendations"], "recommendations")

        # CLV
        clv_out = customer_full[["customer_id", "estimated_clv", "annual_frequency"]].copy()
        save_parquet(clv_out, OUTPUT_DIR / OUTPUT_FILES["clv"], "customer_clv")

        elapsed = time.perf_counter() - start_time
        print()
        print("=" * 60)
        print("  CUSTOMERIQ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"\n  Total time: {elapsed:.1f} seconds")
        print(f"  Outputs written to: {OUTPUT_DIR}")
        print()

        # Summary table
        print("  Output files:")
        for key, filename in OUTPUT_FILES.items():
            path = OUTPUT_DIR / filename
            size = f"{path.stat().st_size / 1024:.1f} KB" if path.exists() else "not generated"
            print(f"    {filename:<40} {size}")

        return True

    except SystemExit:
        # Re-raise SystemExit (dataset missing etc.)
        raise
    except Exception as exc:
        print_error(f"Pipeline failed with an unexpected error:\n  {exc}")
        logger.debug(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
