"""
CustomerIQ — Central Configuration.

All paths are derived relative to this file's location so the project
is portable across machines without hardcoded absolute paths.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — one level above this file (src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUT_DIR: Path = DATA_DIR / "output"

DATABASE_DIR: Path = PROJECT_ROOT / "database"
DATABASE_PATH: Path = DATABASE_DIR / "customeriq.duckdb"

MODELS_DIR: Path = PROJECT_ROOT / "models"
SQL_DIR: Path = PROJECT_ROOT / "sql"

# ---------------------------------------------------------------------------
# Required tables and their minimum required columns
# ---------------------------------------------------------------------------
REQUIRED_TABLES: list[str] = [
    "users",
    "orders",
    "order_items",
    "products",
    "events",
    "inventory_items",
    "distribution_centers",
]

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "users": ["id", "first_name", "last_name", "email", "age", "gender", "country", "state", "city", "created_at"],
    "orders": ["order_id", "user_id", "status", "created_at", "num_of_item"],
    "order_items": ["id", "order_id", "user_id", "product_id", "status", "created_at", "sale_price"],
    "products": ["id", "name", "category", "brand", "department", "cost", "retail_price"],
    "events": ["id", "user_id", "event_type", "created_at"],
    "inventory_items": ["id", "product_id", "created_at", "sold_at", "cost"],
    "distribution_centers": ["id", "name", "latitude", "longitude"],
}

# ---------------------------------------------------------------------------
# Analytical parameters
# ---------------------------------------------------------------------------

# RFM scoring bins (quintile-based)
RFM_BINS: int = 5

# Churn definition:
#   - Feature observation window: all orders before CHURN_CUTOFF_DATE
#   - Prediction window: CHURN_FUTURE_DAYS after last observed order date
#   If CHURN_CUTOFF_DATE is None, it is derived automatically from the data.
CHURN_FUTURE_DAYS: int = 90   # A customer is "churned" if no purchase in next 90 days
CHURN_CUTOFF_DATE: str | None = None  # Set to "YYYY-MM-DD" to override; None = auto

# CLV parameters
CLV_HORIZON_YEARS: float = 2.0   # Estimated customer lifespan used in CLV formula

# Segmentation model
KMEANS_N_CLUSTERS: int = 6
KMEANS_RANDOM_STATE: int = 42

# ML models
ML_RANDOM_STATE: int = 42
ML_TEST_SIZE: float = 0.20       # 20% held out for testing

# Recommendation
MAX_RECOMMENDATIONS: int = 5

# Risk thresholds for churn_probability
RISK_LOW_THRESHOLD: float = 0.30    # < 30%  → Low Risk
RISK_HIGH_THRESHOLD: float = 0.60   # >= 60% → High Risk
# 30% – 60% → Medium Risk

# ---------------------------------------------------------------------------
# Output file names
# ---------------------------------------------------------------------------
OUTPUT_FILES: dict[str, str] = {
    "customer_rfm": "customer_rfm.parquet",
    "customer_segments": "customer_segments.parquet",
    "churn_predictions": "churn_predictions.parquet",
    "churn_feature_importance": "churn_feature_importance.parquet",
    "monthly_kpis": "monthly_kpis.parquet",
    "cohort_retention": "cohort_retention.parquet",
    "category_analytics": "category_analytics.parquet",
    "recommendations": "recommendations.parquet",
    "clv": "customer_clv.parquet",
}

MODEL_FILES: dict[str, str] = {
    "churn_best": "churn_best_model.joblib",
    "churn_scaler": "churn_scaler.joblib",
    "churn_features": "churn_feature_names.joblib",
}
