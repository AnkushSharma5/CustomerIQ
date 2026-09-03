"""
CustomerIQ — Customer Segmentation.

Two-layer segmentation approach:
1. RFM Rule-Based Segmentation (primary, explainable)
   - Assign R/F/M scores (1-5 quintiles)
   - Map score combinations to business segment labels
   - Fully documented, interview-ready rules

2. Optional KMeans Clustering (supplementary)
   - Scale RFM features (StandardScaler)
   - Fit KMeans with KMEANS_N_CLUSTERS
   - Map clusters to business labels

CLV Estimation:
   Estimated CLV = AOV × purchase_frequency_per_year × CLV_HORIZON_YEARS
"""
import logging

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import (
    RFM_BINS,
    KMEANS_N_CLUSTERS,
    KMEANS_RANDOM_STATE,
    CLV_HORIZON_YEARS,
)

logger = logging.getLogger("customeriq.segmentation")


# ---------------------------------------------------------------------------
# RFM Scoring
# ---------------------------------------------------------------------------

def compute_rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign R, F, M scores (1–5) using quintile-based binning.

    Recency: lower is better → score 5 = most recent.
    Frequency: higher is better → score 5 = most frequent.
    Monetary: higher is better → score 5 = highest spender.

    Parameters
    ----------
    rfm : pd.DataFrame
        Must have columns: customer_id, recency, frequency, monetary.

    Returns
    -------
    pd.DataFrame
        Input plus: r_score, f_score, m_score, rfm_score.
    """
    df = rfm.copy()

    # R score: invert so that low recency → high score
    df["r_score"] = _score_quintile(df["recency"], ascending=False)
    # F score
    df["f_score"] = _score_quintile(df["frequency"], ascending=True)
    # M score
    df["m_score"] = _score_quintile(df["monetary"], ascending=True)

    df["rfm_score"] = df["r_score"] + df["f_score"] + df["m_score"]

    logger.info(
        "RFM scores assigned. Score range: %d–%d.", df["rfm_score"].min(), df["rfm_score"].max()
    )
    return df


def _score_quintile(series: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Bin a series into 5 quintile-based scores (1–5).

    ascending=True  → higher values get higher scores (Frequency, Monetary)
    ascending=False → lower values get higher scores (Recency)
    """
    try:
        # Use quantile-based bins; duplicates='drop' handles ties gracefully
        labels = [1, 2, 3, 4, 5] if ascending else [5, 4, 3, 2, 1]
        return pd.qcut(series, q=RFM_BINS, labels=labels, duplicates="drop").astype(int)
    except Exception:
        # Fallback to rank-based scoring if qcut fails (too few unique values)
        ranks = series.rank(method="first", ascending=ascending)
        max_rank = ranks.max()
        return ((ranks / max_rank) * (RFM_BINS - 1)).round().astype(int).clip(1, RFM_BINS)


# ---------------------------------------------------------------------------
# Business Segment Assignment (Rule-Based)
# ---------------------------------------------------------------------------

# Segmentation rules documented here for interview readiness:
#
# Champions         : R=5, F=4-5, M=4-5  → best customers
# Loyal Customers   : F=4-5               → frequent buyers
# Potential Loyalists: R=4-5, F=2-3       → recent, not yet frequent
# New Customers     : R=4-5, F=1          → bought recently, first time
# Promising         : R=3-4               → showing activity
# At Risk           : R=2, F=3-4          → used to buy often, now inactive
# Cannot Lose Them  : R=1, F=4-5          → were top customers, now gone
# Hibernating       : R=2, F=1-2          → long since last purchase
# Lost Customers    : R=1, F=1-2          → very long inactive, low frequency

SEGMENT_RULES: list[dict] = [
    {"name": "Champions",          "r_min": 4, "r_max": 5, "f_min": 4, "f_max": 5, "m_min": 4, "m_max": 5},
    {"name": "Loyal Customers",    "r_min": 3, "r_max": 5, "f_min": 4, "f_max": 5, "m_min": 1, "m_max": 5},
    {"name": "Potential Loyalists","r_min": 4, "r_max": 5, "f_min": 2, "f_max": 3, "m_min": 1, "m_max": 5},
    {"name": "New Customers",      "r_min": 4, "r_max": 5, "f_min": 1, "f_max": 1, "m_min": 1, "m_max": 5},
    {"name": "Promising",          "r_min": 3, "r_max": 3, "f_min": 1, "f_max": 3, "m_min": 1, "m_max": 5},
    {"name": "At Risk",            "r_min": 2, "r_max": 2, "f_min": 3, "f_max": 5, "m_min": 1, "m_max": 5},
    {"name": "Cannot Lose Them",   "r_min": 1, "r_max": 1, "f_min": 4, "f_max": 5, "m_min": 1, "m_max": 5},
    {"name": "Hibernating",        "r_min": 2, "r_max": 2, "f_min": 1, "f_max": 2, "m_min": 1, "m_max": 5},
    {"name": "Lost Customers",     "r_min": 1, "r_max": 1, "f_min": 1, "f_max": 2, "m_min": 1, "m_max": 5},
]


def assign_segments(rfm_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a business segment label to each customer based on R/F/M scores.

    Rules are evaluated in priority order (first match wins).
    Unmatched customers receive "Other".

    Parameters
    ----------
    rfm_scored : pd.DataFrame
        Must have columns: customer_id, r_score, f_score, m_score.

    Returns
    -------
    pd.DataFrame
        Input plus column: segment.
    """
    df = rfm_scored.copy()
    df["segment"] = "Other"

    for rule in SEGMENT_RULES:
        mask = (
            df["r_score"].between(rule["r_min"], rule["r_max"])
            & df["f_score"].between(rule["f_min"], rule["f_max"])
            & df["m_score"].between(rule["m_min"], rule["m_max"])
            & (df["segment"] == "Other")  # only assign if not already labelled
        )
        df.loc[mask, "segment"] = rule["name"]

    seg_counts = df["segment"].value_counts()
    logger.info("Segment distribution:\n%s", seg_counts.to_string())
    return df


# ---------------------------------------------------------------------------
# KMeans Clustering (supplementary)
# ---------------------------------------------------------------------------

def run_kmeans_clustering(
    rfm_scored: pd.DataFrame,
    n_clusters: int = KMEANS_N_CLUSTERS,
    random_state: int = KMEANS_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Run KMeans clustering on scaled RFM features.

    Parameters
    ----------
    rfm_scored : pd.DataFrame
        Must have columns: recency, frequency, monetary.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    pd.DataFrame
        Input plus columns: cluster, cluster_label.
    """
    df = rfm_scored.copy()
    features = ["recency", "frequency", "monetary"]
    X = df[features].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Map clusters to descriptive labels based on cluster centers
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features,
    )
    cluster_centers["cluster"] = range(n_clusters)

    # Sort clusters by overall value (low recency + high freq + high monetary)
    cluster_centers["value_score"] = (
        -cluster_centers["recency"]
        + cluster_centers["frequency"]
        + cluster_centers["monetary"] / cluster_centers["monetary"].max()
    )
    cluster_centers_sorted = cluster_centers.sort_values("value_score", ascending=False)

    cluster_label_map: dict[int, str] = {}
    label_names = [
        "High Value",
        "Loyal Active",
        "Moderate Buyers",
        "Occasional Buyers",
        "Low Engagement",
        "Inactive",
    ][:n_clusters]
    for rank, (_, row) in enumerate(cluster_centers_sorted.iterrows()):
        label = label_names[rank] if rank < len(label_names) else f"Cluster {rank}"
        cluster_label_map[int(row["cluster"])] = label

    df["cluster_label"] = df["cluster"].map(cluster_label_map)
    logger.info("KMeans clustering complete. Cluster distribution:\n%s",
                df["cluster_label"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# CLV Estimation
# ---------------------------------------------------------------------------

def estimate_clv(
    customer_aggregates: pd.DataFrame,
    horizon_years: float = CLV_HORIZON_YEARS,
) -> pd.DataFrame:
    """
    Estimate Customer Lifetime Value (CLV).

    Formula
    -------
        estimated_clv = AOV × annual_purchase_frequency × horizon_years

    Where:
        AOV = average order value (historical)
        annual_purchase_frequency = (total_orders / customer_tenure_years)
        horizon_years = assumed remaining customer lifespan (configurable)

    For customers with tenure < 30 days, we use the dataset average frequency.
    This is clearly labelled as an ESTIMATE, not a guaranteed future value.

    Parameters
    ----------
    customer_aggregates : pd.DataFrame
        Must have: customer_id, total_orders, total_revenue, avg_order_value,
                   tenure_days, first_order_date.
    horizon_years : float
        How many years into the future to project.

    Returns
    -------
    pd.DataFrame
        Input plus column: estimated_clv.
    """
    df = customer_aggregates.copy()

    # Annual purchase frequency
    tenure_years = (df["tenure_days"] / 365.0).clip(lower=0.0833)  # min 1 month
    df["annual_frequency"] = df["total_orders"] / tenure_years

    # Average frequency across dataset (used for very new customers)
    avg_annual_freq = df["annual_frequency"].median()

    # For customers with very short tenure, use the dataset median frequency
    new_customer_mask = df["tenure_days"] < 30
    df.loc[new_customer_mask, "annual_frequency"] = avg_annual_freq

    df["estimated_clv"] = (
        df["avg_order_value"] * df["annual_frequency"] * horizon_years
    ).round(2)

    logger.info(
        "CLV estimated for %d customers. Median CLV: $%.2f",
        len(df),
        df["estimated_clv"].median(),
    )
    return df
