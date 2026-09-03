"""
CustomerIQ — Recommendation Engine.

Approach: Collaborative Filtering based on customer segments + category affinity.

Methodology:
1. For each customer, identify which categories/products they have purchased.
2. For each customer's segment, calculate which categories/products are most
   popular among segment peers.
3. Recommend top-N products that the customer has NOT already purchased,
   ranked by segment affinity score.
4. Each recommendation includes a human-readable explanation.

This is fully explainable — no black-box deep learning.

PERFORMANCE NOTE
-----------------
This module builds recommendations for *all* customers using vectorized
pandas/groupby operations, rather than looping through customers one at a
time in Python (which scales very poorly — fine for a few hundred
customers, but tens of minutes or more at ~100K customers).

The one Python loop that remains here iterates over *segments*
(typically fewer than 10), not customers — each iteration does a
vectorized customer x candidate-product join for just that segment,
keeping every intermediate table small. Text columns (product name,
category, brand) are only attached to the final, already-small result
set (max_recs rows per customer) rather than to any large intermediate
cross-join, which avoids duplicating strings across millions of rows
and keeps peak memory low.
"""
import logging

import numpy as np
import pandas as pd

from src.config import MAX_RECOMMENDATIONS

logger = logging.getLogger("customeriq.recommendation")

# How many top candidates to consider per segment / overall before
# excluding already-purchased products. Only needs to comfortably exceed
# max_recs so that after removing a customer's own purchase history there
# are still enough candidates left.
_CANDIDATE_POOL_MULTIPLIER = 10
_MIN_CANDIDATE_POOL = 50

_OUTPUT_COLS = [
    "customer_id", "product_id", "product_name", "category",
    "brand", "affinity_score", "explanation", "rank",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_recommendations(
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    customer_segments: pd.DataFrame,
    max_recs: int = MAX_RECOMMENDATIONS,
) -> pd.DataFrame:
    """
    Generate product recommendations for all customers.

    Parameters
    ----------
    order_items : pd.DataFrame
        Must have: user_id, product_id, status.
    products : pd.DataFrame
        Must have: id, name, category, brand.
    customer_segments : pd.DataFrame
        Must have: customer_id, segment.
    max_recs : int
        Maximum recommendations per customer.

    Returns
    -------
    pd.DataFrame
        Columns: customer_id, product_id, product_name, category,
                 brand, affinity_score, explanation, rank.
    """
    empty_result = pd.DataFrame(columns=_OUTPUT_COLS)

    if customer_segments.empty or order_items.empty:
        logger.warning("No recommendations generated (empty input).")
        return empty_result

    completed_statuses = {"complete", "shipped", "processing"}
    oi = (
        order_items.loc[order_items["status"].isin(completed_statuses), ["user_id", "product_id"]]
        .rename(columns={"user_id": "customer_id"})
        .copy()
    )
    if oi.empty:
        logger.warning("No recommendations generated (no completed purchases).")
        return empty_result

    customers_df = customer_segments[["customer_id", "segment"]].drop_duplicates()
    seg_map = customers_df.set_index("customer_id")["segment"]
    oi["segment"] = oi["customer_id"].map(seg_map)

    # Deduplicated (customer, product) purchase pairs — used for anti-joins.
    purchased_pairs = oi[["customer_id", "product_id"]].drop_duplicates()

    # --- Segment -> product popularity (numeric only, no text columns) ---
    seg_product_popularity = (
        oi.groupby(["segment", "product_id"]).size().reset_index(name="purchase_count")
    )
    overall_popularity = (
        oi.groupby("product_id").size().reset_index(name="purchase_count")
        .sort_values("purchase_count", ascending=False)
    )

    if seg_product_popularity.empty and overall_popularity.empty:
        logger.warning("No recommendations generated (no purchase data).")
        return empty_result

    pool_size = max(max_recs * _CANDIDATE_POOL_MULTIPLIER, _MIN_CANDIDATE_POOL)

    seg_candidates = (
        seg_product_popularity
        .sort_values(["segment", "purchase_count"], ascending=[True, False])
        .groupby("segment", group_keys=False)
        .head(pool_size)
    )
    overall_candidates = overall_popularity.head(pool_size)[["product_id", "purchase_count"]]

    # --- Primary pass: segment-based recommendations ---
    # Loop over segments (a handful), NOT customers. Each iteration's
    # cross-join is bounded by (customers in that segment) x pool_size,
    # and only numeric columns are involved.
    primary_parts = []
    for segment, cust_group in customers_df.groupby("segment"):
        cand = seg_candidates.loc[seg_candidates["segment"] == segment, ["product_id", "purchase_count"]]
        if cand.empty:
            continue
        pairs = cust_group[["customer_id"]].merge(cand, how="cross")
        pairs = pairs.merge(purchased_pairs, on=["customer_id", "product_id"], how="left", indicator=True)
        pairs = pairs.loc[pairs["_merge"] == "left_only", ["customer_id", "product_id", "purchase_count"]]
        if pairs.empty:
            continue
        pairs = pairs.sort_values(["customer_id", "purchase_count"], ascending=[True, False])
        pairs["rank"] = pairs.groupby("customer_id").cumcount() + 1
        pairs = pairs[pairs["rank"] <= max_recs]
        pairs["source"] = "segment"
        primary_parts.append(pairs)

    primary = (
        pd.concat(primary_parts, ignore_index=True)
        if primary_parts else
        pd.DataFrame(columns=["customer_id", "product_id", "purchase_count", "rank", "source"])
    )

    # --- Backfill pass: only for customers still short of max_recs ---
    existing_counts = (
        primary.groupby("customer_id").size()
        if not primary.empty else pd.Series(dtype=int)
    )
    needed = (max_recs - existing_counts.reindex(customers_df["customer_id"], fill_value=0)).clip(lower=0)
    customers_needing = needed[needed > 0]

    backfill = pd.DataFrame(columns=["customer_id", "product_id", "purchase_count", "rank", "source"])
    if not customers_needing.empty and not overall_candidates.empty:
        backfill_customers = pd.DataFrame({"customer_id": customers_needing.index})
        pairs = backfill_customers.merge(overall_candidates, how="cross")

        already_taken = pd.concat(
            [purchased_pairs, primary[["customer_id", "product_id"]]],
            ignore_index=True,
        ).drop_duplicates()
        pairs = pairs.merge(already_taken, on=["customer_id", "product_id"], how="left", indicator=True)
        pairs = pairs.loc[pairs["_merge"] == "left_only", ["customer_id", "product_id", "purchase_count"]]

        if not pairs.empty:
            pairs = pairs.sort_values(["customer_id", "purchase_count"], ascending=[True, False])
            pairs["seq"] = pairs.groupby("customer_id").cumcount() + 1
            pairs = pairs.merge(customers_needing.rename("needed"), left_on="customer_id", right_index=True)
            pairs = pairs[pairs["seq"] <= pairs["needed"]]
            pairs = pairs.merge(existing_counts.rename("existing"), left_on="customer_id", right_index=True, how="left")
            pairs["existing"] = pairs["existing"].fillna(0).astype(int)
            pairs["rank"] = pairs["existing"] + pairs["seq"]
            pairs["source"] = "overall"
            backfill = pairs[["customer_id", "product_id", "purchase_count", "rank", "source"]]

    selected = pd.concat([primary, backfill], ignore_index=True)
    if selected.empty:
        logger.warning("No recommendations generated after filtering.")
        return empty_result

    # --- Attach text metadata only to the final, small result set ---
    products_small = (
        products[["id", "name", "category", "brand"]]
        .rename(columns={"id": "product_id", "name": "product_name"})
    )
    selected = selected.merge(products_small, on="product_id", how="left")
    selected = selected.merge(customers_df, on="customer_id", how="left")
    selected = selected.dropna(subset=["product_name"])

    if selected.empty:
        logger.warning("No recommendations generated (no matching product metadata).")
        return empty_result

    # Affinity score normalised 0-1 within each customer's recommendation set.
    max_per_customer = selected.groupby("customer_id")["purchase_count"].transform("max")
    selected["affinity_score"] = (selected["purchase_count"] / max_per_customer).fillna(0.0).round(4)

    # Category the customer has already bought from (for explanation text).
    oi_with_cat = oi.merge(products_small[["product_id", "category"]], on="product_id", how="left")
    bought_categories = (
        oi_with_cat[["customer_id", "category"]].dropna().drop_duplicates()
    )
    bought_categories["_bought_category"] = True
    selected = selected.merge(bought_categories, on=["customer_id", "category"], how="left")
    selected["_bought_category"] = selected["_bought_category"].fillna(False)

    selected["explanation"] = _build_explanations_vectorized(selected)

    result = (
        selected[_OUTPUT_COLS]
        .sort_values(["customer_id", "rank"])
        .reset_index(drop=True)
    )

    logger.info(
        "Recommendations generated for %d customers (%d total rows).",
        result["customer_id"].nunique(), len(result),
    )
    return result


def get_customer_recommendations(
    customer_id: int,
    recommendations_df: pd.DataFrame,
    n: int = MAX_RECOMMENDATIONS,
) -> pd.DataFrame:
    """
    Retrieve recommendations for a specific customer from the prebuilt table.
    """
    recs = recommendations_df[recommendations_df["customer_id"] == customer_id]
    return recs.head(n)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_explanations_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized equivalent of the original row-wise explanation builder.

    Produces one of three explanation templates per row, depending on
    whether the product came from the segment-popularity pool or the
    overall-fallback pool, and whether the customer has already bought
    from that product's category.
    """
    segment = df["segment"].fillna("")
    category = df["category"].fillna("")

    trending = "Trending product: popular across all customers in our platform."

    same_category = (
        "Recommended because you've purchased from '" + category +
        "' before, and this product is popular among " + segment +
        " customers like you."
    )

    new_category = (
        "Recommended because customers in the '" + segment +
        "' segment frequently purchase from the '" + category + "' category."
    )

    is_overall = df["source"] == "overall"
    is_same_category = df["_bought_category"].astype(bool) & ~is_overall

    explanation = pd.Series(new_category, index=df.index)
    explanation = explanation.mask(is_same_category, same_category)
    explanation = explanation.mask(is_overall, trending)
    return explanation