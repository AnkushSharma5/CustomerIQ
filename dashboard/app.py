"""
CustomerIQ — Streamlit Dashboard.

Usage:
    streamlit run dashboard/app.py

Pages:
    1. Executive Overview
    2. Customer Segmentation (with Advanced Multi-Attribute Filtering)
    3. Customer 360
    4. Churn Intelligence
    5. Recommendations
    6. Product & Category Analytics
    7. Cohort & Retention
"""
import sys
from pathlib import Path

# Ensure project root is on path
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.config import OUTPUT_DIR, OUTPUT_FILES

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CustomerIQ — AI Customer Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium Design System & CSS Styles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #090d16;
        color: #f1f5f9;
    }

    /* Header styling */
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #64ffda 0%, #7c83fd 50%, #ff6b9d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        font-weight: 400;
        margin-bottom: 24px;
    }

    /* Glassmorphism KPI Card */
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: left;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(100, 255, 218, 0.3);
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 0.78rem;
        font-weight: 500;
        color: #64ffda;
        margin-top: 6px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        border-left: 4px solid #64ffda;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* Streamlit widget overrides */
    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }

    .stMultiSelect span {
        background-color: #334155 !important;
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_output(key: str) -> pd.DataFrame | None:
    """Load an output parquet file. Returns None if not found."""
    path = OUTPUT_DIR / OUTPUT_FILES[key]
    if path.exists():
        return pd.read_parquet(path)
    return None


def require_pipeline() -> bool:
    """Return True if output files exist, else show a warning."""
    rfm = load_output("customer_rfm")
    if rfm is None:
        st.warning(
            "⚠️ Pipeline output not found. Please run `python run_pipeline.py` first, "
            "then refresh this page."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Plotly theme
# ---------------------------------------------------------------------------
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#cbd5e1"),
)


def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_THEME)
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="dashboard-title" style="font-size:1.6rem;">🎯 CustomerIQ</div>', unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#64748b;font-size:0.8rem;margin-top:-10px;'>AI Customer Intelligence Platform</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

PAGE_NAMES = {
    "🏠 Executive Overview": "overview",
    "👥 Customer Segmentation": "segmentation",
    "🔍 Customer 360": "customer360",
    "⚠️ Churn Intelligence": "churn",
    "💡 Recommendations": "recommendations",
    "📦 Product & Category": "products",
    "📅 Cohort & Retention": "cohort",
}
selected_page_label = st.sidebar.radio("Navigate", list(PAGE_NAMES.keys()), label_visibility="collapsed")
page = PAGE_NAMES[selected_page_label]

st.sidebar.markdown("---")
st.sidebar.markdown("##### System Status")

# Show which output files are available
for key in ["customer_rfm", "churn_predictions", "recommendations", "monthly_kpis"]:
    path = OUTPUT_DIR / OUTPUT_FILES[key]
    icon = "🟢" if path.exists() else "🔴"
    st.sidebar.markdown(f"{icon} `{OUTPUT_FILES[key]}`")

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.72rem;color:#64748b;">'
    'theLook eCommerce Dataset<br>© 2024 CustomerIQ Platform</div>',
    unsafe_allow_html=True,
)


# Helper function for rendering KPI Cards
def kpi_card(label: str, value: str, delta: str = "") -> str:
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


# ===========================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ===========================================================================
if page == "overview":
    st.markdown('<div class="dashboard-title">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Real-time performance metrics and high-level customer behavior summary</div>', unsafe_allow_html=True)

    if not require_pipeline():
        st.stop()

    rfm = load_output("customer_rfm")
    kpis = load_output("monthly_kpis")
    segments = load_output("customer_segments")
    churn = load_output("churn_predictions")

    # --- KPI Cards ---
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    total_customers = len(rfm) if rfm is not None else 0
    total_revenue = segments["total_revenue"].sum() if segments is not None and "total_revenue" in segments.columns else 0
    total_orders = segments["total_orders"].sum() if segments is not None and "total_orders" in segments.columns else 0
    avg_order_value = (total_revenue / total_orders) if total_orders > 0 else 0
    high_risk = (churn["risk_level"] == "High Risk").sum() if churn is not None else 0
    repeat_rate = 0.0
    if kpis is not None and "repeat_customer_rate" in kpis.columns:
        repeat_rate = kpis["repeat_customer_rate"].mean() * 100

    with col1:
        st.markdown(kpi_card("Total Customers", f"{total_customers:,}"), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_card("Total Revenue", f"${total_revenue:,.0f}"), unsafe_allow_html=True)
    with col3:
        st.markdown(kpi_card("Total Orders", f"{total_orders:,}"), unsafe_allow_html=True)
    with col4:
        st.markdown(kpi_card("Avg Order Value", f"${avg_order_value:,.2f}"), unsafe_allow_html=True)
    with col5:
        st.markdown(kpi_card("Repeat Rate", f"{repeat_rate:.1f}%"), unsafe_allow_html=True)
    with col6:
        st.markdown(kpi_card("High-Risk Churn", f"{high_risk:,}", "⚠️ Attention Needed"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Charts ---
    row1_col1, row1_col2 = st.columns([2, 1])

    with row1_col1:
        st.markdown('<div class="section-header">Monthly Revenue Trend</div>', unsafe_allow_html=True)
        if kpis is not None:
            fig = px.area(
                kpis, x="year_month", y="total_revenue",
                labels={"year_month": "Month", "total_revenue": "Revenue ($)"},
                color_discrete_sequence=["#64ffda"],
            )
            fig = apply_theme(fig)
            fig.update_traces(fillcolor="rgba(100,255,218,0.12)", line_color="#64ffda", line_width=2.5)
            st.plotly_chart(fig, use_container_width=True)

    with row1_col2:
        st.markdown('<div class="section-header">Customer Segments</div>', unsafe_allow_html=True)
        if rfm is not None and "segment" in rfm.columns:
            seg_counts = rfm["segment"].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            fig = px.pie(
                seg_counts, names="segment", values="count",
                color_discrete_sequence=px.colors.qualitative.Bold,
                hole=0.5,
            )
            fig = apply_theme(fig)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown('<div class="section-header">Monthly Orders Volume</div>', unsafe_allow_html=True)
        if kpis is not None:
            fig = px.bar(
                kpis, x="year_month", y="total_orders",
                labels={"year_month": "Month", "total_orders": "Orders"},
                color_discrete_sequence=["#7c83fd"],
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    with row2_col2:
        st.markdown('<div class="section-header">New vs Repeat Customers</div>', unsafe_allow_html=True)
        if kpis is not None and "new_customers" in kpis.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=kpis["year_month"], y=kpis["new_customers"],
                name="New Customers", marker_color="#64ffda",
            ))
            fig.add_trace(go.Bar(
                x=kpis["year_month"], y=kpis["repeat_customers"],
                name="Repeat Customers", marker_color="#7c83fd",
            ))
            fig.update_layout(barmode="stack", xaxis_title="Month", yaxis_title="Customers")
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 2: CUSTOMER SEGMENTATION (ADVANCED & ROBUST FILTERING)
# ===========================================================================
elif page == "segmentation":
    st.markdown('<div class="dashboard-title">👥 Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Multi-dimensional customer profiling, RFM scoring, and targeted segment filtering</div>', unsafe_allow_html=True)

    if not require_pipeline():
        st.stop()

    rfm = load_output("customer_rfm")
    segments = load_output("customer_segments")

    # Combine segments and rfm safely
    if segments is not None and rfm is not None:
        rfm_cols = [c for c in rfm.columns if c not in segments.columns and c != "customer_id"]
        if rfm_cols:
            df_seg = segments.merge(rfm[["customer_id"] + rfm_cols], on="customer_id", how="left")
        else:
            df_seg = segments.copy()
    elif segments is not None:
        df_seg = segments.copy()
    elif rfm is not None:
        df_seg = rfm.copy()
    else:
        df_seg = None

    if df_seg is None or len(df_seg) == 0:
        st.error("Segmentation data not available. Please run the pipeline first.")
        st.stop()

    # Normalize suffix collisions if present (recency_x / recency_y -> recency)
    for target in ["recency", "frequency", "monetary"]:
        if target not in df_seg.columns:
            if f"{target}_x" in df_seg.columns:
                df_seg[target] = df_seg[f"{target}_x"]
            elif f"{target}_y" in df_seg.columns:
                df_seg[target] = df_seg[f"{target}_y"]
            elif target == "monetary" and "total_revenue" in df_seg.columns:
                df_seg["monetary"] = df_seg["total_revenue"]
            elif target == "frequency" and "total_orders" in df_seg.columns:
                df_seg["frequency"] = df_seg["total_orders"]
            else:
                df_seg[target] = 0.0

    # Fill NaN values in numeric filter columns safely
    df_seg["recency"] = pd.to_numeric(df_seg["recency"], errors="coerce").fillna(0.0)
    df_seg["frequency"] = pd.to_numeric(df_seg["frequency"], errors="coerce").fillna(0).astype(int)
    df_seg["monetary"] = pd.to_numeric(df_seg["monetary"], errors="coerce").fillna(0.0)

    # -----------------------------------------------------------------------
    # Interactive Filter Panel
    # -----------------------------------------------------------------------
    st.markdown("### 🎛️ Interactive Segment Filters")

    with st.expander("🔍 Click to Expand / Collapse Filters", expanded=True):
        # Segments and Countries can each have many distinct values once the
        # real dataset is loaded (vs. only a handful in the small test
        # dataset), so they get their own wide row instead of squeezing into
        # a quarter-width column. Leaving default=[] means "no selection =
        # show all" (see the filtering logic below), which avoids rendering
        # dozens of pre-selected chips that overflow and get truncated.
        col_f1, col_f2 = st.columns(2)

        # 1. Segment multiselect
        all_segments = sorted(df_seg["segment"].dropna().astype(str).unique().tolist()) if "segment" in df_seg.columns else []
        with col_f1:
            selected_segments = st.multiselect(
                "Customer Segments",
                options=all_segments,
                default=[],
                placeholder="All segments",
                help="Select one or more RFM segments to analyze. Leave empty to include all.",
            )

        # 2. Country filter
        all_countries = sorted(df_seg["country"].dropna().astype(str).unique().tolist()) if "country" in df_seg.columns else []
        with col_f2:
            selected_countries = st.multiselect(
                "Countries",
                options=all_countries,
                default=[],
                placeholder="All countries",
                help="Filter customers by geographic location. Leave empty to include all.",
            )

        col_f3, col_f4 = st.columns(2)

        # 3. Gender filter (small, fixed option count — safe to default to all)
        all_genders = sorted(df_seg["gender"].dropna().astype(str).unique().tolist()) if "gender" in df_seg.columns else []
        with col_f3:
            selected_genders = st.multiselect(
                "Gender",
                options=all_genders,
                default=all_genders,
            )

        # 4. Customer Search
        with col_f4:
            search_query = st.text_input("Search Customer ID / Name", placeholder="e.g. 104 or John")

        # Range Sliders (Robust Min/Max calculation)
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            r_min = float(df_seg["recency"].min())
            r_max = float(df_seg["recency"].max())
            if r_min >= r_max:
                r_max = r_min + 1.0
            rec_range = st.slider("Recency (Days Since Last Purchase)", min_value=0.0, max_value=r_max, value=(0.0, r_max))

        with col_r2:
            f_min = int(df_seg["frequency"].min())
            f_max = int(df_seg["frequency"].max())
            if f_min >= f_max:
                f_max = f_min + 1
            freq_range = st.slider("Frequency (Total Orders)", min_value=f_min, max_value=f_max, value=(f_min, f_max))

        with col_r3:
            m_min = float(df_seg["monetary"].min())
            m_max = float(df_seg["monetary"].max())
            if m_min >= m_max:
                m_max = m_min + 10.0
            mon_range = st.slider("Monetary Spend ($)", min_value=0.0, max_value=m_max, value=(0.0, m_max))

    # Apply Filters to DataFrame
    filtered_df = df_seg.copy()

    if selected_segments and "segment" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["segment"].astype(str).isin(selected_segments)]
    if selected_countries and "country" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["country"].astype(str).isin(selected_countries)]
    if selected_genders and "gender" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["gender"].astype(str).isin(selected_genders)]

    filtered_df = filtered_df[
        (filtered_df["recency"].between(rec_range[0], rec_range[1])) &
        (filtered_df["frequency"].between(freq_range[0], freq_range[1])) &
        (filtered_df["monetary"].between(mon_range[0], mon_range[1]))
    ]

    if search_query:
        q = search_query.strip().lower()
        id_match = filtered_df["customer_id"].astype(str).str.lower().str.contains(q)
        name_match = pd.Series(False, index=filtered_df.index)
        if "first_name" in filtered_df.columns:
            name_match = name_match | filtered_df["first_name"].fillna("").astype(str).str.lower().str.contains(q)
        if "last_name" in filtered_df.columns:
            name_match = name_match | filtered_df["last_name"].fillna("").astype(str).str.lower().str.contains(q)
        filtered_df = filtered_df[id_match | name_match]

    # -----------------------------------------------------------------------
    # Filter Summary Bar
    # -----------------------------------------------------------------------
    st.markdown("---")
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)

    n_filtered = len(filtered_df)
    n_total = len(df_seg)
    pct_cust = (n_filtered / n_total * 100) if n_total > 0 else 0

    rev_filtered = filtered_df["monetary"].sum() if n_filtered > 0 else 0.0
    rev_total = df_seg["monetary"].sum() if n_total > 0 else 1.0
    pct_rev = (rev_filtered / rev_total * 100) if rev_total > 0 else 0

    avg_aov = filtered_df["monetary"].mean() if n_filtered > 0 else 0.0
    avg_freq = filtered_df["frequency"].mean() if n_filtered > 0 else 0.0

    mode_seg = filtered_df["segment"].mode() if n_filtered > 0 and "segment" in filtered_df.columns else pd.Series()
    top_segment = str(mode_seg.iloc[0]) if len(mode_seg) > 0 else "N/A"

    with col_s1:
        st.markdown(kpi_card("Filtered Customers", f"{n_filtered:,}", f"{pct_cust:.1f}% of total"), unsafe_allow_html=True)
    with col_s2:
        st.markdown(kpi_card("Filtered Revenue", f"${rev_filtered:,.0f}", f"{pct_rev:.1f}% of revenue"), unsafe_allow_html=True)
    with col_s3:
        st.markdown(kpi_card("Avg Customer Spend", f"${avg_aov:,.2f}"), unsafe_allow_html=True)
    with col_s4:
        st.markdown(kpi_card("Avg Order Count", f"{avg_freq:.1f} orders"), unsafe_allow_html=True)
    with col_s5:
        st.markdown(kpi_card("Dominant Segment", f"{top_segment}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Charts & Visualizations
    # -----------------------------------------------------------------------
    if len(filtered_df) > 0:
        col_c1, col_c2 = st.columns([1, 1])

        with col_c1:
            st.markdown('<div class="section-header">Segment Customer Distribution</div>', unsafe_allow_html=True)
            if "segment" in filtered_df.columns:
                seg_counts = filtered_df["segment"].value_counts().reset_index()
                seg_counts.columns = ["segment", "count"]
                fig = px.bar(
                    seg_counts.sort_values("count", ascending=True),
                    x="count", y="segment", orientation="h",
                    color="count", color_continuous_scale="Teal",
                    labels={"count": "Customer Count", "segment": "Segment"},
                    text_auto=True,
                )
                fig = apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        with col_c2:
            st.markdown('<div class="section-header">Revenue Contribution by Segment</div>', unsafe_allow_html=True)
            if "segment" in filtered_df.columns:
                seg_rev = filtered_df.groupby("segment")["monetary"].sum().reset_index()
                seg_rev.columns = ["segment", "revenue"]
                fig = px.pie(
                    seg_rev, names="segment", values="revenue",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hole=0.4,
                )
                fig = apply_theme(fig)
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)

        # RFM Behaviour Matrix Scatter Plot
        st.markdown('<div class="section-header">RFM Behaviour Matrix (Recency vs. Spend)</div>', unsafe_allow_html=True)
        hover_cols = [c for c in ["customer_id", "frequency", "r_score", "f_score", "m_score", "estimated_clv"] if c in filtered_df.columns]
        fig_scatter = px.scatter(
            filtered_df,
            x="recency",
            y="monetary",
            size="frequency" if (filtered_df["frequency"] > 0).any() else None,
            color="segment" if "segment" in filtered_df.columns else None,
            hover_data=hover_cols,
            labels={"recency": "Recency (Days)", "monetary": "Monetary Spend ($)", "frequency": "Orders"},
            color_discrete_sequence=px.colors.qualitative.Vivid,
            size_max=30,
        )
        fig_scatter = apply_theme(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # -------------------------------------------------------------------
        # Segment Summary Matrix Table
        # -------------------------------------------------------------------
        st.markdown('<div class="section-header">Segment Metrics & Strategy Matrix</div>', unsafe_allow_html=True)
        if "segment" in filtered_df.columns:
            seg_summary = filtered_df.groupby("segment").agg(
                customer_count=("customer_id", "nunique"),
                total_revenue=("monetary", "sum"),
                avg_revenue=("monetary", "mean"),
                avg_recency=("recency", "mean"),
                avg_frequency=("frequency", "mean"),
            ).reset_index()

            seg_summary["revenue_share"] = (seg_summary["total_revenue"] / rev_total * 100).round(1)
            seg_summary["avg_revenue"] = seg_summary["avg_revenue"].round(2)
            seg_summary["avg_recency"] = seg_summary["avg_recency"].round(1)
            seg_summary["avg_frequency"] = seg_summary["avg_frequency"].round(1)

            st.dataframe(
                seg_summary.rename(columns={
                    "segment": "Segment",
                    "customer_count": "Customers",
                    "total_revenue": "Total Spend ($)",
                    "avg_revenue": "Avg Spend ($)",
                    "avg_recency": "Avg Recency (Days)",
                    "avg_frequency": "Avg Orders",
                    "revenue_share": "Rev Share (%)",
                }).sort_values("Total Spend ($)", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        # -------------------------------------------------------------------
        # Detailed Filtered Table & CSV Export
        # -------------------------------------------------------------------
        st.markdown('<div class="section-header">Filtered Customer Segment Details</div>', unsafe_allow_html=True)

        display_cols = [c for c in [
            "customer_id", "first_name", "last_name", "segment", "recency",
            "frequency", "monetary", "r_score", "f_score", "m_score",
            "rfm_score", "estimated_clv", "country", "age", "gender"
        ] if c in filtered_df.columns]

        st.dataframe(
            filtered_df[display_cols].head(300),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Filtered Segment Data (CSV)",
            data=csv_data,
            file_name="customeriq_filtered_segments.csv",
            mime="text/csv",
            help="Download the filtered customer list for targeted marketing campaigns.",
        )
    else:
        st.info("⚠️ No customers match the current filter selection. Adjust the sliders or reset segment filters above.")

    # Documentation section
    with st.expander("📖 View Segmentation Rules Documentation"):
        st.markdown("""
        **RFM Quintile Scoring**: Customers receive scores from 1 (lowest) to 5 (highest) based on Recency, Frequency, and Monetary value.

        | Segment Name | Recency (R) | Frequency (F) | Monetary (M) | Business Action Strategy |
        |--------------|-------------|---------------|--------------|--------------------------|
        | **Champions** | 4–5 | 4–5 | 4–5 | Reward loyalty, early access to new products |
        | **Loyal Customers** | 3–5 | 4–5 | Any | Upsell higher value products, ask for reviews |
        | **Potential Loyalists** | 4–5 | 2–3 | Any | Offer membership, recommend sub-categories |
        | **New Customers** | 4–5 | 1 | Any | Onboarding support, discount on second purchase |
        | **Promising** | 3 | 1–3 | Any | Create brand awareness, limited time offers |
        | **At Risk** | 2 | 3–5 | Any | Re-engagement campaigns, personalized emails |
        | **Cannot Lose Them** | 1 | 4–5 | Any | Win-back calls, aggressive retention offers |
        | **Hibernating** | 2 | 1–2 | Any | Re-create interest with relevant deals |
        | **Lost Customers** | 1 | 1–2 | Any | Low priority re-engagement |
        """)


# ===========================================================================
# PAGE 3: CUSTOMER 360
# ===========================================================================
elif page == "customer360":
    st.markdown('<div class="dashboard-title">🔍 Customer 360 View</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Complete 360° view of customer profile, RFM scores, lifetime value, churn risk, and AI recommendations</div>', unsafe_allow_html=True)

    if not require_pipeline():
        st.stop()

    rfm = load_output("customer_rfm")
    segments = load_output("customer_segments")
    churn = load_output("churn_predictions")
    recs = load_output("recommendations")
    clv = load_output("clv")

    if rfm is None:
        st.error("RFM data not available. Run the pipeline first.")
        st.stop()

    # Customer selector
    customer_ids = sorted(rfm["customer_id"].tolist())
    selected_id = st.selectbox(
        "Select Customer ID",
        options=customer_ids,
        format_func=lambda x: f"Customer #{x}",
        help="Select a customer to view their full profile.",
    )

    if selected_id:
        cust_rfm = rfm[rfm["customer_id"] == selected_id].iloc[0]

        cust_seg = None
        if segments is not None:
            seg_row = segments[segments["customer_id"] == selected_id]
            if len(seg_row) > 0:
                cust_seg = seg_row.iloc[0]

        cust_churn = None
        if churn is not None:
            churn_row = churn[churn["customer_id"] == selected_id]
            if len(churn_row) > 0:
                cust_churn = churn_row.iloc[0]

        cust_clv = None
        if clv is not None:
            clv_row = clv[clv["customer_id"] == selected_id]
            if len(clv_row) > 0:
                cust_clv = clv_row.iloc[0]

        st.markdown(f"## Customer #{selected_id}")
        if cust_seg is not None:
            fname = cust_seg.get("first_name", "") if cust_seg is not None else ""
            lname = cust_seg.get("last_name", "") if cust_seg is not None else ""
            if fname or lname:
                st.markdown(f"### **{fname} {lname}**")

        # Demographics + purchase KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**📍 Profile & Location**")
            if cust_seg is not None:
                for field in ["country", "state", "city", "gender", "age", "email"]:
                    if field in cust_seg.index and pd.notna(cust_seg[field]):
                        st.markdown(f"- **{field.capitalize()}**: `{cust_seg[field]}`")

        with col2:
            st.markdown("**🛒 Purchase Metrics**")
            st.metric("Total Orders", int(cust_rfm.get("frequency", 0)))
            st.metric("Total Revenue", f"${cust_rfm.get('monetary', 0):,.2f}")
            if cust_seg is not None and "avg_order_value" in cust_seg.index:
                st.metric("Avg Order Value", f"${cust_seg['avg_order_value']:,.2f}")

        with col3:
            st.markdown("**📊 RFM Breakdown**")
            st.metric("Recency", f"{cust_rfm.get('recency', 0):.0f} days")
            st.metric("R / F / M Scores", f"{int(cust_rfm.get('r_score', 0))} / {int(cust_rfm.get('f_score', 0))} / {int(cust_rfm.get('m_score', 0))}")
            segment_label = cust_rfm.get("segment", "Unknown")
            st.markdown(f"**Segment**: `{segment_label}`")

        with col4:
            st.markdown("**🔮 AI Predictions**")
            if cust_clv is not None:
                st.metric("Estimated CLV", f"${cust_clv.get('estimated_clv', 0):,.2f}")

            if cust_churn is not None:
                churn_prob = cust_churn.get("churn_probability", 0.0)
                risk = cust_churn.get("risk_level", "Unknown")
                st.metric("Churn Probability", f"{churn_prob:.1%}")
                color = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}.get(risk, "⚪")
                st.markdown(f"**Risk Level**: {color} **{risk}**")

        st.markdown("---")

        # --- Recommendations ---
        st.markdown('<div class="section-header">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
        if recs is not None:
            cust_recs = recs[recs["customer_id"] == selected_id]
            if len(cust_recs) > 0:
                for _, rec in cust_recs.head(5).iterrows():
                    with st.container():
                        rc1, rc2 = st.columns([3, 1])
                        with rc1:
                            st.markdown(f"**{rec.get('product_name', rec.get('name', 'Product'))}**")
                            st.caption(f"Category: {rec.get('category', 'N/A')} | Brand: {rec.get('brand', 'N/A')}")
                            st.caption(f"💬 {rec.get('explanation', '')}")
                        with rc2:
                            score = rec.get("affinity_score", 0)
                            st.metric("Affinity Score", f"{score:.0%}")
                    st.divider()
            else:
                st.info("No recommendations available for this customer.")


# ===========================================================================
# PAGE 4: CHURN INTELLIGENCE
# ===========================================================================
elif page == "churn":
    st.markdown('<div class="dashboard-title">⚠️ Churn Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Machine Learning churn risk assessment and high-risk customer identification</div>', unsafe_allow_html=True)

    if not require_pipeline():
        st.stop()

    churn = load_output("churn_predictions")
    fi = load_output("churn_feature_importance")

    if churn is None:
        st.warning("Churn predictions not available. Run the pipeline with sufficient data.")
        st.stop()

    # KPI summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(kpi_card("Total Scored Customers", f"{len(churn):,}"), unsafe_allow_html=True)
    with col2:
        high_risk = (churn["risk_level"] == "High Risk").sum()
        st.markdown(kpi_card("High Risk 🔴", f"{high_risk:,}", "Action Required"), unsafe_allow_html=True)
    with col3:
        med_risk = (churn["risk_level"] == "Medium Risk").sum()
        st.markdown(kpi_card("Medium Risk 🟡", f"{med_risk:,}"), unsafe_allow_html=True)
    with col4:
        low_risk = (churn["risk_level"] == "Low Risk").sum()
        st.markdown(kpi_card("Low Risk 🟢", f"{low_risk:,}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Risk Level Breakdown</div>', unsafe_allow_html=True)
        risk_counts = churn["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]
        colour_map = {"High Risk": "#ff6b6b", "Medium Risk": "#ffd93d", "Low Risk": "#64ffda"}
        fig = px.bar(
            risk_counts, x="risk_level", y="count",
            color="risk_level", color_discrete_map=colour_map,
            labels={"risk_level": "Risk Level", "count": "Customers"},
            text_auto=True,
        )
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Churn Probability Density</div>', unsafe_allow_html=True)
        fig = px.histogram(
            churn, x="churn_probability", nbins=40,
            color_discrete_sequence=["#7c83fd"],
            labels={"churn_probability": "Churn Probability"},
        )
        fig = apply_theme(fig)
        fig.add_vline(x=0.30, line_dash="dash", line_color="#ffd93d", annotation_text="Medium (30%)")
        fig.add_vline(x=0.60, line_dash="dash", line_color="#ff6b6b", annotation_text="High (60%)")
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    if fi is not None:
        st.markdown('<div class="section-header">Feature Importance (Random Forest Model)</div>', unsafe_allow_html=True)
        fi_sorted = fi.sort_values("importance", ascending=True).tail(10)
        fig = px.bar(
            fi_sorted, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Teal",
            labels={"importance": "Importance Score", "feature": "Feature"},
            text_auto=".3f",
        )
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # High-risk table
    st.markdown('<div class="section-header">High-Risk Customers Action List</div>', unsafe_allow_html=True)
    high_risk_df = churn[churn["risk_level"] == "High Risk"].sort_values(
        "churn_probability", ascending=False
    )
    st.dataframe(
        high_risk_df[["customer_id", "churn_probability", "risk_level"]].head(100),
        use_container_width=True,
        hide_index=True,
    )


# ===========================================================================
# PAGE 5: RECOMMENDATIONS
# ===========================================================================
elif page == "recommendations":
    st.markdown('<div class="dashboard-title">💡 Product Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Collaborative filtering recommendations backed by explainable segment affinity</div>', unsafe_allow_html=True)

    if not require_pipeline():
        st.stop()

    recs = load_output("recommendations")
    rfm = load_output("customer_rfm")

    if recs is None or rfm is None:
        st.warning("Recommendation data not available. Run the pipeline first.")
        st.stop()

    # Customer selector
    customer_ids = sorted(rfm["customer_id"].tolist())
    sel_cust = st.selectbox(
        "Select Customer",
        options=customer_ids,
        format_func=lambda x: f"Customer #{x}",
    )

    if sel_cust:
        cust_info = rfm[rfm["customer_id"] == sel_cust]
        if len(cust_info) > 0:
            ci = cust_info.iloc[0]
            st.info(
                f"**Segment**: {ci.get('segment', 'Unknown')} | "
                f"**Recency**: {ci.get('recency', 0):.0f} days | "
                f"**Frequency**: {ci.get('frequency', 0)} orders | "
                f"**Monetary Spend**: ${ci.get('monetary', 0):,.2f}"
            )

        cust_recs = recs[recs["customer_id"] == sel_cust].head(10)

        if len(cust_recs) > 0:
            st.markdown(f"### Recommended Products for Customer #{sel_cust}")
            for i, (_, rec) in enumerate(cust_recs.iterrows(), 1):
                with st.container():
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        pname = rec.get("product_name", rec.get("name", "Product"))
                        st.markdown(f"**#{i} — {pname}**")
                        st.caption(f"Category: {rec.get('category', 'N/A')} | Brand: {rec.get('brand', 'N/A')}")
                        explanation = rec.get("explanation", "")
                        if explanation:
                            st.markdown(f"> 💬 *{explanation}*")
                    with c2:
                        score = rec.get("affinity_score", 0)
                        st.metric("Affinity", f"{score:.0%}")
                    with c3:
                        st.metric("Rank", f"#{int(rec.get('rank', i))}")
                    st.divider()

    st.markdown("---")
    st.markdown('<div class="section-header">Top Categories by Recommendation Volume</div>', unsafe_allow_html=True)
    cat_counts = recs["category"].value_counts().reset_index().head(15)
    cat_counts.columns = ["category", "count"]
    fig = px.bar(
        cat_counts, x="count", y="category", orientation="h",
        color="count", color_continuous_scale="Teal",
        labels={"count": "Recommendation Count", "category": "Category"},
        text_auto=True,
    )
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 6: PRODUCT & CATEGORY ANALYTICS
# ===========================================================================
elif page == "products":
    st.markdown('<div class="dashboard-title">📦 Product & Category Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Revenue, order volume, and customer reach breakdown across product categories</div>', unsafe_allow_html=True)

    if not require_pipeline():
        st.stop()

    cat = load_output("category_analytics")

    if cat is None:
        st.warning("Category analytics not available. Run the pipeline first.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(kpi_card("Total Categories", f"{len(cat):,}"), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_card("Total Category Revenue", f"${cat['total_revenue'].sum():,.0f}"), unsafe_allow_html=True)
    with col3:
        st.markdown(kpi_card("Total Order Items", f"{cat['order_count'].sum():,}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Revenue by Category (Top 15)</div>', unsafe_allow_html=True)
        top_cat = cat.head(15)
        fig = px.bar(
            top_cat.sort_values("total_revenue", ascending=True),
            x="total_revenue", y="category", orientation="h",
            color="total_revenue", color_continuous_scale="Teal",
            labels={"total_revenue": "Revenue ($)", "category": "Category"},
        )
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Unique Customers by Category (Top 15)</div>', unsafe_allow_html=True)
        fig = px.bar(
            top_cat.sort_values("unique_customers", ascending=True),
            x="unique_customers", y="category", orientation="h",
            color="unique_customers", color_continuous_scale="Blues",
            labels={"unique_customers": "Unique Customers", "category": "Category"},
        )
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Full table
    st.markdown('<div class="section-header">Full Category Performance Matrix</div>', unsafe_allow_html=True)
    st.dataframe(
        cat.rename(columns={
            "category": "Category",
            "total_revenue": "Revenue ($)",
            "order_count": "Orders",
            "unique_customers": "Customers",
            "avg_selling_price": "Avg Price ($)",
            "product_count": "Products",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ===========================================================================
# PAGE 7: COHORT & RETENTION
# ===========================================================================
elif page == "cohort":
    st.markdown('<div class="dashboard-title">📅 Cohort & Retention Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Monthly customer acquisition cohort retention tracking</div>', unsafe_allow_html=True)

    if not require_pipeline():
        st.stop()

    cohort = load_output("cohort_retention")
    kpis = load_output("monthly_kpis")

    if cohort is None:
        st.warning("Cohort data not available. Run the pipeline first.")
        st.stop()

    # Prepare cohort matrix
    if "cohort_month" in cohort.columns:
        cohort_matrix = cohort.set_index("cohort_month")
    else:
        cohort_matrix = cohort.copy()

    numeric_cols = [c for c in cohort_matrix.columns if str(c).lstrip('-').isdigit()]
    cohort_matrix = cohort_matrix[numeric_cols]
    cohort_matrix.columns = [int(c) for c in cohort_matrix.columns]

    max_months = min(12, len(cohort_matrix.columns))
    cohort_matrix = cohort_matrix.iloc[:, :max_months]

    st.markdown('<div class="section-header">Cohort Retention Heatmap (%)</div>', unsafe_allow_html=True)
    fig = px.imshow(
        cohort_matrix * 100,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="Teal",
        labels={"x": "Months Since First Purchase", "y": "Acquisition Cohort", "color": "Retention %"},
        zmin=0, zmax=100,
    )
    fig = apply_theme(fig)
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    if kpis is not None and "new_customers" in kpis.columns:
        st.markdown('<div class="section-header">Customer Acquisition Velocity</div>', unsafe_allow_html=True)
        fig = px.line(
            kpis, x="year_month", y="new_customers",
            labels={"year_month": "Month", "new_customers": "New Customers"},
            color_discrete_sequence=["#64ffda"],
            markers=True,
        )
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)