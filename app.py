"""
Olist E-Commerce Analytics — Streamlit App
Built from the DSA Group 9 capstone notebook.

Data can come from either:
  1. A bundled `data/` folder next to this file (auto-loaded, no upload needed), or
  2. CSVs uploaded via the sidebar (overrides bundled data for that session).

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import re
import tempfile

import duckdb
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Olist E-Commerce Analytics", page_icon="🛒", layout="wide")

REQUIRED_TABLES = [
    "customers", "orders", "order_items", "order_payments",
    "order_reviews", "products", "sellers", "product_category_name_translation",
]

# Known date/timestamp columns per table. Parsed explicitly via pandas
# (rather than relying on DuckDB's auto type-detection, which can silently
# leave a column as VARCHAR/NULL on real-world data with blank values or
# large row counts) so every downstream date comparison is reliable.
DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
    "order_items": ["shipping_limit_date"],
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def table_name_from_filename(filename: str) -> str:
    base = filename.rsplit(".", 1)[0]
    return re.sub(r"olist_|_dataset", "", base)


def find_bundled_csvs() -> dict:
    """Return {table_name: path} for CSVs found in the bundled data/ folder."""
    if not os.path.isdir(DATA_DIR):
        return {}
    paths = {}
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith(".csv"):
            paths[table_name_from_filename(fname)] = os.path.join(DATA_DIR, fname)
    return paths


def load_table(con: duckdb.DuckDBPyConnection, name: str, path: str) -> None:
    """Load one CSV into DuckDB as table `name`.

    For tables with known date columns, reads with pandas and explicitly
    converts those columns with pd.to_datetime(..., format="mixed",
    errors="coerce"). We do NOT rely on read_csv's parse_dates= — if even one
    row has a differently-formatted date (e.g. "2017/10/04" mixed in with
    "2017-10-02"), pandas silently leaves the WHOLE column as plain text with
    no error or warning. Explicit post-hoc to_datetime with format="mixed"
    parses each value independently and is far more robust to messy real data.

    As a final safety net, after loading we verify every date column actually
    ended up as TIMESTAMP in DuckDB, and force-cast with TRY_CAST if not —
    so a date-parsing quirk can degrade (some NULLs) but can never crash the
    app with a BinderException again.
    """
    date_cols = DATE_COLUMNS.get(name, [])
    if date_cols:
        df = pd.read_csv(path)
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
        con.register("_stage", df)
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _stage')
        con.unregister("_stage")
    else:
        con.execute(f"CREATE OR REPLACE TABLE \"{name}\" AS SELECT * FROM read_csv_auto('{path}')")

    # Safety net: force any remaining date/timestamp-named column that isn't
    # actually TIMESTAMP/DATE typed in DuckDB to be cast, so downstream SQL
    # (date_trunc, comparisons, DATE_DIFF) can never hit a type error.
    cols = con.execute(f'DESCRIBE "{name}"').df()
    for _, row in cols.iterrows():
        col, dtype = row["column_name"], row["column_type"]
        looks_like_date = "date" in col.lower() or "timestamp" in col.lower()
        already_typed = dtype in ("TIMESTAMP", "DATE", "TIMESTAMP WITH TIME ZONE")
        if looks_like_date and not already_typed:
            con.execute(
                f'ALTER TABLE "{name}" ALTER COLUMN "{col}" '
                f'TYPE TIMESTAMP USING TRY_CAST("{col}" AS TIMESTAMP)'
            )


@st.cache_resource(show_spinner=False)
def load_data(table_paths: tuple) -> duckdb.DuckDBPyConnection:
    """table_paths: tuple of (table_name, path) pairs — hashable, so this is cached."""
    con = duckdb.connect(database=":memory:")
    for name, path in table_paths:
        load_table(con, name, path)
    return con


@st.cache_resource(show_spinner=False)
def stage_uploaded_csvs(file_records: tuple) -> dict:
    """file_records: tuple of (filename, bytes). Writes each upload to a temp
    dir on disk and returns {table_name: path}."""
    tmp_dir = tempfile.mkdtemp(prefix="olist_")
    paths = {}
    for filename, raw_bytes in file_records:
        name = table_name_from_filename(filename)
        path = os.path.join(tmp_dir, filename)
        with open(path, "wb") as f:
            f.write(raw_bytes)
        paths[name] = path
    return paths


# ---------------------------------------------------------------------------
# SQL marts
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_marts(_con, fingerprint):
    con = _con

    monthly_revenue = con.sql("""
        WITH monthly AS (
            SELECT date_trunc('month', o.order_purchase_timestamp) AS month,
                   SUM(oi.price + oi.freight_value) AS revenue,
                   COUNT(DISTINCT o.order_id) AS order_count
            FROM orders o JOIN order_items oi USING (order_id)
            WHERE o.order_purchase_timestamp IS NOT NULL
            GROUP BY month
        )
        SELECT
            month, revenue, order_count,
            SUM(revenue) OVER (ORDER BY month) AS running_total,
            CASE
                -- Olist's first few months (2016) have only a handful of pilot
                -- orders; a % change off a near-zero base produces a spike of
                -- tens of thousands of percent that swamps every other bar.
                -- Only compute growth % once both months have real volume.
                WHEN order_count >= 10 AND LAG(order_count) OVER (ORDER BY month) >= 10
                THEN ROUND(
                    100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
                    / NULLIF(LAG(revenue) OVER (ORDER BY month), 0), 2
                )
                ELSE NULL
            END AS mom_growth_pct
        FROM monthly
        ORDER BY month
    """).df()

    category_revenue = con.sql("""
        SELECT p.product_category_name, SUM(oi.price + oi.freight_value) AS revenue
        FROM order_items oi JOIN products p ON oi.product_id = p.product_id
        GROUP BY p.product_category_name ORDER BY revenue DESC
    """).df()

    top_sellers = con.sql("""
        SELECT
            p.product_category_name,
            oi.seller_id,
            SUM(oi.price + oi.freight_value) AS total_revenue,
            RANK() OVER (
                PARTITION BY p.product_category_name ORDER BY SUM(oi.price + oi.freight_value) DESC
            ) AS seller_rank
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        GROUP BY p.product_category_name, oi.seller_id
        ORDER BY p.product_category_name, seller_rank
    """).df()

    quartile_summary = con.sql("""
        WITH customer_spending AS (
            SELECT c.customer_unique_id, SUM(oi.price + oi.freight_value) AS total_spent
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            JOIN order_items oi ON o.order_id = oi.order_id
            GROUP BY c.customer_unique_id
        )
        SELECT NTILE(4) OVER (ORDER BY total_spent DESC) AS quartile, total_spent
        FROM customer_spending
    """).df()

    review_distribution = con.sql("""
        SELECT
            CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
                 THEN 'Late' ELSE 'On Time' END AS delivery_status,
            r.review_score, COUNT(*) AS reviews
        FROM orders o JOIN order_reviews r USING (order_id)
        WHERE o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL
        GROUP BY delivery_status, r.review_score
        ORDER BY delivery_status, r.review_score
    """).df()

    ml_data = con.sql("""
        SELECT
            oi.freight_value, p.product_weight_g,
            p.product_length_cm, p.product_height_cm, p.product_width_cm,
            s.seller_zip_code_prefix, c.customer_zip_code_prefix,
            DATE_DIFF('day', o.order_purchase_timestamp, o.order_estimated_delivery_date) AS estimated_delivery_days,
            CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END AS is_late
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN sellers s ON oi.seller_id = s.seller_id
        JOIN products p ON oi.product_id = p.product_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL
    """).df()

    # Diagnostics: null counts on the date columns that drive the charts/model,
    # so a data problem is visible instead of just producing an empty chart.
    diagnostics = con.sql("""
        SELECT
            COUNT(*) AS total_orders,
            COUNT(order_purchase_timestamp) AS non_null_purchase_ts,
            COUNT(order_delivered_customer_date) AS non_null_delivered,
            COUNT(order_estimated_delivery_date) AS non_null_estimated
        FROM orders
    """).df().iloc[0].to_dict()

    return monthly_revenue, category_revenue, top_sellers, quartile_summary, review_distribution, ml_data, diagnostics


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def train_model(ml_data: pd.DataFrame, fingerprint):
    ml_data = ml_data.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = ml_data.drop(columns="is_late")
    y = ml_data["is_late"]

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        return {"error": "Only one delivery outcome (all late or all on-time) is present in this data — there's no variation for a model to learn from."}
    if len(y) < 10:
        return {"error": f"Only {len(y)} delivered orders with complete data were found — too few to train a reliable model."}

    can_stratify = class_counts.min() >= 2
    strat = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )
    scaler = StandardScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000, class_weight="balanced")
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

    return {
        "model": model, "scaler": scaler, "features": list(X.columns),
        "accuracy": accuracy_score(y_test, preds),
        "recall_late": recall_score(y_test, preds, zero_division=0),
        "baseline_accuracy": accuracy_score(y_test, baseline.predict(X_test)),
        "late_rate": y.mean(),
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.title("🛒 Olist E-Commerce Analytics")
st.caption("Revenue, customers, delivery, and a late-delivery risk model — DSA Group 9 capstone.")

bundled_paths = find_bundled_csvs()
bundled_ready = all(t in bundled_paths for t in REQUIRED_TABLES)

with st.sidebar:
    st.header("Data")
    if bundled_ready:
        st.success("Using bundled Olist dataset ✅")
        st.caption(
            "Optional: upload CSVs below to override the bundled data for this "
            "session (e.g. to test a refreshed export). Source: "
            "kaggle.com/datasets/olistbr/brazilian-ecommerce"
        )
    else:
        st.warning(f"No bundled CSVs found in `data/`")
        st.caption("Upload the Olist CSVs (kaggle.com/datasets/olistbr/brazilian-ecommerce)")

    uploaded = st.file_uploader(
        "Select CSVs to override bundled data" if bundled_ready else "Select all 8 CSVs at once",
        type="csv", accept_multiple_files=True,
    )

if uploaded:
    file_records = tuple((f.name, f.getvalue()) for f in uploaded)
    upload_paths = stage_uploaded_csvs(file_records)
    table_paths = {**bundled_paths, **upload_paths}  # uploads override bundled
    st.caption("📤 Using uploaded files for this session (overriding bundled data).")
elif bundled_ready:
    table_paths = bundled_paths
else:
    st.info("👈 Upload the Olist CSVs to get started.")
    st.stop()

missing = [t for t in REQUIRED_TABLES if t not in table_paths]
if missing:
    st.error(f"Missing required table(s): {', '.join(missing)}")
    st.stop()

con = load_data(tuple(sorted(table_paths.items())))

table_names = con.sql("SHOW TABLES").df()["name"].tolist()
fingerprint = tuple(sorted(table_paths.items()))
monthly_revenue, category_revenue, top_sellers, quartile_summary, review_distribution, ml_data, diagnostics = compute_marts(con, fingerprint)
model = train_model(ml_data, fingerprint)

# --- Data health check (visible if something looks off) ---
pct_missing_est = 1 - diagnostics["non_null_estimated"] / diagnostics["total_orders"]
pct_missing_delivered = 1 - diagnostics["non_null_delivered"] / diagnostics["total_orders"]
if pct_missing_est > 0.5 or pct_missing_delivered > 0.9 or monthly_revenue.empty:
    with st.expander("⚠️ Data health check — click for details", expanded=True):
        st.write(diagnostics)
        st.write("`ml_data` rows available for the model:", len(ml_data))
        if "is_late" in ml_data.columns:
            st.write("is_late value counts:", ml_data["is_late"].value_counts().to_dict())

# --- KPIs ---
total_revenue = monthly_revenue["revenue"].sum()
total_orders = con.sql("SELECT COUNT(DISTINCT order_id) n FROM orders").df()["n"][0]
avg_review = con.sql("SELECT AVG(review_score) a FROM order_reviews").df()["a"][0]
late_rate_display = f"{model['late_rate']:.1%}" if "error" not in model else "N/A"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"R$ {total_revenue:,.0f}")
c2.metric("Total Orders", f"{total_orders:,}")
c3.metric("Late Delivery Rate", late_rate_display)
c4.metric("Avg Review Score", f"{avg_review:.2f} / 5")

st.divider()

# --- Dashboard ---
st.subheader("📈 Monthly Revenue")
if monthly_revenue.empty:
    st.info("No monthly revenue data available.")
else:
    mr = monthly_revenue.dropna(subset=["month"]).copy()
    mr["month"] = pd.to_datetime(mr["month"]).dt.strftime("%Y-%m")
    st.bar_chart(mr.set_index("month")["revenue"].rename("Monthly Revenue (R$)"))

    rev_col1, rev_col2 = st.columns(2)
    with rev_col1:
        st.caption("Cumulative Revenue Growth")
        st.line_chart(mr.set_index("month")["running_total"].rename("Running Total (R$)"))
    with rev_col2:
        st.caption("Month-over-Month Growth Rate")
        st.bar_chart(mr.set_index("month")["mom_growth_pct"].rename("MoM Growth (%)"))

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🏆 Top Categories by Revenue")
    top_cat = category_revenue.dropna(subset=["product_category_name"]).head(10)
    st.bar_chart(top_cat.set_index("product_category_name")["revenue"])

with col_b:
    st.subheader("👥 Customer Spend by Tier")
    tier_avg = quartile_summary.groupby("quartile")["total_spent"].mean()
    tier_avg.index = "Tier " + tier_avg.index.astype(str)
    st.bar_chart(tier_avg.rename("Average Spend (R$)"))
    st.caption("Tier 1 = top 25% of customers by total spend. Bars show average spend per customer within each tier.")

st.subheader("🏆 Top Sellers by Category")
cats = sorted(top_sellers["product_category_name"].dropna().unique())
chosen_cat = st.selectbox("Category", cats)
top_sellers_display = top_sellers[top_sellers["product_category_name"] == chosen_cat].head(10).copy()
top_sellers_display["label"] = "#" + top_sellers_display["seller_rank"].astype(str) + " " + top_sellers_display["seller_id"].str[:8]
st.bar_chart(top_sellers_display.set_index("label")["total_revenue"].rename("Revenue (R$)"))

st.subheader("🚚 Review Score by Delivery Status")
if review_distribution.empty:
    st.info("No delivery-status review data available.")
else:
    pivot = review_distribution.pivot(index="review_score", columns="delivery_status", values="reviews").fillna(0)
    pivot_pct = (pivot.div(pivot.sum(axis=0), axis=1) * 100).round(1)
    st.bar_chart(pivot_pct)
    st.caption("% of reviews at each score, split by on-time vs late delivery.")

st.divider()

# --- Model ---
st.subheader("🤖 Late-Delivery Risk Model")
if "error" in model:
    st.warning(f"Model not trained: {model['error']}")
else:
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Baseline Accuracy", f"{model['baseline_accuracy']:.1%}")
    mcol2.metric("Model Accuracy", f"{model['accuracy']:.1%}")
    mcol3.metric("Model Recall (late orders)", f"{model['recall_late']:.1%}")
    st.caption(
        "The baseline model (always predicts 'on time') gets high accuracy but 0% recall on late orders — "
        "it's useless as a risk flag. The balanced logistic regression trades some accuracy for much better "
        "recall, catching far more of the orders that are actually late."
    )

st.divider()

# --- Summary ---
st.subheader("📋 Recommendation")
tier1_avg = quartile_summary.groupby('quartile')['total_spent'].mean().iloc[0]

if "error" in model:
    model_summary = (
        "A late-delivery risk model could not be trained on this data "
        f"({model['error'].lower()}). See the data health check above for details."
    )
else:
    model_summary = (
        f"On delivery: **{model['late_rate']:.1%}** of orders arrive late. A plain accuracy-optimized model "
        f"looks good on paper ({model['baseline_accuracy']:.1%} accuracy) but catches essentially no late "
        f"orders. The class-balanced model catches **{model['recall_late']:.1%}** of them instead, at the "
        "cost of some accuracy — the right trade-off for a checkout-time risk flag, since missing a late "
        "order is costlier than a false alarm."
    )

st.markdown(f"""
Total revenue across the uploaded data is **R$ {total_revenue:,.0f}** across **{total_orders:,} orders**.
Customer spend is concentrated at the top: Tier 1 customers (top 25% by spend) average
**R$ {tier1_avg:,.2f}** each — a strong case for a targeted retention program for that segment.

{model_summary}

**Recommended actions:** (1) launch a Tier-1 retention offer- protecting them protects a disproportionate share of revenue, (2) deploy the balanced model as a live
risk flag rather than the accuracy-optimized one, (3) review fulfillment for the highest-revenue categories to give priority support to the top sellers in similar fashion to the customers.
""")
