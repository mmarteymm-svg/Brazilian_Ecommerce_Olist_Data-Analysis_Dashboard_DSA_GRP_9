"""
Olist E-Commerce Analytics — Streamlit App (simplified)
Built from the DSA Group 9 capstone notebook.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Then upload the Olist CSVs from kaggle.com/datasets/olistbr/brazilian-ecommerce
"""

import re
import os
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

# Bundled dataset location: put the 8 Olist CSVs (everything except
# olist_geolocation_dataset.csv, which the app doesn't use) in a `data/`
# folder next to this script and they'll load automatically — no upload
# needed. The sidebar uploader still works and overrides these for the
# session if the user uploads files there instead.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BUNDLED_FILENAMES = [
    "olist_customers_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def get_bundled_file_records():
    """Read whichever bundled CSVs exist in DATA_DIR. Returns a tuple of
    (filename, bytes), matching the shape st.file_uploader results are
    converted to, so both paths can feed the same load_data()."""
    records = []
    for filename in BUNDLED_FILENAMES:
        path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                records.append((filename, f.read()))
    return tuple(records)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def table_name_from_filename(filename: str) -> str:
    base = filename.rsplit(".", 1)[0]
    return re.sub(r"olist_|_dataset", "", base)


@st.cache_resource(show_spinner=False)
def load_data(file_records):
    """file_records: tuple of (filename, bytes). Writes to disk, loads into DuckDB,
    and force-casts any date/timestamp-named column so downstream SQL never hits a VARCHAR."""
    tmp_dir = tempfile.mkdtemp(prefix="olist_")
    con = duckdb.connect(database=":memory:")

    for filename, raw_bytes in file_records:
        name = table_name_from_filename(filename)
        path = os.path.join(tmp_dir, filename)
        with open(path, "wb") as f:
            f.write(raw_bytes)
        con.execute(f"CREATE OR REPLACE TABLE \"{name}\" AS SELECT * FROM read_csv_auto('{path}')")

        cols = con.execute(f'DESCRIBE "{name}"').df()
        for _, row in cols.iterrows():
            col, dtype = row["column_name"], row["column_type"]
            if ("date" in col.lower() or "timestamp" in col.lower()) and dtype not in (
                "TIMESTAMP", "DATE", "TIMESTAMP WITH TIME ZONE"
            ):
                con.execute(
                    f'ALTER TABLE "{name}" ALTER COLUMN "{col}" '
                    f'TYPE TIMESTAMP USING TRY_CAST("{col}" AS TIMESTAMP)'
                )
    return con


# ---------------------------------------------------------------------------
# SQL marts
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_marts(_con, fingerprint):
    con = _con

    monthly_revenue = con.sql("""
        SELECT date_trunc('month', o.order_purchase_timestamp) AS month,
               SUM(oi.price + oi.freight_value) AS revenue
        FROM orders o JOIN order_items oi USING (order_id)
        GROUP BY month ORDER BY month
    """).df()

    category_revenue = con.sql("""
        SELECT p.product_category_name, SUM(oi.price + oi.freight_value) AS revenue
        FROM order_items oi JOIN products p ON oi.product_id = p.product_id
        GROUP BY p.product_category_name ORDER BY revenue DESC
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
    """).df()

    return monthly_revenue, category_revenue, quartile_summary, review_distribution, ml_data


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

    # Stratified split needs at least 2 members per class; fall back to a plain
    # split if the minority class is too small for that.
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

bundled_records = get_bundled_file_records()

with st.sidebar:
    st.header("Data")
    if bundled_records:
        st.success(f"✅ {len(bundled_records)}/{len(BUNDLED_FILENAMES)} bundled CSVs found in `data/`")
    else:
        st.warning("No bundled CSVs found in `data/`")
    st.caption(
        "Optional: upload CSVs below to override the bundled data for this "
        "session (e.g. to test a refreshed export). "
        "Source: kaggle.com/datasets/olistbr/brazilian-ecommerce"
    )
    uploaded = st.file_uploader("Select CSVs to override bundled data", type="csv", accept_multiple_files=True)

if uploaded:
    file_records = tuple((f.name, f.getvalue()) for f in uploaded)
    st.caption("📤 Using uploaded files for this session (overriding bundled data).")
elif bundled_records:
    file_records = bundled_records
    st.caption("📁 Using bundled data from the app's `data/` folder.")
else:
    file_records = ()

if not file_records:
    st.info("👈 No data available yet — upload the Olist CSVs to get started, or add them to the app's `data/` folder.")
    st.stop()

con = load_data(file_records)

table_names = con.sql("SHOW TABLES").df()["name"].tolist()
missing = [t for t in REQUIRED_TABLES if t not in table_names]
if missing:
    st.error(f"Missing required table(s): {', '.join(missing)}")
    st.stop()

fingerprint = tuple(sorted(table_names))
monthly_revenue, category_revenue, quartile_summary, review_distribution, ml_data = compute_marts(con, fingerprint)
model = train_model(ml_data, fingerprint)

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
st.bar_chart(monthly_revenue.set_index("month")["revenue"])

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🏆 Top Categories by Revenue")
    top_cat = category_revenue.dropna(subset=["product_category_name"]).head(10)
    st.bar_chart(top_cat.set_index("product_category_name")["revenue"])

with col_b:
    st.subheader("👥 Customer Spend by Tier")
    tier_avg = quartile_summary.groupby("quartile")["total_spent"].mean()
    tier_avg.index = "Tier " + tier_avg.index.astype(str)
    st.bar_chart(tier_avg)
    st.caption("Tier 1 = top 25% of customers by total spend.")

st.subheader("🚚 Review Score by Delivery Status")
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
        f"({model['error'].lower()}). Try uploading the full dataset for a complete picture."
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

**Recommended actions:** (1) launch a Tier-1 retention offer, (2) deploy the balanced model as a live
risk flag rather than the accuracy-optimized one, (3) review fulfillment for the highest-revenue categories.
""")
