# Olist E-Commerce Analytics Dashboard

A Streamlit dashboard built from the DSA Group 9 capstone project analyzing the
Brazilian Olist e-commerce dataset. Upload the raw Olist CSVs and get an instant
dashboard covering revenue, customers, delivery performance, and a late-delivery
risk model.

## What it does

- **SQL marts** — DuckDB queries compute monthly revenue, category revenue,
  customer spend quartiles, and review-score distribution by delivery status.
- **Dashboard** — KPI summary (revenue, orders, late-delivery rate, avg review
  score) plus charts for each mart above.
- **Late-delivery risk model** — a class-balanced logistic regression predicting
  whether an order will arrive late, compared against a baseline model.
- **Auto-generated summary** — a plain-language recommendation section that
  updates based on whatever data is uploaded.

## Data

Upload the CSVs from the
[Olist Brazilian E-Commerce dataset on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce):

- `olist_customers_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `product_category_name_translation.csv`

(`olist_geolocation_dataset.csv` is not required.)

Select all files at once in the sidebar uploader. Nothing is stored — everything
runs in memory for the session.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`) and
upload the CSVs.

## Live app

Deployed on Streamlit Community Cloud: *(add your app's URL here once deployed)*

## Tech

Python, Streamlit, DuckDB (SQL marts), scikit-learn (logistic regression),
pandas.
