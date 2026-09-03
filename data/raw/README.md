# theLook eCommerce Dataset

## Overview

CustomerIQ requires the **theLook eCommerce dataset** — a publicly available synthetic e-commerce dataset created by Google and hosted on BigQuery.

Place all required dataset files in this directory (`data/raw/`) before running the pipeline.

---

## Required Files

Place these files here (either `.parquet` or `.csv` format):

| File | Description | Typical Size |
|------|-------------|-------------|
| `users.parquet` / `users.csv` | Customer profiles | ~100K rows |
| `orders.parquet` / `orders.csv` | Order headers | ~125K rows |
| `order_items.parquet` / `order_items.csv` | Individual order line items | ~150K rows |
| `products.parquet` / `products.csv` | Product catalog | ~29K rows |
| `events.parquet` / `events.csv` | Clickstream events | ~2M+ rows |
| `inventory_items.parquet` / `inventory_items.csv` | Inventory records | ~150K rows |
| `distribution_centers.parquet` / `distribution_centers.csv` | Warehouse locations | ~10 rows |

The pipeline auto-detects Parquet (preferred) or CSV format.

---

## How to Obtain the Dataset

### Option 1: Google BigQuery (Official Source)

The official dataset is hosted on BigQuery as a public dataset:

```
bigquery-public-data.thelook_ecommerce
```

**Steps:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Open BigQuery
3. In the Explorer, search for `thelook_ecommerce`
4. Export each required table as CSV or Parquet to Google Cloud Storage
5. Download the files locally

**Required BigQuery tables:**
- `bigquery-public-data.thelook_ecommerce.users`
- `bigquery-public-data.thelook_ecommerce.orders`
- `bigquery-public-data.thelook_ecommerce.order_items`
- `bigquery-public-data.thelook_ecommerce.products`
- `bigquery-public-data.thelook_ecommerce.events`
- `bigquery-public-data.thelook_ecommerce.inventory_items`
- `bigquery-public-data.thelook_ecommerce.distribution_centers`

### Option 2: Kaggle

The dataset may be available on Kaggle:
- Search for "thelook ecommerce" on [kaggle.com/datasets](https://www.kaggle.com/datasets)
- Download and extract the CSV files here

### Option 3: Direct BigQuery Export (Python)

```python
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client(project="YOUR_PROJECT_ID")

tables = [
    "users", "orders", "order_items", "products",
    "events", "inventory_items", "distribution_centers"
]

for table in tables:
    query = f"SELECT * FROM `bigquery-public-data.thelook_ecommerce.{table}`"
    df = client.query(query).to_dataframe()
    df.to_parquet(f"data/raw/{table}.parquet", index=False)
    print(f"Saved {table}: {len(df)} rows")
```

Requires: `pip install google-cloud-bigquery pyarrow`

---

## File Format Notes

- **Parquet is preferred** (faster loading, smaller file size, type-safe)
- CSV is also supported (the loader auto-detects format)
- Do NOT rename the files — the loader expects the exact names above
- Date columns should remain in their original format (ISO 8601)

---

## After Placing the Dataset

Run the pipeline:

```bash
python run_pipeline.py
```

Then start the dashboard:

```bash
streamlit run dashboard/app.py
```

---

## Important Notes

- This is a **synthetic** dataset (not real customer data)
- It is safe to use for portfolio and learning purposes
- Do not commit large data files to Git (they are excluded by `.gitignore`)
