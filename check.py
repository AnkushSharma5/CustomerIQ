import pandas as pd

df = pd.read_parquet("data/output/customer_segments.parquet")
print(df.columns.tolist())
if "f_score" in df.columns:
    print(df["f_score"].value_counts().sort_index())
print(df["frequency"].value_counts(normalize=True).head())
print(df["segment"].value_counts())

print(pd.crosstab(df["frequency"], df["f_score"]))
print(df["r_score"].value_counts().sort_index())
print(df["m_score"].value_counts().sort_index())
print(df[df["segment"] == "Other"][["r_score", "f_score"]].value_counts())