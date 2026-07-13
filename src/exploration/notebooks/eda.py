# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory data analysis: fraud transactions
# MAGIC
# MAGIC A quick look at the data before the ML labs, so you understand what you are modelling.
# MAGIC We read the shared gold table `transactions_enriched` (transaction, card, user and
# MAGIC merchant-category attributes already joined) and answer four questions:
# MAGIC
# MAGIC 1. How much data is there and what does a row look like?
# MAGIC 2. How rare is fraud? (the class imbalance that drives everything later)
# MAGIC 3. What do the numeric features look like?
# MAGIC 4. Does fraud behave differently (by amount, time of day, category)?
# MAGIC
# MAGIC This notebook only reads data. It writes nothing, so it is safe to run any time.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "adoption_workshop")
dbutils.widgets.text("gold_schema", "fraud_gold")

catalog_name = dbutils.widgets.get("catalog_name")
gold_schema = dbutils.widgets.get("gold_schema")

source_table = f"{catalog_name}.{gold_schema}.transactions_enriched"
print(f"Reading from: {source_table}")

from pyspark.sql import functions as F

df = spark.read.table(source_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Shape and schema
# MAGIC How many rows and columns, and what is the grain (one row per transaction).

# COMMAND ----------

print(f"Rows:    {df.count():,}")
print(f"Columns: {len(df.columns)}")

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC A few sample rows to get a feel for the values.

# COMMAND ----------

display(df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. How rare is fraud?
# MAGIC The target `is_fraud` is highly imbalanced: fraud is a fraction of a percent of all
# MAGIC transactions. This is why later labs balance the training data and score on ROC-AUC,
# MAGIC precision and recall instead of accuracy.

# COMMAND ----------

label_counts = (
    df.groupBy("is_fraud")
    .agg(F.count("*").alias("n"))
    .withColumn("pct", F.round(100 * F.col("n") / df.count(), 3))
    .orderBy("is_fraud")
)
display(label_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Numeric feature summary
# MAGIC Basic distribution of the key numeric columns (min, max, mean, stddev).

# COMMAND ----------

numeric_cols = ["amount", "credit_limit", "credit_score", "yearly_income", "current_age"]
present = [c for c in numeric_cols if c in df.columns]
display(df.select(present).describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Does fraud behave differently?
# MAGIC Compare fraud vs non-fraud on a few dimensions. Fraud tends to differ in amount and
# MAGIC time of day, which is the signal the model will pick up.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Average transaction amount by class

# COMMAND ----------

display(
    df.groupBy("is_fraud").agg(
        F.round(F.avg("amount"), 2).alias("avg_amount"),
        F.round(F.expr("percentile_approx(amount, 0.5)"), 2).alias("median_amount"),
        F.round(F.max("amount"), 2).alias("max_amount"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fraud rate by hour of day
# MAGIC If a `transaction_hour` column exists, look at when fraud is more likely.

# COMMAND ----------

if "transaction_hour" in df.columns:
    display(
        df.groupBy("transaction_hour")
        .agg(F.round(100 * F.avg(F.col("is_fraud").cast("double")), 3).alias("fraud_rate_pct"))
        .orderBy("transaction_hour")
    )
else:
    print("No transaction_hour column in this table; skipping.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top merchant categories by fraud rate
# MAGIC Which merchant categories (MCC descriptions) see the most fraud, among categories with
# MAGIC enough volume to be meaningful.

# COMMAND ----------

category_col = next((c for c in ["mcc_description", "mcc", "merchant_category"] if c in df.columns), None)
if category_col:
    display(
        df.groupBy(category_col)
        .agg(
            F.count("*").alias("n"),
            F.round(100 * F.avg(F.col("is_fraud").cast("double")), 3).alias("fraud_rate_pct"),
        )
        .filter(F.col("n") >= 1000)
        .orderBy(F.desc("fraud_rate_pct"))
        .limit(15)
    )
else:
    print("No merchant-category column in this table; skipping.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Takeaways
# MAGIC
# MAGIC - One row per transaction, joined with card, client and merchant-category attributes.
# MAGIC - `is_fraud` is heavily imbalanced, so accuracy alone is misleading.
# MAGIC - Fraud differs by amount, time of day and merchant category, which is the signal the
# MAGIC   feature engineering and training labs build on.
