# Databricks notebook source
# MAGIC %md
# MAGIC # Feature engineering: fraud detection
# MAGIC
# MAGIC **Session:** Model Development & Experimentation
# MAGIC
# MAGIC Real-time scoring has to work for a brand-new transaction that isn't in any table
# MAGIC yet, so features are split the way production systems do:
# MAGIC
# MAGIC 1. **Entity features:** slow-changing card and client attributes, looked up by key at
# MAGIC    serving time. Published to the Unity Catalog Feature Store:
# MAGIC    - `dev.<you>_fraud.card_features` keyed by `card_id` (credit limit, cards issued)
# MAGIC    - `dev.<you>_fraud.client_features` keyed by `client_id` (credit score, income, age)
# MAGIC 2. **On-demand features:** transaction-derived signals (night, online, high-risk MCC,
# MAGIC    amount ratios) computed at request time by UC feature functions, so nothing needs
# MAGIC    the transaction to pre-exist.
# MAGIC
# MAGIC At serving the caller sends only the raw transaction (`card_id`, `client_id`, `amount`,
# MAGIC `transaction_hour`, `mcc`, `use_chip`); the endpoint looks up entity features and runs
# MAGIC the functions. Same definitions at train and serve, so no train/serve skew.
# MAGIC
# MAGIC Docs: [Feature Engineering in Unity Catalog](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/)
# MAGIC | [On-demand features](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/on-demand-features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC Install the Feature Engineering client. On serverless this package is not
# MAGIC pre-installed, so we add it and restart Python. (No-op on ML Runtime clusters.)

# COMMAND ----------

# MAGIC %pip install -q databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog_name", "dev")
dbutils.widgets.text("gold_schema", "fraud_gold")
dbutils.widgets.text("ml_schema", "")

catalog_name = dbutils.widgets.get("catalog_name")
gold_schema = dbutils.widgets.get("gold_schema")
ml_schema = dbutils.widgets.get("ml_schema")

from pyspark.sql import functions as F

# Interactive fallback: jobs pass the resolved personal schema; running standalone derives
# the same per-user name the bundle uses (dev_<short>_fraud) so runs stay isolated.
if not ml_schema:
    _user = spark.range(1).select(F.current_user()).first()[0]
    _short = "".join(c if c.isalnum() else "_" for c in _user.split("@")[0])
    ml_schema = f"dev_{_short}_fraud_ml"

# Read from the shared gold source; write to the personal schema.
source_table = f"{catalog_name}.{gold_schema}.transactions_enriched"
card_feature_table = f"{catalog_name}.{ml_schema}.card_features"
client_feature_table = f"{catalog_name}.{ml_schema}.client_features"

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F

fe = FeatureEngineeringClient()

enriched = spark.read.table(source_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build the entity feature tables
# MAGIC
# MAGIC Slow-changing card and client attributes that exist before a transaction happens, so
# MAGIC they can be looked up by key at serve time:
# MAGIC
# MAGIC - `card_features`, one row per `card_id`: credit limit, cards issued.
# MAGIC - `client_features`, one row per `client_id`: credit score, income, age.

# COMMAND ----------

# One row per entity (deduplicated), keyed by card_id / client_id. This is plain Spark
# aggregation and is provided; publishing these to the Feature Store below is the exercise.
# For simplicity the card/client attributes come from the single transactions_enriched
# table, taking one value per entity with F.first. In production they would come from the
# entity dimension itself (the cards / users tables), because these attribute values can
# change over time, so a per-transaction snapshot is arbitrary and the dimension is the
# source of truth.
card_features = enriched.groupBy("card_id").agg(
    F.first("credit_limit", ignorenulls=True).cast("double").alias("credit_limit"),
    F.first("num_cards_issued", ignorenulls=True).cast("int").alias("num_cards_issued"),
)

client_features = enriched.groupBy("client_id").agg(
    F.first("credit_score", ignorenulls=True).cast("int").alias("credit_score"),
    F.first("yearly_income", ignorenulls=True).cast("double").alias("yearly_income"),
    F.first("current_age", ignorenulls=True).cast("int").alias("current_age"),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Publish the entity feature tables
# MAGIC
# MAGIC Write them to the Feature Store as primary-key tables so they can be resolved by key:
# MAGIC
# MAGIC - at training time (offline join), and
# MAGIC - at serving time, once published to an online store.
# MAGIC
# MAGIC Create on the first run, merge on re-runs. Change Data Feed is enabled so the tables can
# MAGIC sync to an online store later.

# COMMAND ----------


# TODO: create the feature tables (first run) or merge into them (re-runs)
# HINT: use the FeatureEngineeringClient `fe`, primary_keys=["card_id"] / ["client_id"].
# HINT: fe.create_table(name=..., primary_keys=..., df=..., description=...) creates it.
# HINT: if it already exists, fe.write_table(name=..., df=..., mode="merge").
# HINT: enable Change Data Feed (delta.enableChangeDataFeed=true) so it can publish online.
# <-- Your code here

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Register the on-demand feature functions
# MAGIC
# MAGIC UC feature functions (Python UDFs) that compute transaction-derived features from the
# MAGIC request payload itself, so:
# MAGIC
# MAGIC - they work for a transaction never seen before (nothing to look up);
# MAGIC - training and serving call the same functions, so there is no skew.

# COMMAND ----------

# On-demand feature functions are UC Python UDFs. Training and serving call the same
# functions, so there is no train/serve skew. Four are provided; the first (ff_is_night) is
# the exercise.
schema_fqn = f"{catalog_name}.{ml_schema}"

# TODO: register the ff_is_night on-demand feature function as a UC Python UDF
# HINT: CREATE OR REPLACE FUNCTION {schema_fqn}.ff_is_night(transaction_hour INT) RETURNS BOOLEAN
# HINT:   LANGUAGE PYTHON AS $$ <python body that returns a value> $$ (use spark.sql(f"""...""")).
# HINT: is_night(hour) -> hour is not None and (hour < 6 or hour >= 22).
# <-- Your code here

# The remaining on-demand functions (provided).
spark.sql(f"""
CREATE OR REPLACE FUNCTION {schema_fqn}.ff_is_online(use_chip STRING)
RETURNS BOOLEAN
LANGUAGE PYTHON
COMMENT 'On-demand: True for a card-not-present (online) transaction.'
AS $$
return use_chip == 'ONLINE TRANSACTION'
$$
""")

spark.sql(f"""
CREATE OR REPLACE FUNCTION {schema_fqn}.ff_is_high_risk_mcc(mcc BIGINT)
RETURNS BOOLEAN
LANGUAGE PYTHON
COMMENT 'On-demand: True when the merchant category code is in the high-risk set.'
AS $$
return mcc in (4829, 6051, 7995, 5967, 6011, 7801, 7802, 7994)
$$
""")

spark.sql(f"""
CREATE OR REPLACE FUNCTION {schema_fqn}.ff_amount_to_income_ratio(amount DOUBLE, yearly_income DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON
COMMENT 'On-demand: abs(amount) / yearly income (null if income <= 0).'
AS $$
if amount is None or yearly_income is None or yearly_income <= 0:
    return None
return abs(amount) / yearly_income
$$
""")

spark.sql(f"""
CREATE OR REPLACE FUNCTION {schema_fqn}.ff_amount_to_credit_limit_ratio(amount DOUBLE, credit_limit DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON
COMMENT 'On-demand: abs(amount) / credit limit (null if limit <= 0).'
AS $$
if amount is None or credit_limit is None or credit_limit <= 0:
    return None
return abs(amount) / credit_limit
$$
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Document the feature tables (governance)
# MAGIC
# MAGIC Add column comments so the tables are self-describing in Catalog Explorer.
# MAGIC Instructor-provided, not part of the exercise.

# COMMAND ----------

CARD_COMMENTS = {
    "card_id": "Primary key: unique card identifier.",
    "credit_limit": "Card credit limit in USD.",
    "num_cards_issued": "Number of cards issued on the account.",
}
CLIENT_COMMENTS = {
    "client_id": "Primary key: unique client (cardholder) identifier.",
    "credit_score": "Cardholder credit score.",
    "yearly_income": "Cardholder yearly income in USD.",
    "current_age": "Cardholder current age in years.",
}
for column, text in CARD_COMMENTS.items():
    spark.sql(f"ALTER TABLE {card_feature_table} ALTER COLUMN {column} COMMENT '{text}'")
for column, text in CLIENT_COMMENTS.items():
    spark.sql(f"ALTER TABLE {client_feature_table} ALTER COLUMN {column} COMMENT '{text}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production considerations
# MAGIC
# MAGIC This notebook makes two deliberate simplifications:
# MAGIC
# MAGIC - **Point-in-time correctness.** The feature tables have a primary key only (no
# MAGIC   `timestamp_keys`), and the training-set `FeatureLookup`s set no `timestamp_lookup_key`,
# MAGIC   so each transaction joins the entity's current attribute values rather than the values
# MAGIC   as of the transaction time. For attributes that change over time (for example a credit
# MAGIC   score), this risks label leakage. In production, make the feature tables time-series
# MAGIC   tables (add a feature timestamp and `timestamp_keys=[...]`), carry `transaction_ts` on
# MAGIC   the training spine, and set `timestamp_lookup_key="transaction_ts"` so each row joins the
# MAGIC   value effective at its own time.
# MAGIC - **Source of the entity attributes.** Here the card/client attributes are taken from the
# MAGIC   flattened `transactions_enriched` table with `F.first`. In production, source them from
# MAGIC   the entity dimension of record (the `cards` / `users` tables), which is authoritative,
# MAGIC   carries history, and supports the point-in-time joins above.

display(spark.read.table(card_feature_table).limit(10))
display(spark.read.table(client_feature_table).limit(10))
