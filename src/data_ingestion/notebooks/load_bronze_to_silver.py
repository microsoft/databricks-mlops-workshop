# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze -> Silver
# MAGIC
# MAGIC Instructor setup notebook, run before the workshop. Not a lab exercise.
# MAGIC
# MAGIC Turns the Bronze tables into conformed Silver tables (and the enriched Gold
# MAGIC transaction table that feature engineering reads) by:
# MAGIC
# MAGIC - cleaning and casting columns to their real types;
# MAGIC - deduplicating to one row per entity;
# MAGIC - joining the sources into a single labelled transaction table.
# MAGIC
# MAGIC | Layer | Location |
# MAGIC |---|---|
# MAGIC | Bronze (tables) | `dev.fraud_bronze.<table>` |
# MAGIC | Silver (tables) | `dev.fraud_silver.<table>` |

# COMMAND ----------

import logging

logger = logging.getLogger("workshop.silver")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

# COMMAND ----------

dbutils.widgets.text("catalog", "adoption_workshop")
dbutils.widgets.text("bronze_schema", "fraud_bronze")
dbutils.widgets.text("silver_schema", "fraud_silver")
dbutils.widgets.text("gold_schema", "fraud_gold")

catalog = dbutils.widgets.get("catalog")
bronze_schema = dbutils.widgets.get("bronze_schema")
silver_schema = dbutils.widgets.get("silver_schema")
gold_schema = dbutils.widgets.get("gold_schema")

# Bronze source tables.
transactions_bz_fqn = f"{catalog}.{bronze_schema}.transactions"
cards_bz_fqn = f"{catalog}.{bronze_schema}.cards"
users_bz_fqn = f"{catalog}.{bronze_schema}.users"
mcc_codes_bz_fqn = f"{catalog}.{bronze_schema}.mcc_codes"
fraud_labels_bz_fqn = f"{catalog}.{bronze_schema}.fraud_labels"

# Silver target tables (cleaned, conformed, normalized).
transactions_sl_fqn = f"{catalog}.{silver_schema}.transactions"
cards_sl_fqn = f"{catalog}.{silver_schema}.cards"
users_sl_fqn = f"{catalog}.{silver_schema}.users"
mcc_codes_sl_fqn = f"{catalog}.{silver_schema}.mcc_codes"
fraud_labels_sl_fqn = f"{catalog}.{silver_schema}.fraud_labels"

# Gold target table (denormalized, labelled, ML-ready consumption table).
transactions_enriched_gd_fqn = f"{catalog}.{gold_schema}.transactions_enriched"

logger.info(
    "Silver namespace: %s.%s | Gold namespace: %s.%s", catalog, silver_schema, catalog, gold_schema
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target schemas
# MAGIC
# MAGIC - The `fraud_silver` / `fraud_gold` schemas are declared as bundle resources (the
# MAGIC   `shared_infra` block in `databricks.yml`).
# MAGIC - `databricks bundle deploy -t dev` creates them before this job runs.

# COMMAND ----------

transactions_bz = spark.table(transactions_bz_fqn)
cards_bz = spark.table(cards_bz_fqn)
users_bz = spark.table(users_bz_fqn)
mcc_bz = spark.table(mcc_codes_bz_fqn)
fraud_labels_bz = spark.table(fraud_labels_bz_fqn)

display(transactions_bz.limit(10))

# COMMAND ----------

from pyspark.sql.functions import (
    col,
    hour,
    lit,
    regexp_replace,
    to_date,
    to_timestamp,
    trim,
    upper,
    when,
    year,
)

transactions_typed = (
    transactions_bz.dropDuplicates(["id"])
    .withColumnRenamed("id", "transaction_id")
    .withColumn("transaction_ts", to_timestamp(col("date"), "yyyy-MM-dd HH:mm:ss"))
    .withColumn("transaction_date", to_date(col("transaction_ts")))
    .withColumn("transaction_year", year(col("transaction_ts")))
    .withColumn("transaction_hour", hour(col("transaction_ts")))
    .withColumn(
        "amount_decimal", regexp_replace(col("amount"), "[$,\\s]", "").cast("decimal(12,2)")
    )
    .withColumn("use_chip_normalized", upper(trim(col("use_chip"))))
    .withColumn("merchant_city", trim(col("merchant_city")))
    .withColumn("merchant_state", upper(trim(col("merchant_state"))))
    .select(
        "transaction_id",
        "client_id",
        "card_id",
        "transaction_ts",
        "transaction_date",
        "transaction_year",
        "transaction_hour",
        col("amount_decimal").alias("amount"),
        col("use_chip_normalized").alias("use_chip"),
        "merchant_id",
        "merchant_city",
        "merchant_state",
        "zip",
        "mcc",
        "errors",
        "source_year",
    )
)

# COMMAND ----------

silver_cards = (
    cards_bz.dropDuplicates(["id"])
    .withColumnRenamed("id", "card_id")
    .withColumn("card_brand", upper(trim(col("card_brand"))))
    .withColumn("card_type", upper(trim(col("card_type"))))
    .withColumn(
        "has_chip_bool",
        when(upper(trim(col("has_chip"))) == "YES", lit(True)).when(
            upper(trim(col("has_chip"))) == "NO", lit(False)
        ),
    )
    .withColumn(
        "credit_limit_decimal",
        regexp_replace(col("credit_limit"), "[$,\\s]", "").cast("decimal(12,2)"),
    )
    .select(
        "card_id",
        "client_id",
        "card_brand",
        "card_type",
        "has_chip_bool",
        "num_cards_issued",
        "credit_limit_decimal",
        "acct_open_date",
        "expires",
    )
)

silver_users = (
    users_bz.dropDuplicates(["id"])
    .withColumnRenamed("id", "client_id")
    .withColumn("gender", upper(trim(col("gender"))))
    .withColumn(
        "yearly_income_decimal",
        regexp_replace(col("yearly_income"), "[$,\\s]", "").cast("decimal(12,2)"),
    )
    .withColumn(
        "total_debt_decimal", regexp_replace(col("total_debt"), "[$,\\s]", "").cast("decimal(12,2)")
    )
    .select(
        "client_id",
        "current_age",
        "retirement_age",
        "birth_year",
        "birth_month",
        "gender",
        "latitude",
        "longitude",
        "yearly_income_decimal",
        "total_debt_decimal",
        "credit_score",
        "num_credit_cards",
    )
)

silver_mcc_codes = mcc_bz.dropDuplicates(["mcc"]).select("mcc", "mcc_description")

silver_fraud_labels = (
    fraud_labels_bz.dropDuplicates(["transaction_id"])
    .withColumn(
        "is_fraud",
        when(upper(trim(col("fraud_label"))) == "YES", lit(True)).when(
            upper(trim(col("fraud_label"))) == "NO", lit(False)
        ),
    )
    .select("transaction_id", "is_fraud")
)

display(silver_cards)
display(silver_users)
display(silver_fraud_labels)

# COMMAND ----------

silver_transaction_enriched = (
    transactions_typed.alias("t")
    .join(silver_cards.drop("client_id").alias("c"), on="card_id", how="left")
    .join(silver_users.alias("u"), on="client_id", how="left")
    .join(silver_mcc_codes.alias("m"), on="mcc", how="left")
    .join(silver_fraud_labels.alias("f"), on="transaction_id", how="left")
    .select(
        "transaction_id",
        "client_id",
        "card_id",
        "mcc",
        "transaction_ts",
        "transaction_date",
        "transaction_year",
        "transaction_hour",
        "amount",
        "use_chip",
        "merchant_id",
        "merchant_city",
        "merchant_state",
        "zip",
        "errors",
        "source_year",
        "card_brand",
        "card_type",
        "has_chip_bool",
        "num_cards_issued",
        col("credit_limit_decimal").alias("credit_limit"),
        "acct_open_date",
        "expires",
        "current_age",
        "retirement_age",
        "birth_year",
        "birth_month",
        "gender",
        "latitude",
        "longitude",
        col("yearly_income_decimal").alias("yearly_income"),
        col("total_debt_decimal").alias("total_debt"),
        "credit_score",
        "num_credit_cards",
        "mcc_description",
        "is_fraud",
    )
)

# COMMAND ----------


from delta.tables import DeltaTable


def upsert_delta_table(dataframe, table_name, merge_condition):
    if not spark.catalog.tableExists(table_name):
        dataframe.write.format("delta").saveAsTable(table_name)
        logger.info("Created %s", table_name)
        return

    (
        DeltaTable.forName(spark, table_name)
        .alias("target")
        .merge(dataframe.alias("source"), merge_condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    logger.info("Upserted %s", table_name)


# COMMAND ----------

# Governance: table and column comments applied to every curated table so the data
# is self-documenting in Unity Catalog / Catalog Explorer. Column comments are keyed
# by column name and reused across tables (a column means the same thing everywhere).
COLUMN_COMMENTS = {
    "transaction_id": "Unique transaction identifier.",
    "client_id": "Cardholder (user) identifier.",
    "card_id": "Payment card identifier.",
    "transaction_ts": "Transaction timestamp (parsed from the raw string).",
    "transaction_date": "Transaction date (derived from transaction_ts).",
    "transaction_year": "Transaction year (derived from transaction_ts).",
    "transaction_hour": "Hour of day of the transaction 0-23 (derived).",
    "amount": "Transaction amount in USD, cleaned to decimal(12,2).",
    "use_chip": "Entry mode: CHIP / SWIPE / ONLINE (upper-cased).",
    "merchant_id": "Merchant identifier.",
    "merchant_city": "Merchant city (trimmed).",
    "merchant_state": "Merchant state code (upper-cased).",
    "zip": "Merchant postal (ZIP) code.",
    "mcc": "Merchant Category Code.",
    "mcc_description": "Human-readable Merchant Category Code description.",
    "errors": "Raw transaction error flags from source, if any.",
    "source_year": "Partition column: year parsed at ingestion.",
    "card_brand": "Card network brand, e.g. VISA / MASTERCARD (upper-cased).",
    "card_type": "Card type, e.g. DEBIT / CREDIT (upper-cased).",
    "has_chip_bool": "Whether the card has a chip (boolean).",
    "num_cards_issued": "Number of physical cards issued for this account.",
    "credit_limit": "Card credit limit in USD, cleaned to decimal(12,2).",
    "acct_open_date": "Account opening date (MM/yyyy, kept as source string).",
    "expires": "Card expiry (MM/yyyy, kept as source string).",
    "current_age": "Cardholder current age in years.",
    "retirement_age": "Cardholder expected retirement age.",
    "birth_year": "Cardholder birth year.",
    "birth_month": "Cardholder birth month.",
    "gender": "Cardholder gender (upper-cased).",
    "latitude": "Cardholder home latitude.",
    "longitude": "Cardholder home longitude.",
    "yearly_income": "Cardholder yearly income in USD, cleaned to decimal(12,2).",
    "total_debt": "Cardholder total debt in USD, cleaned to decimal(12,2).",
    "credit_score": "Cardholder credit score.",
    "num_credit_cards": "Number of credit cards the cardholder holds.",
    "is_fraud": "Label: whether the transaction is fraudulent (boolean).",
}


def apply_comments(table_name: str, table_comment: str) -> None:
    """Set the table comment and column comments for a curated table."""
    spark.sql(f"COMMENT ON TABLE {table_name} IS '{table_comment}'")
    for column in spark.table(table_name).columns:
        text = COLUMN_COMMENTS.get(column)
        if text:
            spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {column} COMMENT '{text}'")
    logger.info("Applied comments to %s", table_name)


# COMMAND ----------

# (target_fqn, dataframe, merge_condition, table_comment)
table_specs = [
    (
        transactions_sl_fqn,
        transactions_typed,
        "target.transaction_id = source.transaction_id",
        "Silver: cleaned, typed, deduplicated transactions (normalized fact table).",
    ),
    (
        cards_sl_fqn,
        silver_cards,
        "target.card_id = source.card_id",
        "Silver: conformed card dimension (PII such as card number and CVV removed).",
    ),
    (
        users_sl_fqn,
        silver_users,
        "target.client_id = source.client_id",
        "Silver: conformed cardholder dimension with typed financials.",
    ),
    (
        mcc_codes_sl_fqn,
        silver_mcc_codes,
        "target.mcc = source.mcc",
        "Silver: Merchant Category Code lookup (code -> description).",
    ),
    (
        fraud_labels_sl_fqn,
        silver_fraud_labels,
        "target.transaction_id = source.transaction_id",
        "Silver: fraud labels per transaction (boolean is_fraud).",
    ),
    (
        transactions_enriched_gd_fqn,
        silver_transaction_enriched,
        "target.transaction_id = source.transaction_id",
        "Gold: denormalized, labelled transactions joined to card, user and MCC "
        "attributes. ML-ready base table for fraud feature engineering.",
    ),
]

for table_name, dataframe, merge_condition, table_comment in table_specs:
    upsert_delta_table(dataframe, table_name, merge_condition)
    apply_comments(table_name, table_comment)
