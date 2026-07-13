# Databricks notebook source
# MAGIC %md
# MAGIC # Landing -> Bronze
# MAGIC Ingest the raw fraud source files from the landing volume into Bronze Delta
# MAGIC tables, applying explicit schemas and audit columns. Minimal transformation.
# MAGIC
# MAGIC **Setup notebook** run by the instructor before the workshop. Not a lab exercise.
# MAGIC
# MAGIC | Layer | Location |
# MAGIC |---|---|
# MAGIC | Landing (files) | `/Volumes/dev/fraud_landing/raw_data/<dataset>/` |
# MAGIC | Bronze (tables) | `dev.fraud_bronze.<table>` |

# COMMAND ----------

import logging

logger = logging.getLogger("workshop.bronze")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

# COMMAND ----------

dbutils.widgets.text("catalog", "adoption_workshop")
dbutils.widgets.text("bronze_schema", "fraud_bronze")
dbutils.widgets.text("landing_catalog", "dev")
dbutils.widgets.text("landing_schema", "fraud_landing")
dbutils.widgets.text("landing_volume", "raw_data")

catalog = dbutils.widgets.get("catalog")
bronze_schema = dbutils.widgets.get("bronze_schema")

landing_catalog = dbutils.widgets.get("landing_catalog")
landing_schema = dbutils.widgets.get("landing_schema")
landing_volume = dbutils.widgets.get("landing_volume")
src = f"/Volumes/{landing_catalog}/{landing_schema}/{landing_volume}"

# One folder per dataset in the landing volume.
transactions_path = src + "/transactions/transactions_data.csv"
cards_path = src + "/cards"
users_path = src + "/users"
mcc_codes_path = src + "/mcc_codes"
fraud_labels_path = src + "/fraud_labels"

# Bronze tables (no prefix - the schema name conveys the layer).
transactions_fqn = f"{catalog}.{bronze_schema}.transactions"
cards_fqn = f"{catalog}.{bronze_schema}.cards"
users_fqn = f"{catalog}.{bronze_schema}.users"
mcc_codes_fqn = f"{catalog}.{bronze_schema}.mcc_codes"
fraud_labels_fqn = f"{catalog}.{bronze_schema}.fraud_labels"

logger.info("Target catalog/schema : %s.%s", catalog, bronze_schema)
logger.info("Source landing volume : %s", src)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target schemas and volume
# MAGIC The shared `fraud_landing` / `fraud_bronze` schemas and the `raw_data` volume are
# MAGIC declared as bundle resources (the `shared_infra` block in `databricks.yml`) and
# MAGIC created by `databricks bundle deploy -t dev` before this job runs. The raw files are
# MAGIC uploaded into the volume after the deploy, and this notebook just reads them.

# COMMAND ----------

from pyspark.sql.functions import col, current_date, current_timestamp, to_timestamp, year
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType

transactions_schema = StructType(
    [
        StructField("id", LongType(), True),
        StructField("date", StringType(), True),
        StructField("client_id", LongType(), True),
        StructField("card_id", LongType(), True),
        StructField("amount", StringType(), True),
        StructField("use_chip", StringType(), True),
        StructField("merchant_id", LongType(), True),
        StructField("merchant_city", StringType(), True),
        StructField("merchant_state", StringType(), True),
        StructField("zip", StringType(), True),
        StructField("mcc", LongType(), True),
        StructField("errors", StringType(), True),
    ]
)

cards_schema = StructType(
    [
        StructField("id", LongType(), True),
        StructField("client_id", LongType(), True),
        StructField("card_brand", StringType(), True),
        StructField("card_type", StringType(), True),
        StructField("card_number", StringType(), True),
        StructField("expires", StringType(), True),
        StructField("cvv", StringType(), True),
        StructField("has_chip", StringType(), True),
        StructField("num_cards_issued", IntegerType(), True),
        StructField("credit_limit", StringType(), True),
        StructField("acct_open_date", StringType(), True),
        StructField("year_pin_last_changed", IntegerType(), True),
        StructField("card_on_dark_web", StringType(), True),
    ]
)

users_schema = StructType(
    [
        StructField("id", LongType(), True),
        StructField("current_age", IntegerType(), True),
        StructField("retirement_age", IntegerType(), True),
        StructField("birth_year", IntegerType(), True),
        StructField("birth_month", IntegerType(), True),
        StructField("gender", StringType(), True),
        StructField("address", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("per_capita_income", StringType(), True),
        StructField("yearly_income", StringType(), True),
        StructField("total_debt", StringType(), True),
        StructField("credit_score", IntegerType(), True),
        StructField("num_credit_cards", IntegerType(), True),
    ]
)

# COMMAND ----------


def read_csv_with_audit(path: str, schema: StructType):
    return (
        spark.read.format("csv")
        .option("header", "true")
        .schema(schema)
        .load(path)
        .withColumn("_ingest_ts", current_timestamp())
        .withColumn("_ingest_date", current_date())
        .withColumn("_source_file", col("_metadata.file_path"))
    )


transactions_bronze_df = read_csv_with_audit(transactions_path, transactions_schema).withColumn(
    "source_year", year(to_timestamp(col("date"), "yyyy-MM-dd HH:mm:ss"))
)
logger.info("Read transactions from %s", transactions_path)

cards_bronze_df = read_csv_with_audit(cards_path, cards_schema)
logger.info("Read cards from %s", cards_path)

users_bronze_df = read_csv_with_audit(users_path, users_schema)
logger.info("Read users from %s", users_path)

# COMMAND ----------

(
    transactions_bronze_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("source_year")
    .saveAsTable(transactions_fqn)
)
logger.info("Written %s", transactions_fqn)

cards_bronze_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(cards_fqn)
logger.info("Written %s", cards_fqn)

users_bronze_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(users_fqn)
logger.info("Written %s", users_fqn)

# COMMAND ----------

from pyspark.sql.functions import explode, from_json
from pyspark.sql.types import MapType

mcc_map_schema = MapType(StringType(), StringType())

mcc_bronze_df = (
    spark.read.text(mcc_codes_path, wholetext=True)
    .select(
        explode(from_json(col("value"), mcc_map_schema)).alias("mcc", "mcc_description"),
        col("_metadata.file_path").alias("_source_file"),
    )
    .withColumn("mcc", col("mcc").cast("int"))
    .withColumn("_ingest_ts", current_timestamp())
    .withColumn("_ingest_date", current_date())
)
logger.info("Read MCC codes from %s", mcc_codes_path)

# COMMAND ----------

fraud_map_schema = StructType([StructField("target", MapType(StringType(), StringType()))])

fraud_labels_bronze_df = (
    spark.read.schema(fraud_map_schema)
    .json(fraud_labels_path)
    .select(
        explode(col("target")).alias("transaction_id", "fraud_label"),
        col("_metadata.file_path").alias("_source_file"),
    )
    .withColumn("transaction_id", col("transaction_id").cast("int"))
    .withColumn("_ingest_ts", current_timestamp())
    .withColumn("_ingest_date", current_date())
)
logger.info("Read fraud labels from %s", fraud_labels_path)

# COMMAND ----------

mcc_bronze_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    mcc_codes_fqn
)
logger.info("Written %s", mcc_codes_fqn)

fraud_labels_bronze_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(fraud_labels_fqn)
logger.info("Written %s", fraud_labels_fqn)

# COMMAND ----------

# Governance: table-level comments so the raw layer is self-documenting. Bronze keeps
# source fidelity, so we do not comment individual columns here (that happens in silver).
bronze_table_comments = {
    transactions_fqn: "Bronze: raw card transactions ingested as-is from the landing volume, plus audit columns.",
    cards_fqn: "Bronze: raw card metadata ingested as-is from the landing volume, plus audit columns.",
    users_fqn: "Bronze: raw cardholder data ingested as-is from the landing volume, plus audit columns.",
    mcc_codes_fqn: "Bronze: raw Merchant Category Code lookup ingested as-is, plus audit columns.",
    fraud_labels_fqn: "Bronze: raw fraud labels ingested as-is from the landing volume, plus audit columns.",
}

for table_name, comment in bronze_table_comments.items():
    spark.sql(f"COMMENT ON TABLE {table_name} IS '{comment}'")
    logger.info("Commented %s", table_name)
