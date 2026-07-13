# Databricks notebook source
# MAGIC %md
# MAGIC # Train and register the fraud detection model
# MAGIC
# MAGIC **Session:** Model Development & Experimentation
# MAGIC
# MAGIC Assemble a training set from the Unity Catalog Feature Store, train a classifier,
# MAGIC track it with MLflow 3, and register the result to Unity Catalog for the Governance
# MAGIC and Promotion sessions.
# MAGIC
# MAGIC - **Entity features (personal):** `card_features` (by `card_id`) + `client_features` (by `client_id`)
# MAGIC - **On-demand features:** UC functions computed from the request (night, online, high-risk MCC, ratios)
# MAGIC - **Labels (shared):** `dev.fraud_gold.transactions_enriched` (`transaction_id`, `is_fraud`)
# MAGIC - **Model (personal):** registered as `dev.<you>_fraud.fraud_detection`
# MAGIC
# MAGIC Features are looked up from the Feature Store at both training and serving time, so
# MAGIC there is no train/serve skew.
# MAGIC
# MAGIC Docs: [Train models with Feature Engineering](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/train-models-with-feature-store)
# MAGIC | [Manage model lifecycle in UC](https://learn.microsoft.com/azure/databricks/machine-learning/manage-model-lifecycle/)

# COMMAND ----------

dbutils.widgets.text("catalog_name", "adoption_workshop")
dbutils.widgets.text("gold_schema", "fraud_gold")
dbutils.widgets.text("ml_schema", "")
dbutils.widgets.text("experiment_name", "")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("deployment_job_id", "")

catalog_name = dbutils.widgets.get("catalog_name")
gold_schema = dbutils.widgets.get("gold_schema")
ml_schema = dbutils.widgets.get("ml_schema")

from pyspark.sql import functions as F

# Interactive fallback: jobs pass the resolved personal schema; running standalone derives
# the same per-user name the bundle uses (dev_<short>_fraud) so runs stay isolated.
if not ml_schema:
    _user = spark.range(1).select(F.current_user()).first()[0]
    _short = "".join(c if c.isalnum() else "_" for c in _user.split("@")[0])
    ml_schema = f"dev_{_short}_fraud"

# Labels come from the shared gold table; features come from the personal schema.
source_table = f"{catalog_name}.{gold_schema}.transactions_enriched"
card_feature_table = f"{catalog_name}.{ml_schema}.card_features"
client_feature_table = f"{catalog_name}.{ml_schema}.client_features"

# Sensible defaults so the notebook is runnable interactively (outside the DAB job),
# where the experiment/model widgets may be empty.
current_user = spark.range(1).select(F.current_user()).first()[0]
experiment_name = (
    dbutils.widgets.get("experiment_name") or f"/Users/{current_user}/mlops-workshop-fraud"
)
model_name = dbutils.widgets.get("model_name") or f"{catalog_name}.{ml_schema}.fraud_detection"

print(f"Card features:   {card_feature_table}")
print(f"Client features: {client_feature_table}")
print(f"Labels:          {source_table}")
print(f"Experiment: {experiment_name}")
print(f"Model:      {model_name}")

# COMMAND ----------

import mlflow
from databricks.feature_engineering import (
    FeatureEngineeringClient,
    FeatureFunction,
    FeatureLookup,
)
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# MLflow 3: register to the Unity Catalog model registry (the default in MLflow 3, set
# explicitly here so the notebook behaves the same on any runtime).
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_name)
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connect the model to its deployment job
# MAGIC
# MAGIC If the training job passed a `deployment_job_id`, connect it to the registered model
# MAGIC **before** registering a version:
# MAGIC
# MAGIC - registering the new version then auto-triggers the governed evaluate/approve/deploy
# MAGIC   pipeline, even for the first version;
# MAGIC - `create_registered_model` makes the empty model on the first run, later runs just
# MAGIC   `update_registered_model`;
# MAGIC - skipped when run interactively without a job id.
# MAGIC
# MAGIC See [MLflow deployment jobs](https://learn.microsoft.com/azure/databricks/mlflow/deployment-job).

# COMMAND ----------

deployment_job_id = dbutils.widgets.get("deployment_job_id")
if deployment_job_id:
    from mlflow.exceptions import RestException

    _client = MlflowClient()
    try:
        _client.get_registered_model(model_name)
        _client.update_registered_model(model_name, deployment_job_id=deployment_job_id)
    except RestException:
        _client.create_registered_model(model_name, deployment_job_id=deployment_job_id)
    print(f"Connected {model_name} to deployment job {deployment_job_id}.")
else:
    print("No deployment_job_id passed; skipping deployment-job connection (interactive run).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Assemble the training set
# MAGIC
# MAGIC - The spine is the raw transaction: keys (`card_id`, `client_id`), the request fields
# MAGIC   (`amount`, `transaction_hour`, `mcc`, `use_chip`), and the label.
# MAGIC - The Feature Store looks up the card/client entity features and computes the on-demand
# MAGIC   features with the UC functions.
# MAGIC - `create_training_set` records this spec, so serving reproduces the same feature pipeline.

# COMMAND ----------

schema_fqn = f"{catalog_name}.{ml_schema}"

# Cap the training data at a random sample so the notebook runs in a couple of minutes.
# Fraud is extremely rare, so a random sample (not the first N rows) is what keeps enough
# fraud examples in the set; the seed makes it reproducible run-to-run.
SAMPLE_ROWS = 200000

spine = (
    spark.read.table(source_table)
    .select(
        "transaction_id",
        "card_id",
        "client_id",
        F.col("amount").cast("double").alias("amount"),
        "transaction_hour",
        "mcc",
        "use_chip",
        F.col("is_fraud").cast("int").alias("is_fraud"),
    )
    .where(F.col("is_fraud").isNotNull())
    .orderBy(F.rand(42))
    .limit(SAMPLE_ROWS)
)

# The on-demand feature functions are provided (they mirror the UC UDFs registered in the
# feature engineering notebook). Adding the two entity FeatureLookups and assembling the
# training set is the exercise.
feature_functions = [
    FeatureFunction(
        udf_name=f"{schema_fqn}.ff_is_night",
        output_name="is_night",
        input_bindings={"transaction_hour": "transaction_hour"},
    ),
    FeatureFunction(
        udf_name=f"{schema_fqn}.ff_is_online",
        output_name="is_online",
        input_bindings={"use_chip": "use_chip"},
    ),
    FeatureFunction(
        udf_name=f"{schema_fqn}.ff_is_high_risk_mcc",
        output_name="is_high_risk_mcc",
        input_bindings={"mcc": "mcc"},
    ),
    FeatureFunction(
        udf_name=f"{schema_fqn}.ff_amount_to_income_ratio",
        output_name="amount_to_income_ratio",
        input_bindings={"amount": "amount", "yearly_income": "yearly_income"},
    ),
    FeatureFunction(
        udf_name=f"{schema_fqn}.ff_amount_to_credit_limit_ratio",
        output_name="amount_to_credit_limit_ratio",
        input_bindings={"amount": "amount", "credit_limit": "credit_limit"},
    ),
]

# TODO-BEGIN: look up the entity features and assemble the training set
# HINT: FeatureLookup(table_name=card_feature_table, lookup_key="card_id",
# HINT:   feature_names=["credit_limit", "num_cards_issued"]); same for the client table
# HINT:   (client_feature_table, lookup_key="client_id", credit_score/yearly_income/current_age).
# HINT: fe.create_training_set(df=spine, feature_lookups=feature_lookups + feature_functions,
# HINT:   label="is_fraud", exclude_columns=[...]).
# HINT: exclude the keys and raw function inputs so the model only sees features:
# HINT:   transaction_id, card_id, client_id, mcc, use_chip.
feature_lookups = [
    FeatureLookup(
        table_name=card_feature_table,
        lookup_key="card_id",
        feature_names=["credit_limit", "num_cards_issued"],
    ),
    FeatureLookup(
        table_name=client_feature_table,
        lookup_key="client_id",
        feature_names=["credit_score", "yearly_income", "current_age"],
    ),
]

training_set = fe.create_training_set(
    df=spine,
    feature_lookups=feature_lookups + feature_functions,
    label="is_fraud",
    exclude_columns=["transaction_id", "card_id", "client_id", "mcc", "use_chip"],
)
# TODO-END

# Load into pandas and do a standard stratified train/test split (provided: this is ordinary
# scikit-learn, not the MLOps concept this lab is about).
train_pdf = training_set.load_df().toPandas()
X = train_pdf.drop(columns=["is_fraud"])
y = train_pdf["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Rows: {len(train_pdf):,}  |  features: {list(X.columns)}  |  fraud rate: {y.mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train the model and address class imbalance
# MAGIC
# MAGIC Fraud accounts for well under 1% of rows, so a model can reach roughly 99.8% accuracy
# MAGIC by always predicting "not fraud" while detecting none. The workflow:
# MAGIC
# MAGIC - Train a baseline random forest on the raw, imbalanced data to establish the problem.
# MAGIC - Retrain on balanced data (retain every fraud row, down-sample non-fraud to match) so
# MAGIC   the model learns the minority class.
# MAGIC - Evaluate on the original, imbalanced test set so the reported metrics reflect
# MAGIC   production conditions.
# MAGIC
# MAGIC `fe.log_model` packages the model with its feature metadata, infers a signature and
# MAGIC input example, and registers it to Unity Catalog in a single step (no separate
# MAGIC `register_model` call).

# COMMAND ----------


def evaluate(clf, X_eval, y_eval):
    preds = clf.predict(X_eval)
    proba = clf.predict_proba(X_eval)[:, 1]
    return {
        "accuracy": accuracy_score(y_eval, preds),
        "precision": precision_score(y_eval, preds, zero_division=0),
        "recall": recall_score(y_eval, preds, zero_division=0),
        "f1": f1_score(y_eval, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_eval, proba),
    }


class FraudProbabilityModel(mlflow.pyfunc.PythonModel):
    """Serve the fraud PROBABILITY (P(fraud) in [0, 1]) instead of a hard 0/1 class.

    A score is more useful to callers (they pick their own threshold) and lets the
    deployment job's Evaluation task compute ROC AUC on the registered artifact. Wrapping
    the classifier in a pyfunc keeps fe.log_model's feature resolution intact.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


# Naive random forest (instructor-provided): the imbalance trap. Trained on the raw,
# highly-imbalanced data it just learns to predict the majority class, so accuracy looks
# great while recall is ~0 (it never flags a fraud). Logged for comparison, NOT
# registered.
with mlflow.start_run(run_name="rf_imbalanced"):
    naive_params = {"n_estimators": 100, "random_state": 42}
    naive = RandomForestClassifier(**naive_params)
    naive.fit(X_train, y_train)
    naive_metrics = evaluate(naive, X_test, y_test)
    mlflow.log_params({**naive_params, "training_data": "imbalanced"})
    mlflow.log_metrics(naive_metrics)
    print("rf_imbalanced:", naive_metrics)  # high accuracy, low recall

# Fixed random forest: the model that is tracked and registered.
with mlflow.start_run(run_name="rf_balanced") as run:
    # Balance the training rows: keep every fraud row and randomly sample an equal number of
    # non-fraud rows (provided: this is a data-science fix, not the MLOps concept). Evaluation
    # still uses the untouched, imbalanced X_test/y_test so metrics reflect reality.
    fraud_idx = y_train[y_train == 1].index
    legit_idx = y_train[y_train == 0].sample(n=len(fraud_idx), random_state=42).index
    bal_idx = fraud_idx.union(legit_idx)
    X_bal, y_bal = X_train.loc[bal_idx], y_train.loc[bal_idx]

    params = {"n_estimators": 100, "random_state": 42}
    model = RandomForestClassifier(**params)
    model.fit(X_bal, y_bal)
    metrics = evaluate(model, X_test, y_test)

    mlflow.log_params(params)
    mlflow.log_param("training_data", "balanced (1:1 down-sampled)")
    mlflow.log_param("train_rows", int(len(y_bal)))
    mlflow.log_param("training_table", source_table)
    mlflow.log_param("card_feature_table", card_feature_table)
    mlflow.log_param("client_feature_table", client_feature_table)
    mlflow.log_metrics(metrics)

    # Package with feature metadata + an inferred signature/input example (required for
    # safe serving) and register to Unity Catalog in one call. The classifier is wrapped so
    # the served model returns a fraud probability (score), not a hard class.
    # TODO-BEGIN: log and register the model to Unity Catalog with its feature metadata
    # HINT: fe.log_model packages the model with the training_set's feature lookups (so serving
    # HINT:   reproduces the same features) and registers it to UC in a single call.
    # HINT: fe.log_model(model=FraudProbabilityModel(model), artifact_path="model",
    # HINT:   flavor=mlflow.pyfunc, training_set=training_set,
    # HINT:   registered_model_name=model_name, infer_input_example=True)
    fe.log_model(
        model=FraudProbabilityModel(model),
        artifact_path="model",
        flavor=mlflow.pyfunc,
        training_set=training_set,
        registered_model_name=model_name,
        infer_input_example=True,
    )
    # TODO-END

    # MLflow 3: surface the eval metrics on the LoggedModel (and thus the UC model-version
    # page across workspaces), not only on the run.
    try:
        logged = mlflow.last_logged_model()
        if logged is not None:
            mlflow.log_metrics(metrics, model_id=logged.model_id)
    except Exception as exc:  # stay resilient across mlflow 3.x versions
        print(f"(skipped attaching metrics to logged model: {exc})")

print(
    f"imbalanced: recall={naive_metrics['recall']:.3f} precision={naive_metrics['precision']:.3f} "
    f"(accuracy={naive_metrics['accuracy']:.4f})"
)
print(
    f"balanced:   recall={metrics['recall']:.3f} precision={metrics['precision']:.3f} "
    f"(roc_auc={metrics['roc_auc']:.4f})"
)
print(metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion matrix comparison
# MAGIC
# MAGIC Compare the two models on the fraud row (bottom):
# MAGIC
# MAGIC - The baseline model assigns almost all fraud to "predicted legit" (false negatives)
# MAGIC   and detects little fraud.
# MAGIC - The balanced model recovers true positives at the cost of more false positives.
# MAGIC
# MAGIC Both use the same original, imbalanced test set. The figure is logged to the
# MAGIC `rf_balanced` MLflow run so it stays with the model version.

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, clf, title in (
    (axes[0], naive, "rf_imbalanced (naive)"),
    (axes[1], model, "rf_balanced (registered)"),
):
    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, display_labels=["legit", "fraud"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
fig.tight_layout()

# Attach the comparison to the registered model's run so it lives with the model version.
with mlflow.start_run(run_id=run.info.run_id):
    mlflow.log_figure(fig, "confusion_matrices.png")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precision-recall trade-off
# MAGIC
# MAGIC The balanced model achieves high recall but low precision: most flagged transactions
# MAGIC are legitimate. Balancing the data makes the model predict "fraud" far more often,
# MAGIC which recovers real fraud but raises the false-positive rate.
# MAGIC
# MAGIC In production the decision threshold is tuned (for example, score > 0.8 rather than
# MAGIC 0.5) to trade recall for precision. The operating point is chosen from the business
# MAGIC cost of a missed fraud versus a false alarm, and is typically assessed with precision@k
# MAGIC or PR-AUC rather than a single accuracy or recall figure. Because this is a business
# MAGIC decision, the promotion gate in the next section encodes the metric the business
# MAGIC requires.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tag the new version as `challenger` (instructor-provided)
# MAGIC
# MAGIC The training run registered a new model version. Point the `challenger` alias at it to
# MAGIC give a clear before/after comparison. Registering the version also triggers the
# MAGIC deployment job, whose Evaluation task re-scores the version on a fresh holdout and
# MAGIC compares it against the current `@champion` before promotion. Downstream jobs always
# MAGIC load `@champion`, so promotion is an alias reassignment: no code change, and rollback
# MAGIC is immediate.

# COMMAND ----------

client = MlflowClient()
# Unity Catalog's search_model_versions only supports a `name='...'` filter (no run_id
# filtering), so fetch this model's versions and take the highest: the one registered above.
versions = client.search_model_versions(f"name='{model_name}'")
new_version = max(int(mv.version) for mv in versions)
client.set_registered_model_alias(model_name, "challenger", new_version)
print(f"Registered {model_name} version {new_version} and set alias @challenger.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train a challenger
# MAGIC The first model becomes `@champion` once the deployment job promotes it. To produce a
# MAGIC challenger to compete against it, change the model and re-run this whole notebook.
# MAGIC Each run registers a new version and moves `@challenger` to it:
# MAGIC
# MAGIC - In **section 2**, adjust the hyper-parameters in `params`, for example:
# MAGIC   - bump `n_estimators` to `200`, or
# MAGIC   - add `max_depth=8` or `min_samples_leaf=5` to regularise, or
# MAGIC   - swap in a different estimator (e.g. `GradientBoostingClassifier`).
# MAGIC - Optionally add or refine a feature in the feature-engineering notebook and re-run it
# MAGIC   first, so the challenger trains on richer features.
# MAGIC
# MAGIC Registering the new version auto-triggers the **deployment job**: its Evaluation task
# MAGIC compares the challenger's `roc_auc` against the current `@champion` on a fresh holdout
# MAGIC and only lets it through if it wins. A safe, auditable champion/challenger swap with no
# MAGIC downstream code change.
