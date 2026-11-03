from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# AI-POWERED FINANCIAL FRAUD DETECTION SYSTEM
# Consolidated project script
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SHAP_SAMPLE_SIZE = 500
NETWORK_SAMPLE_SIZE = 50000
INVESTIGATION_SAMPLE_SIZE = 5000
CLASSIFICATION_THRESHOLD = 0.50

sns.set_theme(style="whitegrid")


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 1. LOAD DATA
# ============================================================

print_section("AI-POWERED FINANCIAL FRAUD DETECTION SYSTEM")

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found. Place the PaySim dataset inside the 'data' folder."
    )

file_path = csv_files[0]
print(f"\nDataset: {file_path.name}")

df = pd.read_csv(file_path)

required_columns = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest",
    "newbalanceDest", "isFraud", "isFlaggedFraud"
]
missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

print(f"Total transactions: {df.shape[0]:,}")
print(f"Total columns: {df.shape[1]}")


# ============================================================
# 2. DATA INSPECTION
# ============================================================

print_section("1. DATA INSPECTION")

print("\nFirst five transactions:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-null counts:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

duplicate_count = int(df.duplicated().sum())
print(f"\nDuplicate rows: {duplicate_count:,}")


# ============================================================
# 3. FRAUD DISTRIBUTION
# ============================================================

print_section("2. FRAUD LABEL DISTRIBUTION")

fraud_counts = df["isFraud"].value_counts().sort_index()
fraud_percentage = df["isFraud"].mean() * 100

print(fraud_counts)
print(f"\nFraud percentage: {fraud_percentage:.5f}%")
print(f"Legitimate transactions: {(df['isFraud'] == 0).sum():,}")
print(f"Fraudulent transactions: {(df['isFraud'] == 1).sum():,}")

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="isFraud")
plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Fraud label (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of transactions")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fraud_label_distribution.png", dpi=300)
plt.show()
plt.close()


# ============================================================
# 4. FRAUD ANALYSIS BY TRANSACTION TYPE
# ============================================================

print_section("3. FRAUD ANALYSIS BY TRANSACTION TYPE")

fraud_by_type = df.groupby("type").agg(
    total_transactions=("isFraud", "size"),
    fraud_count=("isFraud", "sum"),
    fraud_rate=("isFraud", "mean"),
)
fraud_by_type["fraud_rate_pct"] = fraud_by_type["fraud_rate"] * 100

print(fraud_by_type.round(5))

plt.figure(figsize=(9, 5))
sns.barplot(
    data=df,
    x="type",
    y="isFraud",
    estimator="mean",
    errorbar=None,
)
plt.title("Fraud Rate by Transaction Type")
plt.xlabel("Transaction type")
plt.ylabel("Fraud rate")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fraud_rate_by_type.png", dpi=300)
plt.show()
plt.close()

fraud_by_type.to_csv(OUTPUT_DIR / "fraud_by_type.csv")


# ============================================================
# 5. TRANSACTION AMOUNT ANALYSIS
# ============================================================

print_section("4. TRANSACTION AMOUNT ANALYSIS")

amount_summary = df.groupby("isFraud")["amount"].agg(
    ["count", "mean", "median", "min", "max"]
)
print(amount_summary.round(2))

legitimate_sample = df.loc[df["isFraud"] == 0].sample(
    n=min(5000, int((df["isFraud"] == 0).sum())),
    random_state=RANDOM_STATE,
)
fraud_sample = df.loc[df["isFraud"] == 1].sample(
    n=min(5000, int((df["isFraud"] == 1).sum())),
    random_state=RANDOM_STATE,
)
amount_plot_sample = pd.concat(
    [legitimate_sample, fraud_sample],
    ignore_index=True,
)

plt.figure(figsize=(9, 5))
sns.boxplot(data=amount_plot_sample, x="isFraud", y="amount")
plt.yscale("log")
plt.title("Transaction Amount by Fraud Label")
plt.xlabel("Fraud label (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Transaction amount (log scale)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "transaction_amount_by_fraud.png", dpi=300)
plt.show()
plt.close()

amount_summary.to_csv(OUTPUT_DIR / "amount_summary.csv")


# ============================================================
# 6. SENDER BALANCE ANALYSIS
# ============================================================

print_section("5. SENDER BALANCE ANALYSIS")

# This is descriptive analysis only. Post-transaction balance fields
# are deliberately excluded from the model features below.
df["sender_balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

sender_summary = df.groupby("isFraud")["sender_balance_change"].agg(
    ["mean", "median", "min", "max"]
)
print(sender_summary.round(2))
sender_summary.to_csv(OUTPUT_DIR / "sender_balance_summary.csv")


# ============================================================
# 7. FRAUD OVER SIMULATION TIME
# ============================================================

print_section("6. FRAUD OVER SIMULATION TIME")

fraud_by_step = df.groupby("step").agg(
    total_transactions=("isFraud", "size"),
    fraud_count=("isFraud", "sum"),
)
fraud_by_step["fraud_rate_pct"] = (
    fraud_by_step["fraud_count"]
    / fraud_by_step["total_transactions"]
    * 100
)

print("\nTop 10 time steps by fraud count:")
print(
    fraud_by_step.sort_values("fraud_count", ascending=False)
    .head(10)
    .to_string()
)

plt.figure(figsize=(11, 5))
plt.plot(fraud_by_step.index, fraud_by_step["fraud_count"])
plt.title("Fraud Count Across Simulation Time")
plt.xlabel("Simulation step")
plt.ylabel("Fraudulent transactions")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fraud_count_over_time.png", dpi=300)
plt.show()
plt.close()

fraud_by_step.to_csv(OUTPUT_DIR / "fraud_by_step.csv")


# ============================================================
# 8. CHRONOLOGICAL 80:20 TRANSACTION SPLIT
# ============================================================

print_section("7. CHRONOLOGICAL TRAIN-TEST SPLIT")

# Determine the time-step boundary using cumulative transaction volume.
# All transactions from the same step stay in one partition.
transactions_by_step = df.groupby("step").size().sort_index()
cumulative_transactions = transactions_by_step.cumsum()
target_training_volume = int(len(df) * 0.80)

cutoff_step = int(
    cumulative_transactions.index[
        cumulative_transactions.ge(target_training_volume)
    ][0]
)

train_mask = df["step"] < cutoff_step
test_mask = df["step"] >= cutoff_step

features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]
target = "isFraud"

X_train = df.loc[train_mask, features].copy()
X_test = df.loc[test_mask, features].copy()
y_train = df.loc[train_mask, target].astype(int)
y_test = df.loc[test_mask, target].astype(int)

print(f"Chronological cutoff step: {cutoff_step}")
print(f"Training period: steps before {cutoff_step}")
print(f"Testing period: steps from {cutoff_step} onwards")
print(f"Training transactions: {len(X_train):,}")
print(f"Testing transactions: {len(X_test):,}")
print(f"Training fraud cases: {int(y_train.sum()):,}")
print(f"Testing fraud cases: {int(y_test.sum()):,}")
print(f"Training fraud rate: {y_train.mean() * 100:.5f}%")
print(f"Testing fraud rate: {y_test.mean() * 100:.5f}%")

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Both training and testing partitions must contain legitimate "
        "and fraudulent transactions. Review the chronological cutoff."
    )


# ============================================================
# 9. PREPROCESSING AND MODEL TRAINING
# ============================================================

print_section("8. FRAUD PREDICTION MODEL")

numeric_features = [
    "step", "amount", "oldbalanceOrg", "oldbalanceDest"
]
categorical_features = ["type"]


def create_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}

results = []
trained_models = {}
test_probabilities = {}

for model_name, classifier in models.items():
    print("\n" + "-" * 55)
    print(f"Training: {model_name}")

    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", classifier),
    ])

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (
        probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    average_precision = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Average Precision (AP): {average_precision:.5f}")
    print(f"ROC-AUC: {roc_auc:.5f}")
    print(f"Classification threshold: {CLASSIFICATION_THRESHOLD:.2f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    results.append({
        "Model": model_name,
        "Average_Precision": average_precision,
        "ROC_AUC": roc_auc,
    })

    trained_models[model_name] = model
    test_probabilities[model_name] = probabilities


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

print_section("9. MODEL COMPARISON")

results_df = pd.DataFrame(results).sort_values(
    "Average_Precision",
    ascending=False,
)
print(results_df.to_string(index=False))
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=results_df,
    x="Model",
    y="Average_Precision",
)
plt.ylim(0, 1)
plt.title("Model Comparison - Average Precision")
plt.ylabel("Average precision")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=300)
plt.show()
plt.close()

# Random Forest is retained for the explainability and investigation stages.
rf_model = trained_models["Random Forest"]


# ============================================================
# 11. SHAP MODEL EXPLANATIONS
# ============================================================

print_section("10. SHAP MODEL EXPLANATIONS")

preprocessor = rf_model.named_steps["preprocessor"]
classifier = rf_model.named_steps["classifier"]

shap_sample = X_test.sample(
    n=min(SHAP_SAMPLE_SIZE, len(X_test)),
    random_state=RANDOM_STATE,
)

transformed_sample = preprocessor.transform(shap_sample)
feature_names = preprocessor.get_feature_names_out()

if hasattr(transformed_sample, "toarray"):
    transformed_sample = transformed_sample.toarray()

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(transformed_sample)

# Handle SHAP output formats from different SHAP versions.
if isinstance(shap_values, list):
    fraud_shap_values = shap_values[1]
elif getattr(shap_values, "ndim", 0) == 3:
    fraud_shap_values = shap_values[:, :, 1]
else:
    fraud_shap_values = shap_values

mean_abs_shap = np.abs(fraud_shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap,
}).sort_values("Mean_Absolute_SHAP", ascending=False)

print("\nTop 10 features influencing fraud predictions:")
print(shap_importance.head(10).to_string(index=False))

shap_importance.to_csv(
    OUTPUT_DIR / "shap_feature_importance.csv",
    index=False,
)

top_features = shap_importance.head(10).sort_values(
    "Mean_Absolute_SHAP"
)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Mean_Absolute_SHAP"],
)
plt.xlabel("Mean absolute SHAP value")
plt.title("Top Features Influencing Fraud Predictions")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_feature_importance.png", dpi=300)
plt.show()
plt.close()


# ============================================================
# 12. TRANSACTION-LEVEL INVESTIGATION
# ============================================================

print_section("11. TRANSACTION-LEVEL INVESTIGATION")

# Select the highest-scoring transaction from a reproducible test sample.
# The actual label is shown only for retrospective evaluation.
investigation_sample = test_mask[test_mask].index
investigation_indices = pd.Series(investigation_sample).sample(
    n=min(INVESTIGATION_SAMPLE_SIZE, len(investigation_sample)),
    random_state=RANDOM_STATE,
).to_numpy()

investigation_X = df.loc[investigation_indices, features]
investigation_probabilities = rf_model.predict_proba(investigation_X)[:, 1]

selected_position = int(np.argmax(investigation_probabilities))
selected_index = investigation_indices[selected_position]
transaction = df.loc[[selected_index], features].copy()

fraud_probability = float(investigation_probabilities[selected_position])

if fraud_probability >= 0.70:
    risk_level = "HIGH"
elif fraud_probability >= 0.30:
    risk_level = "MEDIUM"
else:
    risk_level = "LOW"

actual_label = int(df.loc[selected_index, "isFraud"])

print("\nSelected transaction details:")
print(transaction.to_string(index=False))
print(f"\nEstimated fraud probability: {fraud_probability * 100:.4f}%")
print(f"Illustrative risk category: {risk_level}")
print(
    "Actual dataset label (retrospective evaluation only):",
    "Fraud" if actual_label == 1 else "Legitimate",
)

investigation_result = transaction.copy()
investigation_result["estimated_fraud_probability"] = fraud_probability
investigation_result["risk_level"] = risk_level
investigation_result["actual_label_for_evaluation_only"] = actual_label

investigation_result.to_csv(
    OUTPUT_DIR / "transaction_investigation.csv",
    index=False,
)

# Save the fitted Random Forest pipeline for the Streamlit application.
model_path = OUTPUT_DIR / "fraud_model.joblib"
joblib.dump(rf_model, model_path)
print(f"\nTrained Random Forest model saved to: {model_path}")


# ============================================================
# 13. ACCOUNT NETWORK ANALYSIS (WITHOUT FRAUD LABELS)
# ============================================================

print_section("12. ACCOUNT NETWORK ANALYSIS")

# Build the exploratory network from a random transaction sample.
# Fraud labels are deliberately not used to rank accounts.
network_columns = [
    "nameOrig", "nameDest", "amount", "type"
]
network_sample = df[network_columns].sample(
    n=min(NETWORK_SAMPLE_SIZE, len(df)),
    random_state=RANDOM_STATE,
).copy()

# Aggregate repeated transfers between the same sender-recipient pair.
edge_summary = network_sample.groupby(
    ["nameOrig", "nameDest"],
    as_index=False,
).agg(
    transaction_count=("amount", "size"),
    total_amount=("amount", "sum"),
)

G = nx.from_pandas_edgelist(
    edge_summary,
    source="nameOrig",
    target="nameDest",
    edge_attr=["transaction_count", "total_amount"],
    create_using=nx.DiGraph(),
)

print(f"Accounts in sampled network: {G.number_of_nodes():,}")
print(f"Unique directed account connections: {G.number_of_edges():,}")

# Distinct incoming/outgoing counterparties.
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

account_metrics = pd.DataFrame({
    "Account": list(G.nodes()),
    "Incoming_Counterparties": [
        in_degree[node] for node in G.nodes()
    ],
    "Outgoing_Counterparties": [
        out_degree[node] for node in G.nodes()
    ],
})

# Transaction counts and amounts by sender and recipient.
outgoing = edge_summary.groupby("nameOrig").agg(
    Outgoing_Transaction_Count=("transaction_count", "sum"),
    Outgoing_Amount=("total_amount", "sum"),
)

incoming = edge_summary.groupby("nameDest").agg(
    Incoming_Transaction_Count=("transaction_count", "sum"),
    Incoming_Amount=("total_amount", "sum"),
)

account_metrics = account_metrics.set_index("Account")
account_metrics = account_metrics.join(outgoing).join(incoming).fillna(0)
account_metrics = account_metrics.reset_index()

account_metrics["Total_Observed_Transactions"] = (
    account_metrics["Outgoing_Transaction_Count"]
    + account_metrics["Incoming_Transaction_Count"]
)

account_metrics["Total_Observed_Amount"] = (
    account_metrics["Outgoing_Amount"]
    + account_metrics["Incoming_Amount"]
)

account_metrics = account_metrics.sort_values(
    [
        "Total_Observed_Transactions",
        "Incoming_Counterparties",
        "Outgoing_Counterparties",
    ],
    ascending=False,
)

print("\nMost active accounts in the sampled network:")
print(account_metrics.head(15).to_string(index=False))

account_metrics.to_csv(
    OUTPUT_DIR / "account_network_analysis.csv",
    index=False,
)

print(
    "\nNetwork ranking is based on observed activity, "
    "not confirmed or predicted fraud."
)


# ============================================================
# 14. TEMPORAL FRAUD VALIDATION
# ============================================================

print_section("13. TEMPORAL FRAUD VALIDATION")

temporal_summary = df.groupby("step").agg(
    total_transactions=("isFraud", "size"),
    fraud_count=("isFraud", "sum"),
).reset_index()

temporal_summary["fraud_rate_pct"] = (
    temporal_summary["fraud_count"]
    / temporal_summary["total_transactions"]
    * 100
)

temporal_summary["period"] = np.where(
    temporal_summary["step"] < cutoff_step,
    "Training",
    "Testing",
)

period_summary = temporal_summary.groupby("period").agg(
    first_step=("step", "min"),
    last_step=("step", "max"),
    total_transactions=("total_transactions", "sum"),
    fraud_count=("fraud_count", "sum"),
)

period_summary["overall_fraud_rate_pct"] = (
    period_summary["fraud_count"]
    / period_summary["total_transactions"]
    * 100
)

print("\nTransaction-weighted fraud summary by period:")
print(period_summary.to_string())

print("\nTime steps with the highest transaction volume:")
print(
    temporal_summary.sort_values(
        "total_transactions",
        ascending=False,
    ).head(10).to_string(index=False)
)

print("\nTime steps with the highest fraud rates:")
print(
    temporal_summary.sort_values(
        "fraud_rate_pct",
        ascending=False,
    ).head(10).to_string(index=False)
)

temporal_summary.to_csv(
    OUTPUT_DIR / "temporal_fraud_summary.csv",
    index=False,
)
period_summary.to_csv(
    OUTPUT_DIR / "temporal_period_summary.csv",
)

plt.figure(figsize=(12, 5))
plt.plot(
    temporal_summary["step"],
    temporal_summary["fraud_count"],
    label="Fraud count",
)
plt.axvline(
    cutoff_step,
    linestyle="--",
    label=f"Train/test boundary: step {cutoff_step}",
)
plt.xlabel("Simulation step")
plt.ylabel("Fraudulent transactions")
plt.title("Fraud Distribution Across Simulation Time")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fraud_temporal_validation.png", dpi=300)
plt.show()
plt.close()


# ============================================================
# 15. PHASE 4: THRESHOLD OPTIMISATION AND MODEL EVALUATION
# ============================================================

print_section("PHASE 4: THRESHOLD OPTIMISATION AND MODEL EVALUATION")

# Use the Random Forest probabilities already calculated for X_test.
# Keep the chronological order: earlier test steps are used for threshold
# calibration, and later test steps are reserved for evaluation.
rf_test_probabilities = test_probabilities["Random Forest"]
test_steps = X_test["step"].to_numpy()
unique_test_steps = np.sort(np.unique(test_steps))

if len(unique_test_steps) < 2:
    raise ValueError(
        "Phase 4 requires at least two distinct test-period time steps."
    )

midpoint_position = max(1, len(unique_test_steps) // 2)
calibration_end_step = unique_test_steps[midpoint_position - 1]
evaluation_start_step = unique_test_steps[midpoint_position]

calibration_mask = test_steps <= calibration_end_step
evaluation_mask = test_steps >= evaluation_start_step

y_calibration = y_test.to_numpy()[calibration_mask]
y_evaluation = y_test.to_numpy()[evaluation_mask]
probabilities_calibration = rf_test_probabilities[calibration_mask]
probabilities_evaluation = rf_test_probabilities[evaluation_mask]

print(f"Calibration period: test steps up to {calibration_end_step}")
print(f"Evaluation period: test steps from {evaluation_start_step}")
print(f"Calibration transactions: {len(y_calibration):,}")
print(f"Calibration fraud cases: {int(y_calibration.sum()):,}")
print(f"Evaluation transactions: {len(y_evaluation):,}")
print(f"Evaluation fraud cases: {int(y_evaluation.sum()):,}")

if y_calibration.sum() == 0 or y_evaluation.sum() == 0:
    raise ValueError(
        "Both Phase 4 periods must contain fraud cases. Review the test-period split."
    )


def calculate_threshold_metrics(actual, probabilities, threshold):
    """Calculate fraud-class metrics and confusion-matrix counts."""
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "Threshold": float(threshold),
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1_score,
        "True_Positives": int(tp),
        "False_Positives": int(fp),
        "True_Negatives": int(tn),
        "False_Negatives": int(fn),
        "Total_Alerts": int(tp + fp),
    }


# Compare thresholds on the calibration period only.
threshold_values = np.round(np.arange(0.05, 1.00, 0.05), 2)
threshold_results = [
    calculate_threshold_metrics(
        y_calibration,
        probabilities_calibration,
        threshold,
    )
    for threshold in threshold_values
]

threshold_results_df = pd.DataFrame(threshold_results)
threshold_results_df.to_csv(
    OUTPUT_DIR / "threshold_optimization.csv",
    index=False,
)

print("\nThreshold comparison on calibration period:")
print(
    threshold_results_df[
        ["Threshold", "Precision", "Recall", "F1_Score", "Total_Alerts"]
    ].to_string(index=False, formatters={
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1_Score": "{:.4f}".format,
    })
)

# Select the threshold with the highest calibration F1-score.
# If F1 scores tie, prefer higher precision, then the higher threshold.
best_threshold_row = threshold_results_df.sort_values(
    ["F1_Score", "Precision", "Threshold"],
    ascending=[False, False, False],
).iloc[0]
selected_threshold = float(best_threshold_row["Threshold"])

print(f"\nSelected candidate threshold: {selected_threshold:.2f}")
print("Selection basis: highest F1-score on the calibration period")

# Evaluate the selected threshold on the later, untouched evaluation period.
evaluation_metrics = calculate_threshold_metrics(
    y_evaluation,
    probabilities_evaluation,
    selected_threshold,
)
baseline_evaluation_metrics = calculate_threshold_metrics(
    y_evaluation,
    probabilities_evaluation,
    CLASSIFICATION_THRESHOLD,
)

evaluation_comparison = pd.DataFrame([
    {"Evaluation": "Default threshold", **baseline_evaluation_metrics},
    {"Evaluation": "Selected threshold", **evaluation_metrics},
])
evaluation_comparison.to_csv(
    OUTPUT_DIR / "threshold_evaluation_comparison.csv",
    index=False,
)

print("\nEvaluation-period comparison:")
print(
    evaluation_comparison[
        [
            "Evaluation", "Threshold", "Precision", "Recall", "F1_Score",
            "True_Positives", "False_Positives", "False_Negatives", "Total_Alerts",
        ]
    ].to_string(index=False, formatters={
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1_Score": "{:.4f}".format,
    })
)

# Plot precision, recall and F1-score across calibration thresholds.
plt.figure(figsize=(10, 6))
plt.plot(
    threshold_results_df["Threshold"],
    threshold_results_df["Precision"],
    marker="o",
    label="Precision",
)
plt.plot(
    threshold_results_df["Threshold"],
    threshold_results_df["Recall"],
    marker="o",
    label="Recall",
)
plt.plot(
    threshold_results_df["Threshold"],
    threshold_results_df["F1_Score"],
    marker="o",
    label="F1-score",
)
plt.axvline(
    selected_threshold,
    linestyle="--",
    label=f"Selected threshold: {selected_threshold:.2f}",
)
plt.xlabel("Classification threshold")
plt.ylabel("Score")
plt.title("Threshold Optimisation - Calibration Period")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "threshold_performance.png", dpi=300)
plt.show()
plt.close()

# Save the confusion matrix for the later evaluation period.
evaluation_predictions = (
    probabilities_evaluation >= selected_threshold
).astype(int)
evaluation_cm = confusion_matrix(
    y_evaluation,
    evaluation_predictions,
    labels=[0, 1],
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    evaluation_cm,
    annot=True,
    fmt=",d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title(f"Evaluation Confusion Matrix (Threshold {selected_threshold:.2f})")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "threshold_evaluation_confusion_matrix.png",
    dpi=300,
)
plt.show()
plt.close()

# Save the chosen threshold and its calibration details for later use by Streamlit.
model_config = {
    "model_file": "fraud_model.joblib",
    "model_name": "Random Forest",
    "selected_threshold": selected_threshold,
    "default_threshold": CLASSIFICATION_THRESHOLD,
    "selection_method": "Highest F1-score on the earlier calibration portion of the test period",
    "calibration_end_step": int(calibration_end_step),
    "evaluation_start_step": int(evaluation_start_step),
    "calibration_f1_score": float(best_threshold_row["F1_Score"]),
    "evaluation_precision": float(evaluation_metrics["Precision"]),
    "evaluation_recall": float(evaluation_metrics["Recall"]),
    "evaluation_f1_score": float(evaluation_metrics["F1_Score"]),
    "note": "PaySim is synthetic. Threshold is a candidate decision setting, not a validated banking policy. Reassess against operational alert capacity and future data.",
}

with open(OUTPUT_DIR / "fraud_model_config.json", "w", encoding="utf-8") as config_file:
    json.dump(model_config, config_file, indent=4)

print("\nPhase 4 files saved:")
for filename in [
    "threshold_optimization.csv",
    "threshold_evaluation_comparison.csv",
    "threshold_performance.png",
    "threshold_evaluation_confusion_matrix.png",
    "fraud_model_config.json",
]:
    print(f"- {filename}")

print(
    "\nImportant: the selected threshold was calibrated on the earlier part "
    "of the test period and evaluated on the later part. It is a candidate "
    "threshold and should be reviewed against business alert capacity."
)


# ============================================================
# 16. FINAL OUTPUT SUMMARY
# ============================================================

print_section("PROJECT EXECUTION COMPLETED")

print("Generated files are saved in:")
print(OUTPUT_DIR)

print("\nKey output files:")
output_files = [
    "fraud_by_type.csv",
    "amount_summary.csv",
    "fraud_by_step.csv",
    "model_comparison.csv",
    "model_comparison.png",
    "shap_feature_importance.csv",
    "shap_feature_importance.png",
    "transaction_investigation.csv",
    "fraud_model.joblib",
    "account_network_analysis.csv",
    "temporal_fraud_summary.csv",
    "temporal_period_summary.csv",
    "fraud_temporal_validation.png",
    "threshold_optimization.csv",
    "threshold_evaluation_comparison.csv",
    "threshold_performance.png",
    "threshold_evaluation_confusion_matrix.png",
    "fraud_model_config.json",
]

for filename in output_files:
    print(f"- {filename}")

print("\nNote:")
print(
    "Fraud risk thresholds are illustrative. Model scores and network "
    "activity indicators are decision-support signals, not proof of fraud."
)
print(
    "PaySim is a synthetic dataset; results should not be presented as "
    "validated real-world banking performance."
)
print("\nAll stages completed successfully.")
