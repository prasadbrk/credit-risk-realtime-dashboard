"""
Complete ML Pipeline with Polars
Gold Layer - Model Training & Evaluation
"""

import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_curve
)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

print("="*70)
print("GOLD LAYER - ML MODEL TRAINING (Polars + XGBoost)")
print("="*70)

# Configuration
SILVER_PATH = "./data/parquet/silver/features"
MODEL_DIR = "models"
RESULTS_DIR = "results"

# Create directories
Path(MODEL_DIR).mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)

# ============================================================================
# STEP 1: Load Silver Features
# ============================================================================

print("\n[1/5] Loading Silver features...")

# Lazy load for efficiency
df = pl.scan_parquet(f"{SILVER_PATH}/**/*.parquet").collect()

print(f"✓ Loaded {len(df):,} records")
print(f"  Columns: {len(df.columns)}")

# ============================================================================
# STEP 2: Feature Selection & Preparation
# ============================================================================

print("\n[2/5] Preparing features for ML...")

# Define feature sets
NUMERIC_FEATURES = [
    'credit_score', 'loan_amount', 'monthly_income', 'interest_rate',
    'loan_term_months', 'current_dpd', 'max_dpd_12m', 'late_payments_12m',
    'payment_history_pct', 'existing_emi', 'existing_loans', 'ltv_ratio',
    'dti_ratio', 'age', 'dependents', 'credit_history_years', 'emi_burden_pct',
    'total_debt_burden', 'payment_consistency_score', 'payment_behavior_score',
    'per_capita_income', 'employment_stability_score', 'combined_risk_score',
    'loan_vintage_months', 'days_since_disbursement', 'loan_maturity_pct'
]

CATEGORICAL_FEATURES = [
    # 'loan_purpose',
    'employment_type', 'education', 'city_tier', 'marital_status'
]

TARGET = 'is_default'

# Select features + target
feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
available_cols = [col for col in feature_cols if col in df.columns]

df_ml = df.select(available_cols)

# Convert to Pandas for sklearn compatibility
print("  Converting to Pandas for sklearn...")
df_pandas = df_ml.to_pandas()

# Prepare feature matrix
X = df_pandas[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
y = df_pandas[TARGET]

# Encode categorical features
print("  Encoding categorical features...")
label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Handle missing values
X = X.fillna(X.median())

print(f"✓ Feature matrix shape: {X.shape}")
print(f"  Features: {len(X.columns)}")
print(f"  Default rate: {y.mean():.2%}")

# ============================================================================
# STEP 3: Train-Test Split
# ============================================================================

print("\n[3/5] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training set: {len(X_train):,} records ({y_train.mean():.2%} default rate)")
print(f"  Test set: {len(X_test):,} records ({y_test.mean():.2%} default rate)")

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f"  Class imbalance ratio: {scale_pos_weight:.2f}")

# ============================================================================
# STEP 4: Train XGBoost Model
# ============================================================================
print("\n[4/5] Training XGBoost classifier...")

model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20  # Now a parameter in XGBoost 2.0+
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print(f"✓ Model trained (best iteration: {model.best_iteration})")

# ============================================================================
# STEP 5: Evaluate Model
# ============================================================================

print("\n[5/5] Evaluating model performance...")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f} {'✓ Excellent!' if auc > 0.85 else '✓ Good!'}")
print("="*70)

# Classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"                Predicted")
print(f"              No    Yes")
print(f"Actual No   {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"       Yes  {cm[1,0]:5d}  {cm[1,1]:5d}")

# Feature importance
feature_importance = pl.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort('importance', descending=True)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15))

# ============================================================================
# Save Model Artifacts
# ============================================================================

print("\n" + "="*70)
print("SAVING MODEL ARTIFACTS")
print("="*70)

# Save model
model_file = os.path.join(MODEL_DIR, 'xgboost_credit_model.pkl')
joblib.dump(model, model_file)
print(f"✓ Model: {model_file}")

# Save encoders
encoders_file = os.path.join(MODEL_DIR, 'label_encoders.pkl')
joblib.dump(label_encoders, encoders_file)
print(f"✓ Encoders: {encoders_file}")

# Save feature list
features_file = os.path.join(MODEL_DIR, 'model_features.txt')
with open(features_file, 'w') as f:
    f.write('\n'.join(X_train.columns.tolist()))
print(f"✓ Features: {features_file}")

# Save feature importance (as Parquet for efficiency)
importance_file = os.path.join(RESULTS_DIR, 'feature_importance.parquet')
feature_importance.write_parquet(importance_file)
print(f"✓ Feature importance: {importance_file}")

# Also save as CSV for compatibility
importance_csv = os.path.join(RESULTS_DIR, 'feature_importance.csv')
feature_importance.write_csv(importance_csv)
print(f"✓ Feature importance (CSV): {importance_csv}")

# ============================================================================
# Generate Visualizations
# ============================================================================

print("\nGenerating visualizations...")

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
cm_file = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
plt.savefig(cm_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Confusion matrix: {cm_file}")

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Model (AUC = {auc:.4f})', linewidth=2, color='#2563eb')
plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
roc_file = os.path.join(RESULTS_DIR, 'roc_curve.png')
plt.savefig(roc_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ ROC curve: {roc_file}")

# 3. Feature Importance
plt.figure(figsize=(10, 8))
top_20 = feature_importance.head(20).to_pandas().sort_values('importance')
plt.barh(range(len(top_20)), top_20['importance'], color='#6366f1')
plt.yticks(range(len(top_20)), top_20['feature'])
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
fi_file = os.path.join(RESULTS_DIR, 'feature_importance.png')
plt.savefig(fi_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Feature importance plot: {fi_file}")

# Save model summary
summary_file = os.path.join(RESULTS_DIR, 'model_summary.txt')
with open(summary_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("CREDIT RISK MODEL - TRAINING SUMMARY (Polars Pipeline)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Dataset:\n")
    f.write(f"  Total Records: {len(df):,}\n")
    f.write(f"  Training Set: {len(X_train):,}\n")
    f.write(f"  Test Set: {len(X_test):,}\n")
    f.write(f"  Default Rate: {y.mean():.2%}\n\n")
    f.write(f"Features:\n")
    f.write(f"  Numeric: {len(NUMERIC_FEATURES)}\n")
    f.write(f"  Categorical: {len(CATEGORICAL_FEATURES)}\n")
    f.write(f"  Total: {len(X_train.columns)}\n\n")
    f.write(f"Model Performance:\n")
    f.write(f"  Accuracy:  {accuracy:.4f}\n")
    f.write(f"  Precision: {precision:.4f}\n")
    f.write(f"  Recall:    {recall:.4f}\n")
    f.write(f"  F1 Score:  {f1:.4f}\n")
    f.write(f"  AUC-ROC:   {auc:.4f}\n\n")
    f.write(f"Pipeline Technology:\n")
    f.write(f"  Data Processing: Polars (Rust-based, parallel)\n")
    f.write(f"  Storage Format: Parquet (partitioned)\n")
    f.write(f"  ML Framework: XGBoost\n")

print(f"✓ Summary: {summary_file}")

# ============================================================================
# Final Summary
# ============================================================================

print("\n" + "="*70)
print("✅ GOLD LAYER COMPLETE - PIPELINE FINISHED!")
print("="*70)
print(f"\nModel Artifacts:")
print(f"  📁 {MODEL_DIR}/")
print(f"     ├── xgboost_credit_model.pkl")
print(f"     ├── label_encoders.pkl")
print(f"     └── model_features.txt")
print(f"\nResults:")
print(f"  📁 {RESULTS_DIR}/")
print(f"     ├── confusion_matrix.png")
print(f"     ├── roc_curve.png")
print(f"     ├── feature_importance.png")
print(f"     ├── feature_importance.parquet")
print(f"     ├── feature_importance.csv")
print(f"     └── model_summary.txt")

print(f"\n🚀 Next Steps:")
print(f"  1. Launch dashboard: streamlit run dashboards/streamlit_app.py")
print(f"  2. Review results in {RESULTS_DIR}/ folder")
print(f"  3. Commit to GitHub")

print(f"\n📊 Your Model Stats for LinkedIn:")
print(f"  • Technology: Polars (10-100x faster than Pandas)")
print(f"  • AUC-ROC: {auc:.2f} ({'Excellent' if auc > 0.85 else 'Good'} performance)")
print(f"  • Features: {len(X_train.columns)} engineered features")
print(f"  • Dataset: {len(df):,} loan records")
print(f"  • Storage: Partitioned Parquet (Snappy compression)")
print(f"  • Processing: Parallel, zero-copy Arrow format")

print("\n" + "="*70)
print("✓ Polars pipeline demonstrates modern data engineering!")
print("="*70)