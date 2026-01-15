"""Simplified local pipeline for testing"""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

print("Loading data...")
df = pd.read_csv('data/loan_portfolio_sample.csv')
print(f"Loaded {len(df):,} records")

# Simple feature selection
numeric_features = [
    'credit_score', 'loan_amount', 'monthly_income', 
    'current_dpd', 'late_payments_12m', 'dti_ratio'
]

categorical_features = ['employment_type', 'loan_purpose']

# Prepare features
X = df[numeric_features + categorical_features].copy()
y = df['is_default']

# Encode categoricals
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X = X.fillna(X.median())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Train model
print("\nTraining XGBoost model...")
model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\nModel Performance:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgboost_credit_model.pkl')
print("\n✓ Model saved to models/xgboost_credit_model.pkl")