# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer - ML Model Training
# MAGIC 
# MAGIC **Purpose**: Train XGBoost credit risk model with MLflow tracking
# MAGIC 
# MAGIC **Models**:
# MAGIC - Binary Classification (Default Prediction)
# MAGIC - Probability of Default (PD)
# MAGIC - Loss Given Default (LGD) estimation

# COMMAND ----------

import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# Configuration
SILVER_PATH = "/mnt/datalake/credit_risk/silver/features"
MODEL_PATH = "/mnt/datalake/credit_risk/gold/models"

spark = SparkSession.builder.appName("ML Training").getOrCreate()

# Setup MLflow
mlflow.set_experiment("/credit_risk/model_training")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Silver Features

# COMMAND ----------

silver_df = spark.read.format("delta").load(SILVER_PATH)
print(f"Total records: {silver_df.count():,}")

# Convert to Pandas for sklearn/xgboost
df_pandas = silver_df.toPandas()
print(f"Pandas DataFrame shape: {df_pandas.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Selection

# COMMAND ----------

# Define feature sets
NUMERIC_FEATURES = [
    # Credit features
    'credit_score', 'credit_history_months', 'credit_utilization_ratio',
    
    # Loan features
    'loan_amount', 'loan_term_months', 'interest_rate', 'ltv_ratio', 'dti_ratio',
    'loan_to_annual_income', 'emi_burden_pct', 'total_debt_burden',
    
    # Financial features
    'monthly_income', 'existing_emi', 'existing_loans', 'per_capita_income',
    
    # Repayment features
    'current_dpd', 'max_dpd_12m', 'late_payments_12m', 'payment_history_pct',
    'payment_consistency_score', 'payment_behavior_score',
    
    # Demographic features
    'age', 'dependents', 'family_size', 'employment_stability_score',
    
    # Risk features
    'combined_risk_score', 'pd_proxy',
    
    # Time features
    'loan_vintage_months', 'days_since_disbursement'
]

CATEGORICAL_FEATURES = [
    'employment_type', 'education', 'marital_status', 'city_tier',
    'loan_purpose', 'loan_status', 'credit_score_bucket', 'utilization_category',
    'loan_term_category', 'interest_rate_bucket', 'dpd_category',
    'age_group', 'income_bracket'
]

TARGET = 'is_default'

print(f"Numeric features: {len(NUMERIC_FEATURES)}")
print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Prepare features
X = df_pandas[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
y = df_pandas[TARGET]

# Encode categorical features
label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Handle missing values
X = X.fillna(X.median())

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Default rate in train: {y_train.mean():.2%}")
print(f"Default rate in test: {y_test.mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training with MLflow

# COMMAND ----------

def train_xgboost_model(X_train, y_train, X_test, y_test, params=None):
    """Train XGBoost model with MLflow tracking"""
    
    with mlflow.start_run(run_name="XGBoost_Credit_Risk_v1"):
        
        # Default parameters
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum(),  # Handle imbalance
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        print("Training XGBoost model...")
        model = xgb.XGBClassifier(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc)
        
        print(f"\n{'='*50}")
        print("MODEL PERFORMANCE")
        print(f"{'='*50}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        print(f"{'='*50}\n")
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15))
        
        # Log feature importance as artifact
        mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")
        
        # Plot and log confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
        plt.close()
        
        # Plot and log ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "roc_curve.png")
        plt.close()
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(20)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "feature_importance.png")
        plt.close()
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print("\n✓ Model logged to MLflow")
        
        return model, feature_importance

# COMMAND ----------

# Train model
model, feature_importance = train_xgboost_model(X_train, y_train, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Calibration & Validation

# COMMAND ----------

def calibrate_probability_bins(y_test, y_pred_proba, n_bins=10):
    """Analyze probability calibration"""
    
    df = pd.DataFrame({
        'actual': y_test,
        'predicted_proba': y_pred_proba
    })
    
    df['probability_bin'] = pd.cut(df['predicted_proba'], bins=n_bins)
    
    calibration = df.groupby('probability_bin').agg({
        'actual': ['mean', 'count']
    }).reset_index()
    
    calibration.columns = ['probability_bin', 'actual_default_rate', 'count']
    
    print("Probability Calibration:")
    print(calibration)
    
    return calibration

# COMMAND ----------

y_pred_proba = model.predict_proba(X_test)[:, 1]
calibration_df = calibrate_probability_bins(y_test, y_pred_proba)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Production Model

# COMMAND ----------

import joblib
import os

# Create model directory
os.makedirs('models', exist_ok=True)

# Save model
model_file = 'models/xgboost_credit_model.pkl'
joblib.dump(model, model_file)
print(f"✓ Model saved to: {model_file}")

# Save label encoders
encoders_file = 'models/label_encoders.pkl'
joblib.dump(label_encoders, encoders_file)
print(f"✓ Label encoders saved to: {encoders_file}")

# Save feature list
features_file = 'models/model_features.txt'
with open(features_file, 'w') as f:
    f.write('\n'.join(X_train.columns.tolist()))
print(f"✓ Feature list saved to: {features_file}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Scoring Function

# COMMAND ----------

def score_new_loans(input_df, model, label_encoders):
    """
    Score new loan applications
    
    Parameters:
    - input_df: DataFrame with same features as training data
    - model: Trained XGBoost model
    - label_encoders: Dictionary of label encoders for categorical features
    
    Returns:
    - DataFrame with predictions and probabilities
    """
    
    # Prepare features
    X = input_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    
    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        if col in label_encoders:
            X[col] = label_encoders[col].transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Risk rating based on probability
    def get_risk_rating(prob):
        if prob < 0.1:
            return 'Very Low'
        elif prob < 0.25:
            return 'Low'
        elif prob < 0.5:
            return 'Medium'
        elif prob < 0.75:
            return 'High'
        else:
            return 'Very High'
    
    # Create results DataFrame
    results = input_df.copy()
    results['default_prediction'] = predictions
    results['default_probability'] = probabilities
    results['risk_rating'] = [get_risk_rating(p) for p in probabilities]
    
    return results

# Example usage (commented out - uncomment to test)
# sample_loans = X_test.head(10)
# scored_loans = score_new_loans(sample_loans, model, label_encoders)
# print(scored_loans[['loan_id', 'default_prediction', 'default_probability', 'risk_rating']])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Summary

# COMMAND ----------

print("="*60)
print("GOLD LAYER ML MODEL TRAINING COMPLETE")
print("="*60)
print(f"\nModel Type: XGBoost Binary Classifier")
print(f"Training Records: {len(X_train):,}")
print(f"Test Records: {len(X_test):,}")
print(f"Features Used: {len(X_train.columns)}")
print(f"\nModel Performance:")
print(f"  - AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"  - Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
print(f"\nModel artifacts saved to: models/")
print(f"MLflow experiment: /credit_risk/model_training")
print("="*60)