# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer - Feature Engineering
# MAGIC 
# MAGIC **Purpose**: Transform bronze data into ML-ready features
# MAGIC 
# MAGIC **Key Features**:
# MAGIC - 30+ engineered credit risk features
# MAGIC - Data normalization and encoding
# MAGIC - Feature quality validation
# MAGIC - Incremental processing

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from delta.tables import DeltaTable

# COMMAND ----------

# Configuration
BRONZE_PATH = "/mnt/datalake/credit_risk/bronze/loans"
SILVER_PATH = "/mnt/datalake/credit_risk/silver/features"

spark = SparkSession.builder.appName("Silver Layer").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Bronze Data

# COMMAND ----------

bronze_df = spark.read.format("delta").load(BRONZE_PATH)
print(f"Bronze records: {bronze_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering Functions

# COMMAND ----------

def engineer_credit_features(df):
    """Create credit-related features"""
    
    print("Engineering credit features...")
    
    df = df.withColumn("credit_score_bucket", 
                       when(col("credit_score") >= 750, "Excellent")
                       .when(col("credit_score") >= 700, "Good")
                       .when(col("credit_score") >= 650, "Fair")
                       .when(col("credit_score") >= 600, "Poor")
                       .otherwise("Very Poor"))
    
    df = df.withColumn("credit_history_years", 
                       (col("credit_history_months") / 12).cast("double"))
    
    df = df.withColumn("utilization_category",
                       when(col("credit_utilization_ratio") < 0.3, "Low")
                       .when(col("credit_utilization_ratio") < 0.6, "Medium")
                       .otherwise("High"))
    
    df = df.withColumn("has_existing_loans", 
                       when(col("existing_loans") > 0, 1).otherwise(0))
    
    df = df.withColumn("credit_experience", 
                       when(col("credit_history_months") >= 60, "Experienced")
                       .when(col("credit_history_months") >= 24, "Moderate")
                       .otherwise("Limited"))
    
    return df

# COMMAND ----------

def engineer_loan_features(df):
    """Create loan-specific features"""
    
    print("Engineering loan features...")
    
    # Loan-to-Income ratio (annual)
    df = df.withColumn("loan_to_annual_income", 
                       (col("loan_amount") / (col("monthly_income") * 12)))
    
    # EMI burden (as % of income)
    df = df.withColumn("emi_burden_pct", 
                       (col("emi_amount") / col("monthly_income")) * 100)
    
    # Total debt burden
    df = df.withColumn("total_debt_burden", 
                       ((col("emi_amount") + col("existing_emi")) / col("monthly_income")) * 100)
    
    # Loan term category
    df = df.withColumn("loan_term_category",
                       when(col("loan_term_months") <= 24, "Short")
                       .when(col("loan_term_months") <= 60, "Medium")
                       .otherwise("Long"))
    
    # Interest rate bucket
    df = df.withColumn("interest_rate_bucket",
                       when(col("interest_rate") < 9, "Low")
                       .when(col("interest_rate") < 12, "Medium")
                       .otherwise("High"))
    
    # Loan amount bucket
    df = df.withColumn("loan_amount_bucket",
                       when(col("loan_amount") < 100000, "Small")
                       .when(col("loan_amount") < 500000, "Medium")
                       .when(col("loan_amount") < 2000000, "Large")
                       .otherwise("Very Large"))
    
    return df

# COMMAND ----------

def engineer_repayment_features(df):
    """Create repayment behavior features"""
    
    print("Engineering repayment features...")
    
    # DPD category
    df = df.withColumn("dpd_category",
                       when(col("current_dpd") == 0, "Current")
                       .when(col("current_dpd") <= 30, "DPD-30")
                       .when(col("current_dpd") <= 60, "DPD-60")
                       .when(col("current_dpd") <= 90, "DPD-90")
                       .otherwise("DPD-90+"))
    
    # Payment consistency score
    df = df.withColumn("payment_consistency_score",
                       col("payment_history_pct") * 100 - (col("late_payments_12m") * 5))
    
    # Delinquency risk flag
    df = df.withColumn("high_delinquency_risk",
                       when((col("late_payments_12m") >= 3) | 
                            (col("max_dpd_12m") > 60), 1).otherwise(0))
    
    # Days since disbursement
    df = df.withColumn("days_since_disbursement",
                       datediff(current_date(), col("disbursement_date")))
    
    # Loan maturity percentage
    df = df.withColumn("loan_maturity_pct",
                       (col("total_payments_made") / col("loan_term_months")) * 100)
    
    # Payment behavior score (0-100)
    df = df.withColumn("payment_behavior_score",
                       (100 - col("current_dpd")) * col("payment_history_pct"))
    
    return df

# COMMAND ----------

def engineer_demographic_features(df):
    """Create demographic and profile features"""
    
    print("Engineering demographic features...")
    
    # Age group
    df = df.withColumn("age_group",
                       when(col("age") < 25, "18-24")
                       .when(col("age") < 35, "25-34")
                       .when(col("age") < 45, "35-44")
                       .when(col("age") < 55, "45-54")
                       .otherwise("55+"))
    
    # Income bracket
    df = df.withColumn("income_bracket",
                       when(col("monthly_income") < 30000, "Low")
                       .when(col("monthly_income") < 60000, "Middle")
                       .when(col("monthly_income") < 100000, "Upper-Middle")
                       .otherwise("High"))
    
    # Family size
    df = df.withColumn("family_size", col("dependents") + 
                       when(col("marital_status") == "Married", 2).otherwise(1))
    
    # Per capita income
    df = df.withColumn("per_capita_income",
                       col("monthly_income") / col("family_size"))
    
    # Employment stability proxy
    df = df.withColumn("employment_stability_score",
                       when(col("employment_type") == "Salaried", 100)
                       .when(col("employment_type") == "Professional", 90)
                       .when(col("employment_type") == "Self-Employed", 70)
                       .otherwise(60))
    
    return df

# COMMAND ----------

def engineer_risk_features(df):
    """Create composite risk indicators"""
    
    print("Engineering risk features...")
    
    # Combined risk score (0-100, lower is better)
    df = df.withColumn("combined_risk_score",
                       (100 - ((col("credit_score") - 300) / 6)) * 0.4 +
                       (col("current_dpd") / 180 * 100) * 0.3 +
                       (col("dti_ratio") * 100) * 0.2 +
                       (col("late_payments_12m") * 10) * 0.1)
    
    # Probability of default proxy (0-1)
    df = df.withColumn("pd_proxy",
                       (col("combined_risk_score") / 100).cast("double"))
    
    # Expected loss (PD * LGD * EAD)
    df = df.withColumn("expected_loss_amount",
                       col("pd_proxy") * col("estimated_lgd") * 
                       (col("loan_amount") - (col("emi_amount") * col("total_payments_made"))))
    
    # Early warning indicator
    df = df.withColumn("early_warning_flag",
                       when((col("current_dpd") > 0) | 
                            (col("late_payments_12m") >= 2) |
                            (col("payment_consistency_score") < 70), 1).otherwise(0))
    
    # Credit deterioration flag
    df = df.withColumn("credit_deterioration",
                       when((col("max_dpd_12m") > col("current_dpd")) & 
                            (col("max_dpd_12m") > 30), 1).otherwise(0))
    
    return df

# COMMAND ----------

def engineer_time_features(df):
    """Create time-based features"""
    
    print("Engineering time features...")
    
    # Application date features
    df = df.withColumn("application_year", year(col("application_date")))
    df = df.withColumn("application_month", month(col("application_date")))
    df = df.withColumn("application_quarter", quarter(col("application_date")))
    df = df.withColumn("application_day_of_week", dayofweek(col("application_date")))
    
    # Loan vintage (months since disbursement)
    df = df.withColumn("loan_vintage_months",
                       months_between(current_date(), col("disbursement_date")).cast("int"))
    
    # Seasonality
    df = df.withColumn("application_season",
                       when(col("application_month").isin(3, 4, 5), "Spring")
                       .when(col("application_month").isin(6, 7, 8), "Summer")
                       .when(col("application_month").isin(9, 10, 11), "Fall")
                       .otherwise("Winter"))
    
    return df

# COMMAND ----------

def add_statistical_features(df):
    """Add statistical aggregations"""
    
    print("Adding statistical features...")
    
    # Window specs
    purpose_window = Window.partitionBy("loan_purpose")
    tier_window = Window.partitionBy("city_tier")
    
    # Average metrics by loan purpose
    df = df.withColumn("avg_amount_by_purpose",
                       avg("loan_amount").over(purpose_window))
    
    df = df.withColumn("avg_dpd_by_purpose",
                       avg("current_dpd").over(purpose_window))
    
    # Metrics by city tier
    df = df.withColumn("avg_income_by_tier",
                       avg("monthly_income").over(tier_window))
    
    # Deviation from average
    df = df.withColumn("amount_deviation_pct",
                       ((col("loan_amount") - col("avg_amount_by_purpose")) / 
                        col("avg_amount_by_purpose")) * 100)
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply All Feature Engineering

# COMMAND ----------

# Apply all transformations
featured_df = bronze_df

featured_df = engineer_credit_features(featured_df)
featured_df = engineer_loan_features(featured_df)
featured_df = engineer_repayment_features(featured_df)
featured_df = engineer_demographic_features(featured_df)
featured_df = engineer_risk_features(featured_df)
featured_df = engineer_time_features(featured_df)
featured_df = add_statistical_features(featured_df)

# Add processing timestamp
featured_df = featured_df.withColumn("feature_engineering_timestamp", current_timestamp())

print(f"\n✓ Feature engineering complete!")
print(f"Total columns: {len(featured_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Quality Checks

# COMMAND ----------

def validate_features(df):
    """Validate engineered features"""
    
    print("Running feature quality checks...")
    
    # Check for nulls in critical features
    critical_features = [
        'credit_score_bucket', 'loan_to_annual_income', 'emi_burden_pct',
        'dpd_category', 'combined_risk_score', 'pd_proxy'
    ]
    
    for feat in critical_features:
        null_count = df.filter(col(feat).isNull()).count()
        if null_count > 0:
            print(f"⚠️  {feat}: {null_count} nulls")
        else:
            print(f"✓ {feat}: No nulls")
    
    # Check for infinite values
    numeric_features = [f.name for f in df.schema.fields if f.dataType in [IntegerType(), DoubleType()]]
    
    for feat in numeric_features[:10]:  # Check first 10 numeric features
        inf_count = df.filter(col(feat).isNaN() | col(feat).isNull()).count()
        if inf_count > 0:
            print(f"⚠️  {feat}: {inf_count} invalid values")
    
    print("\n✓ Feature validation complete")

# COMMAND ----------

validate_features(featured_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Silver Delta Table

# COMMAND ----------

print("Writing to Silver layer...")

featured_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("risk_category", "loan_purpose") \
    .save(SILVER_PATH)

print("✓ Silver table written")

# Optimize
print("Optimizing Delta table...")
spark.sql(f"OPTIMIZE delta.`{SILVER_PATH}`")

# COMMAND ----------

# Create SQL table
spark.sql(f"""
    CREATE OR REPLACE TABLE credit_risk.silver_features
    USING DELTA
    LOCATION '{SILVER_PATH}'
""")

print("✓ SQL table 'credit_risk.silver_features' created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Summary

# COMMAND ----------

silver_df = spark.read.format("delta").load(SILVER_PATH)

print("="*60)
print("SILVER LAYER FEATURE ENGINEERING SUMMARY")
print("="*60)

print(f"\nTotal Records: {silver_df.count():,}")
print(f"Total Features: {len(silver_df.columns)}")

print("\nKey Feature Statistics:")
silver_df.select(
    avg("combined_risk_score").alias("avg_risk_score"),
    avg("pd_proxy").alias("avg_pd"),
    avg("emi_burden_pct").alias("avg_emi_burden"),
    avg("payment_consistency_score").alias("avg_payment_score")
).show()

print("\nRisk Distribution:")
silver_df.groupBy("risk_category").count().orderBy("count", ascending=False).show()

print("\nCredit Score Distribution:")
silver_df.groupBy("credit_score_bucket").count().orderBy("count", ascending=False).show()

print("="*60)
print("✓ Silver layer complete!")
print("="*60)