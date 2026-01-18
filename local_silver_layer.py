"""
Silver Layer - Clean Spark + Delta Lake Implementation
"""

import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages io.delta:delta-core_2.12:2.4.0 pyspark-shell'

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from delta import *

print("="*70)
print("SILVER LAYER - FEATURE ENGINEERING (Spark + Delta Lake)")
print("="*70)

# Configuration
BRONZE_PATH = "./data/delta/bronze/loans"
SILVER_PATH = "./data/delta/silver/features"

# Initialize Spark
print("\n[1/4] Initializing Spark session with Delta Lake...")

builder = SparkSession.builder \
    .appName("Silver Layer") \
    .master("local[*]") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", "4g")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("✓ Spark session created")

# Load Bronze data
print(f"\n[2/4] Loading Bronze data from {BRONZE_PATH}...")
bronze_df = spark.read.format("delta").load(BRONZE_PATH)
record_count = bronze_df.count()
print(f"✓ Loaded {record_count:,} records")

# Feature Engineering
print("\n[3/4] Engineering features...")

df = bronze_df

# Credit features
print("  → Credit features...")
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

# Loan features
print("  → Loan features...")
df = df.withColumn("loan_to_annual_income", 
                   (col("loan_amount") / (col("monthly_income") * 12)))

df = df.withColumn("emi_burden_pct", 
                   (col("emi_amount") / col("monthly_income")) * 100)

df = df.withColumn("total_debt_burden", 
                   ((col("emi_amount") + col("existing_emi")) / col("monthly_income")) * 100)

df = df.withColumn("loan_term_category",
                   when(col("loan_term_months") <= 24, "Short")
                   .when(col("loan_term_months") <= 60, "Medium")
                   .otherwise("Long"))

df = df.withColumn("interest_rate_bucket",
                   when(col("interest_rate") < 9, "Low")
                   .when(col("interest_rate") < 12, "Medium")
                   .otherwise("High"))

df = df.withColumn("loan_amount_bucket",
                   when(col("loan_amount") < 100000, "Small")
                   .when(col("loan_amount") < 500000, "Medium")
                   .when(col("loan_amount") < 2000000, "Large")
                   .otherwise("Very Large"))

# Repayment features
print("  → Repayment features...")
df = df.withColumn("dpd_category",
                   when(col("current_dpd") == 0, "Current")
                   .when(col("current_dpd") <= 30, "DPD-30")
                   .when(col("current_dpd") <= 60, "DPD-60")
                   .when(col("current_dpd") <= 90, "DPD-90")
                   .otherwise("DPD-90+"))

df = df.withColumn("payment_consistency_score",
                   col("payment_history_pct") * 100 - (col("late_payments_12m") * 5))

df = df.withColumn("high_delinquency_risk",
                   when((col("late_payments_12m") >= 3) | 
                        (col("max_dpd_12m") > 60), 1).otherwise(0))

df = df.withColumn("days_since_disbursement",
                   datediff(current_date(), col("disbursement_date")))

df = df.withColumn("loan_maturity_pct",
                   (col("total_payments_made") / col("loan_term_months")) * 100)

df = df.withColumn("payment_behavior_score",
                   (100 - col("current_dpd")) * col("payment_history_pct"))

# Demographic features
print("  → Demographic features...")
df = df.withColumn("age_group",
                   when(col("age") < 25, "18-24")
                   .when(col("age") < 35, "25-34")
                   .when(col("age") < 45, "35-44")
                   .when(col("age") < 55, "45-54")
                   .otherwise("55+"))

df = df.withColumn("income_bracket",
                   when(col("monthly_income") < 30000, "Low")
                   .when(col("monthly_income") < 60000, "Middle")
                   .when(col("monthly_income") < 100000, "Upper-Middle")
                   .otherwise("High"))

df = df.withColumn("family_size", col("dependents") + 
                   when(col("marital_status") == "Married", 2).otherwise(1))

df = df.withColumn("per_capita_income",
                   col("monthly_income") / col("family_size"))

df = df.withColumn("employment_stability_score",
                   when(col("employment_type") == "Salaried", 100)
                   .when(col("employment_type") == "Professional", 90)
                   .when(col("employment_type") == "Self-Employed", 70)
                   .otherwise(60))

# Risk features
print("  → Risk features...")
df = df.withColumn("combined_risk_score",
                   (100 - ((col("credit_score") - 300) / 6)) * 0.4 +
                   (col("current_dpd") / 180 * 100) * 0.3 +
                   (col("dti_ratio") * 100) * 0.2 +
                   (col("late_payments_12m") * 10) * 0.1)

df = df.withColumn("pd_proxy",
                   (col("combined_risk_score") / 100).cast("double"))

df = df.withColumn("expected_loss_amount",
                   col("pd_proxy") * col("estimated_lgd") * 
                   (col("loan_amount") - (col("emi_amount") * col("total_payments_made"))))

df = df.withColumn("early_warning_flag",
                   when((col("current_dpd") > 0) | 
                        (col("late_payments_12m") >= 2) |
                        (col("payment_consistency_score") < 70), 1).otherwise(0))

# Time features
print("  → Time features...")
df = df.withColumn("application_year", year(col("application_date")))
df = df.withColumn("application_month", month(col("application_date")))
df = df.withColumn("application_quarter", quarter(col("application_date")))

df = df.withColumn("loan_vintage_months",
                   months_between(current_date(), col("disbursement_date")).cast("int"))

# Statistical features
print("  → Statistical aggregation features...")
purpose_window = Window.partitionBy("loan_purpose")
tier_window = Window.partitionBy("city_tier")

df = df.withColumn("avg_amount_by_purpose",
                   avg("loan_amount").over(purpose_window))

df = df.withColumn("avg_dpd_by_purpose",
                   avg("current_dpd").over(purpose_window))

df = df.withColumn("avg_income_by_tier",
                   avg("monthly_income").over(tier_window))

df = df.withColumn("amount_deviation_pct",
                   ((col("loan_amount") - col("avg_amount_by_purpose")) / 
                    col("avg_amount_by_purpose")) * 100)

# Add processing timestamp
df = df.withColumn("feature_engineering_timestamp", current_timestamp())

print(f"✓ Feature engineering complete!")
print(f"  Total columns: {len(df.columns)}")

# Feature validation
print("\nValidating features...")
critical_features = [
    'credit_score_bucket', 'loan_to_annual_income', 'emi_burden_pct',
    'dpd_category', 'combined_risk_score', 'pd_proxy'
]

for feat in critical_features:
    null_count = df.filter(col(feat).isNull()).count()
    status = "✓" if null_count == 0 else "⚠"
    print(f"  {status} {feat}: {'No nulls' if null_count == 0 else f'{null_count} nulls'}")

# Write to Silver Delta table
print(f"\n[4/4] Writing to Silver layer at {SILVER_PATH}...")

df.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("risk_category", "loan_purpose") \
    .option("overwriteSchema", "true") \
    .save(SILVER_PATH)

print("✓ Data written to Delta Lake")

# Optimize
print("Optimizing Delta table...")
DeltaTable.forPath(spark, SILVER_PATH).optimize().executeCompaction()
print("✓ Optimization complete")

# Summary
silver_df = spark.read.format("delta").load(SILVER_PATH)
final_count = silver_df.count()

print("\n" + "="*70)
print("SILVER LAYER SUMMARY")
print("="*70)
print(f"Total Records: {final_count:,}")
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

print("="*70)
print("✓ Silver layer complete!")
print("="*70)

spark.stop()