# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer - Raw Data Ingestion
# MAGIC 
# MAGIC **Purpose**: Ingest raw loan data into Delta Lake bronze layer
# MAGIC 
# MAGIC **Key Features**:
# MAGIC - Incremental loading with watermark
# MAGIC - Data quality checks
# MAGIC - Schema enforcement
# MAGIC - Audit columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup & Configuration

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import DeltaTable
from datetime import datetime

# COMMAND ----------

# Configuration
BRONZE_PATH = "/mnt/datalake/credit_risk/bronze/loans"
SOURCE_PATH = "/mnt/datalake/credit_risk/raw/loan_portfolio_data.csv"

# Initialize Spark session with Delta Lake
spark = SparkSession.builder \
    .appName("Bronze Layer Ingestion") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Schema

# COMMAND ----------

loan_schema = StructType([
    # Customer Information
    StructField("customer_id", StringType(), False),
    StructField("age", IntegerType(), False),
    StructField("employment_type", StringType(), False),
    StructField("education", StringType(), False),
    StructField("marital_status", StringType(), False),
    StructField("dependents", IntegerType(), True),
    StructField("city_tier", StringType(), True),
    
    # Financial Profile
    StructField("monthly_income", IntegerType(), False),
    StructField("credit_score", IntegerType(), False),
    StructField("existing_emi", IntegerType(), True),
    StructField("credit_history_months", IntegerType(), True),
    StructField("existing_loans", IntegerType(), True),
    StructField("credit_utilization_ratio", DoubleType(), True),
    
    # Loan Details
    StructField("loan_id", StringType(), False),
    StructField("loan_purpose", StringType(), False),
    StructField("loan_amount", IntegerType(), False),
    StructField("loan_term_months", IntegerType(), False),
    StructField("interest_rate", DoubleType(), False),
    StructField("application_date", DateType(), False),
    StructField("loan_status", StringType(), False),
    StructField("disbursement_date", DateType(), True),
    
    # Repayment Behavior
    StructField("emi_amount", IntegerType(), False),
    StructField("current_dpd", IntegerType(), False),
    StructField("max_dpd_12m", IntegerType(), True),
    StructField("late_payments_12m", IntegerType(), True),
    StructField("payment_history_pct", DoubleType(), True),
    StructField("total_payments_made", IntegerType(), True),
    
    # Risk Labels
    StructField("is_default", IntegerType(), False),
    StructField("risk_category", StringType(), False),
    StructField("estimated_lgd", DoubleType(), True),
    
    # Derived Features
    StructField("ltv_ratio", DoubleType(), True),
    StructField("dti_ratio", DoubleType(), True)
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Source Data

# COMMAND ----------

def read_source_data(path, schema):
    """Read CSV data with schema enforcement"""
    
    df = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("inferSchema", "false") \
        .schema(schema) \
        .load(path)
    
    # Add audit columns
    df = df.withColumn("ingestion_timestamp", current_timestamp()) \
           .withColumn("source_file", input_file_name()) \
           .withColumn("record_hash", sha2(concat_ws("||", *df.columns), 256))
    
    return df

# COMMAND ----------

# Read data
print(f"Reading data from: {SOURCE_PATH}")
raw_df = read_source_data(SOURCE_PATH, loan_schema)

# Show sample
print(f"\nTotal records: {raw_df.count():,}")
display(raw_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

def run_quality_checks(df):
    """Run data quality validations"""
    
    checks = []
    
    # Check 1: No nulls in key columns
    key_columns = ['customer_id', 'loan_id', 'loan_amount', 'credit_score']
    for col in key_columns:
        null_count = df.filter(col_(col).isNull()).count()
        checks.append({
            'check': f'Null check - {col}',
            'passed': null_count == 0,
            'details': f'{null_count} nulls found'
        })
    
    # Check 2: Valid credit score range
    invalid_score = df.filter((col("credit_score") < 300) | (col("credit_score") > 900)).count()
    checks.append({
        'check': 'Credit score range (300-900)',
        'passed': invalid_score == 0,
        'details': f'{invalid_score} invalid scores'
    })
    
    # Check 3: Valid DPD values
    invalid_dpd = df.filter(col("current_dpd") < 0).count()
    checks.append({
        'check': 'DPD non-negative',
        'passed': invalid_dpd == 0,
        'details': f'{invalid_dpd} negative values'
    })
    
    # Check 4: Loan amount > 0
    invalid_amount = df.filter(col("loan_amount") <= 0).count()
    checks.append({
        'check': 'Loan amount positive',
        'passed': invalid_amount == 0,
        'details': f'{invalid_amount} invalid amounts'
    })
    
    # Check 5: Valid dates
    invalid_dates = df.filter(
        (col("application_date") > current_date()) |
        (col("disbursement_date") < col("application_date"))
    ).count()
    checks.append({
        'check': 'Valid date logic',
        'passed': invalid_dates == 0,
        'details': f'{invalid_dates} invalid dates'
    })
    
    # Display results
    checks_df = spark.createDataFrame(checks)
    display(checks_df)
    
    failed_checks = [c for c in checks if not c['passed']]
    
    if failed_checks:
        print(f"\n⚠️  {len(failed_checks)} quality check(s) failed!")
        return False
    else:
        print("\n✓ All quality checks passed!")
        return True

# COMMAND ----------

# Run quality checks
quality_passed = run_quality_checks(raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Bronze Delta Table

# COMMAND ----------

def write_to_bronze(df, path, mode="append"):
    """Write data to Bronze Delta table with merge logic"""
    
    # Check if table exists
    if DeltaTable.isDeltaTable(spark, path):
        print("Delta table exists. Performing merge operation...")
        
        # Load existing table
        delta_table = DeltaTable.forPath(spark, path)
        
        # Merge logic: Update if record_hash exists, insert if new
        delta_table.alias("target") \
            .merge(
                df.alias("source"),
                "target.loan_id = source.loan_id AND target.record_hash = source.record_hash"
            ) \
            .whenNotMatchedInsertAll() \
            .execute()
        
        print("✓ Merge completed")
        
    else:
        print("Creating new Delta table...")
        
        # Write as new table
        df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .partitionBy("loan_status") \
            .save(path)
        
        print("✓ Table created")
    
    # Optimize table
    print("Optimizing Delta table...")
    spark.sql(f"OPTIMIZE delta.`{path}`")
    
    # Update statistics
    print("Updating statistics...")
    spark.sql(f"ANALYZE TABLE delta.`{path}` COMPUTE STATISTICS")
    
    return True

# COMMAND ----------

if quality_passed:
    write_to_bronze(raw_df, BRONZE_PATH)
    
    # Verify write
    bronze_df = spark.read.format("delta").load(BRONZE_PATH)
    print(f"\n✓ Bronze table record count: {bronze_df.count():,}")
    
    # Show table details
    display(spark.sql(f"DESCRIBE DETAIL delta.`{BRONZE_PATH}`"))
else:
    print("❌ Data quality checks failed. Skipping write to Bronze.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Bronze Table View

# COMMAND ----------

# Create or replace SQL view
spark.sql(f"""
    CREATE OR REPLACE TABLE credit_risk.bronze_loans
    USING DELTA
    LOCATION '{BRONZE_PATH}'
""")

print("✓ SQL table 'credit_risk.bronze_loans' created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

print("="*60)
print("BRONZE LAYER INGESTION SUMMARY")
print("="*60)

bronze_df = spark.read.format("delta").load(BRONZE_PATH)

print(f"\nTotal Records: {bronze_df.count():,}")
print(f"Columns: {len(bronze_df.columns)}")
print(f"Partitions: {bronze_df.rdd.getNumPartitions()}")

print("\nLoan Status Distribution:")
bronze_df.groupBy("loan_status").count().orderBy(desc("count")).show()

print("\nRisk Category Distribution:")
bronze_df.groupBy("risk_category").count().orderBy(desc("count")).show()

print("\nDefault Rate:")
default_rate = bronze_df.agg(avg("is_default")).collect()[0][0]
print(f"{default_rate:.2%}")

print("="*60)
print("✓ Bronze layer ingestion complete!")
print("="*60)