import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import DeltaTable
from delta import configure_spark_with_delta_pip
from datetime import datetime

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
BRONZE_PATH = os.path.join(DATA_DIR, "delta", "bronze", "loans")
SOURCE_PATH = os.path.join(DATA_DIR, "raw", "loans_data.csv")  # This matches the generator script

print("="*60)
print("🚀 STARTING BRONZE LAYER INGESTION (LOCAL)")
print("="*60)

# --- INITIALIZE SPARK WITH DELTA ---
print("\n[1/7] Initializing Spark session...")
builder = SparkSession.builder \
    .appName("Bronze Layer Ingestion") \
    .master("local[*]") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.driver.memory", "2g")

# This creates the Spark session with correct Delta JARs
spark = configure_spark_with_delta_pip(builder).getOrCreate()
print(f"✓ Spark {spark.version} initialized")

# --- DEFINE SCHEMA ---
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

# --- READ SOURCE DATA ---
def read_source_data(path, schema):
    """Read CSV data with schema enforcement"""
    print(f"\n[2/7] Reading data from: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found at {path}. Did you run generate_loan_data.py?")

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

raw_df = read_source_data(SOURCE_PATH, loan_schema)
print(f"Total records read: {raw_df.count():,}")

# --- DATA QUALITY CHECKS ---
def run_quality_checks(df):
    """Run data quality validations"""
    print("\n[3/7] Running quality checks...")
    
    checks = []
    
    # Check 1: No nulls in key columns
    key_columns = ['customer_id', 'loan_id', 'loan_amount', 'credit_score']
    for col_name in key_columns:
        null_count = df.filter(col(col_name).isNull()).count()
        checks.append({
            'check': f'Null check - {col_name}',
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
    
    # Print results locally
    failed_checks = [c for c in checks if not c['passed']]
    
    for c in checks:
        status = "✅" if c['passed'] else "❌"
        print(f"{status} {c['check']}: {c['details']}")
    
    if failed_checks:
        print(f"\n⚠️  {len(failed_checks)} quality check(s) failed!")
        return False
    else:
        print("\n✓ All quality checks passed!")
        return True

quality_passed = run_quality_checks(raw_df)

# --- WRITE TO BRONZE DELTA TABLE ---
def write_to_bronze(df, path):
    """Write data to Bronze Delta table with merge logic"""
    print(f"\n[4/7] Writing to Bronze Delta table at {path}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Check if table exists
    if DeltaTable.isDeltaTable(spark, path):
        print("Delta table exists. Performing merge operation...")
        
        # Load existing table
        delta_table = DeltaTable.forPath(spark, path)
        
        # Merge logic
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
    
    # Optimize table (Try/Except because OPTIMIZE is sometimes Databricks-specific depending on Delta version)
    try:
        print("Optimizing Delta table...")
        spark.sql(f"OPTIMIZE delta.`{path}`") 
    except Exception as e:
        print("⚠ Optimization skipped (likely requires full Databricks runtime), skipping...")

    return True

if quality_passed:
    write_to_bronze(raw_df, BRONZE_PATH)
    
    # Verify write
    bronze_df = spark.read.format("delta").load(BRONZE_PATH)
    print(f"\n[5/7] Verification - Bronze table record count: {bronze_df.count():,}")
else:
    print("❌ Data quality checks failed. Skipping write to Bronze.")

# --- SUMMARY STATISTICS ---
print("\n" + "="*60)
print("BRONZE LAYER INGESTION SUMMARY")
print("="*60)

if quality_passed:
    bronze_df = spark.read.format("delta").load(BRONZE_PATH)

    print(f"\nTotal Records: {bronze_df.count():,}")
    print(f"Columns: {len(bronze_df.columns)}")
    print(f"Partitions: {bronze_df.rdd.getNumPartitions()}")

    print("\nLoan Status Distribution:")
    bronze_df.groupBy("loan_status").count().orderBy(desc("count")).show()

    print("\nRisk Category Distribution:")
    bronze_df.groupBy("risk_category").count().orderBy(desc("count")).show()
    
    print("="*60)
    print("✓ Bronze layer ingestion complete!")
    print("="*60)

spark.stop()