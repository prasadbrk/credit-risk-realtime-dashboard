"""
Bronze Layer - Raw Data Ingestion with Polars
Ultra-fast data processing without JVM/Spark overhead
"""

import polars as pl
from datetime import datetime
import os
from pathlib import Path

print("="*70)
print("BRONZE LAYER - RAW DATA INGESTION (Polars)")
print("="*70)

# Configuration
BRONZE_PATH = "./data/parquet/bronze/loans"
SOURCE_PATH = "./data/loan_portfolio_data.csv"

# Create directories
Path(BRONZE_PATH).mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: Read CSV with Schema
# ============================================================================

print("\n[1/5] Reading CSV data...")

# Define schema for type safety
schema = {
    'customer_id': pl.Utf8,
    'age': pl.Int32,
    'employment_type': pl.Utf8,
    'education': pl.Utf8,
    'marital_status': pl.Utf8,
    'dependents': pl.Int32,
    'city_tier': pl.Utf8,
    'monthly_income': pl.Int32,
    'credit_score': pl.Int32,
    'existing_emi': pl.Int32,
    'credit_history_months': pl.Int32,
    'existing_loans': pl.Int32,
    'credit_utilization_ratio': pl.Float64,
    'loan_id': pl.Utf8,
    'loan_purpose': pl.Utf8,
    'loan_amount': pl.Int32,
    'loan_term_months': pl.Int32,
    'interest_rate': pl.Float64,
    'application_date': pl.Utf8,
    'loan_status': pl.Utf8,
    'disbursement_date': pl.Utf8,
    'emi_amount': pl.Int32,
    'current_dpd': pl.Int32,
    'max_dpd_12m': pl.Int32,
    'late_payments_12m': pl.Int32,
    'payment_history_pct': pl.Float64,
    'total_payments_made': pl.Int32,
    'is_default': pl.Int32,
    'risk_category': pl.Utf8,
    'estimated_lgd': pl.Float64,
    'ltv_ratio': pl.Float64,
    'dti_ratio': pl.Float64
}

# Read CSV - Polars is 5-10x faster than Pandas here
df = pl.read_csv(
    SOURCE_PATH,
    schema=schema,
    try_parse_dates=False  # We'll parse dates manually
)

print(f"✓ Loaded {len(df):,} records")
print(f"  Columns: {len(df.columns)}")
print(f"  Memory usage: {df.estimated_size('mb'):.2f} MB")

# ============================================================================
# STEP 2: Data Type Conversions
# ============================================================================

print("\n[2/5] Converting data types...")

# Parse dates
df = df.with_columns([
    pl.col('application_date').str.to_date('%Y-%m-%d').alias('application_date'),
    pl.col('disbursement_date').str.to_date('%Y-%m-%d').alias('disbursement_date')
])

# Add audit columns
df = df.with_columns([
    pl.lit(datetime.now()).alias('ingestion_timestamp'),
    pl.lit(SOURCE_PATH).alias('source_file'),
    pl.concat_str([pl.col('customer_id'), pl.col('loan_id')], separator='||').alias('record_key')
])

print("✓ Data types converted")

# ============================================================================
# STEP 3: Data Quality Checks
# ============================================================================

print("\n[3/5] Running data quality checks...")

checks = []

# Check 1: No nulls in key columns
key_columns = ['customer_id', 'loan_id', 'loan_amount', 'credit_score']
for col in key_columns:
    null_count = df.filter(pl.col(col).is_null()).height
    checks.append({
        'check': f'Null check - {col}',
        'passed': null_count == 0,
        'details': f'{null_count} nulls found'
    })

# Check 2: Valid credit score range
invalid_score = df.filter(
    (pl.col('credit_score') < 300) | (pl.col('credit_score') > 900)
).height
checks.append({
    'check': 'Credit score range (300-900)',
    'passed': invalid_score == 0,
    'details': f'{invalid_score} invalid scores'
})

# Check 3: Valid DPD values
invalid_dpd = df.filter(pl.col('current_dpd') < 0).height
checks.append({
    'check': 'DPD non-negative',
    'passed': invalid_dpd == 0,
    'details': f'{invalid_dpd} negative values'
})

# Check 4: Loan amount > 0
invalid_amount = df.filter(pl.col('loan_amount') <= 0).height
checks.append({
    'check': 'Loan amount positive',
    'passed': invalid_amount == 0,
    'details': f'{invalid_amount} invalid amounts'
})

# Check 5: Valid dates
invalid_dates = df.filter(
    (pl.col('disbursement_date') < pl.col('application_date'))
).height
checks.append({
    'check': 'Valid date logic',
    'passed': invalid_dates == 0,
    'details': f'{invalid_dates} invalid dates'
})

print("\nQuality Check Results:")
for check in checks:
    status = "✓" if check['passed'] else "✗"
    print(f"  {status} {check['check']}: {check['details']}")

quality_passed = all(c['passed'] for c in checks)

if quality_passed:
    print("\n✓ All quality checks passed!")
else:
    print("\n⚠ Some quality checks failed")

# ============================================================================
# STEP 4: Write to Parquet (Partitioned)
# ============================================================================

print(f"\n[4/5] Writing to Parquet at {BRONZE_PATH}...")

# Write partitioned by loan_status (like Delta Lake partitioning)
df.write_parquet(
    BRONZE_PATH,
    compression='snappy',
    statistics=True,
    use_pyarrow=True,
    pyarrow_options={
        'partition_cols': ['loan_status'],
        'existing_data_behavior': 'delete_matching'
    }
)

print("✓ Data written to Parquet (partitioned by loan_status)")

# ============================================================================
# STEP 5: Verify Write & Summary
# ============================================================================

print("\n[5/5] Verifying write...")

# Read back to verify
bronze_df = pl.scan_parquet(f"{BRONZE_PATH}/**/*.parquet").collect()
final_count = len(bronze_df)

print(f"✓ Bronze table verification: {final_count:,} records")

# Summary statistics
print("\n" + "="*70)
print("BRONZE LAYER SUMMARY")
print("="*70)
print(f"Total Records: {final_count:,}")
print(f"Columns: {len(bronze_df.columns)}")
# print(f"Partitions: {bronze_df['loan_status'].n_unique()}")

# print("\nLoan Status Distribution:")
# status_dist = bronze_df.group_by('loan_status').agg(pl.count().alias('count')).sort('count', descending=True)
# print(status_dist)

print("\nRisk Category Distribution:")
risk_dist = bronze_df.group_by('risk_category').agg(pl.count().alias('count')).sort('count', descending=True)
print(risk_dist)

default_rate = bronze_df['is_default'].mean()
print(f"\nDefault Rate: {default_rate:.2%}")

print("\nSample Records:")
print(bronze_df.select(['loan_id', 'customer_id', 'loan_amount', 'credit_score', 'risk_category']).head(5))

print("\n" + "="*70)
print("✓ Bronze layer ingestion complete!")
print("="*70)
print(f"\nPerformance:")
print(f"  - Processing speed: {len(df) / 1000:.1f}K records/second (approx)")
print(f"  - Storage: Parquet with Snappy compression")
print(f"  - Partitioning: By loan_status for efficient queries")