"""
Silver Layer - Feature Engineering with Polars
Blazing-fast transformations using Polars expressions
"""

import polars as pl
from datetime import datetime
from pathlib import Path

print("="*70)
print("SILVER LAYER - FEATURE ENGINEERING (Polars)")
print("="*70)

# Configuration
BRONZE_PATH = "./data/parquet/bronze/loans"
SILVER_PATH = "./data/parquet/silver/features"

# Create directory
Path(SILVER_PATH).mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: Load Bronze Data
# ============================================================================

print("\n[1/4] Loading Bronze data...")

# Lazy loading for better performance
df = pl.scan_parquet(f"{BRONZE_PATH}/**/*.parquet")

# Collect to trigger computation
df = df.collect()

print(f"✓ Loaded {len(df):,} records")

# ============================================================================
# STEP 2: Feature Engineering (Polars Expressions)
# ============================================================================

print("\n[2/4] Engineering features...")

# Credit Features
print("  → Credit features...")
df = df.with_columns([
    pl.when(pl.col('credit_score') >= 750).then(pl.lit('Excellent'))
      .when(pl.col('credit_score') >= 700).then(pl.lit('Good'))
      .when(pl.col('credit_score') >= 650).then(pl.lit('Fair'))
      .when(pl.col('credit_score') >= 600).then(pl.lit('Poor'))
      .otherwise(pl.lit('Very Poor'))
      .alias('credit_score_bucket'),
    
    (pl.col('credit_history_months') / 12).alias('credit_history_years'),
    
    pl.when(pl.col('credit_utilization_ratio') < 0.3).then(pl.lit('Low'))
      .when(pl.col('credit_utilization_ratio') < 0.6).then(pl.lit('Medium'))
      .otherwise(pl.lit('High'))
      .alias('utilization_category'),
    
    (pl.col('existing_loans') > 0).cast(pl.Int32).alias('has_existing_loans')
])

# Loan Features
print("  → Loan features...")
df = df.with_columns([
    (pl.col('loan_amount') / (pl.col('monthly_income') * 12)).alias('loan_to_annual_income'),
    
    ((pl.col('emi_amount') / pl.col('monthly_income')) * 100).alias('emi_burden_pct'),
    
    (((pl.col('emi_amount') + pl.col('existing_emi')) / pl.col('monthly_income')) * 100)
        .alias('total_debt_burden'),
    
    pl.when(pl.col('loan_term_months') <= 24).then(pl.lit('Short'))
      .when(pl.col('loan_term_months') <= 60).then(pl.lit('Medium'))
      .otherwise(pl.lit('Long'))
      .alias('loan_term_category'),
    
    pl.when(pl.col('interest_rate') < 9).then(pl.lit('Low'))
      .when(pl.col('interest_rate') < 12).then(pl.lit('Medium'))
      .otherwise(pl.lit('High'))
      .alias('interest_rate_bucket'),
    
    pl.when(pl.col('loan_amount') < 100000).then(pl.lit('Small'))
      .when(pl.col('loan_amount') < 500000).then(pl.lit('Medium'))
      .when(pl.col('loan_amount') < 2000000).then(pl.lit('Large'))
      .otherwise(pl.lit('Very Large'))
      .alias('loan_amount_bucket')
])

# Repayment Features
print("  → Repayment features...")
df = df.with_columns([
    pl.when(pl.col('current_dpd') == 0).then(pl.lit('Current'))
      .when(pl.col('current_dpd') <= 30).then(pl.lit('DPD-30'))
      .when(pl.col('current_dpd') <= 60).then(pl.lit('DPD-60'))
      .when(pl.col('current_dpd') <= 90).then(pl.lit('DPD-90'))
      .otherwise(pl.lit('DPD-90+'))
      .alias('dpd_category'),
    
    (pl.col('payment_history_pct') * 100 - (pl.col('late_payments_12m') * 5))
        .alias('payment_consistency_score'),
    
    ((pl.col('late_payments_12m') >= 3) | (pl.col('max_dpd_12m') > 60))
        .cast(pl.Int32).alias('high_delinquency_risk'),
    
    (pl.lit(datetime.now().date()) - pl.col('disbursement_date')).dt.total_days()
        .alias('days_since_disbursement'),
    
    ((pl.col('total_payments_made') / pl.col('loan_term_months')) * 100)
        .alias('loan_maturity_pct'),
    
    ((100 - pl.col('current_dpd')) * pl.col('payment_history_pct'))
        .alias('payment_behavior_score')
])

# Demographic Features
print("  → Demographic features...")
df = df.with_columns([
    pl.when(pl.col('age') < 25).then(pl.lit('18-24'))
      .when(pl.col('age') < 35).then(pl.lit('25-34'))
      .when(pl.col('age') < 45).then(pl.lit('35-44'))
      .when(pl.col('age') < 55).then(pl.lit('45-54'))
      .otherwise(pl.lit('55+'))
      .alias('age_group'),
    
    pl.when(pl.col('monthly_income') < 30000).then(pl.lit('Low'))
      .when(pl.col('monthly_income') < 60000).then(pl.lit('Middle'))
      .when(pl.col('monthly_income') < 100000).then(pl.lit('Upper-Middle'))
      .otherwise(pl.lit('High'))
      .alias('income_bracket'),
    
    (pl.col('dependents') + 
     pl.when(pl.col('marital_status') == 'Married').then(2).otherwise(1))
        .alias('family_size')
])

# Add per_capita_income after family_size
df = df.with_columns([
    (pl.col('monthly_income') / pl.col('family_size')).alias('per_capita_income')
])

# Employment stability score
df = df.with_columns([
    pl.when(pl.col('employment_type') == 'Salaried').then(100)
      .when(pl.col('employment_type') == 'Professional').then(90)
      .when(pl.col('employment_type') == 'Self-Employed').then(70)
      .otherwise(60)
      .alias('employment_stability_score')
])

# Risk Features
print("  → Risk features...")
df = df.with_columns([
    # Combined risk score
    (
        (100 - ((pl.col('credit_score') - 300) / 6)) * 0.4 +
        (pl.col('current_dpd') / 180 * 100) * 0.3 +
        (pl.col('dti_ratio') * 100) * 0.2 +
        (pl.col('late_payments_12m') * 10) * 0.1
    ).alias('combined_risk_score')
])

df = df.with_columns([
    (pl.col('combined_risk_score') / 100).alias('pd_proxy')
])

df = df.with_columns([
    (
        pl.col('pd_proxy') * pl.col('estimated_lgd') * 
        (pl.col('loan_amount') - (pl.col('emi_amount') * pl.col('total_payments_made')))
    ).alias('expected_loss_amount'),
    
    (
        (pl.col('current_dpd') > 0) | 
        (pl.col('late_payments_12m') >= 2) |
        (pl.col('payment_consistency_score') < 70)
    ).cast(pl.Int32).alias('early_warning_flag')
])

# Time Features
print("  → Time features...")
df = df.with_columns([
    pl.col('application_date').dt.year().alias('application_year'),
    pl.col('application_date').dt.month().alias('application_month'),
    pl.col('application_date').dt.quarter().alias('application_quarter'),
    
    ((pl.lit(datetime.now().date()) - pl.col('disbursement_date')).dt.total_days() / 30)
        .cast(pl.Int32).alias('loan_vintage_months')
])

# Statistical Aggregation Features
print("  → Statistical aggregation features...")

# Average by loan purpose
purpose_stats = df.group_by('loan_purpose').agg([
    pl.col('loan_amount').mean().alias('avg_amount_by_purpose'),
    pl.col('current_dpd').mean().alias('avg_dpd_by_purpose')
])

df = df.join(purpose_stats, on='loan_purpose', how='left')

# Average by city tier
tier_stats = df.group_by('city_tier').agg([
    pl.col('monthly_income').mean().alias('avg_income_by_tier')
])

df = df.join(tier_stats, on='city_tier', how='left')

# Amount deviation
df = df.with_columns([
    (((pl.col('loan_amount') - pl.col('avg_amount_by_purpose')) / 
      pl.col('avg_amount_by_purpose')) * 100).alias('amount_deviation_pct')
])

# Add processing timestamp
df = df.with_columns([
    pl.lit(datetime.now()).alias('feature_engineering_timestamp')
])

print(f"✓ Feature engineering complete!")
print(f"  Total columns: {len(df.columns)}")

# ============================================================================
# STEP 3: Feature Validation
# ============================================================================

print("\n[3/4] Validating features...")

critical_features = [
    'credit_score_bucket', 'loan_to_annual_income', 'emi_burden_pct',
    'dpd_category', 'combined_risk_score', 'pd_proxy'
]

for feat in critical_features:
    null_count = df.filter(pl.col(feat).is_null()).height
    status = "✓" if null_count == 0 else "⚠"
    print(f"  {status} {feat}: {'No nulls' if null_count == 0 else f'{null_count} nulls'}")

# ============================================================================
# STEP 4: Write to Parquet
# ============================================================================

print(f"\n[4/4] Writing to Parquet at {SILVER_PATH}...")

# Write partitioned by risk_category and loan_purpose
df.write_parquet(
    SILVER_PATH,
    compression='snappy',
    statistics=True,
    use_pyarrow=True,
    pyarrow_options={
        'partition_cols': ['risk_category', 'loan_purpose'],
        'existing_data_behavior': 'delete_matching'
    }
)

print("✓ Data written to Parquet (partitioned by risk_category, loan_purpose)")

# Verify write
silver_df = pl.scan_parquet(f"{SILVER_PATH}/**/*.parquet").collect()
final_count = len(silver_df)

print(f"✓ Silver table verification: {final_count:,} records")

# Summary
print("\n" + "="*70)
print("SILVER LAYER SUMMARY")
print("="*70)
print(f"Total Records: {final_count:,}")
print(f"Total Features: {len(silver_df.columns)}")

print("\nKey Feature Statistics:")
stats = silver_df.select([
    pl.col('combined_risk_score').mean().alias('avg_risk_score'),
    pl.col('pd_proxy').mean().alias('avg_pd'),
    pl.col('emi_burden_pct').mean().alias('avg_emi_burden'),
    pl.col('payment_consistency_score').mean().alias('avg_payment_score')
])
print(stats)

# print("\nRisk Distribution:")
# risk_dist = silver_df.group_by('risk_category').agg(
#     pl.count().alias('count')
# ).sort('count', descending=True)
# print(risk_dist)

print("\nCredit Score Distribution:")
score_dist = silver_df.group_by('credit_score_bucket').agg(
    pl.count().alias('count')
).sort('count', descending=True)
print(score_dist)

print("\n" + "="*70)
print("✓ Silver layer complete!")
print("="*70)
print(f"\nPerformance Benefits:")
print(f"  - Lazy evaluation for memory efficiency")
print(f"  - Parallel processing on all CPU cores")
print(f"  - 10-100x faster than Pandas for large datasets")
print(f"  - Arrow-based zero-copy data exchange")