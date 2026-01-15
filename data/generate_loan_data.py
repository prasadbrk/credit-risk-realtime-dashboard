"""
Generate synthetic loan data for credit risk modeling
Simulates realistic banking loan portfolio
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_LOANS = 50000  # Start with 50K loans
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2024, 12, 31)

def generate_customer_profile():
    """Generate realistic customer demographics"""
    age = np.random.normal(38, 12, NUM_LOANS).clip(18, 75).astype(int)
    
    employment_types = ['Salaried', 'Self-Employed', 'Business', 'Professional']
    employment_weights = [0.55, 0.25, 0.15, 0.05]
    
    education_levels = ['Graduate', 'Post-Graduate', 'Under-Graduate', 'Doctorate']
    education_weights = [0.45, 0.30, 0.20, 0.05]
    
    marital_status = ['Married', 'Single', 'Divorced']
    marital_weights = [0.60, 0.35, 0.05]
    
    return pd.DataFrame({
        'customer_id': [f'CUST{str(i).zfill(8)}' for i in range(1, NUM_LOANS + 1)],
        'age': age,
        'employment_type': np.random.choice(employment_types, NUM_LOANS, p=employment_weights),
        'education': np.random.choice(education_levels, NUM_LOANS, p=education_weights),
        'marital_status': np.random.choice(marital_status, NUM_LOANS, p=marital_weights),
        'dependents': np.random.choice([0, 1, 2, 3, 4], NUM_LOANS, p=[0.25, 0.30, 0.25, 0.15, 0.05]),
        'city_tier': np.random.choice(['Tier1', 'Tier2', 'Tier3'], NUM_LOANS, p=[0.40, 0.35, 0.25])
    })

def generate_financial_profile(customer_df):
    """Generate financial metrics based on customer profile"""
    
    # Income based on employment and education
    base_income = np.random.normal(50000, 20000, NUM_LOANS).clip(15000, 200000)
    
    # Adjust based on employment type
    income_multipliers = {
        'Salaried': 1.0,
        'Self-Employed': 1.2,
        'Business': 1.5,
        'Professional': 1.8
    }
    
    monthly_income = np.array([
        base_income[i] * income_multipliers[customer_df.loc[i, 'employment_type']]
        for i in range(NUM_LOANS)
    ])
    
    # Credit score (300-900 range, realistic distribution)
    credit_score = np.random.normal(680, 80, NUM_LOANS).clip(300, 900).astype(int)
    
    # Existing EMI (0 to 40% of income)
    existing_emi = (monthly_income * np.random.uniform(0, 0.4, NUM_LOANS)).astype(int)
    
    # Credit history length (in months)
    credit_history_months = np.random.exponential(60, NUM_LOANS).clip(0, 240).astype(int)
    
    # Number of existing loans
    existing_loans = np.random.choice([0, 1, 2, 3, 4, 5], NUM_LOANS, p=[0.20, 0.30, 0.25, 0.15, 0.08, 0.02])
    
    return pd.DataFrame({
        'monthly_income': monthly_income.astype(int),
        'credit_score': credit_score,
        'existing_emi': existing_emi,
        'credit_history_months': credit_history_months,
        'existing_loans': existing_loans,
        'credit_utilization_ratio': np.random.uniform(0.1, 0.9, NUM_LOANS).round(2)
    })

def generate_loan_details(customer_df, financial_df):
    """Generate loan application details"""
    
    loan_purposes = ['Home', 'Auto', 'Personal', 'Education', 'Business']
    purpose_weights = [0.30, 0.25, 0.25, 0.15, 0.05]
    
    loan_purpose = np.random.choice(loan_purposes, NUM_LOANS, p=purpose_weights)
    
    # Loan amount based on income and purpose
    income = financial_df['monthly_income'].values
    
    amount_multipliers = {
        'Home': (60, 100),  # 60-100x monthly income
        'Auto': (12, 24),
        'Personal': (3, 12),
        'Education': (12, 36),
        'Business': (24, 60)
    }
    
    loan_amount = np.array([
        income[i] * np.random.uniform(*amount_multipliers[loan_purpose[i]])
        for i in range(NUM_LOANS)
    ]).astype(int)
    
    # Loan term in months
    term_options = {
        'Home': [180, 240, 300, 360],
        'Auto': [36, 48, 60, 84],
        'Personal': [12, 24, 36, 48],
        'Education': [60, 84, 120, 180],
        'Business': [36, 60, 84, 120]
    }
    
    loan_term = np.array([
        np.random.choice(term_options[loan_purpose[i]])
        for i in range(NUM_LOANS)
    ])
    
    # Interest rate based on credit score
    base_rate = 8.5
    credit_score = financial_df['credit_score'].values
    interest_rate = base_rate + (750 - credit_score) / 50  # Higher score = lower rate
    interest_rate = interest_rate.clip(7.0, 18.0).round(2)
    
    # Application dates
    days_range = (END_DATE - START_DATE).days
    application_dates = [START_DATE + timedelta(days=random.randint(0, days_range)) 
                         for _ in range(NUM_LOANS)]
    
    # Loan status
    loan_status = np.random.choice(
        ['Active', 'Closed', 'Written-Off', 'Current'],
        NUM_LOANS,
        p=[0.25, 0.55, 0.05, 0.15]
    )
    
    return pd.DataFrame({
        'loan_id': [f'LOAN{str(i).zfill(8)}' for i in range(1, NUM_LOANS + 1)],
        'loan_purpose': loan_purpose,
        'loan_amount': loan_amount,
        'loan_term_months': loan_term,
        'interest_rate': interest_rate,
        'application_date': application_dates,
        'loan_status': loan_status,
        'disbursement_date': [d + timedelta(days=random.randint(7, 30)) for d in application_dates]
    })

def generate_repayment_behavior(loan_df, financial_df):
    """Generate realistic repayment patterns"""
    
    # Calculate EMI
    principal = loan_df['loan_amount'].values
    rate = loan_df['interest_rate'].values / 1200  # Monthly rate
    tenure = loan_df['loan_term_months'].values
    
    emi = (principal * rate * (1 + rate)**tenure) / ((1 + rate)**tenure - 1)
    emi = emi.astype(int)
    
    # Days past due (DPD) - key credit risk indicator
    # Lower credit score = higher chance of delinquency
    credit_score = financial_df['credit_score'].values
    dpd_probability = (750 - credit_score) / 450  # Normalized probability
    dpd_probability = dpd_probability.clip(0, 1)
    
    current_dpd = np.array([
        np.random.choice([0, 0, 0, 30, 60, 90, 120, 180], 
                        p=[0.70, 0.10, 0.05, 0.08, 0.03, 0.02, 0.01, 0.01])
        if np.random.random() < dpd_probability[i] else 0
        for i in range(NUM_LOANS)
    ])
    
    # Max DPD in last 12 months
    max_dpd_12m = np.maximum(current_dpd, 
                             np.array([np.random.choice([0, 30, 60, 90]) 
                                      for _ in range(NUM_LOANS)]))
    
    # Number of late payments
    late_payments_12m = np.random.poisson(dpd_probability * 3, NUM_LOANS).clip(0, 12)
    
    # Payment history (% of on-time payments)
    payment_history_pct = (1 - dpd_probability) * np.random.uniform(0.85, 1.0, NUM_LOANS)
    payment_history_pct = payment_history_pct.clip(0.5, 1.0).round(2)
    
    return pd.DataFrame({
        'emi_amount': emi,
        'current_dpd': current_dpd,
        'max_dpd_12m': max_dpd_12m,
        'late_payments_12m': late_payments_12m,
        'payment_history_pct': payment_history_pct,
        'total_payments_made': np.random.randint(0, loan_df['loan_term_months'].values, NUM_LOANS)
    })

def calculate_risk_labels(df):
    """Calculate target variable: default risk"""
    
    # Simple rule-based default definition
    # Default if: DPD > 90 days OR written-off status
    df['is_default'] = (
        (df['current_dpd'] > 90) | 
        (df['loan_status'] == 'Written-Off')
    ).astype(int)
    
    # Risk category
    def categorize_risk(row):
        if row['is_default'] == 1:
            return 'High'
        elif row['current_dpd'] > 30:
            return 'Medium'
        elif row['credit_score'] < 650 or row['late_payments_12m'] > 2:
            return 'Medium'
        else:
            return 'Low'
    
    df['risk_category'] = df.apply(categorize_risk, axis=1)
    
    # Estimated Loss Given Default (LGD)
    df['estimated_lgd'] = np.where(
        df['is_default'] == 1,
        np.random.uniform(0.40, 0.70, len(df)),  # 40-70% loss on defaults
        0.0
    ).round(2)
    
    return df

def main():
    """Generate complete synthetic dataset"""
    
    print("Generating synthetic loan portfolio data...")
    print(f"Number of loans: {NUM_LOANS:,}")
    
    # Generate all components
    print("\n1. Generating customer profiles...")
    customer_df = generate_customer_profile()
    
    print("2. Generating financial profiles...")
    financial_df = generate_financial_profile(customer_df)
    
    print("3. Generating loan details...")
    loan_df = generate_loan_details(customer_df, financial_df)
    
    print("4. Generating repayment behavior...")
    repayment_df = generate_repayment_behavior(loan_df, financial_df)
    
    # Combine all dataframes
    print("5. Combining datasets...")
    complete_df = pd.concat([
        customer_df,
        financial_df,
        loan_df,
        repayment_df
    ], axis=1)
    
    # Calculate risk labels
    print("6. Calculating risk labels...")
    complete_df = calculate_risk_labels(complete_df)
    
    # Add some derived features
    complete_df['ltv_ratio'] = (complete_df['loan_amount'] / 
                                 (complete_df['monthly_income'] * 12)).round(2)
    complete_df['dti_ratio'] = ((complete_df['emi_amount'] + complete_df['existing_emi']) / 
                                 complete_df['monthly_income']).round(2)
    
    # Data quality checks
    print("\n7. Running data quality checks...")
    print(f"   - Total records: {len(complete_df):,}")
    print(f"   - Default rate: {complete_df['is_default'].mean():.2%}")
    print(f"   - Missing values: {complete_df.isnull().sum().sum()}")
    
    print("\n   Risk distribution:")
    print(complete_df['risk_category'].value_counts())
    
    print("\n   Loan purpose distribution:")
    print(complete_df['loan_purpose'].value_counts())
    
    # Save to CSV
    output_file = 'data/loan_portfolio_data.csv'
    complete_df.to_csv(output_file, index=False)
    print(f"\n✓ Data saved to: {output_file}")
    
    # Save sample for quick testing (1000 records)
    sample_df = complete_df.sample(n=min(1000, NUM_LOANS), random_state=42)
    sample_file = 'data/loan_portfolio_sample.csv'
    sample_df.to_csv(sample_file, index=False)
    print(f"✓ Sample data saved to: {sample_file}")
    
    # Display summary statistics
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"\nNumeric columns summary:")
    print(complete_df[['age', 'monthly_income', 'credit_score', 'loan_amount', 
                       'interest_rate', 'current_dpd']].describe())
    
    print("\n" + "="*60)
    print("✓ Data generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()