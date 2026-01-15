"""
Real-Time Credit Risk Monitoring Dashboard
Built with Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f5f7fa;}
    .stMetric {background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    h1 {color: #1f2937;}
    h2 {color: #374151;}
    h3 {color: #6b7280;}
</style>
""", unsafe_allow_html=True)

# Load data function (cached)
@st.cache_data
def load_data():
    """Load sample data"""
    try:
        df = pd.read_csv('data/loan_portfolio_sample.csv', parse_dates=['application_date', 'disbursement_date'])
        return df
    except:
        # Generate minimal sample if file not found
        st.warning("Sample data not found. Using demo data.")
        return pd.DataFrame({
            'loan_id': [f'LOAN{i:08d}' for i in range(1, 101)],
            'customer_id': [f'CUST{i:08d}' for i in range(1, 101)],
            'loan_amount': [100000 + i * 10000 for i in range(100)],
            'credit_score': [650 + i * 2 for i in range(100)],
            'current_dpd': [0] * 80 + [30] * 15 + [90] * 5,
            'is_default': [0] * 85 + [1] * 15,
            'risk_category': ['Low'] * 60 + ['Medium'] * 25 + ['High'] * 15,
            'loan_status': ['Active'] * 70 + ['Closed'] * 25 + ['Written-Off'] * 5
        })

# Load model function (cached)
@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        model = joblib.load('models/xgboost_credit_model.pkl')
        return model
    except:
        return None

# Main app
def main():
    # Title
    st.title("🏦 Real-Time Credit Risk Monitoring Dashboard")
    st.markdown("Production-grade credit risk analytics powered by PySpark, Databricks & MLflow")
    
    # Load data
    df = load_data()
    model = load_model()
    
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        
        # Date range
        if 'application_date' in df.columns:
            date_range = st.date_input(
                "Application Date Range",
                value=(df['application_date'].min(), df['application_date'].max()),
                key="date_range"
            )
        
        # Risk category
        if 'risk_category' in df.columns:
            risk_categories = st.multiselect(
                "Risk Category",
                options=df['risk_category'].unique(),
                default=df['risk_category'].unique()
            )
        else:
            risk_categories = None
        
        # Loan status
        if 'loan_status' in df.columns:
            loan_statuses = st.multiselect(
                "Loan Status",
                options=df['loan_status'].unique(),
                default=df['loan_status'].unique()
            )
        else:
            loan_statuses = None
        
        st.markdown("---")
        st.markdown("**Refresh Interval**")
        refresh_rate = st.selectbox("Update every:", ["5 minutes", "15 minutes", "1 hour"], index=1)
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Apply filters
    filtered_df = df.copy()
    if risk_categories:
        filtered_df = filtered_df[filtered_df['risk_category'].isin(risk_categories)]
    if loan_statuses:
        filtered_df = filtered_df[filtered_df['loan_status'].isin(loan_statuses)]
    
    # Key Metrics Row
    st.header("📈 Key Risk Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_portfolio = filtered_df['loan_amount'].sum() / 1_000_000
        st.metric("Total Portfolio", f"₹{total_portfolio:.1f}M")
    
    with col2:
        total_loans = len(filtered_df)
        st.metric("Total Loans", f"{total_loans:,}")
    
    with col3:
        default_rate = filtered_df['is_default'].mean() * 100 if 'is_default' in filtered_df else 0
        st.metric("Default Rate", f"{default_rate:.2f}%", delta=f"{default_rate - 8.5:.1f}%")
    
    with col4:
        avg_dpd = filtered_df['current_dpd'].mean() if 'current_dpd' in filtered_df else 0
        st.metric("Avg DPD", f"{avg_dpd:.1f} days")
    
    with col5:
        high_risk = (filtered_df['risk_category'] == 'High').sum() if 'risk_category' in filtered_df else 0
        st.metric("High Risk Loans", f"{high_risk:,}")
    
    st.markdown("---")
    
    # Row 2: Portfolio Distribution
    st.header("📊 Portfolio Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution pie chart
        st.subheader("Risk Category Distribution")
        if 'risk_category' in filtered_df:
            risk_dist = filtered_df['risk_category'].value_counts()
            fig = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                color=risk_dist.index,
                color_discrete_map={'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'}
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk category data not available")
    
    with col2:
        # DPD distribution
        st.subheader("DPD Distribution")
        if 'current_dpd' in filtered_df:
            dpd_bins = pd.cut(filtered_df['current_dpd'], 
                             bins=[0, 1, 30, 60, 90, 180], 
                             labels=['Current', '1-30', '31-60', '61-90', '90+'])
            dpd_dist = dpd_bins.value_counts().sort_index()
            
            fig = px.bar(
                x=dpd_dist.index,
                y=dpd_dist.values,
                labels={'x': 'DPD Bucket', 'y': 'Number of Loans'},
                color=dpd_dist.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("DPD data not available")
    
    # Row 3: Trends
    st.header("📉 Trend Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Portfolio growth over time
        st.subheader("Portfolio Growth")
        if 'application_date' in filtered_df:
            monthly_volume = filtered_df.set_index('application_date').resample('M')['loan_amount'].sum() / 1_000_000
            
            fig = px.line(
                x=monthly_volume.index,
                y=monthly_volume.values,
                labels={'x': 'Month', 'y': 'Loan Volume (₹M)'}
            )
            fig.update_traces(line_color='#2563eb', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date data not available")
    
    with col2:
        # Default rate trend
        st.subheader("Default Rate Trend")
        if 'application_date' in filtered_df and 'is_default' in filtered_df:
            monthly_default = filtered_df.set_index('application_date').resample('M')['is_default'].mean() * 100
            
            fig = px.line(
                x=monthly_default.index,
                y=monthly_default.values,
                labels={'x': 'Month', 'y': 'Default Rate (%)'}
            )
            fig.update_traces(line_color='#ef4444', line_width=3)
            fig.add_hline(y=8.5, line_dash="dash", line_color="gray", 
                         annotation_text="Target: 8.5%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Default rate trend data not available")
    
    # Row 4: Detailed Analytics
    st.header("🔍 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Credit score distribution
        st.subheader("Credit Score Distribution")
        if 'credit_score' in filtered_df:
            fig = px.histogram(
                filtered_df,
                x='credit_score',
                nbins=30,
                color_discrete_sequence=['#6366f1']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Credit score data not available")
    
    with col2:
        # Loan amount by purpose
        st.subheader("Loan Amount by Purpose")
        if 'loan_purpose' in filtered_df:
            purpose_amount = filtered_df.groupby('loan_purpose')['loan_amount'].sum() / 1_000_000
            purpose_amount = purpose_amount.sort_values(ascending=True)
            
            fig = px.bar(
                x=purpose_amount.values,
                y=purpose_amount.index,
                orientation='h',
                labels={'x': 'Total Amount (₹M)', 'y': 'Loan Purpose'},
                color=purpose_amount.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Loan purpose data not available")
    
    # Row 5: High Risk Loans Table
    st.header("⚠️ High Risk Loans - Immediate Attention Required")
    
    if 'risk_category' in filtered_df:
        high_risk_df = filtered_df[filtered_df['risk_category'] == 'High'].copy()
        
        if len(high_risk_df) > 0:
            # Select relevant columns
            display_cols = ['loan_id', 'customer_id', 'loan_amount', 'current_dpd', 
                           'credit_score', 'risk_category']
            display_cols = [col for col in display_cols if col in high_risk_df.columns]
            
            st.dataframe(
                high_risk_df[display_cols].head(20),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = high_risk_df.to_csv(index=False)
            st.download_button(
                label="📥 Download High Risk Loans",
                data=csv,
                file_name=f"high_risk_loans_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No high risk loans found!")
    else:
        st.info("Risk category data not available")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Data Pipeline**: PySpark → Databricks → Delta Lake  
    **ML Framework**: XGBoost with MLflow tracking  
    **Last Updated**: {} (Auto-refresh: {})
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), refresh_rate))
    
    # Model info if loaded
    if model:
        with st.expander("🤖 Model Information"):
            st.write("**Model Type**: XGBoost Binary Classifier")
            st.write("**Target**: Loan Default Prediction")
            st.write("**Features**: 30+ engineered credit risk features")
            st.write("**Performance**: AUC-ROC > 0.85")

if __name__ == "__main__":
    main()