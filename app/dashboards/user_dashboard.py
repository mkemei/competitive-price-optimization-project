# dashboards/user_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from optimizer.price_optimizer import PriceOptimizer

# ==========================
# Load Model + Optimizer
# ==========================
@st.cache_resource
def load_optimizer():
    model = joblib.load("models/best_model.pkl")
    encoder = joblib.load("models/encoder.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return PriceOptimizer(model=model, encoder=encoder, feature_columns=feature_columns)

optimizer = load_optimizer()

# ==========================
# Load Sales Data
# ==========================
@st.cache_data
def load_sales_data():
    df_path = "data/latest/sales.csv" if os.path.exists("data/latest/sales.csv") else "data/cleaned/sales.csv"
    if not os.path.exists(df_path):
        return None
    df = pd.read_csv(df_path)
    df['product'] = df['product'].str.lower().str.strip()
    return df

# ==========================
# Batch Optimization Helper
# ==========================
def batch_optimize_products(df, optimizer):
    if df.empty:
        return pd.DataFrame(columns=[
            'product', 'current_price', 'optimal_price', 'expected_demand', 'expected_profit'
        ])
    
    if 'product' not in df.columns:
        df['product'] = "unknown"
    
    latest_data = df.loc[df.groupby("product")["date"].idxmax()]
    batch_results = []
    for _, row in latest_data.iterrows():
        product_name = row.get("product", "unknown")
        result = optimizer.optimize_product(row, product_name=product_name)
        batch_results.append(result)
    
    return pd.DataFrame(batch_results)

# ==========================
# Main Dashboard
# ==========================
def render_user_dashboard():
    st.title("🚀 Price Optimization Dashboard")
    st.markdown("Ultra-robust daily optimization with conservative lift checks")
    
    df = load_sales_data()
    if df is None:
        st.error("Sales data file not found!")
        return
    
    # Single product selection
    products = sorted(df['product'].unique())
    product = st.selectbox("📦 Select Product", products)
    
    product_data = df[df['product'] == product]
    default_price = float(product_data['unit_price'].median())
    current_price = st.number_input("💰 Current Price (KES)", value=default_price, step=1.0)
    
    if st.button("🧠 OPTIMIZE PRICE"):
        row = product_data.iloc[-1]
        result = optimizer.optimize_product(row)
        price_change = ((result["optimal_price"] - current_price) / current_price) * 100
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Optimal Price", f"KES {result['optimal_price']:.0f}", f"{price_change:+.1f}%")
        col2.metric("📦 Expected Demand", f"{result['expected_demand']:.1f} units")
        col3.metric("💵 Expected Profit", f"{result['expected_profit']:.0f} KES")
        
        if price_change > 5:
            st.success("✅ IMPLEMENT IMMEDIATELY")
        else:
            st.info("📊 Monitor before implementing")
        
        # Save recommendation
        recs_path = "data/latest/recommendations.csv"
        new_rec = pd.DataFrame([{
            "Product": product,
            "Recommended Price": result['optimal_price'],
            "Expected Lift": f"{price_change:+.1f}%"
        }])
        if os.path.exists(recs_path):
            recs = pd.read_csv(recs_path)
            recs = recs[recs['Product'] != product]
            recs = pd.concat([recs, new_rec], ignore_index=True)
        else:
            recs = new_rec
        recs.to_csv(recs_path, index=False)
        st.success(f"💾 Recommendation saved to {recs_path}")
        
        # Visualization
        st.subheader("📊 Optimization Insights")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0,0].hist(df['unit_price'], bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0,0].set_title('Unit Price Distribution')
        axes[0,1].scatter(df['unit_price'], df['unit_price']*0.7, c='orange', alpha=0.6)
        axes[0,1].set_title('Price vs Cost (COGS)')
        axes[1,0].scatter(result['expected_demand'], result['expected_profit'], color='purple', alpha=0.7)
        axes[1,0].set_title('Expected Demand vs Profit')
        axes[1,1].barh([product[:20]], [price_change], color='gold')
        axes[1,1].set_title('Price Change % Recommendation')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show recent recommendations
        st.subheader("📈 Recent Recommendations")
        st.dataframe(recs.sort_values('Expected Lift', ascending=False), use_container_width=True)
    
    # Batch optimization
    if st.checkbox("⚡ Optimize All Products"):
        st.info("This may take a few seconds for all products...")
        batch_df = batch_optimize_products(df, optimizer)
        st.subheader("📊 All Products Optimization Summary")
        st.dataframe(batch_df.round(2), use_container_width=True)