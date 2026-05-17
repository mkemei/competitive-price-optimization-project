from itertools import product

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import shap
import holidays
import matplotlib.pyplot as plt
from datetime import datetime
from optimizer.price_optimizer import PriceOptimizerDefault
import pytz
from src.utils import PreprocessingPipeline

# ==========================
# Load Model + Optimizer
# ==========================
@st.cache_resource
def load_optimizer():
    # Load the new model and the specific preprocessing pipeline
    model = joblib.load("models/best_model.pkl")
    pipeline = joblib.load("models/pipeline.pkl") 
    
    df = load_sales_data()
    # Calculate median prices for 'price_vs_avg_ratio'
    train_price_stats = df.groupby('product')['unit_price'].median().to_dict()
    
    return PriceOptimizerDefault(
        model=model, 
        pipeline=pipeline, 
        historical_df=df, 
        train_price_stats=train_price_stats
    )

# ==========================
# Load Sales Data
# ==========================
@st.cache_data
def load_sales_data():
    df_path = "data/cleaned/sales.csv"
    if not os.path.exists(df_path):
        return None
    df = pd.read_csv(df_path)
    df['product'] = df['product'].str.lower().str.strip()
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    return df

# ==========================
# Load Latest Competitor Data
# ==========================
@st.cache_data
def load_latest_competitor_data(path="data/competitor_prices"):
    if not os.path.exists(path):
        return None
    files = sorted([f for f in os.listdir(path) if f.endswith('.csv')], reverse=True)
    if not files:
        return None
    latest_file = os.path.join(path, files[0])
    df = pd.read_csv(latest_file)
    df['matched_user_product_original'] = df['matched_user_product_original'].str.lower().str.strip()
    
    # Dynamic price column selection based on availability
    price_cols = [c for c in ['jumia_price','totshoppe_price','peekaboo_price'] if c in df.columns]
    df['comp_min_price'] = df[price_cols].min(axis=1, skipna=True)
    df['comp_avg_price'] = df[price_cols].mean(axis=1, skipna=True)
    return df[['matched_user_product_original','comp_min_price','comp_avg_price']]

# ==========================
# Product Sales & Price Trend
# ==========================
def render_product_trends(df, product):
    product_df = df[df['product'] == product].sort_values('date')
    if product_df.empty:
        st.info("No sales data available for this product.")
        return

    # Aggregate weekly for cleaner visualization (matches model logic)
    product_df['week'] = product_df['date'].dt.to_period('W').dt.start_time
    sales_trend = product_df.groupby('week')['net_quantity'].sum().reset_index()
    price_trend = product_df.groupby('week')['unit_price'].mean().reset_index()

    st.subheader(f"📊 Weekly Trends: {product.title()}")
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(sales_trend.set_index('week')['net_quantity'], height=250)
        st.caption("Weekly Units Sold")
    with col2:
        st.line_chart(price_trend.set_index('week')['unit_price'], height=250)
        st.caption("Avg Weekly Price")

def render_shap_explanation(model, pipeline, df, product, price, competitor_price):
    st.write("---")
    st.subheader("🧠 AI Decision Logic")
    
    try:
        # 1. Get historical data and ensure types are correct
        prod_df = df[df['product'] == product].copy()
        prod_df['date'] = pd.to_datetime(prod_df['date'])
        prod_df = prod_df.sort_values('date')
        
        # Capture metadata for the pipeline
        p_type = prod_df['product_type'].iloc[0] if 'product_type' in prod_df.columns else 'Unknown'

        # 2. Weekly Aggregation
        df_w = prod_df.groupby(pd.Grouper(key='date', freq='W-MON')).agg({
            'net_quantity': 'sum', 
            'unit_price': 'mean', 
            'historical_average_competitor_price': 'mean'
        }).reset_index()

        # 3. Add MISSING FEATURES (Including 'year' for the Pipeline)
        df_w['product'] = product
        df_w['product_type'] = p_type
        
        # Time Identity (Fixes the "['year'] not in index" error)
        df_w['year'] = df_w['date'].dt.year
        df_w['month'] = df_w['date'].dt.month
        df_w['week_of_year'] = df_w['date'].dt.isocalendar().week.astype(int)
        
        # Holiday Logic
        ke_hols = holidays.Kenya(years=[2024, 2025, 2026])
        df_w['is_holiday'] = df_w['date'].apply(lambda x: 1 if x in ke_hols else 0)
        df_w['is_pre_holiday'] = df_w['date'].apply(lambda x: 1 if (x + pd.Timedelta(days=1)) in ke_hols else 0)
        
        # Lag/Rolling Math
        df_w['net_quantity_lag_1'] = df_w['net_quantity'].shift(1).fillna(0)
        df_w['unit_price_lag_1'] = df_w['unit_price'].shift(1).fillna(df_w['unit_price'])
        df_w['rolling_mean_net_quantity_4w'] = df_w['net_quantity'].rolling(4, min_periods=1).mean().fillna(0)
        df_w['rolling_median_net_quantity_4w'] = df_w['net_quantity'].rolling(4, min_periods=1).median().fillna(0)
        
        # 4. Inject Scenario Values into the latest week
        last_row = df_w.iloc[-1:].copy()
        last_row['unit_price'] = price
        last_row['historical_average_competitor_price'] = competitor_price
        last_row['competitor_gap'] = price - competitor_price
        last_row['price_vs_avg_ratio'] = price / (df_w['unit_price'].mean() + 1e-6)
        
        # 5. Transform & SHAP
        X_proc = pipeline.transform(last_row)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_proc)
        
        # Handle Output Shapes
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)): base_val = base_val[0]
        curr_shap = shap_values[0] if isinstance(shap_values, list) else shap_values
        if len(curr_shap.shape) > 1: curr_shap = curr_shap.flatten()

        # 6. Plot Waterfall
        fig = plt.figure(figsize=(10, 5))
        shap.plots._waterfall.waterfall_legacy(
            base_val, curr_shap, feature_names=X_proc.columns.tolist(), max_display=10, show=False
        )
        plt.title(f"Demand Drivers: {product.title()}", fontsize=12)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Feature Alignment Error: {e}")

# ==========================
# POS UTILITY FUNCTION
# ==========================
import pandas as pd
import pytz
from datetime import datetime
import os

def record_sale_to_csv(product_name, unit_price, qty):
    sales_path = "data/cleaned/sales.csv"
    
    # 1. Standardize Inputs
    unit_price = float(unit_price)
    qty = int(qty)
    product_clean = product_name.lower().strip()

    # 2. Timezone & Formatting
    nairobi_tz = pytz.timezone('Africa/Nairobi')
    now = datetime.now(nairobi_tz)
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S%z")
    timestamp_formatted = timestamp_str[:-2] + ":" + timestamp_str[-2:]

    # 3. Product Type Lookup (Look for existing type in history)
    p_type = "General" 
    if os.path.exists(sales_path):
        try:
            # We read the whole file to find the type, but only the columns we need
            df_history = pd.read_csv(sales_path)
            match = df_history[df_history['product'].str.lower().str.strip() == product_clean]
            if not match.empty:
                p_type = match['product_type'].iloc[0]
        except Exception as e:
            print(f"Lookup Error: {e}")

    # 4. Competitor Context
    comp_df = load_latest_competitor_data()
    comp_min, comp_avg = unit_price, unit_price
    if comp_df is not None:
        c_match = comp_df[comp_df['matched_user_product_original'] == product_clean]
        if not c_match.empty:
            comp_min = c_match['comp_min_price'].values[0]
            comp_avg = c_match['comp_avg_price'].values[0]

    # 5. Financials 
    discount_per = 0.0 
    net_sales = unit_price * qty 

    # 6. Create Row 
    new_row_data = {
        'sale_id': int(datetime.now().timestamp()),
        'date': timestamp_formatted,
        'product_type': p_type,
        'product': product_clean,
        'unit_price': unit_price, 
        'net_quantity': qty,
        'discount_percentage': discount_per, 
        #historical_min_competitor_price': comp_min,
        'historical_average_competitor_price': comp_avg,
        'net_sales': net_sales
    }

    # 7. Write to CSV with Column Enforcement
    df_new = pd.DataFrame([new_row_data])
    
    # Define the strict column order to prevent the shift
    column_order = [
        'sale_id', 'date', 'product_type', 'product',  'unit_price',
        'net_quantity', 'discount_percentage',
        'historical_average_competitor_price','net_sales'
    ]
    
    # Reindex ensures that even if the dict was out of order, the CSV isn't
    df_new = df_new.reindex(columns=column_order)

    file_exists = os.path.isfile(sales_path)
    df_new.to_csv(sales_path, mode='a', index=False, header=not file_exists)
    
    return True

# ==========================
# Main Dashboard
# ==========================
def render_user_dashboard():
    st.title("🚀 Price Optimization Dashboard")
    
    df = load_sales_data()
    if df is None:
        st.error("Sales data file not found!")
        return

    # Initialize components
    optimizer = load_optimizer()
    competitor_df = load_latest_competitor_data()

    st.divider()

    # Product Selection
    products = sorted(df['product'].unique())
    product = st.selectbox("📦 Select Product to Optimize", products)

    # UI Inputs for current context
    col_a, col_b = st.columns(2)
    with col_a:
        default_price = float(df[df['product'] == product]['unit_price'].iloc[-1])
        current_price = st.number_input("💰 Store Price (KES)", value=default_price, step=10.0)
    
    product_data = df[df['product'] == product]
    historical_avg_price = float(product_data["unit_price"].mean())
    minimum_allowed_price = historical_avg_price * 0.60

    if current_price < minimum_allowed_price:
      st.error(
        f"⚠️ Unreasonable current price entered. "
        f"The current price cannot be below 60% of the historical average price.\n\n"
        f"Historical average: KES {historical_avg_price:,.2f}\n"
        f"Minimum allowed: KES {minimum_allowed_price:,.2f}\n"
        f"Entered price: KES {current_price:,.2f}"
      )
      st.stop()

    with col_b:
        # Fetch current competitor context
        comp_match = competitor_df[competitor_df['matched_user_product_original'] == product] if competitor_df is not None else pd.DataFrame()
        c_min = float(comp_match['comp_min_price'].iloc[0]) if not comp_match.empty else current_price
        adj_comp_min = st.number_input("📉 Competitor Min Price (KES)", value=c_min, step=10.0)

    # =====================================================================
    # ADDED FEATURE: Economic Parameter Simulation Control Interface
    # =====================================================================
    st.markdown("### 🔧 Market Simulation Controls")
    with st.expander("📊 Elasticity Overlay Adjustments (Presentation Simulation Mode)", expanded=False):
        st.info("Manually overlay extra log-linear price sensitivity parameters to simulate severe economic fluctuations.")
        
        sim_elasticity = st.slider(
            "Own-Price Elasticity Override (α)", 
            min_value=0.0, 
            max_value=3.0, 
            value=0.0, 
            step=0.1,
            help="0.0 relies purely on Random Forest features. Values > 0.0 mathematically penalize high pricing."
        )
        sim_comp_elasticity = st.slider(
            "Competitor Cross-Price Elasticity Override (β)", 
            min_value=0.0, 
            max_value=3.0, 
            value=0.0, 
            step=0.1,
            help="Values > 0.0 shift consumer demand dynamically based on the competitor price gap."
        )

    # Direct implementation update mapping straight into the backend configuration memory layer
    optimizer.config['elasticity'] = sim_elasticity
    optimizer.config['comp_elasticity'] = sim_comp_elasticity
    # =====================================================================

    # OPTIMIZE BUTTON
    if st.button("🧠 CALCULATE OPTIMAL PRICE"):
        res = optimizer.optimize_product(
            product_name=product, 
            current_price=current_price, 
            competitor_price=adj_comp_min
        )
        
        # Display Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🎯 Current Price", f"KES {res['current_price']:.0f}")
        m2.metric("🎯 Optimal Price", f"KES {res['optimal_price']:.0f}")
        m3.metric("📦 Predicted demand at Current", f"{res['current_demand']:.1f} units")
        m4.metric("📦 Expected Weekly Demand", f"{res['expected_demand_opt']:.1f} units")
        m5.metric("📈 Revenue Lift", f"{res['revenue_lift_pct']}%")

        if res['revenue_lift_pct'] > 0:
            st.success("✅ Improvement detected with suggested price.")
        else:
            st.warning("⚠️ Current price is already near optimal or market demand is low.")

     # Demand and Revenue Curve
    st.subheader("📈 Price Optimization Curve")
    
    # 1. Establish robust default baselines if an optimization hasn't run yet
    low = historical_avg_price * 0.5
    high = historical_avg_price * 1.5
    lower_bound = current_price * 0.8
    upper_bound = current_price * 1.3
    opt_price = current_price
    
    # 2. Safely look for your optimization dictionary under common variable names
    optimization_output = None
    if 'res' in locals() and locals()['res'] is not None:
        optimization_output = locals()['res']
    elif 'result' in locals() and locals()['result'] is not None:
        optimization_output = locals()['result']
    
    # 3. If optimization was successfully run, extract your exact mathematical bounds
    if optimization_output is not None and isinstance(optimization_output, dict):
        lower_bound = optimization_output.get("lower_bound", lower_bound)
        upper_bound = optimization_output.get("upper_bound", upper_bound)
        opt_price = optimization_output.get("optimal_price", current_price)
    
    # 4. Generate the visualization matrix
    prices = np.linspace(low, high, 20)
    demands = [optimizer.predict_demand(product_name=product, price=p, competitor_price=adj_comp_min) for p in prices]
    revenues = [p * d for p, d in zip(prices, demands)]
    
    # 5. Build the Matplotlib figure explicitly to maintain Streamlit thread-safety
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot Demand Profile (Left Axis)
    ax1.plot(prices, demands, "b--", linewidth=2, label="Demand")
    ax1.set_xlabel("Price (KES)")
    ax1.set_ylabel("Expected Demand (Units)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    
    # Plot Revenue Profile (Right Axis via twinx)
    ax2 = ax1.twinx()
    ax2.plot(prices, revenues, "g-", linewidth=3, label="Revenue")
    ax2.set_ylabel("Expected Revenue (KES)", color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    
    # 6. Add formatted vertical benchmark lines
    ax1.axvline(current_price, color="black", linestyle=":", linewidth=2, label=f"Current: KES {current_price:,.2f}")
    if opt_price is not None:
        ax1.axvline(opt_price, color="red", linestyle="--", linewidth=2, label=f"Optimal: KES {opt_price:,.2f}")
    
    # 7. Merge legends from both overlapping axes cleanly
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    
    plt.title(f"Price Optimization Space: {str(product).title()}")
    fig.tight_layout()
    
    # 8. Render natively in Streamlit and explicitly wipe the figure memory
    st.pyplot(fig)
    plt.close(fig) 

    st.divider()

    # Trend Visualization
    render_product_trends(df, product)

    render_shap_explanation(
        optimizer.model,    # 1. model
        optimizer.pipeline, # 2. pipeline
        df,                 # 3. df (your loaded sales data)
        product,            # 4. product name
        opt_price,      # 5. suggested price
        adj_comp_min        # 6. competitor price
    )

if __name__ == "__main__":
    render_user_dashboard()