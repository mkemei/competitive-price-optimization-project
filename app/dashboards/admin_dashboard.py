import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import shap
import yaml
import bcrypt
import sys
import subprocess
import holidays
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

# -----------------------------
# 1. PIPELINE CLASS DEFINITION
# -----------------------------
class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, ohe_fitted_instance, numerical_cols_train, categorical_cols_train, final_column_order):
        self.ohe = ohe_fitted_instance
        self.numerical_cols_train = numerical_cols_train
        self.categorical_cols_train = categorical_cols_train
        self.final_column_order = final_column_order
        self.encoded_feature_names = self.ohe.get_feature_names_out(self.categorical_cols_train)

    def fit(self, X, y=None): 
        return self 

    def transform(self, X):
        if isinstance(X, (dict, pd.Series)): 
            X = pd.DataFrame([X])
        
        X_copy = X.copy()
        
        # Remove target and date if they exist to prevent pipeline interference
        cols_to_drop = [col for col in ['date', 'net_quantity'] if col in X_copy.columns]
        X_copy = X_copy.drop(columns=cols_to_drop, errors='ignore')
        
        # Separate features
        X_cat = X_copy[self.categorical_cols_train].copy()
        X_num = X_copy[self.numerical_cols_train].copy()
        
        # One-Hot Encoding
        X_cat_encoded = self.ohe.transform(X_cat)
        X_cat_df = pd.DataFrame(X_cat_encoded, columns=self.encoded_feature_names, index=X_copy.index)
        
        # Merge and enforce the column order used during training
        processed_df = pd.concat([X_num, X_cat_df], axis=1)
        return processed_df.reindex(columns=self.final_column_order, fill_value=0)

# Monkey Patching for Pickle/Joblib compatibility
import __main__
__main__.PreprocessingPipeline = PreprocessingPipeline
sys.modules['utils'] = __main__ 

# -----------------------------
# 2. WEEKLY DATA PREPARATION (The Resampling Logic)
# -----------------------------
def prepare_data_for_model(df):
    """
    Performs the notebook-style weekly aggregation and chronological sorting.
    This ensures features like Lags and Rolling Averages are contextually correct.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Initial Chronological Sort
    df = df.sort_values('date')
    
    # Weekly Aggregation per Product (Anchor to Monday)
    df_weekly = df.groupby(['product', pd.Grouper(key='date', freq='W-MON')]).agg({
        'net_quantity': 'sum',
        'unit_price': 'mean',
        'historical_average_competitor_price': 'mean',
        'product_type': 'first'
    }).reset_index()
    
    # Final Sort to ensure time-series integrity
    df_weekly = df_weekly.sort_values('date')
    
    # --- Feature Engineering ---
    df_weekly['year'] = df_weekly['date'].dt.year
    df_weekly['week_of_year'] = df_weekly['date'].dt.isocalendar().week.astype(int)
    
    # Kenyan Holiday logic for 2026 Prediction context
    ke_hols = holidays.Kenya(years=[2024, 2025, 2026])
    df_weekly['is_holiday'] = df_weekly['date'].apply(lambda x: 1 if x in ke_hols else 0)
    df_weekly['is_pre_holiday'] = df_weekly['date'].apply(lambda x: 1 if (x + pd.Timedelta(days=1)) in ke_hols else 0)
    
    # Shift-based features (Calculated per product group)
    df_weekly = df_weekly.sort_values(['product', 'date'])
    df_weekly['net_quantity_lag_1'] = df_weekly.groupby('product')['net_quantity'].shift(1).fillna(0)
    df_weekly['unit_price_lag_1'] = df_weekly.groupby('product')['unit_price'].shift(1).fillna(df_weekly['unit_price'])
    
    # 4-Week Rolling Average
    df_weekly['rolling_mean_net_quantity_4w'] = df_weekly.groupby('product')['net_quantity'].transform(lambda x: x.rolling(4, min_periods=1).mean()).fillna(0)
    df_weekly['rolling_median_net_quantity_4w'] = df_weekly.groupby('product')['net_quantity'].transform(lambda x: x.rolling(4, min_periods=1).median()).fillna(0)
     
    # Gap and Ratio Metrics
    if 'historical_average_competitor_price' not in df_weekly.columns:
        df_weekly['historical_average_competitor_price'] = df_weekly['unit_price']
        
    df_weekly['competitor_gap'] = df_weekly['unit_price'] - df_weekly['historical_average_competitor_price']
    
    avg_price = df_weekly.groupby('product')['unit_price'].transform('mean')
    df_weekly['price_vs_avg_ratio'] = df_weekly['unit_price'] / (avg_price + 1e-6)
    
    return df_weekly

# -----------------------------
# 3. SECURITY & CREDENTIAL HELPERS
# -----------------------------
def save_credentials(users):
    config = {
        "credentials": {"usernames": users},
        "cookie": {"name": "price_optimizer_auth", "key": "secure_2026", "expiry_days": 7}
    }
    with open('credentials.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

# -----------------------------
# 4. ARTIFACT LOADING
# -----------------------------
@st.cache_resource
def load_admin_artifacts():
    try:
        model_path = "models/best_model.pkl"
        pipe_path = "models/pipeline.pkl"
        
        if os.path.exists(model_path) and os.path.exists(pipe_path):
            model = joblib.load(model_path)
            pipeline = joblib.load(pipe_path) 
            return model, pipeline, pipeline.final_column_order
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
    return None, None, None

# -----------------------------
# 5. TAB RENDERING FUNCTIONS
# -----------------------------

def render_model_performance(model, pipeline):
    st.subheader("📊 Performance Monitoring (Weekly Aggregation)")
    df_path = "data/cleaned/sales.csv"
    
    if os.path.exists(df_path):
        df_raw = pd.read_csv(df_path)
        df_ready = prepare_data_for_model(df_raw)
        
        try:
            X_proc = pipeline.transform(df_ready)
            y_true = df_ready['net_quantity'].values
            # Reversing Log-Transformation for human-readable units
            y_pred = np.expm1(model.predict(X_proc))

            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            r2 = 1 - ((y_true - y_pred)**2).sum() / ((y_true - y_true.mean())**2).sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("MAE (Units)", f"{mae:.2f}")
            col2.metric("RMSE", f"{rmse:.2f}")
            col3.metric("R² Score", f"{r2:.3f}")
            
            st.write("### Predicted vs Actual (Weekly Samples)")
            comp_df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred}).tail(20)
            st.line_chart(comp_df)
        except Exception as e:
            st.error(f"Performance Analysis Error: {e}")
    else:
        st.warning("No sales data available for performance tracking.")


def render_feature_importance(model, feature_columns):
    st.subheader("📈 Top Sales Drivers")
    
    if hasattr(model, 'feature_importances_'):
        # 1. Prepare and Sort Data
        fi_df = pd.DataFrame({
            "Feature": feature_columns, 
            "Importance": model.feature_importances_
        })
        
        # Sort by importance and take top 15
        fi_df = fi_df.sort_values("Importance", ascending=False).head(15)
        
        # 2. Setup Plotting (Professional Styling)
        # Using a slightly larger vertical size to accommodate feature names
        fig, ax = plt.subplots(figsize=(10, 7))
        
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=fi_df, 
            palette='viridis', 
            ax=ax
        )
        
        # 3. Add Formatting
        ax.set_title('Top 15 Feature Importances (Random Forest)', fontsize=15, pad=15)
        ax.set_xlabel('Importance Score (Gini)', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        # Add a subtle grid on the X-axis for readability
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Clean up the chart spines
        sns.despine(left=True, bottom=True)
        
        # 4. Render in Streamlit
        st.pyplot(fig)
        
        # Optional: Add a simple expander to see the raw numbers
        with st.expander("View Numerical Importance Scores"):
            st.dataframe(fi_df.reset_index(drop=True), use_container_width=True)
            
    else:
        st.info("Feature importance is not supported by this model type.")

def render_user_management():
    st.subheader("👥 User Access Control")
    try:
        with open('credentials.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Ensure we are accessing the dictionary correctly
        if config and 'credentials' in config and 'usernames' in config['credentials']:
            users = config['credentials']['usernames']
        else:
            users = {}
    except Exception as e:
        st.error(f"Error reading credentials: {e}")
        users = {}

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Existing Users")
        if not users:
            st.info("No users found in credentials.yaml")
        
        for u, d in list(users.items()):
            # SAFE GET: Use .get() to prevent KeyError if 'name' or 'roles' is missing
            display_name = d.get('name', u)
            user_roles = d.get('roles', ['user']) # Default to ['user'] if missing
            
            with st.expander(f"👤 {display_name} (@{u})"):
                st.text(f"Roles: {', '.join(user_roles) if isinstance(user_roles, list) else user_roles}")
                
                if st.button(f"Delete Account: {u}", key=f"del_{u}"):
                    del users[u]
                    save_credentials(users)
                    st.success(f"User {u} removed.")
                    st.rerun()

    with col2:
        st.markdown("#### Create New User")
        new_u = st.text_input("New Username", key="new_user_input")
        new_p = st.text_input("New Password", type="password", key="new_pass_input")
        
        # Optional: Select role during creation
        new_role = st.selectbox("Assign Role", ["user", "admin"])
        
        if st.button("Register User"):
            if new_u and new_p:
                # Clean the username (lowercase, no spaces)
                clean_u = new_u.lower().strip()
                
                if clean_u in users:
                    st.error("User already exists!")
                else:
                    hashed = bcrypt.hashpw(new_p.encode(), bcrypt.gensalt()).decode()
                    
                    # STRICT STRUCTURE: Ensuring all keys exist
                    users[clean_u] = {
                        "name": new_u, 
                        "password": hashed, 
                        "roles": [new_role], # Saved as a list to match auth standards
                        "email": ""
                    }
                    
                    save_credentials(users)
                    st.success(f"User {clean_u} created successfully.")
                    st.rerun()
            else:
                st.error("Please provide both username and password.")

def render_data_retraining():
    st.subheader("🔄 Strategic Retraining")
    
    st.markdown("#### 1. Upload Fresh Sales Data")
    uploaded_file = st.file_uploader("Upload CSV for Training", type="csv")
    if uploaded_file and st.button("💾 Replace Current Sales CSV"):
        pd.read_csv(uploaded_file).to_csv("data/cleaned/sales.csv", index=False)
        st.cache_resource.clear()
        st.success("File uploaded and cache cleared.")

    st.divider()
    
    st.markdown("#### 2. Trigger Model Update")
    st.info("Retrain Model on latest dataset.")
    
    if st.button("🚀 Execute Training Pipeline"):
        with st.spinner("Processing Weekly Aggregation and Fitting Model..."):
            try:
                # Target the script inside the src folder
                train_script_path = os.path.join("src", "train_model.py")
                
                if not os.path.exists(train_script_path):
                    st.error(f"Critical Error: File not found at {train_script_path}")
                    return

                result = subprocess.run(
                    [sys.executable, train_script_path], 
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode == 0:
                    st.success("✅ Model Training Complete!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error(f"❌ Training Script Failed: {result.stderr}")
            except Exception as e:
                st.error(f"❌ System Execution Error: {e}")

# -----------------------------
# 6. MAIN DASHBOARD RENDER
# -----------------------------
def render_admin_dashboard():
    st.header("🔧 Admin Control Panel")    
    model, pipeline, feature_columns = load_admin_artifacts()
    
    if model is None:
        st.warning("System Warning: Model artifacts (best_model.pkl / pipeline.pkl) are missing. Some features are disabled.")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance", 
        "📈 Features", 
        "👥 User Mgmt", 
        "🔄 Retraining"
    ])
    
    with tab1:
        if model: render_model_performance(model, pipeline)
    
    with tab2:
        if model: render_feature_importance(model, feature_columns)
                
    with tab3:
        render_user_management()
        
    with tab4:
        render_data_retraining()