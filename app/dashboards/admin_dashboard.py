import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import shap
import streamlit_authenticator as stauth  
import yaml
from yaml.loader import SafeLoader
import bcrypt

# -----------------------------
# Load Model & Artifacts
# -----------------------------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load("models/best_model.pkl")
    encoder = joblib.load("models/encoder.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, encoder, feature_columns

model, encoder, feature_columns = load_model_artifacts()


# -----------------------------
# ADMIN DASHBOARD
# -----------------------------
def render_admin_dashboard():
    st.header("🔧 Admin Control Panel")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "📈 Feature Importance", "⚡ SHAP Explainability", "👥 User Management"])

    with tab1:
        render_model_performance()
    with tab2:
        render_feature_importance()
    with tab3:
        render_shap_explainability()
    with tab4:  # 👥 USER MANAGEMENT TAB
        render_user_management()


# -----------------------------
# LIVE MODEL PERFORMANCE
# -----------------------------
def render_model_performance():
    st.subheader("Live Model Performance")

    # Load recent data
    df_path = "data/cleaned/sales.csv"
    if not os.path.exists(df_path):
        st.warning("Sales data not found to compute metrics.")
        return

    df = pd.read_csv(df_path)
    if "net_quantity" not in df.columns:
        st.warning("No actual target column (net_quantity) found in data.")
        return

    X = df.drop(columns=['net_quantity', 'date'], errors='ignore')
    y_true = df['net_quantity'].values

    # Preprocess categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        X_cat_encoded = encoder.transform(X[cat_cols])
        X_cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=encoder.get_feature_names_out(cat_cols), index=X.index)
        X = pd.concat([X.drop(columns=cat_cols), X_cat_encoded_df], axis=1)

    # Align columns
    missing_cols = set(feature_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[feature_columns]

    # Predict
    y_pred = model.predict(X)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R² Score", f"{r2:.3f}")


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
def render_feature_importance():
    st.subheader("Feature Importance (XGBoost)")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        st.bar_chart(fi_df.set_index("Feature"))
    else:
        st.info("Feature importances not available for this model.")


# -----------------------------
# SHAP EXPLAINABILITY
# -----------------------------
def render_shap_explainability():
    st.subheader("SHAP Explainability")

    df_path = "data/cleaned/sales.csv"
    if not os.path.exists(df_path):
        st.warning("Sales data not found for SHAP analysis.")
        return

    df = pd.read_csv(df_path)
    X = df.drop(columns=['net_quantity', 'date'], errors='ignore')

    cat_cols = X.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        X_cat_encoded = encoder.transform(X[cat_cols])
        X_cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=encoder.get_feature_names_out(cat_cols), index=X.index)
        X = pd.concat([X.drop(columns=cat_cols), X_cat_encoded_df], axis=1)

    missing_cols = set(feature_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[feature_columns]

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.text("SHAP summary plot:")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(fig)


def render_user_management():
    st.subheader("👥 User Management")
    
    # Load current users
    try:
        with open('credentials.yaml', 'r') as f:
            config = yaml.safe_load(f)
        users = config['credentials']['usernames']
    except:
        users = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Current Users")
        for username, data in users.items():
            with st.expander(f"👤 {data['name']} (@{username})"):
                st.write(f"**Email**: {data['email']}")
                st.write(f"**Role**: {data.get('roles', ['user'])[0] if data.get('roles') else 'user'}")
                
                if st.button(f"🗑️ Delete {username}", key=f"del_{username}"):
                    del users[username]
                    save_credentials(users)
                    st.success(f"✅ Deleted {username}")
                    st.rerun()
    
    with col2:
        st.subheader("➕ Add New User")
        new_username = st.text_input("Username")
        new_email = st.text_input("Email")
        new_name = st.text_input("Full Name")
        new_password = st.text_input("Password", type="password")
        new_role = st.selectbox("Role", ["user", "admin"])
        
        if st.button("Create User"):
            if new_username and new_password:
                hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                
                users[new_username] = {
                    "name": new_name,
                    "email": new_email,
                    "password": hashed_pw,
                    "roles": [new_role]
                }
                
                save_credentials(users)
                st.success(f"✅ Created {new_username}")
                st.rerun()
            else:
                st.error("❌ Username and password required")

def save_credentials(users):
    config = {
        "credentials": {"usernames": users},
        "cookie": {
            "name": "price_optimizer_auth",
            "key": "super_secure_key_2026_change_in_prod",
            "expiry_days": 7
        }
    }
    with open('credentials.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
