import streamlit as st
import pandas as pd
import os
from datetime import datetime

from auth import AuthManager
from dashboards.user_dashboard import render_user_dashboard, load_sales_data, record_sale_to_csv
from dashboards.admin_dashboard import render_admin_dashboard
from scraper_engine import main as run_scraper_pipeline

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(
    page_title="Price Optimization Platform",
    page_icon="📈",
    layout="wide"
)

# ==========================================================
# APP CLASS
# ==========================================================
class PriceOptimizerApp:

    def __init__(self):
        self.auth = AuthManager()

        # Session defaults
        st.session_state.setdefault("authentication_status", None)
        st.session_state.setdefault("user_role", None)
        st.session_state.setdefault("user_name", None)

    # -----------------------------
    # ENTRY
    # -----------------------------
    def run(self):
        if st.session_state.get("authentication_status"):
            self.show_main_app()
        else:
            self.show_login()

    # -----------------------------
    # LOGIN
    # -----------------------------
    def show_login(self):
        st.title("🔐 Price Optimization Platform")
        st.markdown("price optimization system.")

        user_info = self.auth.login()
        if user_info:
            st.rerun()

    # -----------------------------
    # MAIN APP (LAYOUT & NAVIGATION)
    # -----------------------------
    def show_main_app(self):
        role = st.session_state.get("user_role") or "user"
        name = st.session_state.get("user_name") or "User"

        # Header
        st.markdown(f"""
        <div style="background: linear-gradient(90deg,#4b6cb7,#182848);
                    padding:20px; border-radius:10px; color:white">
        <h2>👋 Welcome, {name}</h2>
        <p>Access Level: <b>{role.upper()}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar
        with st.sidebar:
            st.header("⚙️ Control Panel")
            st.write(f"👤 {name} ({role})")
            st.write(f"🔐 **{role.upper()}**")
            st.divider()

            # Admin-only data sync
            if role == "admin":
                st.subheader("📡 Market Data")
                if st.button("🔄 Sync Competitor Prices"):
                    with st.spinner("Scraping market data..."):
                        run_scraper_pipeline("data/pos_sales_data.csv")
                        st.success("✅ Market data updated")
                        st.cache_data.clear()
                st.divider()
            # Navigation
            if role == "admin":
                navigations= ["📊 Dashboard", "🛒 Competitor Intelligence"]
            else:
                navigations = ["📊 Dashboard", "🛒 Competitor Intelligence", "🛍️ POS Terminal"]
            # Navigation
            page = st.radio(
                "📂 Navigation",
                 navigations
            )

            st.divider()
            self.auth.logout()

        # -----------------------------
        # PAGE ROUTING
        # -----------------------------
        if page == "📊 Dashboard":
            if role == "admin":
                render_admin_dashboard()
            else:
                render_user_dashboard()

        elif page == "🛒 Competitor Intelligence":
            self.render_competitor_dashboard()
            
        elif page == "🛍️ POS Terminal":
            self.render_pos_system()

    # ==========================================================
    # CLASS METHODS 
    # ==========================================================

    @st.cache_data
    def load_market_data(_self):
        path = "data/competitor_prices"
        if not os.path.exists(path):
            return None
        files = sorted(os.listdir(path), reverse=True)
        if not files:
            return None
        latest_file = os.path.join(path, files[0])
        return pd.read_csv(latest_file)

    def render_competitor_dashboard(self):
        st.title("🇰🇪 Competitor Price Intelligence")
        df = self.load_market_data()
        if df is None:
            st.warning("No market data available. Please sync data from sidebar.")
            return

        # KPI METRICS
        col1, col2 = st.columns(2)
        with col1: st.metric("Total Products", len(df))
        with col2:
            coverage = df[['jumia_price', 'totshoppe_price', 'peekaboo_price']].notna().mean().mean() * 100
            st.metric("Market Coverage", f"{coverage:.1f}%")

        st.divider()
        
        product_list = ["All Products"] + sorted(df['matched_user_product_original'].unique())
        selected_product = st.selectbox("🔍 Select Product", options=product_list)

        filtered_df = df.copy()
        if selected_product != "All Products":
            filtered_df = filtered_df[filtered_df['matched_user_product_original'] == selected_product]

        st.subheader("📊 Market Pricing Table")
        st.dataframe(filtered_df, use_container_width=True)

    # ----------------------------------------------------------
    # POS SYSTEM
    # ----------------------------------------------------------
    def render_pos_system(self):
        """POS View"""
        st.title("🛍️ POS Terminal")
        
        # Load the latest sales data to get product lists and last prices
        sales_df = load_sales_data()
        
        if sales_df is not None:
            st.info("Record sales here to update the historical dataset for 2026.")
            product_list = sorted(sales_df['product'].unique())
            
            with st.container(border=True):
                # Use a form to group inputs and allow 'clear_on_submit'
                with st.form("pos_terminal_form", clear_on_submit=True):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        product = st.selectbox("Select Product to Sell", options=product_list)
                    with col2:
                        # Auto-suggest the last recorded price for this item
                        last_p = sales_df[sales_df['product'] == product]['unit_price'].iloc[-1] if not sales_df[sales_df['product'] == product].empty else 0.0
                        price = st.number_input("Selling Price (KES)", min_value=0.0, value=float(last_p), step=10.0)
                    with col3:
                        qty = st.number_input("Quantity Sold", min_value=1, step=1)
                        
                    submitted = st.form_submit_button("✅ Complete Sale", use_container_width=True)
                    
                    if submitted:
                        if price > 0:
                            # Calls  function to record ale
                            if record_sale_to_csv(product, price, qty):
                                st.success(f"Transaction Recorded: {qty} units of {product}")
                                st.balloons()
                                st.cache_data.clear() # Forces Dashboard graphs to update
                        else:
                            st.warning("Please enter a valid price.")
            
            # Show a history log of recent entries below the form
            st.divider()
            st.subheader("🕒 Recent Transactions (2026 Live Log)")
            st.dataframe(sales_df.sort_values('date', ascending=False).head(10), use_container_width=True)
        else:
            st.error("Could not load sales data from 'data/cleaned/sales.csv'. Please check file path.")

# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    app = PriceOptimizerApp()
    app.run()