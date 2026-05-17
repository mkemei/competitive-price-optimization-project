import streamlit as st
from auth import AuthManager
from dashboards.user_dashboard import render_user_dashboard
from dashboards.admin_dashboard import render_admin_dashboard
from pos_handler import record_sale_to_csv

st.set_page_config(page_title="Price Optimization Platform", page_icon="📈", layout="wide")

class PriceOptimizerApp:
    def __init__(self):
        self.auth = AuthManager()
        st.session_state.setdefault("authentication_status", None)
        st.session_state.setdefault("user_role", None)
        st.session_state.setdefault("user_name", None)

    def run(self):
        auth_status = st.session_state.get("authentication_status")
        if auth_status:
            self.show_main_app()
        else:
            self.show_login()

    def show_login(self):
        st.title("🔐 Price Optimization Platform")
        st.markdown("Welcome to the **AI-powered retail price optimization system**. Please login to continue.")
        user_info = self.auth.login()
        if user_info:
            st.rerun()

    def show_main_app(self):
        role = st.session_state.get("user_role") or "user"
        name = st.session_state.get("user_name") or "User"

        st.markdown(f"""
        <div style="background: linear-gradient(90deg,#4b6cb7,#182848);
                    padding:20px; border-radius:10px; color:white">
        <h2>👋 Welcome, {name}</h2>
        <p>Access Level: <b>{role.upper()}</b></p>
        </div>
        """, unsafe_allow_html=True)

        with st.sidebar:
            st.header("Navigation")
            st.write(f"User: **{name}**")
            st.write(f"Role: **{role}**")
            st.divider()
            self.auth.logout()

        if role == "admin":
            render_admin_dashboard()
        else:
            render_user_dashboard()

if __name__ == "__main__":
    app = PriceOptimizerApp()
    app.run()

def render_pos_system(sales_df):
    st.divider()
    st.subheader("🛒 Point of Sale (POS)")
    st.info("Record daily sales here. This data is used by the Admin to refresh the AI model.")

    # Get product list for the dropdown
    product_list = sorted(sales_df['product'].unique())

    with st.form("new_sale_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            product = st.selectbox("Select Product", options=product_list)
        with col2:
            price = st.number_input("Selling Price (KES)", min_value=0.0, step=10.0)
        with col3:
            qty = st.number_input("Quantity Sold", min_value=1, step=1)
            
        submitted = st.form_submit_button("✅ Complete Sale")
        
        if submitted:
            if price > 0:
                success = record_sale_to_csv(product, price, qty)
                if success:
                    st.success(f"Transaction Recorded: {qty} units of {product}")
                    # Clear cache so the trends/charts refresh with new data
                    st.cache_data.clear()
            else:
                st.warning("Please enter a valid price.")