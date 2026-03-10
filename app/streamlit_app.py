import streamlit as st
from auth import AuthManager
from dashboards.user_dashboard import render_user_dashboard
from dashboards.admin_dashboard import render_admin_dashboard

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