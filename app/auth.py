import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import bcrypt


class AuthManager:

    def __init__(self, credentials_path="credentials.yaml"):
        self.credentials_path = os.path.abspath(credentials_path)
        self.authenticator = None
        self.config = None
        self._load_authenticator()

    # --------------------------------
    # LOAD AUTH CONFIG
    # --------------------------------
    def _load_authenticator(self):

        if not os.path.exists(self.credentials_path):
            self._create_default_credentials()

        with open(self.credentials_path) as file:
            self.config = yaml.load(file, Loader=SafeLoader)

        self.authenticator = stauth.Authenticate(
            self.config["credentials"],
            self.config["cookie"]["name"],
            self.config["cookie"]["key"],
            self.config["cookie"]["expiry_days"],
        )

    # --------------------------------
    # CREATE DEFAULT USERS
    # --------------------------------
    def _create_default_credentials(self):

        creds = {
            "usernames": {
                "admin": {
                    "name": "System Administrator",
                    "email": "admin@retail.co.ke",
                    "password": "admin123",
					"role": "admin"
                },
                "user1": {
                    "name": "Store Manager",
                    "email": "manager@retail.co.ke",
                    "password": "user123",
					"role": "user"
                },
            }
        }

        # hash passwords

        passwords = [user["password"] for user in creds["usernames"].values()]
        hashed_passwords = stauth.Hasher(passwords).generate()
        
            # Assign back to creds
        for i, username in enumerate(creds["usernames"]):
             creds["usernames"][username]["password"] = hashed_passwords[i]
        
        config = {
            "credentials": {
                "usernames": usernames
            },
            "cookie": {
                "name": "price_optimizer_auth",
                "key": "super_secure_key_2026",
                "expiry_days": 1,
            }
        }

        with open(self.credentials_path, "w") as file:
            yaml.dump(config, file)

    # --------------------------------
    # LOGIN
    # --------------------------------
    def login(self):

        # Latest API requires keyword argument
        self.authenticator.login(location="main")

        auth_status = st.session_state.get("authentication_status")

        if auth_status:

            username = st.session_state.get("username")
            name = st.session_state.get("name")

            role = "admin" if username == "admin" else "user"

            st.session_state["user_role"] = role
            st.session_state["user_name"] = name

            return {
                "username": username,
                "name": name,
                "role": role,
            }

        elif auth_status is False:
            st.error("❌ Username or password incorrect")

        return None

    # --------------------------------
    # LOGOUT
    # --------------------------------
    def logout(self):

        if self.authenticator:
            self.authenticator.logout(location="sidebar")


    # ------------------------------------------------
    # CREATE NEW USER
    # ------------------------------------------------
    def create_user(self, username, name, email, password):

        with open(self.credentials_path) as file:
            config = yaml.load(file, Loader=SafeLoader)

        usernames = config["credentials"]["usernames"]

        if username in usernames:
            st.error("User already exists")
            return False

        # Hash password
        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        usernames[username] = {
            "name": name,
            "email": email,
            "password": hashed_pw,
            "role": role
        }

        with open(self.credentials_path, "w") as file:
            yaml.dump(config, file)

        # Reload authenticator
        self._load_authenticator()

        return True