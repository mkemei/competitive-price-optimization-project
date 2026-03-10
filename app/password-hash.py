import streamlit_authenticator as stauth

# 1. Create a dummy credentials structure
credentials = {
    'usernames': {
        'admin': {'password': 'admin123'},
        'user1': {'password': 'user123'}
    }
}

# 2. Pass the WHOLE dictionary to the hasher
# This method modifies the dictionary in-place
stauth.Hasher.hash_passwords(credentials)

# 3. Print the results
for username, info in credentials['usernames'].items():
    print(f"User: {username}")
    print(f"Hashed Password: {info['password']}\n")