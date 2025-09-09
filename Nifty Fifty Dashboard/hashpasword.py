import streamlit_authenticator as stauth

hashed_pw = stauth.Hasher(['aiman123']).generate()
print(hashed_pw)