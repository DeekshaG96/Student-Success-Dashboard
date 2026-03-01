import streamlit_authenticator as stauth
import toml

secrets = toml.load(".streamlit/secrets.toml")
h = secrets["credentials"]["usernames"]["student"]["password"]

print("Loaded Hash:", h)

from passlib.context import CryptContext
ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

print("Valid?", ctx.verify("SIT_Student_2026", h))
