import streamlit_authenticator as stauth
import toml

secrets_dict = toml.load(".streamlit/secrets.toml")
authenticator = stauth.Authenticate(
    secrets_dict["credentials"],
    secrets_dict["cookie"]["name"],
    secrets_dict["cookie"]["key"],
    secrets_dict["cookie"].get("expiry_days", 30),
    secrets_dict.get("preauthorized", None)
)

authenticator.username = "student"
authenticator.password = "SIT_Student_2026"

try:
    result = authenticator._check_pw()
    print("Does password check pass? ", result)
except Exception as e:
    print("Exception checking password:", e)

# Let's also verify if there's any weird bcrypt hash conversion bug
print("Stored hash:", authenticator.credentials['usernames']['student']['password'])
