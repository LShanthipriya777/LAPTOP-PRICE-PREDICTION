import streamlit as st
import numpy as np
import joblib

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# ---------- CSS STYLING ----------
st.markdown("""
<style>
.stApp {
    background-color: #f3e8ff;
}

/* Titles h1 all black */
h1 {
    color: black !important;
    text-align: center;
}

/* Input labels */
label {
    color: black !important;
    font-weight: 600;
}

/* Input box text and background */
.stTextInput input {
    color: black !important;
    background-color: white !important;
}

/* Paragraphs like "New user?" */
p {
    color: black !important;
}

/* Buttons */
div.stButton > button:first-child {
    background-color: #7d5fff;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 16px;
    border: none;
}
div.stButton > button:hover {
    background-color: #5f3dc4;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
try:
    model = joblib.load("laptop_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("Model or scaler file not found")
    st.stop()

# ---------- SESSION ----------
if "users" not in st.session_state:
    st.session_state["users"] = {}
if "login" not in st.session_state:
    st.session_state["login"] = False
if "page" not in st.session_state:
    st.session_state["page"] = "login"

# ---------- MAIN PAGE ----------
def main_page():
    st.markdown("<h1>Laptop Price Prediction</h1>", unsafe_allow_html=True)

    brand = st.selectbox("Brand", ["Dell", "HP", "Lenovo", "Apple", "Asus", "Acer"])
    processor_speed = st.number_input("Processor Speed (GHz)", 1.0, 5.0)
    ram = st.number_input("RAM (GB)", 2, 64)
    storage = st.number_input("Storage (GB)", 128, 2048)
    weight = st.number_input("Weight (kg)", 1.0, 5.0)

    if st.button("Predict Price"):
        data = np.array([[processor_speed, ram, storage, weight]])
        data = scaler.transform(data)
        price = model.predict(data)
        st.success(f"Predicted Price for {brand} = ${price[0]:.2f}")

    if st.button("Logout"):
        st.session_state["login"] = False
        st.session_state["page"] = "login"

# ---------- LOGIN PAGE ----------
def login_page():
    st.markdown("<h1>Login</h1>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in st.session_state["users"] and st.session_state["users"][username] == password:
            st.session_state["login"] = True
            st.success("Login Successful")
        else:
            st.error("Invalid Username or Password")

    st.write("New user?")
    if st.button("Go to Register"):
        st.session_state["page"] = "register"

# ---------- REGISTER PAGE ----------
def register_page():
    st.markdown("<h1>Register</h1>", unsafe_allow_html=True)

    new_user = st.text_input("Create Username")
    new_pass = st.text_input("Create Password", type="password")

    if st.button("Register"):
        if new_user in st.session_state["users"]:
            st.error("Username already exists")
        else:
            st.session_state["users"][new_user] = new_pass
            st.success("Registration Successful")
            st.session_state["page"] = "login"

    if st.button("Back to Login"):
        st.session_state["page"] = "login"

# ---------- NAVIGATION ----------
if st.session_state["login"]:
    main_page()
else:
    if st.session_state["page"] == "login":
        login_page()
    else:
        register_page()