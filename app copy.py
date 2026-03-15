import streamlit as st
import numpy as np
import joblib

# --- Load model and scaler safely ---
try:
    model = joblib.load("laptop_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("❌ Model or scaler file not found")
    st.stop()


if "users" not in st.session_state:
    st.session_state["users"] = {}  # store users as {username: password}
if "login" not in st.session_state:
    st.session_state["login"] = False


def main_page():
    st.title("💻 Laptop Price Prediction")
    
    st.write("Enter Laptop Specifications")
    
    brand = st.selectbox("Brand", ["Dell", "HP", "Lenovo", "Apple", "Asus", "Acer"])
    processor_speed = st.number_input("Processor Speed (GHz)", 1.0, 5.0)
    ram = st.number_input("RAM (GB)", 2, 64)
    storage = st.number_input("Storage (GB)", 128, 2048)
    weight = st.number_input("Weight (kg)", 1.0, 5.0)
    
    if st.button("Predict Price"):
        new_data = np.array([[processor_speed, ram, storage, weight]])
        new_data_scaled = scaler.transform(new_data)
        prediction = model.predict(new_data_scaled)
        st.success(f"💰 Predicted Price for {brand} = ${prediction[0]:.2f}")

def login_page():
    st.title("🔐 Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username in st.session_state["users"] and st.session_state["users"][username] == password:
            st.session_state["login"] = True
            st.success("✅ Login Successful")
        else:
            st.error("❌ Invalid username or password")
    
    st.write("Don't have an account?")
    if st.button("Go to Registration"):
        st.session_state["page"] = "register"

def register_page():
    st.title("📝 Register")
    
    new_username = st.text_input("Choose Username")
    new_password = st.text_input("Choose Password", type="password")
    
    if st.button("Register"):
        if new_username in st.session_state["users"]:
            st.error("❌ Username already exists")
        else:
            st.session_state["users"][new_username] = new_password
            st.success("✅ Registration Successful. Please login.")
            st.session_state["page"] = "login"
    
    if st.button("Back to Login"):
        st.session_state["page"] = "login"

# --- Page Navigation ---
if "page" not in st.session_state:
    st.session_state["page"] = "login"

if st.session_state["login"]:
    main_page()
else:
    if st.session_state["page"] == "login":
        login_page()
    elif st.session_state["page"] == "register":
        register_page()