import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# -------------------------------
# Page Config & Custom Styling
# -------------------------------
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for a modern look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2E86C1;
        color: white;
        font-weight: bold;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# Load model & preprocessing files
# -------------------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("model.h5")
    with open("onehot_encoder_geo.pkl", "rb") as f:
        ohe = pickle.load(f)
    with open("label_encoder_gender.pkl", "rb") as f:
        le = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        sc = pickle.load(f)
    return model, ohe, le, sc

model, onehot_encoder_geo, label_encoder_gender, scaler = load_assets()

# -------------------------------
# Header Section
# -------------------------------
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3143/3143460.png", width=100)
with col2:
    st.title("ChurnGuard™ Predictor")
    st.caption("Advanced Neural Network Analysis for Banking Retention")

st.divider()

# -------------------------------
# Input Form (Multi-Column Layout)
# -------------------------------
st.subheader("📋 Customer Profile")

with st.container():
    # Row 1
    c1, c2, c3 = st.columns(3)
    with c1:
        geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
        gender = st.selectbox("👤 Gender", label_encoder_gender.classes_)
    with c2:
        age = st.slider("🎂 Age", 18, 92, 35)
        tenure = st.slider("⏳ Tenure (Years)", 0, 10, 5)
    with c3:
        credit_score = st.number_input("💳 Credit Score", 300, 900, 650)
        num_of_products = st.selectbox("📦 Number of Products", [1, 2, 3, 4])

    # Row 2
    c4, c5, c6 = st.columns(3)
    with c4:
        balance = st.number_input("🏦 Account Balance ($)", min_value=0.0, format="%.2f")
    with c5:
        salary = st.number_input("💰 Estimated Salary ($)", min_value=0.0, format="%.2f")
    with c6:
        st.write("🛠️ **Status Flags**")
        has_cr_card = st.checkbox("Has Credit Card", value=True)
        is_active = st.checkbox("Is Active Member", value=True)

# -------------------------------
# Data Processing
# -------------------------------
# Map boolean to int
cr_card_int = 1 if has_cr_card else 0
active_int = 1 if is_active else 0

# Encode Input
gender_encoded = label_encoder_gender.transform([gender])[0]
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))

# Construct DataFrame
input_data = pd.DataFrame({
    "Gender": [gender_encoded], "Age": [age], "Balance": [balance],
    "CreditScore": [credit_score], "EstimatedSalary": [salary], "Tenure": [tenure],
    "NumOfProducts": [num_of_products], "HasCrCard": [cr_card_int], "IsActiveMember": [active_int]
})

input_df = pd.concat([input_data, geo_encoded_df], axis=1)
input_df = input_df[scaler.feature_names_in_] # Match original order

# -------------------------------
# Prediction Section
# -------------------------------
st.markdown("---")
if st.button("🚀 ANALYZE CUSTOMER RETENTION"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled, verbose=0)
    prob = float(prediction[0][0])
    
    # UI Logic for results
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(label="Churn Probability", value=f"{prob*100:.1f}%")
        if prob > 0.5:
            st.error("⚠️ HIGH RISK")
        else:
            st.success("💎 LOYAL CUSTOMER")

    with res_col2:
        st.write("### Insights")
        st.progress(prob)
        if prob > 0.5:
            st.write("This customer shows behavior patterns similar to previous churned users. **Action recommended:** Outreach with loyalty offers.")
        else:
            st.write("This customer is currently stable. Maintain current engagement levels.")

# -------------------------------
# Footer
# -------------------------------
st.sidebar.markdown("### 📊 Model Stats")
st.sidebar.info("Model Accuracy: ~86%\nType: Sequential ANN")