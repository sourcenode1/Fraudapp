import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from catboost import Pool
import warnings
import os

warnings.filterwarnings("ignore")
GLOBAL_SEED = 50
np.random.seed(GLOBAL_SEED)

# ---------------- Load model ----------------
@st.cache_resource(show_spinner=True)
def load_model(path="final_AIML_model.pkl"):
    if not os.path.exists(path):
        st.error(f"Model file '{path}' not found in the repo!")
        return None
    return joblib.load(path)

model = load_model()
if model is None:
    st.stop()

# ---------------- Load sample input ----------------
@st.cache_data
def load_sample_input(path="warranty_claim_fraud_detection_cleaned.csv"):
    if not os.path.exists(path):
        st.warning(f"Sample input CSV '{path}' not found in the repo!")
        return pd.DataFrame()
    return pd.read_csv(path)

sample_input = load_sample_input()
if sample_input.empty:
    st.warning("No sample input loaded. Manual entry will still work.")

# ---------------- Feature mappings ----------------
days_mapping = {'more than 30': 4, '15 to 30': 3, '8 to 15': 2, '1 to 7': 1, 'none': 0, 'NA': 0}
claims_mapping = {'none': 0, '1': 1, '2 to 4': 3, 'more than 4': 5, 'NA': 0}
vehicle_age_mapping = {'new': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
                       '5 years': 5, '6 years': 6, '7 years': 7, 'more than 7': 8, 'NA': -1}
policyholder_age_mapping = {'16 to 17': 1, '18 to 20': 2, '21 to 25': 3, '26 to 30': 4, '31 to 35': 5,
                            '36 to 40': 6, '41 to 50': 7, '51 to 65': 8, 'over 65': 9, 'NA': -1}
suppliments_mapping = {'none': 0, '1 to 2': 2, '3 to 5': 4, 'more than 5': 6, 'NA': 0}
address_change_mapping = {'no change': 0, 'under 6 months': 1, '1 year': 2, '2 to 3 years': 3,
                          '4 to 8 years': 4, 'more than 8 years': 5, 'NA': 0}
number_of_cars_mapping = {'1 vehicle': 1, '2 vehicles': 2, '3 to 4': 3, '5 to 8': 5,
                          'more than 8': 9, 'NA': 0}

categorical_cols = ['Month', 'DayOfWeek', 'Make', 'AccidentArea',
                    'DayOfWeekClaimed', 'MonthClaimed', 'Sex', 'MaritalStatus',
                    'Fault', 'PolicyType', 'VehicleCategory', 'VehiclePrice',
                    'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'BasePolicy']

# ---------------- Preprocess input ----------------
def preprocess_input(df):
    df = df.copy()
    df['Days_Policy_Accident'] = df['Days_Policy_Accident'].map(days_mapping).fillna(0)
    df['Days_Policy_Claim'] = df['Days_Policy_Claim'].map(days_mapping).fillna(0)
    df['PastNumberOfClaims'] = df['PastNumberOfClaims'].map(claims_mapping).fillna(0)
    df['AgeOfVehicle'] = df['AgeOfVehicle'].map(vehicle_age_mapping).fillna(-1)
    df['AgeOfPolicyHolder'] = df['AgeOfPolicyHolder'].map(policyholder_age_mapping).fillna(-1)
    df['NumberOfSuppliments'] = df['NumberOfSuppliments'].map(suppliments_mapping).fillna(0)
    df['AddressChange_Claim'] = df['AddressChange_Claim'].map(address_change_mapping).fillna(0)
    df['NumberOfCars'] = df['NumberOfCars'].map(number_of_cars_mapping).fillna(0)

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('NA')
    return df

# ---------------- Streamlit UI ----------------
st.title("🚨 Warranty Claim Fraud Detection")
st.markdown("""
This app detects **fraudulent warranty claims** using a CatBoost ML model.
You can input data manually or upload a CSV file.
""")

st.sidebar.header("Input Options")
input_method = st.sidebar.radio("Select Input Method:", ["Manual Entry", "Upload CSV"])
input_df = None

if input_method == "Manual Entry":
    st.subheader("🔧 Manual Input")
    manual_input = {}
    for col in sample_input.columns:
        if col in categorical_cols:
            options = sorted(sample_input[col].dropna().unique())
            manual_input[col] = st.selectbox(f"{col}", options)
        else:
            try:
                mean_val = float(sample_input[col].dropna().astype(float).mean())
            except Exception:
                mean_val = 0.0
            manual_input[col] = st.number_input(f"{col}", value=mean_val)
    input_df = pd.DataFrame([manual_input])
    input_df = preprocess_input(input_df)

elif input_method == "Upload CSV":
    st.subheader("📤 Upload CSV File")
    uploaded_file = st.file_uploader("Upload your input CSV", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        input_df = preprocess_input(input_df)

if input_df is not None and not input_df.empty:
    st.write("### Input Data Preview")
    st.dataframe(input_df)

    # Prepare CatBoost Pool
    cat_indices = [input_df.columns.get_loc(c) for c in categorical_cols if c in input_df.columns]
    pool = Pool(input_df, cat_features=cat_indices)

    # Predict
    proba = model.predict_proba(pool)[:, 1]

    # Threshold
    st.subheader("⚖️ Adjust Decision Threshold")
    threshold = st.slider("Probability Threshold", 0.0, 1.0, 0.31, 0.01)
    preds = (proba > threshold).astype(int)
    labels = np.where(preds == 1, "⚠️ Fraud", "✅ Not Fraud")

    results_df = input_df.copy()
    results_df['Fraud_Probability'] = proba
    results_df['Prediction'] = labels

    st.write("### Prediction Results")
    st.dataframe(results_df[['Fraud_Probability', 'Prediction']])
    st.success(f"Latest prediction: **{labels[-1]}** with probability **{proba[-1]:.2f}**")

    # SHAP explanation
    st.subheader("🔍 SHAP Explanation")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure()
        shap.summary_plot(shap_values, input_df, show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")

# ---------------- Model Performance ----------------
st.subheader("📊 Model Performance Visualizations")
col1, col2 = st.columns(2)
with col1:
    if os.path.exists("pr_curve.png"):
        st.image("pr_curve.png", caption="Precision-Recall Curve", use_column_width=True)
with col2:
    if os.path.exists("shap_summary.png"):
        st.image("shap_summary.png", caption="SHAP Summary Plot", use_column_width=True)

st.markdown("---")
if os.path.exists("AIML_fraud_detection_report.pdf"):
    with open("AIML_fraud_detection_report.pdf", "rb") as f:
        st.download_button("📄 Download PDF Report", f, file_name="AIML_fraud_detection_report.pdf")
else:
    st.info("PDF report not found in the repo.")

st.caption("Developed as a Proof-of-Concept for warranty claim fraud detection using AI/ML.")
