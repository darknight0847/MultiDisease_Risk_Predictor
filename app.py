import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import shap
import warnings

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(page_title="MedScan AI — Multi-Disease Risk Predictor", layout="wide")

# --- 1. LOAD MODELS ---
@st.cache_resource
def load_models_and_data():
    models = {
        'Diabetes':      joblib.load('saved_models/Diabetes_model.joblib'),
        'Heart Disease': joblib.load('saved_models/Heart_Disease_model.joblib'),
        'Liver Disease': joblib.load('saved_models/Liver_Disease_model.joblib'),
    }
    background_data = joblib.load('saved_models/background_data.joblib')
    return models, background_data

models, background_data = load_models_and_data()

# --- 2. CONFIGURATIONS ---
MODEL_FEATURES = {
    'Diabetes':      ['RIDAGEYR', 'LBDGLUSI', 'LBXGLU', 'BMXWAIST', 'BMXBMI', 'BPXSY1', 'URDACT'],
    'Heart Disease': ['RIDAGEYR', 'URDACT', 'BMXWAIST', 'RIAGENDR', 'LBXTC'],
    'Liver Disease': ['RIDAGEYR', 'LBXGLU', 'BPXSY1', 'URXUMA', 'LBDGLUSI'],
}

# --- 3. UI LAYOUT ---
st.title("MedScan AI: Multi-Disease Risk Predictor")
st.markdown("Enter patient biomarkers to get simultaneous risk assessments for Diabetes, Heart Disease, and Liver Disease — with AI-explainable SHAP insights.")

st.sidebar.header("Patient Biomarkers")

# Define inputs with strictly enforced ranges
gender_str = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender = 1.0 if gender_str == "Male" else 2.0

age = st.sidebar.number_input("Age (yrs)", min_value=1.0, max_value=120.0, value=45.0, help="Range: 1 - 120 years")
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, help="Range: 10 - 100")
sys_bp = st.sidebar.number_input("Systolic BP", min_value=70.0, max_value=250.0, value=120.0, help="Range: 70 - 250 mmHg")
dia_bp = st.sidebar.number_input("Diastolic BP", min_value=40.0, max_value=150.0, value=80.0, help="Range: 40 - 150 mmHg")
waist = st.sidebar.number_input("Waist (cm)", min_value=40.0, max_value=250.0, value=90.0, help="Range: 40 - 250 cm")
glucose = st.sidebar.number_input("Glucose (mg/dL)", min_value=40.0, max_value=600.0, value=100.0, help="Range: 40 - 600 mg/dL")
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", min_value=50.0, max_value=600.0, value=180.0, help="Range: 50 - 600 mg/dL")
albumin = st.sidebar.number_input("Urine Albumin", min_value=0.0, max_value=1000.0, value=10.0, help="Range: 0 - 1000")
acr = st.sidebar.number_input("ACR (Albumin Creatinine Ratio)", min_value=0.0, max_value=10000.0, value=15.0, help="Range: 0 - 10000")

if st.sidebar.button("Run Analysis", type="primary"):
    with st.spinner("Analyzing..."):
        all_inputs = {
            'RIAGENDR':  gender,
            'RIDAGEYR':  age,
            'LBXGLU':   glucose,
            'LBDGLUSI': glucose * 0.0555, # Convert mg/dL to mmol/L for specific models
            'BMXBMI':   bmi,
            'BMXWAIST': waist,
            'LBXTC':    cholesterol,
            'BPXSY1':   sys_bp,
            'BPXDI1':   dia_bp,
            'URXUMA':   albumin,
            'URDACT':   acr,
        }

        st.subheader("Risk Assessment Report")
        cols = st.columns(3)

        for idx, (name, model) in enumerate(models.items()):
            # Select only the specific features needed for this disease model
            feats = MODEL_FEATURES[name]
            df = pd.DataFrame([{k: all_inputs[k] for k in feats}], columns=feats)

            # Predict probability of the disease (Class 1)
            prob = model.predict_proba(df)[0][1]
            risk_pct = round(prob * 100, 1)

            with cols[idx]:
                st.markdown(f"### {name}")
                if risk_pct > 70:
                    st.error(f"**HIGH RISK**: {risk_pct}% probability")
                elif risk_pct > 30:
                    st.warning(f"**MODERATE RISK**: {risk_pct}% probability")
                else:
                    st.success(f"**LOW RISK**: {risk_pct}% probability")

                # SHAP Explanation Generation
                predict_fn = lambda x: model.predict_proba(x)[:, 1]
                explainer = shap.Explainer(predict_fn, background_data[name])
                shap_values = explainer(df)

                with st.expander(f"View AI Explanation (SHAP Waterfall)"):
                    fig = plt.figure(figsize=(7, 4))
                    shap.plots.waterfall(shap_values[0], show=False)
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)

        st.info("⚠️ For research & educational purposes only. Powered by XGBoost + SHAP explainability. Risk scores reflect model probability, not clinical diagnosis.")