import os
import joblib
import shap
import pandas as pd

# 1. Import your existing class
from Medical_Risk_Predictor_App import HealthRiskPredictor

# 2. Setup paths
DATA_DIRECTORY = r"C:\Users\Dev Patel\Desktop\jupyter projects\Health_risk_predictor\Health_App_Web"

print("⏳ Loading data (this is fast)...")
predictor = HealthRiskPredictor(data_dir=DATA_DIRECTORY)
predictor.load_and_merge_data()

# 3. Extract the background data manually
background_data = {}
print("🧠 Generating SHAP background baselines...")

for name, config in predictor.models_config.items():
    target = config['target']
    features = config['features']
    
    # Isolate clean data for this specific target
    df_clean = predictor.df_final.dropna(subset=[target])
    X = df_clean[features]
    
    # Take 100 representative samples (Standard practice for SHAP)
    background_data[name] = shap.sample(X, 100)
    print(f" -> Sampled 100 patients for {name}")

# 4. Save it to your existing folder
os.makedirs('saved_models', exist_ok=True)
joblib.dump(background_data, 'saved_models/background_data.joblib')

print("✅ background_data.joblib successfully saved! You can start Flask now.")