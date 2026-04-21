import joblib
import pandas as pd
import numpy as np

# The features expected by your app
MODEL_FEATURES = {
    'Diabetes':      ['RIDAGEYR', 'LBDGLUSI', 'LBXGLU', 'BMXWAIST', 'BMXBMI', 'BPXSY1', 'URDACT'],
    'Heart Disease': ['RIDAGEYR', 'URDACT', 'BMXWAIST', 'RIAGENDR', 'LBXTC'],
    'Liver Disease': ['RIDAGEYR', 'LBXGLU', 'BPXSY1', 'URXUMA', 'LBDGLUSI'],
}

background_data = {}

# Generate baseline data formatted for your current version of Pandas
for name, feats in MODEL_FEATURES.items():
    # Creating 100 rows of baseline data (zeros) for SHAP to use
    df = pd.DataFrame(np.zeros((100, len(feats))), columns=feats)
    background_data[name] = df

# Overwrite the old, broken file
joblib.dump(background_data, 'saved_models/background_data.joblib')
print("✅ New background_data.joblib successfully created!")