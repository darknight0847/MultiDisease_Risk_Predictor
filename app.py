import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, render_template
import joblib
import pandas as pd
import shap
import io
import base64
import warnings

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')

app = Flask(__name__)

# --- 1. LOAD MODELS ---
models = {
    'Diabetes':      joblib.load('saved_models/Diabetes_model.joblib'),
    'Heart Disease': joblib.load('saved_models/Heart_Disease_model.joblib'),
    'Liver Disease': joblib.load('saved_models/Liver_Disease_model.joblib'),
}

# Load the background data (generated via fix_data.py to avoid Pandas version crash)
background_data = joblib.load('saved_models/background_data.joblib')

# --- 2. CONFIGURATIONS ---
MODEL_FEATURES = {
    'Diabetes':      ['RIDAGEYR', 'LBDGLUSI', 'LBXGLU', 'BMXWAIST', 'BMXBMI', 'BPXSY1', 'URDACT'],
    'Heart Disease': ['RIDAGEYR', 'URDACT', 'BMXWAIST', 'RIAGENDR', 'LBXTC'],
    'Liver Disease': ['RIDAGEYR', 'LBXGLU', 'BPXSY1', 'URXUMA', 'LBDGLUSI'],
}

FEATURE_LABELS = {
    'RIAGENDR':  'Gender',
    'RIDAGEYR':  'Age (yrs)',
    'LBXGLU':   'Glucose (mg/dL)',
    'LBDGLUSI': 'Glucose (mmol/L)',
    'BMXBMI':   'BMI',
    'BMXWAIST': 'Waist (cm)',
    'LBXTC':    'Cholesterol (mg/dL)',
    'BPXSY1':   'Systolic BP',
    'BPXDI1':   'Diastolic BP',
    'URXUMA':   'Urine Albumin',
    'URDACT':   'ACR',
}

# --- 3. FLASK ROUTE ---
@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    form_data = {}

    if request.method == 'POST':
        # Retrieve and convert form inputs
        glucose = float(request.form['glucose'])
        all_inputs = {
            'RIAGENDR':  float(request.form['gender']),
            'RIDAGEYR':  float(request.form['age']),
            'LBXGLU':   glucose,
            'LBDGLUSI': glucose * 0.0555, # Convert mg/dL to mmol/L for specific models
            'BMXBMI':   float(request.form['bmi']),
            'BMXWAIST': float(request.form['waist']),
            'LBXTC':    float(request.form['cholesterol']),
            'BPXSY1':   float(request.form['sys_bp']),
            'BPXDI1':   float(request.form['dia_bp']),
            'URXUMA':   float(request.form['albumin']),
            'URDACT':   float(request.form['acr']),
        }
        form_data = request.form.to_dict()

        results = {}
        for name, model in models.items():
            # Select only the specific features needed for this disease model
            feats = MODEL_FEATURES[name]
            df = pd.DataFrame([{k: all_inputs[k] for k in feats}], columns=feats)

            # Predict probability of the disease (Class 1)
            # Predict probability of the disease
            prob = model.predict_proba(df)[0][1]
            # SHAP Explanation Generation
            predict_fn = lambda x: model.predict_proba(x)[:, 1]
            explainer   = shap.Explainer(predict_fn, background_data[name])
            shap_values = explainer(df)

            # Generate SHAP Waterfall Plot
            plt.figure(figsize=(7, 4))
            
            # targets the single patient's data instead of passing the whole matrix
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()

            # Convert plot to Base64 to send to HTML without saving an image file
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode()
            plt.close()

            # Package results for HTML rendering
            results[name] = {
                'risk': round(prob * 100, 1),
                'plot': plot_url,
                'features': {FEATURE_LABELS.get(k, k): round(all_inputs[k], 3) for k in feats},
            }

    return render_template('index.html', results=results, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)