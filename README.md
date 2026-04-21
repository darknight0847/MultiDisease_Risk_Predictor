# MedScan AI — Multi-Disease Risk Predictor

A Flask web app that predicts risk for **Diabetes**, **Heart Disease**, and **Liver Disease**
using pre-trained XGBoost models with SHAP explainability.

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open your browser at:
#    http://127.0.0.1:5000
```

## Input Fields

| Field | Description | Typical Range |
|-------|-------------|---------------|
| Gender | 1 = Male, 2 = Female | 1 or 2 |
| Age | Years | 18–85 |
| BMI | Body Mass Index | 15–55 |
| Systolic BP | mmHg | 90–200 |
| Diastolic BP | mmHg | 50–130 |
| Waist | Circumference in cm | 60–160 |
| Glucose | Blood glucose mg/dL | 70–400 |
| Cholesterol | Total cholesterol mg/dL | 100–350 |
| Urine Albumin | μg/mL | 1–2000 |
| ACR | Albumin-to-Creatinine Ratio | 1–3000 |

## Files

```
Health_App_Web/
├── app.py                  ← Flask application
├── requirements.txt
├── README.md
├── saved_models/
│   ├── Diabetes_model.joblib
│   ├── Heart_Disease_model.joblib
│   ├── Liver_Disease_model.joblib
│   └── background_data.joblib   ← SHAP background (auto-generated)
└── templates/
    └── index.html
```

> ⚠️ For educational/research purposes only. Not a substitute for medical advice.
