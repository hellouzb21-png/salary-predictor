import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Salary Predictor API with SHAP")

# Load models at startup
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
try:
    pipeline = joblib.load(os.path.join(model_dir, 'pipeline.pkl'))
    explainer = joblib.load(os.path.join(model_dir, 'explainer.pkl'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
except Exception as e:
    print(f"Warning: Could not load models on startup. Did you run train.py? Error: {e}")
    pipeline, explainer, feature_names = None, None, None

class PredictionRequest(BaseModel):
    years_experience: float
    education_level: str  # 'Bachelors', 'Masters', 'PhD'
    location: str         # 'San Francisco', 'New York', 'Austin', 'Remote', 'London'
    is_manager: int       # 0 or 1

class PredictionResponse(BaseModel):
    predicted_salary: float
    shap_base_value: float
    shap_values: dict

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": pipeline is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    # Convert request to DataFrame
    input_df = pd.DataFrame([{
        'years_experience': request.years_experience,
        'education_level': request.education_level,
        'location': request.location,
        'is_manager': request.is_manager
    }])
    
    # 1. Predict
    pred = pipeline.predict(input_df)[0]
    
    # 2. SHAP Explanation
    # We must transform the input first to pass it to the TreeExplainer
    preprocessor = pipeline.named_steps['preprocessor']
    X_transformed = preprocessor.transform(input_df)
    
    # Calculate SHAP values
    shap_vals = explainer.shap_values(X_transformed)
    
    # explainer.expected_value could be an array or float depending on the model version
    base_val = explainer.expected_value
    if isinstance(base_val, list) or isinstance(base_val, np.ndarray):
        base_val = float(base_val[0])
    else:
        base_val = float(base_val)
        
    # Map SHAP values to feature names
    shap_dict = {feat: float(val) for feat, val in zip(feature_names, shap_vals[0])}
    
    return PredictionResponse(
        predicted_salary=float(pred),
        shap_base_value=base_val,
        shap_values=shap_dict
    )
