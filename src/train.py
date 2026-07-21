import pandas as pd
import numpy as np
import os
import joblib
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

def train_model():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'salary_data.csv')
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['salary'])
    y = df['salary']
    
    # Identify categorical and numerical columns
    categorical_cols = ['education_level', 'location']
    numerical_cols = ['years_experience', 'is_manager']
    
    # Preprocessing for categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Build the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Model Performance -> MAE: ${mae:,.2f} | R2: {r2:.3f}")
    
    # SHAP Explainer
    # We need to compute SHAP values on the transformed data
    X_train_transformed = preprocessor.fit_transform(X_train)
    # Get feature names from preprocessor
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = list(cat_features) + numerical_cols
    
    # Create a TreeExplainer for the XGBoost model
    explainer = shap.TreeExplainer(pipeline.named_steps['regressor'])
    
    # Save the pipeline, explainer, and feature names
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(pipeline, os.path.join(model_dir, 'pipeline.pkl'))
    joblib.dump(explainer, os.path.join(model_dir, 'explainer.pkl'))
    joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.pkl'))
    
    print("Model and explainer saved successfully.")

if __name__ == "__main__":
    train_model()
