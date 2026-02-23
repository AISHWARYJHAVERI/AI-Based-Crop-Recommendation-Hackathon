import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'yield_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'crop_yield_model.joblib')

def train_model():
    """Reads yield_data.csv, trains a Random Forest model, and saves it."""
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return False
    
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Define features and target
    X = df.drop('Yield_Tonnes_Per_Acre', axis=1)
    y = df['Yield_Tonnes_Per_Acre']
    
    # Preprocessing for categorical and numerical data
    categorical_features = ['Crop', 'Season']
    numerical_features = ['N', 'P', 'K', 'pH', 'Temp', 'Humidity', 'Rainfall']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create the pipeline with Random Forest
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training Random Forest model...")
    model_pipeline.fit(X_train, y_train)
    
    # Calculate Metrics
    y_pred = model_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nModel Metrics:\n- MSE: {mse:.4f}\n- R2 Score: {r2:.4f}")
    
    # Save Metrics to a text file
    metrics_path = os.path.join(BASE_DIR, 'model_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Mean Squared Error: {mse:.4f}\n")
        f.write(f"R2 Score: {r2:.4f}\n")

    # Generate Visualizations
    plots_dir = os.path.join(os.path.dirname(BASE_DIR), 'backend', 'static', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(10, 6))
    regressor = model_pipeline.named_steps['regressor']
    # Handle feature names from ColumnTransformer
    features = numerical_features + list(model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))
    importances = regressor.feature_importances_
    # Sort top 10
    indices = np.argsort(importances)[::-1][:10]
    sns.barplot(x=importances[indices], y=np.array(features)[indices], palette='viridis')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    plt.close()
    
    # 2. Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='green')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title('Actual vs Predicted Yield (Tonnes/Acre)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'))
    plt.close()
    
    print(f"Visualizations saved to {plots_dir}")
    
    # Save the model
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model_pipeline, MODEL_PATH)
    print("Training complete.")
    return True

def predict_yield(crop, season, n, p, k, ph, temp, humidity, rainfall):
    """Loads the saved model and returns a yield prediction."""
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training a new one...")
        if not train_model():
            return None
            
    model = joblib.load(MODEL_PATH)
    
    # Prepare input for model
    input_df = pd.DataFrame([{
        'Crop': crop,
        'Season': season,
        'N': n,
        'P': p,
        'K': k,
        'pH': ph,
        'Temp': temp,
        'Humidity': humidity,
        'Rainfall': rainfall
    }])
    
    # Predict using model
    prediction = model.predict(input_df)
    return float(prediction[0])

if __name__ == "__main__":
    # Test the functionality
    print("--- Running new_model.py as a standalone script ---")
    if train_model():
        test_crop = "Rice"
        test_season = "Kharif"
        print(f"\nTesting prediction for {test_crop} in {test_season}...")
        res = predict_yield(test_crop, test_season, 80, 40, 40, 6.0, 28, 80, 200)
        if res is not None:
            print(f"Predicted Yield: {res:.2f} Tonnes Per Acre")
        else:
            print("Prediction failed.")
