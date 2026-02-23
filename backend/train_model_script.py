import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# 1. Load the data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'yield_data.csv')
df = pd.read_csv(data_path)

# 2. Define features and target
X = df.drop('Yield_Tonnes_Per_Acre', axis=1)
y = df['Yield_Tonnes_Per_Acre']

# 3. Preprocessing for categorical and numerical data
categorical_features = ['Crop', 'Season']
numerical_features = ['N', 'P', 'K', 'pH', 'Temp', 'Humidity', 'Rainfall']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Create the pipeline with Random Forest
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# 6. Save the model
joblib.dump(model_pipeline, 'crop_yield_model.joblib')

print("Model trained and saved successfully as 'crop_yield_model.joblib'")

# Verify prediction
test_input = pd.DataFrame([{
    'Crop': 'Rice',
    'Season': 'Kharif',
    'N': 80, 'P': 40, 'K': 40, 'pH': 6.0,
    'Temp': 28, 'Humidity': 80, 'Rainfall': 200
}])
prediction = model_pipeline.predict(test_input)
print(f"Test Prediction for Rice/Kharif: {prediction[0]:.2f} Tonnes")
