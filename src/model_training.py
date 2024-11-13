# src/model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_model():
    print("Starting model training...")
    
    # Load the processed data
    df = pd.read_csv('data/processed_data.csv')
    print("Processed data loaded.")

    df.columns = df.columns.astype(str)

    # Separate features and target variable
    X = df.drop('log_price', axis=1)
    y = df['log_price']
    print(f"Features and target variable separated. X shape: {X.shape}, y shape: {y.shape}")

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )
    print(f"Data split into training, validation, and test sets.")

    # Scale the features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)
    x_test_scaled = scaler.transform(x_test)
    print("Feature scaling completed.")

    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    x_valid_scaled = pd.DataFrame(x_valid_scaled, columns=x_train.columns)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_train.columns)

    # Train the model
    print("Training the model...")
    try:
        model = XGBRegressor(random_state=42)
        model.fit(x_train_scaled, y_train)
        print("Model training completed.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return  # Exit the function if training fails

    # Evaluate the model
    y_pred = model.predict(x_valid_scaled)
    mse = mean_squared_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    print(f"Validation Mean Squared Error: {mse}")
    print(f"Validation R-squared: {r2}")

    # Save the model and scaler
    print("Saving the model and scaler...")
    try:
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/car_price_predictor_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        feature_names = x_train.columns.tolist()
        joblib.dump(feature_names, 'models/feature_names.pkl')
        print("Model and scaler saved successfully.")
    except Exception as e:
        print(f"Error saving model or scaler: {e}")

if __name__ == "__main__":
    train_model()
