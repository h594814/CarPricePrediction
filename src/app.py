# src/app.py

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import os


# Load the saved model and preprocessing objects
model_path = os.path.join('models', 'car_price_predictor_model.pkl')
scaler_path = os.path.join('models', 'scaler.pkl')
encoder_path = os.path.join('models', 'ordinal_encoder.pkl')
feature_names_path = os.path.join('models', 'feature_names.pkl')

# Check if model files exist
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}. Please run 'model_training.py' first.")
    exit(1)
if not os.path.exists(scaler_path):
    print(f"Scaler file not found at {scaler_path}. Please run 'model_training.py' first.")
    exit(1)
if not os.path.exists(encoder_path):
    print(f"Encoder file not found at {encoder_path}. Please run 'data_preprocessing.py' first.")
    exit(1)
if not os.path.exists(feature_names_path):
    print(f"Feature names file not found at {feature_names_path}. Please run 'model_training.py' first.")
    exit(1)

try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

try:
    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit(1)

try:
    ordinal_encoder = joblib.load(encoder_path)
    print("Ordinal Encoder loaded successfully.")
except Exception as e:
    print(f"Error loading encoder: {e}")
    exit(1)

try:
    feature_names = joblib.load(feature_names_path)
    print("Feature names loaded successfully.")
except Exception as e:
    print(f"Error loading feature names: {e}")
    exit(1)

# # Define input choices
df_raw = pd.read_csv('data/used_cars2.csv')  # Load raw data to extract unique values
# brand_choices = df_raw['brand'].unique().tolist()
# fuel_type_choices = df_raw['fuel_type'].unique().tolist()
# transmission_choices = df_raw['transmission'].unique().tolist()
# color_choices = ['black', 'white', 'red', 'blue', 'silver', 'gray', 'brown', 'yellow', 'green', 'orange', 'other']
brand_choices = df_raw['brand'].unique().tolist()
fuel_type_choices = ['Electric', 'Hybrid', 'Diesel', 'E85 Flex Fuel', 'Gasoline', 'not supported']
transmission_choices = ['Automatic', 'Manual', 'Other']
color_choices = ['black', 'white', 'red', 'blue', 'silver', 'gray', 'brown', 'yellow', 'green', 'orange', 'other']


def predict_price(brand, milage, fuel_type, transmission, ext_col, int_col,
                  age, horsepower, engine_size, num_cylinders, accident_history, clean_title):
    try:
        # Create a DataFrame from the input
        input_data = pd.DataFrame({
            'brand': [brand],
            'fuel_type': [fuel_type],
            'transmission': [transmission],
            'ext_col': [ext_col],
            'int_col': [int_col],
            'Age': [age],
            'horsepower': [horsepower],
            'engine_size': [engine_size],
            'num_cylinders': [num_cylinders],
            'accident_encoded': [int(accident_history)],
            'clean_title_encoded': [int(clean_title)],
            'log_milage': [np.log1p(milage)],
        })

        # Simplify colors
        def define_color(text):
            colors = ['black', 'white', 'red', 'blue', 'silver', 'gray', 'brown', 'yellow', 'green', 'orange']
            if pd.isnull(text):
                return 'other'
            text = text.lower()
            for color in colors:
                if color in text:
                    return color
            return 'other'
        
        input_data['int_col'] = input_data['int_col'].apply(define_color)
        input_data['ext_col'] = input_data['ext_col'].apply(define_color)

        # Encode categorical variables using the loaded OrdinalEncoder
        categorical_cols = ['brand', 'fuel_type', 'transmission', 'int_col', 'ext_col']
        input_data[categorical_cols] = ordinal_encoder.transform(input_data[categorical_cols])

        # Ensure all column names are strings
        input_data.columns = input_data.columns.astype(str)

        # Debugging: Print input data details
        print("\n--- Input Data ---")
        print(input_data)
        print("Columns:", input_data.columns.tolist())
        print("Data Types:\n", input_data.dtypes)

        # Check if the model has 'feature_names_in_'
        if hasattr(model, 'feature_names_in_'):
            print("\nModel's feature_names_in_:")
            print(model.feature_names_in_)
            expected_features = model.feature_names_in_
        else:
            print("Model does not have 'feature_names_in_' attribute.")
            return "An error occurred: Model lacks 'feature_names_in_' attribute."

        # Verify that all expected features are present in input_data
        missing_features = set(expected_features) - set(input_data.columns)
        extra_features = set(input_data.columns) - set(expected_features)
        if missing_features:
            print(f"Missing features: {missing_features}")
            return f"An error occurred: Missing features {missing_features}"
        if extra_features:
            print(f"Extra features: {extra_features}")
            # Optionally, you can drop extra features if necessary
            # input_data = input_data.drop(list(extra_features), axis=1)

        # Align input_data with model's expected features
        input_data = input_data[feature_names]

        # Debugging: Print aligned input data
        print("\n--- Aligned Input Data ---")
        print(input_data)

        # Scale the features
        input_scaled = scaler.transform(input_data)
        print("\nFeatures scaled successfully.")

        # Make prediction
        prediction_log_price = model.predict(input_scaled)
        predicted_price = np.expm1(prediction_log_price)[0]  # Inverse of np.log1p

        print(f"\nPredicted log_price: {prediction_log_price}")
        print(f"Predicted price: {predicted_price}")

        return f"The predicted price is ${predicted_price:,.2f}"

    except Exception as e:
        print("Error during prediction:", e)
        return f"An error occurred: {e}"

iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(choices=brand_choices, label="Brand"),
        gr.Number(label="Mileage"),
        gr.Dropdown(choices=fuel_type_choices, label="Fuel Type"),
        gr.Dropdown(choices=transmission_choices, label="Transmission"),
        gr.Dropdown(choices=color_choices, label="Exterior Color"),
        gr.Dropdown(choices=color_choices, label="Interior Color"),
        gr.Number(label="Age"),
        gr.Number(label="Horsepower"),
        gr.Number(label="Engine Size (L)"),
        gr.Number(label="Number of Cylinders"),
        gr.Radio(choices=[0, 1], label="Accident History (0 = No, 1 = Yes)"),
        gr.Radio(choices=[0, 1], label="Clean Title (0 = No, 1 = Yes)"),
    ],
    outputs=gr.Textbox(label="Predicted Price"),
    title="Used Car Price Predictor",
    description="Enter the details of the car to predict its price.",
)

if __name__ == "__main__":
    iface.launch()
