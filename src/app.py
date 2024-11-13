# src/app.py

import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load the saved model and preprocessing objects
model = joblib.load('models/car_price_predictor_model.pkl')
scaler = joblib.load('models/scaler.pkl')
ordinal_encoder = joblib.load('models/ordinal_encoder.pkl')

# Define input choices
df_raw = pd.read_csv('data/used_cars2.csv')  # Load raw data to extract unique values
brand_choices = df_raw['brand'].unique().tolist()
fuel_type_choices = df_raw['fuel_type'].unique().tolist()
transmission_choices = df_raw['transmission'].unique().tolist()
color_choices = ['black', 'white', 'red', 'blue', 'silver', 'gray', 'brown', 'yellow', 'green', 'orange', 'other']

def predict_price(brand, milage, fuel_type, transmission, ext_col, int_col,
                  age, horsepower, engine_size, num_cylinders,
                  accident_history, clean_title):
    try:
        # Create a DataFrame from the input
        input_data = pd.DataFrame({
            'brand': [brand],
            'milage': [milage],
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
        })

        # Simplify colors (if necessary)
        def define_color(text):
            colors = [
                'black', 'white', 'red', 'blue', 'silver', 'gray', 'brown', 'yellow', 'green', 'orange'
            ]
            text = text.lower()
            for color in colors:
                if color in text:
                    return color
            return 'other'
        input_data['int_col'] = input_data['int_col'].apply(define_color)
        input_data['ext_col'] = input_data['ext_col'].apply(define_color)

        # Preprocess input data
        input_data['log_milage'] = np.log1p(input_data['milage'])
        input_data = input_data.drop('milage', axis=1)

        # Encode categorical variables using the loaded OrdinalEncoder
        categorical_cols = ['brand', 'fuel_type', 'transmission', 'int_col', 'ext_col']
        input_data[categorical_cols] = ordinal_encoder.transform(input_data[categorical_cols])

        # Ensure the order of columns matches training data
        input_data = input_data[model.feature_names_in_]

        # Scale the features
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction_log_price = model.predict(input_scaled)
        predicted_price = np.expm1(prediction_log_price)[0]  # Inverse of np.log1p

        return f"The predicted price is ${predicted_price:,.2f}"

    except Exception as e:
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
