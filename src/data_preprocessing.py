# src/data_preprocessing.py

import numpy as np
import pandas as pd
import re
import datetime
from sklearn.preprocessing import OrdinalEncoder
import joblib
import os


def preprocess_data():
    # Load data
    df = pd.read_csv('data/used_cars2.csv')

    # Data Cleaning and Preprocessing Steps
    # 1. Clean 'milage' and 'price' columns
    df['milage'] = df['milage'].str.replace(',', '').str.replace(' mi.', '').astype(int)
    df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(int)

    # 2. Handle missing values
    df['accident'] = df['accident'].fillna('None reported')
    df['clean_title'] = df['clean_title'].fillna('No')
    df['fuel_type'] = df['fuel_type'].fillna('not supported')
    df['fuel_type'] = df['fuel_type'].replace('-', None)

    # 3. Infer fuel type
    def infer_fuel_type(row):
        if pd.isnull(row['fuel_type']):
            engine_info = str(row['engine']).lower()
            if 'electric' in engine_info:
                return 'Electric'
            elif 'hybrid' in engine_info or 'gas/electric' in engine_info:
                return 'Hybrid'
            elif 'diesel' in engine_info:
                return 'Diesel'
            elif 'flex fuel' in engine_info:
                return 'E85 Flex Fuel'
            else:
                return 'Gasoline'  # Default to 'Gasoline' if not determinable
        else:
            return row['fuel_type']
    df['fuel_type'] = df.apply(infer_fuel_type, axis=1)

    # 4. Calculate car age
    current_year = datetime.datetime.now().year
    df['Age'] = current_year - df['model_year']
    df.drop(columns='model_year', axis=1, inplace=True)

    # 5. Simplify transmission types
    df['transmission'] = df['transmission'].replace(['-', 'F'], None)
    def simplify_transmission(trans):
        if pd.isnull(trans):
            return 'Other'
        trans_lower = trans.lower()
        if ('automatic' in trans_lower) or ('a/t' in trans_lower) or ('auto' in trans_lower) or ('cvt' in trans_lower):
            return 'Automatic'
        elif ('manual' in trans_lower) or ('m/t' in trans_lower):
            return 'Manual'
        else:
            return 'Other'
    df['transmission'] = df['transmission'].apply(simplify_transmission)

    # 6. Extract engine information
    def extract_engine_info(engine):
        if pd.isnull(engine):
            return pd.Series([np.nan, np.nan, np.nan])
        else:
            hp_match = re.search(r'(\d+(\.\d+)?)HP', engine)
            horsepower = float(hp_match.group(1)) if hp_match else np.nan

            size_match = re.search(r'(\d+(\.\d+)?)L', engine)
            engine_size = float(size_match.group(1)) if size_match else np.nan

            cyl_match = re.search(r'([V|I|Flat|Straight]\s*\d+)', engine)
            if cyl_match:
                cyl_info = cyl_match.group(1)
                cyl_num_match = re.search(r'\d+', cyl_info)
                num_cylinders = int(cyl_num_match.group()) if cyl_num_match else np.nan
            else:
                num_cylinders = np.nan

            return pd.Series([horsepower, engine_size, num_cylinders])
    df[['horsepower', 'engine_size', 'num_cylinders']] = df['engine'].apply(extract_engine_info)
    df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())
    df['engine_size'] = df['engine_size'].fillna(df['engine_size'].median())
    df['num_cylinders'] = df['num_cylinders'].fillna(df['num_cylinders'].median())
    df = df.drop('engine', axis=1)

    # 7. Simplify colors
    def base_color(df):
        colors = [
            'black', 'white', 'red', 'blue', 'silver', 'gray', 'brown', 'yellow', 'green', 'orange'
        ]
        df['int_col'] = df['int_col'].str.lower()
        df['ext_col'] = df['ext_col'].str.lower()

        def define_color(text):
            if pd.isnull(text):
                return 'other'
            for color in colors:
                if color in text:
                    return color
            return 'other'

        df['int_col'] = df['int_col'].apply(define_color)
        df['ext_col'] = df['ext_col'].apply(define_color)
        return df
    df = base_color(df)

    # 8. Encode 'accident' and 'clean_title'
    df['accident_encoded'] = df['accident'].apply(lambda x: 1 if 'At least 1 accident' in x else 0)
    df['clean_title_encoded'] = df['clean_title'].apply(lambda x: 1 if x == 'Yes' else 0)
    df = df.drop(['accident', 'clean_title'], axis=1)

    # 9. Log transformation
    df['log_milage'] = np.log1p(df['milage'])
    df['log_price'] = np.log1p(df['price'])

    # 10. Drop unnecessary columns
    df = df.drop(['milage', 'price', 'model'], axis=1)

    # 11. Encode categorical variables using OrdinalEncoder
    categorical_cols = ['brand', 'fuel_type', 'transmission', 'int_col', 'ext_col']
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[categorical_cols] = ordinal_encoder.fit_transform(df[categorical_cols])

    df.columns = df.columns.astype(str)

    # Save the processed data and encoder
    df.to_csv('data/processed_data.csv', index=False)
    joblib.dump(ordinal_encoder, 'models/ordinal_encoder.pkl')

    print("Data preprocessing completed and saved.")

if __name__ == "__main__":
    preprocess_data()
