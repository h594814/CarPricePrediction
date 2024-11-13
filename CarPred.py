# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


#pip install --upgrade gradio
df = pd.read_csv('Data/used_cars2.csv')
df.head()
df.info()
df.describe(include='all')
#**Data Preprocessing**
# Removing commas and ' mi.' from mileage column, also converting to int
df['milage'] = df['milage'].str.replace(',', '').str.replace(' mi.', '').astype(int)
# Cleaning price column
df['price'] = df['price'].str.replace('$', '').str.replace(',','').astype(int)
# Checking for missing values
df.isnull().sum()
#Filling missing values. Assuming for accidents 'No' and 'No' for clean titles
df['accident'] = df['accident'].fillna('None reported')
df['clean_title'] = df['clean_title'].fillna('No')
df['fuel_type'] = df['fuel_type'].fillna('not supported')
df['fuel_type'] = df['fuel_type'].replace('-', None)
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

df['fuel_type'].isnull().sum()
df['clean_title'].value_counts()
df['fuel_type'].value_counts()
df['accident'].value_counts()
df.head()
import datetime
cur_data = datetime.datetime.now()
df['Age'] = cur_data.year - df['model_year']
df['Age'].value_counts()
df.drop(columns='model_year', axis=1, inplace=True)
df['transmission'].value_counts()
df['transmission'].unique()
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
df['transmission'].value_counts()
df['engine'].value_counts()

# Extract engine information
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

print("Columns in the DataFrame:")
print(df.columns.tolist())
df.head()
df['horsepower'].isna().sum()
df['ext_col'].value_counts()
df['int_col'].value_counts()
#Defining 10 base colors
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

      
df.head()
df['int_col'].value_counts()
df['ext_col'].value_counts()
df.info()
from sklearn.preprocessing import LabelEncoder

# List of categorical columns to encode
categorical_cols = ['brand', 'model', 'fuel_type', 'transmission', 'int_col', 'ext_col']

# Initialize the LabelEncoder
le = LabelEncoder()

# Apply Label Encoding to each categorical column
for col in categorical_cols:
    df[col] = df[col].astype(str)  # Ensure all data is string type
    df[col] = le.fit_transform(df[col])
print(df.info())

X = df.drop('price', axis=1)
y = df['price']
def encode_accident(value):
    if 'At least 1 accident' in value:
        return 1
    else:
        return 0

df['accident_encoded'] = df['accident'].apply(encode_accident)
def encode_clean_title(value):
    if value == 'Yes':
        return 1
    else:
        return 0

df['clean_title_encoded'] = df['clean_title'].apply(encode_clean_title)
df = df.drop(['accident', 'clean_title'], axis=1)
df.info()

X = df.drop('price', axis=1)
y = df['price']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor

# Initialize the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


importances = rf.feature_importances_


feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': importances})


feature_importances = feature_importances.sort_values(by='importance', ascending=False)


print(feature_importances)


plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importances')
plt.show()
df.isnull().sum()

df = df.drop(['model'], axis=1)
df.columns.tolist()

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=5,
    scoring='neg_mean_squared_error'
)
print("Cross-Validation MSE:", -cv_scores)
print("Average CV MSE:", -cv_scores.mean())
import numpy as np


line_start = min(y_test.min(), y_pred.min())
line_end = max(y_test.max(), y_pred.max())
diagonal = np.linspace(line_start, line_end, 100)

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(diagonal, diagonal, color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.legend()
plt.show()
print("Actual Prices (y_test):")
print(y_test.describe())

print("\nPredicted Prices (y_pred):")
print(pd.Series(y_pred).describe())

max_price = 100000 

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)

plt.plot([0, max_price], [0, max_price], 'r--')


plt.xlim(0, max_price)
plt.ylim(0, max_price)

plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices (Limited Range)')
plt.show()
import numpy as np

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)

# Plot diagonal line
plt.plot([min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())],
         [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())], 'r--')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Actual Prices (log scale)')
plt.ylabel('Predicted Prices (log scale)')
plt.title('Actual vs. Predicted Prices (Log Scale)')
plt.show()
print(df.columns)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df['log_milage'] = np.log1p(df['milage'])
df['log_price'] = np.log1p(df['price'])
df['Age'] = df['Age'] 

df_plot = df[['log_milage', 'log_price', 'Age']]

sns.pairplot(df_plot)
plt.show()

df['price'] = np.log10(df['price'])
sns.histplot(df['price'])
plt.show()
df.head()
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
encoder = ce.BinaryEncoder(cols=['brand'])
df_binary_encoded = encoder.fit_transform(df)
le = LabelEncoder()
categorical_columns = ['fuel_type', 'ext_col', 'int_col']

for col in categorical_columns:
    df_binary_encoded[col] = le.fit_transform(df_binary_encoded[col])
df_binary_encoded.head()
df['log_milage'] = np.log1p(df['milage'])
df['log_price'] = np.log1p(df['price'])

import category_encoders as ce
from sklearn.preprocessing import LabelEncoder

# Apply BinaryEncoder to 'brand'
encoder = ce.BinaryEncoder(cols=['brand'])
df_binary_encoded = encoder.fit_transform(df)

# Apply LabelEncoder to specified categorical columns
le = LabelEncoder()
categorical_columns = ['fuel_type', 'ext_col', 'int_col']

for col in categorical_columns:
    df_binary_encoded[col] = le.fit_transform(df_binary_encoded[col])

x = df_binary_encoded.drop(['price', 'log_price'], axis=1)
y = df_binary_encoded['price']
x.shape,y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=0.1,random_state=42)
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
x_scaled = scale.fit_transform(x_train)
x_valid_scaled = scale.transform(x_valid)

print("Columns in x_train:")
print(x_train.columns.tolist())
print("Number of columns in x_train:", x_train.shape[1])

print("'log_price' in x_train columns:", 'log_price' in x_train.columns)

columns = [
    'brand_0', 'brand_1', 'brand_2', 'brand_3', 'brand_4', 'brand_5', 
    'milage', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'Age',
    'Horsepower', 'engine_size', 'num_cylinders', 'accident_encoded', 'clean_title_encoded', 'log_milage'
]

x_scaled_df = pd.DataFrame(data=x_scaled, columns=columns)

x_scaled_df.head()

print(x_train.columns.tolist())

print("x_scaled shape:", x_scaled.shape)
print("x_valid_scaled shape:", x_valid_scaled.shape)
print("Number of columns in 'columns' list:", len(columns))

x_scaled_df = pd.DataFrame(data=x_scaled, columns=columns)
x_valid_scaled_df = pd.DataFrame(data=x_valid_scaled, columns=columns)

from xgboost import XGBRegressor
model1 = XGBRegressor()
model1.fit(x_scaled, y_train)
y_pred = model1.predict(x_valid_scaled)

from sklearn.metrics import mean_squared_error, r2_score
r2_score(y_valid, y_pred)

import pandas as pd
import matplotlib.pyplot as plt

feature_importances = pd.DataFrame({
    'feature': x_train.columns,
    'importance': model1.feature_importances_
})

feature_importances = feature_importances.sort_values(by='importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feature_importances['feature'], feature_importances['importance'])
plt.title('Feature Importances from XGBoost Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

import matplotlib.pyplot as plt

# Predict on validation set
y_pred = model1.predict(x_valid_scaled_df)

# Plot actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_valid, y_pred, alpha=0.5)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()

import seaborn as sns

# Calculate residuals
residuals = y_valid - y_pred

# Histogram of residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


#Deployment
import joblib

joblib.dump(model1, 'car_price_predictor_model.pkl')

joblib.dump(scale, 'scaler.pkl')

joblib.dump(encoder, 'binary_encoder.pkl')
joblib.dump(le, 'label_encoder.pkl')


brand_choices = df['brand'].unique().tolist()
fuel_type_choices = df['fuel_type'].unique().tolist()
transmission_choices = df['transmission'].unique().tolist()
color_choices = ['black', 'white', 'red', 'blue', 'silver', 'gray', 'brown', 'yellow', 'green', 'orange', 'other']

import gradio as gr
import joblib
import numpy as np
import pandas as pd


model = joblib.load('car_price_predictor_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('binary_encoder.pkl')
le = joblib.load('label_encoder.pkl')

def predict_price(brand, milage, fuel_type, transmission, ext_col, int_col, age, horsepower, engine_size, num_cylinders, accident_history, clean_title):
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
        'accident_encoded': [accident_history],
        'clean_title_encoded': [clean_title],
    })

   
    input_data = encoder.transform(input_data)

    for col in ['fuel_type', 'ext_col', 'int_col']:
        input_data[col] = le.transform(input_data[col])

    input_data['log_milage'] = np.log1p(input_data['milage'])

    input_data = input_data.drop('milage', axis=1)

    input_data = input_data[model.feature_names_in_]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

 
    predicted_price = prediction[0]

    return f"The predicted price is ${predicted_price:,.2f}"

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

# Launch the interface
iface.launch()

