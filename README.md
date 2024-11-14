# Used Car Predictor

## 🖥️ Live Demo
Try out the **Used Car Price Predictor** app live [here](https://huggingface.co/spaces/h594814/car-price-predictor).

Business Objective
The primary goal of the Used Car Price Predictor project is to develop a machine learning-based tool that accurately estimates the resale value of used cars based on various features. This solution aims to streamline the car valuation process for both sellers and buyers, enhancing decision-making and ensuring fair pricing in the automotive market.

Usage and Existing Solutions
Currently, car pricing relies heavily on manual appraisals by experts or generic online valuation tools that may not account for specific vehicle nuances. Without machine learning, estimating a car's price involves subjective judgments based on limited data points. The Used Car Price Predictor leverages historical sales data and advanced algorithms to provide precise and data-driven price estimates, outperforming traditional methods in accuracy and efficiency.

Performance Measurement via Business Metrics
The performance of the predictor is measured using the Mean Absolute Error (MAE) between the predicted prices and actual sale prices. A lower MAE indicates higher accuracy, directly translating to increased trust and usability of the tool for stakeholders.

System Components and Integration
The machine learning model serves as the core component, integrated into a Gradio-based web interface for user interaction. Additional components include data preprocessing scripts, a model training pipeline, and deployment infrastructure on Hugging Face Spaces. Changes in the data preprocessing stage, such as feature engineering, can impact model performance, necessitating synchronized updates across the pipeline.

Stakeholders
Sellers: Individuals looking to sell their used cars at fair market prices.
Buyers: Potential car buyers seeking accurate price estimations to make informed purchasing decisions.

Tentative Timeline and Milestones
Given the one-week timeframe, the project was structured with daily milestones to ensure timely completion:

Day 1:
Data Collection and Exploration: Gathered and examined the used car sales dataset to understand feature distributions and identify potential issues.
Day 2:
Data Preprocessing and Feature Engineering: Cleaned the data, handled missing values, and engineered relevant features to enhance model performance.
Day 3:
Model Selection and Training: Experimented with various machine learning models, starting with simple baselines and progressing to more complex algorithms like XGBoost.
Day 4:
Model Evaluation and Optimization: Assessed model performance using appropriate metrics and fine-tuned hyperparameters to improve accuracy.
Day 5:
Deployment Setup: Configured the Gradio interface and prepared the deployment environment on Hugging Face Spaces.
Day 6:
Testing and Validation: Deployed the model, conducted thorough testing to ensure functionality, and gathered initial user feedback.
Day 7:
Documentation and Final Reporting: Compiled project documentation, updated the README with deployment links, and finalized the assignment report.

Required Resources
Computational Resources: Access to a personal computer with sufficient processing power for model training and deployment.
Data: Comprehensive dataset of used car sales, including features like brand, mileage, fuel type, transmission, color, age, horsepower, engine size, number of cylinders, accident history, and clean title status.
Tools: Python, Jupyter Notebooks, Scikit-learn, XGBoost, Gradio, GitHub, Hugging Face Spaces.

METRICS
Business Metric Performance
The project is considered successful if the Used Car Price Predictor achieves a Mean Absolute Error (MAE) of less than $2,500. This threshold ensures that the model's price predictions are sufficiently accurate to be practically useful for dealerships and individual sellers.
Current Performance: $8,753.89
R-squared (R²):0.86
The model explains 86% of the variance in used car prices, demonstrating strong explanatory power. However, 14% of the variance remains unexplained, indicating potential areas for model enhancement.

DATA
Data and Labels
The dataset comprises historical used car sales data, including features such as:

Categorical Features: Brand, Fuel Type, Transmission, Exterior Color, Interior Color, Accident History, Clean Title.
Numerical Features: Mileage, Age, Horsepower, Engine Size (L), Number of Cylinders.
Labels: The target variable is the sale price of the used cars.

Data Collection and Availability
Data was sourced from an existing used car sales database, containing approximately 4000 records. For supervised learning, the ground truth labels are the actual sale prices. To ensure label accuracy:

Data Cleaning: Removed outliers and inconsistent entries.

Data Representation and Preprocessing
Data Cleaning: Handled missing values and corrected erroneous data entries.
Feature Engineering: Created new features like log-transformed mileage to stabilize variance.
Encoding: Applied Ordinal Encoding for categorical variables.
Scaling: Standardized numerical features using StandardScaler to ensure uniformity across features.

MODELING
Machine Learning Models Explored
Linear Regression: As a baseline to establish initial performance metrics.
Decision Trees: To capture non-linear relationships in the data.
Random Forests: For improved accuracy and robustness over single decision trees.
XGBoost Regressor: Selected for its superior performance and efficiency in handling large datasets.

DEPLOYMENT
Deployment Strategy
The final model was deployed using Gradio on Hugging Face Spaces, providing a user-friendly web interface for real-time predictions. Link to live demo at the top.

Usage of Predictions
Users input specific car features through the Gradio interface, and the model returns an estimated sale price. This facilitates informed decision-making for both buyers and sellers in the used car market.

Monitoring and Maintenance
Monitoring: Utilized Hugging Face's built-in logging to track prediction requests and model performance.
Maintenance: Regularly update the model with new sales data to maintain accuracy.

REFERENCES
Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html
XGBoost Documentation: https://xgboost.readthedocs.io/en/latest/
Gradio Documentation: https://gradio.app/get_started
Hugging Face Spaces: https://huggingface.co/spaces
Used Car Sales Data: https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset/data


