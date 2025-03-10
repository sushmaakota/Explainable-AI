## Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
yield_df = pd.read_csv("Yield Forecasting Data.csv")

# Label encoding the data
yield_original_columns = yield_df.select_dtypes(include='object').columns

yield_label_encoders = {}
for col in yield_original_columns:
    yield_label_encoders[col] = LabelEncoder()
    yield_df[col] = yield_label_encoders[col].fit_transform(yield_df[col])

# Selecting relevant columns
yield_col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
yield_df = yield_df[yield_col]

# Splitting into features and target
yield_X = yield_df.drop('hg/ha_yield', axis=1)
yield_y = yield_df['hg/ha_yield']

# Scaling the features
yield_scaler = StandardScaler()
yield_X_scaled = yield_scaler.fit_transform(yield_X)

# Split the data into train and test sets
yield_X_train, yield_X_test, yield_y_train, yield_y_test = train_test_split(yield_X_scaled, yield_y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor model
yield_random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
yield_random_forest_model.fit(yield_X_train, yield_y_train)

# Define feature names
feature_names = ["Year", "Rainfall", "Pesticides", "Temp", "Area", "Item"]

# Define mappings
area_mapping = {i: country for i, country in enumerate([
    "Albania", "Algeria", "Angola", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
    "Bangladesh", "Belarus", "Belgium", "Botswana", "Brazil", "Bulgaria", "Burkina Faso", "Burundi", "Cameroon", "Canada",
    "Central African Republic", "Chile", "Colombia", "Croatia", "Denmark", "Dominican Republic", "Ecuador", "Egypt", "El Salvador",
    "Eritrea", "Estonia", "Finland", "France", "Germany", "Ghana", "Greece", "Guatemala", "Guinea", "Guyana", "Haiti",
    "Honduras", "Hungary", "India", "Indonesia", "Iraq", "Ireland", "Italy", "Jamaica", "Japan", "Kazakhstan", "Kenya",
    "Latvia", "Lebanon", "Lesotho", "Libya", "Lithuania", "Madagascar", "Malawi", "Malaysia", "Mali", "Mauritania", "Mauritius",
    "Mexico", "Montenegro", "Morocco", "Mozambique", "Namibia", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger",
    "Norway", "Pakistan", "Papua New Guinea", "Peru", "Poland", "Portugal", "Qatar", "Romania", "Rwanda", "Saudi Arabia",
    "Senegal", "Slovenia", "South Africa", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Tajikistan",
    "Thailand", "Tunisia", "Turkey", "Uganda", "Ukraine", "United Kingdom", "Uruguay", "Zambia", "Zimbabwe"
])}

item_mapping = {i: crop for i, crop in enumerate([
    "Cassava", "Maize", "Plantains and others", "Potatoes", "Rice, paddy", "Sorghum", "Soybeans", "Sweet potatoes", "Wheat", "Yams"
])}

def predict_yield_and_explain(year, rainfall, pesticides, temp, area, item): 
    yield_input_data = np.array([[year, rainfall, pesticides, temp, area, item]])
    yield_inputs_scaled = yield_scaler.transform(yield_input_data)
    yield_predicted = yield_random_forest_model.predict(yield_inputs_scaled)[0]
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=yield_scaler.transform(yield_X_train),
        feature_names=feature_names,
        mode="regression"
    )
    
    exp = explainer.explain_instance(
        yield_inputs_scaled[0], yield_random_forest_model.predict, num_features=len(feature_names)
    )
    return yield_predicted, exp

# Streamlit interface
def main():
    st.title("Yield Forecasting with LIME 📊")
    st.sidebar.header("Enter Yield Forecasting Parameters")
    
    year = st.sidebar.number_input("Year", min_value=1990, max_value=2050, value=1990, step=1)
    rainfall = st.sidebar.number_input("Average Rainfall (mm)", min_value=0.0, max_value=3000.0, value=0.0, step=10.0)
    pesticides = st.sidebar.number_input("Pesticides Used (tonnes)", min_value=0.0, max_value=5000.0, value=0.0, step=10.0)
    temp = st.sidebar.number_input("Average Temperature (°C)", min_value=-0.0, max_value=50.0, value=0.0, step=0.1)
    
    selected_area = st.sidebar.selectbox("Select Area", list(area_mapping.values()))
    selected_item = st.sidebar.selectbox("Select Item", list(item_mapping.values()))
    
    area = list(area_mapping.keys())[list(area_mapping.values()).index(selected_area)]
    item = list(item_mapping.keys())[list(item_mapping.values()).index(selected_item)]
    
    if st.sidebar.button("Predict Yield"):
        yield_prediction, explanation = predict_yield_and_explain(year, rainfall, pesticides, temp, area, item)
        st.success(f"Predicted Yield: {yield_prediction:.2f}")
        
        st.subheader("LIME Explanation")
        explanation_html = explanation.as_html()
        st.components.v1.html(explanation_html, height=600)
        fig = explanation.as_pyplot_figure()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
