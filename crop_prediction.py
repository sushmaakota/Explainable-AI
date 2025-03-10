## Importing necessary libraries for the web app
import streamlit as st
import pickle
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('Crop_recommendation.csv')

# Label encoding the categorical columns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Store original column names
original_columns = df.select_dtypes(include='object').columns
label_encoders = {}
for col in original_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Splitting features and target
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Train the XGBoost model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Define the crop dictionary
crop_dict = {
    0: 'Apple', 1: 'Banana', 2: 'Blackgram', 3: 'Chickpea', 4: 'Coconut',
    5: 'Coffee', 6: 'Cotton', 7: 'Grapes', 8: 'Jute', 9: 'Kidneybeans',
    10: 'Lentil', 11: 'Maize', 12: 'Mango', 13: 'Mothbeans', 14: 'Moongbeans',
    15: 'Muskmelon', 16: 'Orange', 17: 'Papaya', 18: 'Pigeonpeas', 19: 'Pomegranate',
    20: 'Rice', 21: 'Watermelon'
}

# Prediction function
def predict_with_RF(N, P, K, temperature, humidity, ph, rainfall):
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = rf_model.predict(input_features)
    predicted_crop = crop_dict[prediction[0]]
    return predicted_crop

# LIME Explanation function
def explain_with_lime(input_features):
    explainer = LimeTabularExplainer(X_train.values, training_labels=y_train, mode="classification", 
                                     feature_names=X_train.columns, class_names=np.unique(y_train), discretize_continuous=True)
    exp = explainer.explain_instance(input_features[0], rf_model.predict_proba, num_features=5)
    return exp

## Streamlit code for the web app interface
def main():  
    # Setting the title of the web app
    st.markdown("<h1 style='text-align: center;'>SMART CROP RECOMMENDATIONS🌱</h1>", unsafe_allow_html=True)
    
    st.sidebar.title("CropPredict")
    st.sidebar.header("Find out the most suitable crop to grow in your farm 👨‍")
    
    # Input fields
    nitrogen = st.sidebar.number_input("Nitrogen", min_value=0, max_value=140, value=0, step=1)
    phosphorus = st.sidebar.number_input("Phosphorus", min_value=0, max_value=145, value=0, step=1)
    potassium = st.sidebar.number_input("Potassium", min_value=0, max_value=205, value=0, step=1)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
    inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    
    if st.sidebar.button("Predict"):
        if np.isnan(inputs).any() or (inputs == 0).all():
            st.error("Please fill in all input fields with valid values before predicting.")
        else:
            prediction = predict_with_RF(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            st.success(f"The recommended crop is: {prediction}")
            
            # Generate LIME explanation
            st.subheader("LIME Explanation")
            input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            exp = explain_with_lime(input_features)
            exp_html = exp.as_html()
            st.components.v1.html(exp_html, height=600)
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)

## Running the main function
if __name__ == '__main__':
    main()
