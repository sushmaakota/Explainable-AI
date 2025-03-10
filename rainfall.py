## Import necessary libraries
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd

# Load the saved ANN model and scaler
rain_model = tf.keras.models.load_model('ann_model.h5')
rain_scaler = joblib.load('scaler.pkl')

# Load the dataset to get feature names
rain_data = pd.read_csv('train_data____1.csv')
rain_feature_columns = rain_data.columns[:-1]  # All columns except the target
rain_X = rain_data[rain_feature_columns].values
rain_y = rain_data['target'].values

# Define the location mapping
location_mapping = {
    "Adelaide": 0, "Albany": 1, "Albury": 2, "AliceSprings": 3, "BadgerysCreek": 4, "Ballarat": 5, "Bendigo": 6,
    "Brisbane": 7, "Cairns": 8, "Canberra": 9, "Cobar": 10, "CoffsHarbour": 11, "Dartmoor": 12, "Darwin": 13,
    "GoldCoast": 14, "Hobart": 15, "Katherine": 16, "Launceston": 17, "Melbourne": 18, "MelbourneAirport": 19,
    "Mildura": 20, "Moree": 21, "MountGambier": 22, "MountGinini": 23, "Newcastle": 24, "Nhil": 25, "NorahHead": 26,
    "NorfolkIsland": 27, "Nuriootpa": 28, "PearceRAAF": 29, "Penrith": 30, "Perth": 31, "PerthAirport": 32,
    "Portland": 33, "Richmond": 34, "Sale": 35, "SalmonGums": 36, "Sydney": 37, "SydneyAirport": 38, "Townsville": 39,
    "Tuggeranong": 40, "Uluru": 41, "WaggaWagga": 42, "Walpole": 43, "Watsonia": 44, "Williamtown": 45,
    "Witchcliffe": 46, "Wollongong": 47, "Woomera": 48
}

# Initialize LIME Explainer
rain_explainer = LimeTabularExplainer(
    rain_X,
    training_labels=rain_y,
    feature_names=rain_feature_columns,
    class_names=['No Rain', 'Rain'],
    mode='classification'
)

# Function for prediction
def predict_rain_ann(user_input):
    rain_input_scaled = rain_scaler.transform(np.array(user_input).reshape(1, -1))
    rain_prediction_prob = rain_model.predict(rain_input_scaled).ravel()[0]
    rain_prediction_label = "Rain (1)" if rain_prediction_prob > 0.5 else "No Rain (0)"
    
    if rain_prediction_label == "Rain (1)":
        rain_suggestion = ("Rain is expected tomorrow. It's advisable to carry an umbrella, avoid outdoor activities if possible, "
                      "and ensure your home is protected from potential water accumulation. Plan ahead for any travel to avoid disruptions.")
    else:
        rain_suggestion = ("No rain is forecasted tomorrow. It’s a good day to engage in outdoor activities. However, stay prepared "
                      "for any sudden weather changes and keep yourself updated with the latest weather reports.")
    
    return rain_prediction_label, rain_prediction_prob, rain_suggestion

# Function to generate LIME explanation
def explain_rain_prediction(user_input):
    rain_user_input_scaled = rain_scaler.transform(np.array(user_input).reshape(1, -1))
    
    def predict_proba_for_lime(input_data):
        rain_probabilities = rain_model.predict(input_data)
        return np.hstack((1 - rain_probabilities, rain_probabilities))
    
    rain_explanation = rain_explainer.explain_instance(
        rain_user_input_scaled[0],
        predict_proba_for_lime,
        num_features=5
    )
    return rain_explanation

# Streamlit interface
def main():
    st.title("Rainfall Prediction with ANN 🌦️")
    st.sidebar.header("Enter Weather Parameters")
    
    user_input = []
    
    # Dropdown for Location Selection
    selected_location = st.sidebar.selectbox("Select Location", list(location_mapping.keys()))
    user_input.append(location_mapping[selected_location])
    
    # Other Input Fields
    input_fields = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed", 
                    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", 
                    "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RISK_MM"]
    
    for field in input_fields:
        value = st.sidebar.number_input(f"{field}", value=0.0, step=0.1)
        user_input.append(value)
    
    if st.sidebar.button("Predict"):
        prediction_label, prediction_prob, suggestion = predict_rain_ann(user_input)
        st.success(f"Prediction: {prediction_label} ({prediction_prob:.2f} probability)")
        st.info(suggestion)
        
        # Generate and display LIME explanation
        st.subheader("LIME Explanation")
        explanation = explain_rain_prediction(user_input)
        explanation_html = explanation.as_html()
        st.components.v1.html(explanation_html, height=600)
        fig = explanation.as_pyplot_figure()
        st.pyplot(fig)

if __name__ == "__main__":
    main()