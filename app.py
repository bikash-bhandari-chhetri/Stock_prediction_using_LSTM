import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the saved LSTM model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('lstm_stock_price_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Load the saved scaler
@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load('scaler.pkl')
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}")

# Load model and scaler
model = load_model()
scaler = load_scaler()
st.title("Stock Price Prediction")

st.write("""
         Enter the stock data below to predict the next day's closing price.
         """)

# Input fields for user to enter data
day = st.number_input('Day', min_value=1, max_value=31, step=1)
month = st.number_input('Month', min_value=1, max_value=12, step=1)
year = st.number_input('Year', min_value=2000, max_value=2100, step=1)
open_value = st.number_input('Open Value', min_value=0.0, format="%.2f")
high_value = st.number_input('High Value', min_value=0.0, format="%.2f")
low_value = st.number_input('Low Value', min_value=0.0, format="%.2f")
last_value = st.number_input('Last Value (Closing Price)', min_value=0.0, format="%.2f")
change_prev_close_percentage = st.number_input('Change Previous Close Percentage', min_value=-100.0, format="%.2f")
turnover = st.number_input('Turnover', min_value=0.0, format="%.2f")

def predict_next_day(day, month, year, open_value, high_value, low_value, last_value, change_prev_close_percentage, turnover):
    try:
        # Combine input data into a DataFrame
        input_data = pd.DataFrame([[open_value, high_value, low_value, last_value, change_prev_close_percentage, turnover, year, month, day]], 
                                  columns=['open_value', 'high_value', 'low_value', 'last_value', 'change_prev_close_percentage', 'turnover', 'year', 'month', 'day'])

        # Scale the input data
        scaled_input = scaler.transform(input_data)

        # Reshape input data to fit model input shape (batch_size, time_steps, features)
        scaled_input_reshaped = scaled_input.reshape((1, scaled_input.shape[0], scaled_input.shape[1]))

        # Predict using the LSTM model
        prediction = model.predict(scaled_input_reshaped)

        # Inverse transform to get the actual predicted value
        predicted_value = scaler.inverse_transform(np.concatenate((prediction, np.zeros((prediction.shape[0], len(input_data.columns) - 1))), axis=1))[:, 0]

        return predicted_value[0]

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Button to trigger prediction
if st.button('Predict Next Day Price'):
    predicted_price = predict_next_day(day, month, year, open_value, high_value, low_value, last_value, change_prev_close_percentage, turnover)

    if predicted_price is not None:
        st.success(f"Predicted Closing Price for the Next Day: {predicted_price:.2f}")
    else:
        st.error("Prediction failed. Please check your input values.")

st.write("Note: This prediction is based on historical data and may not be accurate. Use at your own risk.")