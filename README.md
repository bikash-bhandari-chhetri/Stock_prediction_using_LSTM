# Stock_prediction_using_LSTM
This project aims to predict the next day's closing price of a stock using historical stock data. The prediction model is built using a Long Short-Term Memory (LSTM) network, a type of recurrent neural network (RNN) particularly effective for time-series forecasting tasks.

## Table of Contents
- [Overvuew](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Cleaning](#data-cleaning)
- [Data Preprocessing](#data-preprocessing)
- [Data Visualization](#data-visualization)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Streamlit Application](#streamlit-application)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)

## Overview
The stock price prediction project is designed to forecast the next day's closing price based on historical data using machine learning techniques, specifically LSTM (Long Short-Term Memory) networks. The workflow of the project is divided into five main parts: data cleaning, data preprocessing, data visualization, model building, and model evaluation. Additionally, an interactive Streamlit app is provided for users to make predictions using their input data.

## Features
* **Data Cleaning:** Removal of missing values, duplicate rows, and handling of outliers.
* **Data Preprocessing:** Feature engineering, scaling, and transformation to prepare the data for training.
* **Data Visualization:** Exploratory data analysis using various plots to understand the data better.
* **LSTM Model for Time-Series Forecasting**: A neural network model designed to learn from sequential data and predict future stock prices.
* **Model Evaluation:** Assessing the model's performance using various metrics and visualizations.
* **Interactive Streamlit App:** Allows users to input stock data and receive real-time predictions.

## Installation
1. **Clone the repository:** 
  https://github.com/bikash-bhandari-chhetri/Stock_prediction_using_LSTM.git
  cd Stock_prediction_using_LSTM
2. **Install the required dependencies:**
  pip install -r requirements.txt
3. **Ensure you have the following files in your project directory:**
  * Stock_Prediction_using_LSTM(Data_Cleaning).ipynb: Jupyter notebook for data cleaning
  * Stock_Prediction_using_LSTM(Data_Preprocessing).ipynb: Jupyter notebook for data preprocessing
  * Stock_Prediction_using_LSTM(Data_Visualization).ipynb: Jupyter notebook for data visualization
  * Stock_Prediction_using_LSTM(Model_Building_and_Training)ipynb.ipynb: Jupyter notebook for model building     and training
  * Stock_Prediction_using_LSTM(Model_Evaluation_and_Prediction).ipynb: Jupyter notebook for model           
    evaluation and prediction
  * scaler.pkl: Scaler object for data normalization
  * lstm_stock_price_model.keras: Trained LSTM model
  * app.py: Streamlit app script
  * stock_data.csv: Original dataset
  * cleaned_stock_data.csv: Cleaned dataset
  * preprocessed_stock_data.csv: Preprocessed dataset
  * requirements.txt: txt file containing all the required dependencies
