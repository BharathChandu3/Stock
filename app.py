from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from textblob import TextBlob
import math
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
import os
import time

app = Flask(__name__)
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/team')
def team():
    return render_template('team.html')

def calculate_intervals(predictions, error_std):
    lower_bound = predictions - 1.96 * error_std
    upper_bound = predictions + 1.96 * error_std
    return lower_bound, upper_bound

def get_recommendation(latest_price, predictions):
    avg_prediction = np.mean(predictions)
    if avg_prediction > latest_price * 1.02:  # If the average prediction is more than 2% higher than the latest price
        return "Buy"
    elif avg_prediction < latest_price * 0.98:  # If the average prediction is more than 2% lower than the latest price
        return "Sell"
    else:
        return "Hold"

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def fetch_news(ticker):
    stock = yf.Ticker(ticker)
    news = stock.news
    return news

def analyze_news(news):
    sentiments = []
    for article in news:
        if 'summary' in article:
            sentiment = analyze_sentiment(article['summary'])
            sentiments.append(sentiment)
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    if avg_sentiment > 0.1:
        return "Positive"
    elif avg_sentiment < -0.1:
        return "Negative"
    else:
        return "Neutral"

def save_graph(actual, predicted, ticker):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='Actual Prices', color='red')
    plt.plot(predicted, label='Predicted Prices', color='blue')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    image_path = f'static/{ticker}_prediction.png'
    plt.savefig(image_path)
    plt.close()
    return image_path

# ARIMA model training
def arima_algo(df, prediction_window, forecast_horizon):
    data = df.reset_index()
    data['Price'] = data['Close']
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
    data = data.fillna(data.bfill())

    quantity = data['Price'].values
    size = int(len(quantity) * 0.80)
    train, test = quantity[0:size], quantity[size:len(quantity)]
    history = [x for x in train]
    predictions = []
    intervals = []

    for t in range(min(len(test), forecast_horizon)):
        model = ARIMA(history, order=(6, 1, 0))
        model_fit = model.fit()
        output = model_fit.get_forecast(steps=prediction_window)
        yhat = output.predicted_mean[0]
        conf_int = output.conf_int()[0]
        predictions.append(yhat)
        intervals.append(conf_int)
        history.append(test[t])

    latest_price = data['Price'].iloc[-1] if not data.empty else None
    return latest_price, predictions, intervals, test[:forecast_horizon]

# LSTM+GRU model training
def lstm_gru_algo(df, prediction_window):
    training_set = df['Close'].values.reshape(-1, 1)
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    X_train, y_train = [], []

    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-7:i, 0])
        y_train.append(training_set_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    regressor = Sequential()
    regressor.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.1))
    regressor.add(GRU(50))
    regressor.add(Dropout(0.1))
    regressor.add(Dense(1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, epochs=25, batch_size=32)

    predictions = []
    X_forecast = X_train[-1:]
    for _ in range(prediction_window):
        forecasted_stock_price = regressor.predict(X_forecast)
        predictions.append(sc.inverse_transform(forecasted_stock_price)[0, 0])
        forecasted_stock_price = np.reshape(forecasted_stock_price, (1, 1, 1))
        X_forecast = np.append(X_forecast[:, 1:, :], forecasted_stock_price, axis=1)

    return predictions, training_set[-prediction_window:].flatten()

# Linear Regression model training
def linear_regression_algo(df, prediction_window):
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['Days']]
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    last_day = df['Days'].iloc[-1]
    future_days = np.arange(last_day + 1, last_day + 1 + prediction_window).reshape(-1, 1)
    predictions = model.predict(future_days)
    return predictions, y[-prediction_window:].values

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_post():
    ticker = request.form['company']
    period = int(request.form['period'])
    model_type = request.form['model']

    retries = 3
    for attempt in range(retries):
        df = yf.download(ticker, period='2y')
        if not df.empty:
            break
        time.sleep(1)
    else:
        return jsonify({'error': 'Failed to download stock data. Please try again later.'}), 500

    if model_type == 'ARIMA':
        latest_price, predictions, intervals, actual_values = arima_algo(df, period, period)
    elif model_type == 'LSTM+GRU':
        predictions, actual_values = lstm_gru_algo(df, period)
        latest_price = df['Close'].iloc[-1] if not df.empty else None
        predictions = [float(p) for p in predictions]  # Convert float32 to float
    elif model_type == 'Linear Regression':
        predictions, actual_values = linear_regression_algo(df, period)
        latest_price = df['Close'].iloc[-1] if not df.empty else None
        predictions = [float(p) for p in predictions]  # Convert float32 to float
    
    rmse = math.sqrt(mean_squared_error(actual_values, predictions))
    mae = mean_absolute_error(actual_values, predictions)
    
    news = fetch_news(ticker)
    news_sentiment = analyze_news(news)
    news_data = [{
        'title': article.get('title', 'No Title'),
        'summary': article.get('summary', 'No Summary'),
        'link': article.get('link', '#')
    } for article in news]

    graph_path = save_graph(actual_values, predictions, ticker)
    
    return jsonify({
        'todays_price': float(latest_price) if latest_price is not None else None,
        'predictions': predictions,
        'actual_values': actual_values.tolist(),
        'rmse': rmse,
        'mae': mae,
        'news': news_data,
        'news_sentiment': news_sentiment,
        'graph_path': graph_path
    })

@app.route('/recommendation', methods=['POST'])
def recommendation():
    latest_price = float(request.form['latest_price'])
    predictions = request.form.getlist('predictions')
    predictions = [float(p) for p in predictions]
    recommendation = get_recommendation(latest_price, predictions)
    return jsonify({'recommendation': recommendation})

@app.route('/graph/<path:filename>')
def graph(filename):
    return send_file(os.path.join('static', filename))

if __name__ == '__main__':
    app.run(debug=True)