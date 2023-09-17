import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
import yfinance as yf
import numpy as np
from pandas_datareader import data as pdr
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import multiprocessing
import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
import multiprocessing
import joblib
import os
from pathlib import Path
import sys

# Check if there are enough command-line arguments
# if len(sys.argv) != 4:
#     print("Usage: python your_script.py stock_symbol start_date end_date")
#     sys.exit(1)

n_cores = multiprocessing.cpu_count()

# Check if GPU is available
# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found, using CPU")

yf.pdr_override()
np.random.seed(5805)

def fetch_stock_data(stocks, start_date, end_date):
    df = pd.DataFrame()
    for stock_symbol in stocks:
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        stock_data['Symbol'] = stock_symbol  # Add a column to identify the stock
        df = pd.concat([df, stock_data])
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def calculate_sma(data, window_size):
    data['SMA'] = data['Close'].rolling(window=window_size, min_periods=1).mean()
    return data

def calculate_macd(data, short_period, long_period, signal_period):
    data['ShortEMA'] = data['Close'].ewm(span=short_period).mean()
    data['LongEMA'] = data['Close'].ewm(span=long_period).mean()
    data['MACD'] = data['ShortEMA'] - data['LongEMA']
    data['Signal Line'] = data['MACD'].ewm(span=signal_period).mean()
    data['MACD Histogram'] = data['MACD'] - data['Signal Line']
    return data

def calculate_daily_returns(data):
    data['Daily_Return'] = data['Close'].pct_change()
    return data

def make_trading_decision(data, short_period, long_period):
    signals = [1]  # Initialize with "Buy" for the first row
    
    for i in range(1, len(data)):
        if (
            data['SMA'][i] > data['SMA'][i - 1] and
            data['MACD'][i] > data['Signal Line'][i] and
            data['MACD'][i - 1] <= data['Signal Line'][i - 1] and
            data['Daily_Return'][i] > 0
        ):
            # Buy signal
            signals.append(1)
        elif (
            data['SMA'][i] < data['SMA'][i - 1] and
            data['MACD'][i] < data['Signal Line'][i] and
            data['MACD'][i - 1] >= data['Signal Line'][i - 1] and
            data['Daily_Return'][i] < 0
        ):
            # Sell signal
            signals.append(-1)
        else:
            # Hold signal
            signals.append(0)

    # Add the signals as a new column to the DataFrame
    data['Signal'] = signals

    return data

def predict_signal(model, X_train, y_train_signal, X_test, threshold=0.08000000000000007):
    # Train the given model on X_train and y_train_signal
    model.fit(X_train, y_train_signal)
    
    # Make predictions using the trained model for 'Signal'
    ml_predictions_signal = model.predict(X_test)
    
    # Convert model predictions into buy/sell/hold signals based on the threshold
    signals = []
    for pred in ml_predictions_signal:
        if pred > threshold:
            signals.append(1)
        elif pred < -threshold:
            signals.append(-1)
        else:
            signals.append(0)
    
    # Return the buy/sell/hold signals for 'Signal'
    return signals

def predict_close(model, X_train, y_train_close, X_test):
    # Train the given model on X_train and y_train_close
    model.fit(X_train, y_train_close)
    
    # Make predictions using the trained model for 'Close'
    ml_predictions_close = model.predict(X_test)
    
    # Return the predicted close values
    return ml_predictions_close

# stocks = sys.argv[1]
# start_date = sys.argv[2]
# end_date = sys.argv[3]

# try:
#     stock_data = yf.download(stocks, start=start_date, end=end_date)
# except Exception as e:
#     print(f"Failed to fetch data for {stocks}: {e}")
#     sys.exit(1)

# # Check if the DataFrame is empty
# if stock_data.empty:
#     print(f"No data available for {stocks} in the specified date range.")
#     sys.exit(1)

if __name__ == "__main__":
    # Define the stocks to analyze
    stocks = ['HP']
    # print("Stock Symbol:", stocks)
    # print("Start Date:", start_date)
    # print("End Date:", end_date)
    # stocks=input("Enter a stock")

    # print("Define the date range for data retrieval")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    # with pd.ExcelWriter("stock_predictions.xlsx", engine='xlsxwriter') as writer:
    for stock_symbol in stocks:
    # stock_symbol=stocks
        df = fetch_stock_data([stock_symbol], start_date, end_date)
        window_size = 250
        short_term_period = 12
        long_term_period = 26
        signal_period = 9

        df = calculate_sma(df, window_size)
        df = calculate_macd(df, short_term_period, long_term_period, signal_period)
        df = calculate_daily_returns(df)
        make_trading_decision(df, window_size, short_term_period)
        df = df.dropna()

        features = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'SMA', 'MACD', 'Daily_Return']
        target_variable = ['Close', 'Signal']
        X = df[features]
        y = df[target_variable]

        if not X.empty:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            print("No data available for scaling.")

        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)
        
        X_test = X[df['Date'] >= '2015-01-01']
        y_test = y[df['Date'] >= '2015-01-01']
        y_test['Date'] = df[df['Date'] >= '2015-01-01']['Date']
        
        X_train = X[(df['Date'] >= start_date) & (df['Date'] < '2015-01-01')]
        y_train = y[(df['Date'] >= start_date) & (df['Date'] < '2015-01-01')]

        y_train = y_train[1:]
        X_train = X_train[:-1]  
        y_test = y_test[1:]
        X_test = X_test[:-1]  

        # model = Sequential()
        # model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
        # model.add(Dense(256, activation='relu'))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(1, activation='linear'))
        # model.compile(optimizer='adam', loss='mean_absolute_error')
        # model.fit(X_train, y_train['Close'], epochs=300, batch_size=64)
        # predicted_close_nn = model.predict(X_test).flatten()
        # nn_model_filename = f"{stock_symbol}_neural_network_model.pkl"
        # joblib.dump(model, nn_model_filename)
        # mae_nn = mean_absolute_error(y_test['Close'], predicted_close_nn)

        

        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train['Close'])
        y_pred_lr = lr_model.predict(X_test)
        lr_model_filename = f"{stock_symbol}_linear_regression_model.pkl"
        joblib.dump(lr_model, lr_model_filename)
        mae_lr = mean_absolute_error(y_test['Close'], y_pred_lr)
        print(len(y_test['Close']), len(y_pred_lr))

        rf_model = RandomForestRegressor(n_estimators=10, random_state=0)
        rf_model.fit(X_train,y_train['Close'])
        y_pred_rf = rf_model.predict(X_test)
        rf_model_filename = f"{stock_symbol}_random_forest_model.pkl"
        joblib.dump(rf_model, rf_model_filename)
        mae_rf = mean_absolute_error(y_test['Close'], y_pred_rf)

        gb_model = GradientBoostingRegressor()
        gb_model.fit(X_train, y_train['Close'])
        y_pred_gb = gb_model.predict(X_test)
        gb_model_filename = f"{stock_symbol}_gradient_boost.pkl"
        joblib.dump(gb_model, gb_model_filename)
        mae_gb = mean_absolute_error(y_test['Close'], y_pred_gb)

        print(f"Stock: {stock_symbol}")
        # print("Mean absolute error neural network: ", mae_nn)
        print("Mean absolute error linear regression: ", mae_lr)
        print("Mean absolute error Random Forest: ", mae_rf)
        print("Mean absolute error Gradient Boosting Regressor: ", mae_gb)
        print()
        
        results_df = pd.DataFrame({
            "Date": y_test['Date'],
            "Actual_Close": y_test['Close'],
            # "Predicted_Close_NN": predicted_close_nn,
            "Predicted_Close_LR": y_pred_lr,
            "Predicted_Close_RF": y_pred_rf,
            "Predicted_Close_GB": y_pred_gb
        })

        stock_results = pd.DataFrame({
            "Stock": [stock_symbol] * len(y_test['Close']),
            "Date": y_test['Date'],
            "Actual_Close": y_test['Close'],
            # "Predicted_Close_NN": predicted_close_nn,
            "Predicted_Close_LR": y_pred_lr,
            "Predicted_Close_RF": y_pred_rf,
            "Predicted_Close_GB": y_pred_gb
        })

        results_df = pd.concat([results_df, stock_results], ignore_index=True)
        stock_data = results_df[results_df['Stock'] == stock_symbol]
        stock_data.to_csv(f"{stock_symbol}_stock_data.csv", index=False)


        # while True:
        #     input_date = input(f"Enter a date for {stock_symbol} (YYYY-MM-DD) to predict the close price (or 'q' to quit): ")
        #     if input_date.lower() == 'q':
        #         break
            
        #     try:
        #         # Convert the input date to a datetime object
        #         input_date = pd.to_datetime(input_date)
                
        #         # Prepare the input data for prediction
        #         input_data = df[df['Date'] == input_date][features]
        #         input_data = scaler.transform(input_data)
                
        #         # Predict using the trained models
        #         nn_prediction = model.predict(input_data)[0][0]
        #         lr_prediction = lr_model.predict(input_data)[0]

        #         print(f"Predicted Close Price (Neural Network): {nn_prediction:.2f}")
        #         print(f"Predicted Close Price (Linear Regression): {lr_prediction:.2f}")
        #     except ValueError:
        #         print("Invalid date format. Please use 'YYYY-MM-DD' format.")
        #     except KeyError:
        #         print("Date not found in the dataset. Please enter a date within the available range.")
        # print(f"Finished predictions for {stock_symbol}.")
        

