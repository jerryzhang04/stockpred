import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Select a page:", ["Stock Predictor", "Trading Script"])

# Setting the default dates
today = pd.to_datetime("today")
ten_years_ago = today - pd.DateOffset(years=10)

# Allow users to choose dates
start_date = st.date_input('Start Date', ten_years_ago)
end_date = st.date_input('End Date', today)

start = start_date.strftime('%Y-%m-%d')
end = end_date.strftime('%Y-%m-%d')

userInput = st.text_input('Enter Stock Ticker', 'AAPL')

if selection == "Stock Predictor":

    st.title('Stock Trend Prediction')

    


    df = data.DataReader(userInput, 'stooq', start, end)
    df = df[::-1]

    #describing data
    st.subheader(f'Data from {start} to {end}')
    if df.empty:
        st.write(f"No data available for ticker {userInput} for the provided date range.")
    else:
        st.write(df.describe())

    #visualizations
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(df.Close, 'b')
    st.pyplot(fig)


    #splitting data into training and testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)



    #loading my model
    model = load_model('my_finallmodel.keras')

    #testing part
    past_100_days = data_training.tail(100)

    final_df = pd.concat([past_100_days, data_testing]).reset_index(drop=True)
    input_data = scaler.transform(final_df)

    x_test =[]
    y_test=[]

    for i in range(100,input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i,0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)

    scale_factor = 1/scaler.scale_[0]
    y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate the rolling bias between predicted and actual values
    rolling_bias = pd.Series(y_test.flatten() - y_predicted.flatten()).rolling(window=10).mean().fillna(0)

    # Adjust the predicted values based on the rolling bias
    y_predicted_adjusted = y_predicted.flatten() + rolling_bias


    #final graph
    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)


    from sklearn.preprocessing import MinMaxScaler
    original_scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = original_scaler.fit_transform(data_training)

    def predict_future_days(model, data, days_into_future):
        future_predictions = []
        
        last_100_days = data[-100:].values
        for _ in range(days_into_future):
            last_100_days_normalized = original_scaler.transform(last_100_days.reshape(-1, 1))
            last_100_days_normalized = last_100_days_normalized.reshape((1, last_100_days_normalized.shape[0], 1))
            
            prediction = model.predict(last_100_days_normalized)
            
            future_predictions.append(prediction[0][0])
            
            # Add the prediction to the list of last 100 days and remove the first value
            last_100_days = np.append(last_100_days, prediction)
            last_100_days = last_100_days[1:]
            
        scale_factor = 1/original_scaler.scale_[0]
        future_predictions = np.array(future_predictions).reshape(-1, 1) * scale_factor
        
        return future_predictions

    
    N = st.number_input('How many days into the future would you like to predict?', min_value=1, value=1)
    future_preds = predict_future_days(model, final_df['Close'], N)
    st.subheader('Predictions for the next days:')
    st.write(future_preds)


elif selection == "Trading Script":
    st.title('Trading Script')

    # Load stock data
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')
    df = data.DataReader(userInput, 'stooq', start, end)
    df = df[::-1]

    # Calculate short and long moving averages
    short_window = st.slider('Short Moving Average Window', min_value=5, max_value=50, value=40, step=5)
    long_window = st.slider('Long Moving Average Window', min_value=51, max_value=200, value=100, step=5)
    initial_capital = st.number_input('Initial Capital', value=5000.0, step=1000.0)


    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    signals['short_mavg'] = df['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = df['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)   
    signals['positions'] = signals['signal'].diff()

    # Plotting the strategy results
    st.subheader('Stock Price with Trading Signals')
    fig3 = plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Price')
    plt.plot(df.index, signals['short_mavg'], label=f'Short {short_window}D MA')
    plt.plot(df.index, signals['long_mavg'], label=f'Long {long_window}D MA')
    plt.plot(signals.loc[signals.positions == 1.0].index, signals.short_mavg[signals.positions == 1.0], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(signals.loc[signals.positions == -1.0].index, signals.short_mavg[signals.positions == -1.0], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    plt.legend()
    st.pyplot(fig3)

    # Performance metrics
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['Stock'] = signals['signal']
    portfolio = positions.multiply(signals['price'], axis=0)
    pos_diff = positions.diff()
    portfolio['holdings'] = (positions.multiply(signals['price'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(signals['price'], axis=0)).sum(axis=1).cumsum()   
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']
    portfolio['returns'] = portfolio['total'].pct_change()

    st.text('Performance Metrics')
    st.write(f"Final Portfolio Value: {portfolio['total'][-1]:.2f}")
    st.write(f"Total Returns: {(portfolio['total'][-1]-initial_capital)/initial_capital*100:.2f}%")

    # Plotting portfolio value
    st.subheader('Portfolio Value Over Time')
    fig4 = plt.figure(figsize=(12,6))
    plt.plot(portfolio['total'])
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    st.pyplot(fig4)

    
