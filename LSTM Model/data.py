import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler


# Download and preprocess data
def download_and_preprocess_data():
    stock_name = "Tata Motors"
    data = yf.download(tickers='TATAMOTORS.NS', start='2012-03-11', end='2022-07-10')

    data['RSI'] = ta.rsi(data.Close, length=15)
    data['EMAF'] = ta.ema(data.Close, length=20)
    data['EMAM'] = ta.ema(data.Close, length=100)
    data['EMAS'] = ta.ema(data.Close, length=150)

    data['Target'] = data['Adj Close'] - data.Open
    data['Target'] = data['Target'].shift(-1)

    data['TargetClass'] = [1 if row['Target'] > 0 else 0 for index, row in data.iterrows()]


    data['TargetNextClose'] = data['Adj Close'].shift(-1)

    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)

    data_set = data.iloc[:, 0:11]

    sc = MinMaxScaler(feature_range=(0, 1))
    data_set_scaled = sc.fit_transform(data_set)

    X = []
    backcandles = 30

    for j in range(8):
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):
            X[j].append(data_set_scaled[i - backcandles:i, j])

    X = np.moveaxis(X, [0], [2])

    X = np.array(X)
    yi = np.array(data_set_scaled[backcandles:, -1])
    y = np.reshape(yi, (len(yi), 1))

    splitlimit = int(len(X) * 0.8)
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]

    return X_train, X_test, y_train, y_test, sc, stock_name


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler,stock_name = download_and_preprocess_data()
