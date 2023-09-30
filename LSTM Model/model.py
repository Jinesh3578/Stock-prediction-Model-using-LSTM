import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Activation
from keras import optimizers
import plotly.graph_objs as go
import plotly.offline as pyo
from data import download_and_preprocess_data



# Define the LSTM model
def build_lstm_model(backcandles):
    lstm_input = Input(shape=(backcandles, 8), name='lstm_input')
    inputs = LSTM(150, name='first_layer')(lstm_input)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss='mse')

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, stock_name = download_and_preprocess_data()

    np.random.seed(10)
    model = build_lstm_model(backcandles=30)
    model.fit(x=X_train, y=y_train, batch_size=15, epochs=30, shuffle=True, validation_split=0.1)

    y_pred = model.predict(X_test)

    # Create traces for actual and predicted data
    trace_actual = go.Scatter(x=np.arange(len(y_test)), y=y_test.flatten(), mode='lines', name='Actual')
    trace_pred = go.Scatter(x=np.arange(len(y_test)), y=y_pred.flatten(), mode='lines', name='Predicted')

    data = [trace_actual, trace_pred]

    layout = go.Layout(title=f'{stock_name} - Actual vs. Predicted', xaxis=dict(title='Time'), yaxis=dict(title='Value'))

    fig = go.Figure(data=data, layout=layout)

    # Save the plot to an HTML file or display it in your Jupyter Notebook
    pyo.plot(fig, filename='actual_vs_predicted.html')

