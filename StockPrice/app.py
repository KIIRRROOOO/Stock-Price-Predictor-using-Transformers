import io
import base64
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import yfinance as yf

app = Flask(__name__)

def build_transformer_model(input_shape, num_heads=4, ff_dim=64, dropout_rate=0.1):
    """
    Build a simple Transformer encoder model for time series prediction.
    """
    inputs = Input(shape=input_shape)

    # Multi-Head Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feed Forward Network
    ffn_output = Dense(ff_dim, activation='relu')(attention_output)
    ffn_output = Dense(input_shape[-1])(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    sequence_output = LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)

    # Pooling + Dense for prediction
    pooled_output = GlobalAveragePooling1D()(sequence_output)
    outputs = Dense(1)(pooled_output)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def run_transformer(ticker="BTC-USD"):
    # Training period
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime.now()

    data = yf.download(ticker, start=start, end=end)
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    prediction_days = 60
    x_train, y_train = [], []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build Transformer model
    model = build_transformer_model(input_shape=(x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

    # Testing
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()

    test_data = yf.download(ticker, start=test_start, end=test_end)
    actual_prices = test_data["Close"].values

    total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Prediction
    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual_prices, label="Actual Price", color='black')
    ax.plot(prediction_prices, label="Predicted Price", color='red')
    ax.legend()
    ax.set_title(f"{ticker} Price Prediction (Transformer Model)")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return encoded


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    ticker = data.get("ticker")
    img = run_transformer(ticker)
    return jsonify({"image": img})


if __name__ == "__main__":
    app.run(debug=True)
