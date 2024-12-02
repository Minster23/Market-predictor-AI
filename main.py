# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

timestamps = []
actual_prices = []  # You should have the actual future prices for accuracy calculation
predicted_prices = []

# Load the data
file_path = '/content/market.csv'  # Update with your CSV file path
data = pd.read_csv(file_path)

# Convert timestamp to datetime and sort
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values('timestamp')

# Use 'close' as the feature for prediction
data_close = data[['close']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data_close)

# Prepare sequences for LSTM
def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(sequence_length, len(data)):
        x.append(data[i-sequence_length:i, 0])  # Past sequence_length points
        y.append(data[i, 0])  # The next point
    return np.array(x), np.array(y)

sequence_length = 60  # Use the past 60 timesteps for predictions
x, y = create_sequences(data_normalized, sequence_length)

# Split into train and test
train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM input
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, epochs=80, batch_size=32, validation_data=(x_test, y_test))

# Predict on test data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Evaluate predictions
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot actual vs. predicted
plt.figure(figsize=(14, 7))
plt.plot(y_test_scaled, color='blue', label='Actual Prices')
plt.plot(predictions, color='red', label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Market Price Prediction')
plt.legend()
plt.show()

# Predict the next 10 days
last_sequence = np.append(last_sequence[:, 1:, :], next_price.reshape(1, 1, 1), axis=1)

steps_to_predict = 10  # Define how many future steps you want to predict
future_predictions = []

for i in range(steps_to_predict):
    # Make the prediction
    next_price = model.predict(last_sequence)

    # Track the predicted price
    future_predictions.append(next_price[0, 0])  # Save the predicted price

    # Check if there's an actual price to compare
    if i < len(actual_prices):
        actual_price = actual_prices[i]  # Get actual price for this timestep

        # Calculate the error for this prediction (using Mean Squared Error or other metrics)
        mse = mean_squared_error([actual_price], [next_price[0, 0]])

        # Print the timestamp, predicted value, actual value, and error
        print(f"Timestamp: {i}, Predicted: {next_price[0, 0]}, Actual: {actual_price}, MSE: {mse}")
    else:
        # If no actual price is available, just print the predicted value
        print(f"Timestamp: {i}, Predicted: {next_price[0, 0]}")

    # Append the predicted value to the sequence for the next prediction
    next_price = next_price.reshape(1, 1, 1)  # Reshape next_price to be (1, 1, 1)
    last_sequence = np.concatenate([last_sequence[:, 1:, :], next_price], axis=1)


future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
print("Next 10 days predicted prices:", future_predictions)
