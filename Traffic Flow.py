import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

# Load your dataset
data = pd.read_csv('TrafficTwoMonth.csv')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Select relevant features and label
features = data[['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']].values
labels = data['Traffic Situation'].values

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Convert labels to one-hot encoding
encoder = OneHotEncoder(sparse=False)
labels_one_hot = encoder.fit_transform(labels.reshape(-1, 1))

# Create sequences for LSTM
def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(labels[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 20  # Adjust based on your requirement
X, y = create_sequences(features_scaled, labels_one_hot, time_steps)

# Check the shapes of X and y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Define the CustomConv1D class
class CustomConv1D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.kernel_size = kernel_size
        self.weights = np.random.rand(out_channels, in_channels, kernel_size) * 0.01
        self.bias = np.random.rand(out_channels) * 0.01

    def forward(self, x):
        batch_size, seq_length, in_channels = x.shape
        out_length = seq_length - self.kernel_size + 1
        out = np.zeros((batch_size, out_length, self.weights.shape[0]))

        for b in range(batch_size):
            for o in range(self.weights.shape[0]):  # For each output channel
                for t in range(out_length):
                    out[b, t, o] = np.sum(x[b, t:t + self.kernel_size, :] * self.weights[o].T) + self.bias[o]

        return out

# Define the CustomLSTM class
class CustomLSTM:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.Wf = np.random.rand(hidden_size, input_size + hidden_size) * 0.01
        self.Wi = np.random.rand(hidden_size, input_size + hidden_size) * 0.01
        self.Wc = np.random.rand(hidden_size, input_size + hidden_size) * 0.01
        self.Wo = np.random.rand(hidden_size, input_size + hidden_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        h_prev = h_prev.reshape(self.hidden_size, 1)  # Ensure h_prev is (hidden_size, 1)
        combined = np.vstack((h_prev, x.reshape(-1, 1)))  # Stack h_prev and x vertically
        ft = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
        it = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
        ct_tilde = np.tanh(np.dot(self.Wc, combined) + self.bc)
        c = ft * c_prev + it * ct_tilde
        ot = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
        h = ot * np.tanh(c)

        # Apply dropout (manually)
        dropout_mask = (np.random.rand(*h.shape) > 0.2).astype(np.float32)  # 20% dropout
        h *= dropout_mask  # Zero out some activations

        return h, c

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Define the CNN-LSTM model
class CNNLSTMModel:
    def __init__(self, conv_layer, lstm_layer, output_size):
        self.conv_layer = conv_layer
        self.lstm_layer = lstm_layer
        self.output_size = output_size
        # Initialize weights for the output layer
        self.W_output = np.random.uniform(-0.1, 0.1, (self.output_size, self.lstm_layer.hidden_size))
        self.b_output = np.zeros((self.output_size, 1))

    def forward(self, x):
        conv_output = self.conv_layer.forward(x)
        batch_size, seq_length, _ = conv_output.shape
        h_prev = np.zeros((self.lstm_layer.hidden_size, batch_size))
        c_prev = np.zeros((self.lstm_layer.hidden_size, batch_size))
        
        for t in range(seq_length):
            h_prev, c_prev = self.lstm_layer.forward(conv_output[:, t, :], h_prev, c_prev)
        
        # Apply the output layer transformation
        output = np.dot(self.W_output, h_prev) + self.b_output
        return output.T  # Return the output for the last time step

# Define the training function
def train(model, X, y, epochs, learning_rate):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            x_sample = X[i:i+1]  # Take one sample
            y_sample = y[i:i+1]  # Corresponding label
            
            # Forward pass
            output = model.forward(x_sample)
            
            # Compute loss
            loss = mean_squared_error(y_sample, output)
            total_loss += loss

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X)}")

    print("Training complete.")

# Define the evaluation function
def evaluate(model, X, y):
    predictions = []
    for i in range(len(X)):
        x_sample = X[i:i+1]
        output = model.forward(x_sample)
        predictions.append(output)

    predictions = np.array(predictions).reshape(-1, 4)  # Ensure predictions shape is (990, 4)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
    return accuracy

# Define the prediction function
def predict(model, X):
    predictions = []
    for i in range(len(X)):
        x_sample = X[i:i+1]
        output = model.forward(x_sample)
        predictions.append(output)

    return np.array(predictions)

# Initialize the model with adjusted hyperparameters
conv_layer = CustomConv1D(in_channels=5, out_channels=20, kernel_size=3)  # Increased out_channels
lstm_layer = CustomLSTM(input_size=20, hidden_size=50)  # Increased hidden_size
model = CNNLSTMModel(conv_layer, lstm_layer, output_size=4)

# Train the model
train(model, X, y, epochs=10, learning_rate=0.001)

# Evaluate the model
accuracy = evaluate(model, X, y)
print(f"Evaluation Accuracy: {accuracy}")

# Make predictions
predictions = predict(model, X)
print("Sample Predictions:", predictions[:5])