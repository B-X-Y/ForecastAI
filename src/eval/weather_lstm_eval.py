import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# File paths
data_path = "../../data/weather/processed/weather_dataset_eval.csv"
model_save_path = "../../weights/weather/weather_lstm.pth"
log_file_name = "../../outputs/weather/weather_dataset_pred.csv"

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the data
print("Loading data...")
new_data = pd.read_csv(data_path)

# Select relevant features and target column
features = ["tempmin", "tempmax", "humidity", "windspeed", "pressure", "cloudcover", "precip", "dew", "solarradiation"]
target = "temp"  # Target column for prediction

# Drop rows with missing values
print("Dropping rows with missing values...")
missing_count_before = new_data.shape[0]
new_data = new_data[features + [target, "datetime"]].dropna()
missing_count_after = new_data.shape[0]
print(f"Dropped {missing_count_before - missing_count_after} rows due to missing values.")

# Normalize the features and target separately
print("Normalizing features and target...")
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
new_data[features] = feature_scaler.fit_transform(new_data[features])
new_data[[target]] = target_scaler.fit_transform(new_data[[target]])

# Convert data to sequences for time-series prediction
sequence_length = 3  # Use data from the last 3 days
forecast_horizon = 7  # Predict the next 7 days
def create_sequences(data, target, seq_length, horizon, datetimes):
    sequences = []
    labels = []
    dates = []
    for i in range(len(data) - seq_length - horizon):
        # Sequence of past data points
        sequences.append(data[i:i + seq_length])
        # Corresponding future target values
        labels.append(target[i + seq_length:i + seq_length + horizon])
        # Datetime values for the prediction window
        dates.append(datetimes[i + seq_length:i + seq_length + horizon])
    return np.array(sequences), np.array(labels), np.array(dates)

print("Creating sequences...")
# Prepare the sequences, labels, and associated datetimes
X_new, y_new, dates_new = create_sequences(
    new_data[features].values, new_data[target].values, sequence_length, forecast_horizon, new_data["datetime"].values
)

# Define a PyTorch Dataset class for the weather data
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Input features
        self.y = torch.tensor(y, dtype=torch.float32)  # Target labels

    def __len__(self):
        return len(self.X)  # Number of samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # Return a single sample

# Create datasets and dataloaders for training, validation, and testing
print("Creating datasets and dataloaders...")
new_dataset = WeatherDataset(X_new, y_new)
new_loader = DataLoader(new_dataset, batch_size=64, shuffle=False)

# Define the LSTM model
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)  # LSTM layer
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer

    def forward(self, x):
        out, _ = self.lstm(x)  # Pass input through LSTM
        out = self.fc(out[:, -1, :])  # Take the output of the last LSTM time step
        return out

# Set model hyperparameters
input_size = len(features)  # Number of input features
hidden_size = 128  # Number of LSTM hidden units
num_layers = 2  # Number of LSTM layers
output_size = forecast_horizon  # Number of output predictions (forecast horizon)

# Define a custom weighted loss function for handling different forecast weights
def weighted_loss(predictions, targets):
    weights = torch.tensor([1.0, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3], device=predictions.device)  # Custom weights for each day
    return (weights * (predictions - targets) ** 2).mean()

criterion = weighted_loss

# Function to load a saved model
def load_model(path=model_save_path):
    model = WeatherLSTM(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load(path))  # Load the model's state dictionary
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {path}")
    return model

# Save predictions, actuals, and other logs to a CSV file
def save_logs(predictions, actuals, dates, file_name=log_file_name):
    import pandas as pd
    # Inverse transform predictions and actuals to original scale
    original_features = feature_scaler.inverse_transform(X_new[:, -1, :])  # Last step of each sequence
    predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, forecast_horizon))
    actuals_original = target_scaler.inverse_transform(actuals.reshape(-1, forecast_horizon))

    offsets = predictions_original - actuals_original  # Calculate prediction offsets

    logs = pd.DataFrame({
        "datetime": dates[:, 0],  # Start datetime for each prediction
        **{f"day_{i+1}_pred": predictions_original[:, i] for i in range(forecast_horizon)},
        **{f"day_{i+1}_actual": actuals_original[:, i] for i in range(forecast_horizon)},
        **{f"day_{i+1}_offset": offsets[:, i] for i in range(forecast_horizon)}
    })

    for i, feature_name in enumerate(features):
        logs[feature_name] = original_features[:, i]  # Add original feature values

    logs.to_csv(file_name, index=False)  # Save logs to a CSV file
    print(f"Logs saved to {file_name}")

# Evaluate the model on the test set
def evaluate_model(model, test_loader):
    model.to(device)  # Ensure model is on the correct device
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    predictions, actuals = [], []
    print("Evaluating the model on the test set...")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()

            predictions.append(outputs.cpu().numpy())  # Collect predictions
            actuals.append(y_batch.cpu().numpy())  # Collect actual values

    predictions = np.concatenate(predictions, axis=0)  # Combine all predictions
    actuals = np.concatenate(actuals, axis=0)  # Combine all actuals
    test_loss /= len(test_loader)  # Average test loss
    print(f"Test Loss: {test_loss:.4f}")
    save_logs(predictions, actuals, dates_new)  # Save predictions and actuals
    return predictions, actuals

# Load the trained model
model = load_model(model_save_path)

# Evaluate the model on the test set
predictions, actuals = evaluate_model(model, new_loader)
