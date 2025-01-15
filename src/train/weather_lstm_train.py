import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# File paths
data_path = "../../data/weather/processed/weather_dataset_train.csv"
model_save_path = "../../weights/weather/weather_lstm.pth"
log_file_name = "../../outputs/weather/weather_dataset_test.csv"

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the data
print("Loading data...")
data = pd.read_csv(data_path)

# Select relevant features and target column
features = ["tempmin", "tempmax", "humidity", "windspeed", "pressure", "cloudcover", "precip", "dew", "solarradiation"]
target = "temp"  # Target column for prediction

# Drop rows with missing values
print("Dropping rows with missing values...")
missing_count_before = data.shape[0]
data = data[features + [target, "datetime"]].dropna()
missing_count_after = data.shape[0]
print(f"Dropped {missing_count_before - missing_count_after} rows due to missing values.")

# Normalize the features and target separately
print("Normalizing features and target...")
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
data[features] = feature_scaler.fit_transform(data[features])
data[[target]] = target_scaler.fit_transform(data[[target]])

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
X, y, datetimes = create_sequences(
    data[features].values, data[target].values, sequence_length, forecast_horizon, data["datetime"].values
)

# Split the data into training, validation, and test sets
print("Splitting data into train, validation, and test sets...")
X_train, X_temp, y_train, y_temp, dates_train, dates_temp = train_test_split(
    X, y, datetimes, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test, dates_val, dates_test = train_test_split(
    X_temp, y_temp, dates_temp, test_size=0.5, random_state=42
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
train_dataset = WeatherDataset(X_train, y_train)
val_dataset = WeatherDataset(X_val, y_val)
test_dataset = WeatherDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

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
num_epochs = 100  # Number of training epochs
learning_rate = 0.001  # Learning rate for optimizer

# Initialize the model
print("Initializing the LSTM model...")
model = WeatherLSTM(input_size, hidden_size, num_layers, output_size).to(device)

# Define a custom weighted loss function for handling different forecast weights
def weighted_loss(predictions, targets):
    weights = torch.tensor([1.0, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3], device=predictions.device)  # Custom weights for each day
    return (weights * (predictions - targets) ** 2).mean()

criterion = weighted_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Initialize TensorBoard writer for logging
print("Initializing TensorBoard writer...")
tensorboard_writer = SummaryWriter()

# Function to save the model
def save_model(model, path=model_save_path):
    torch.save(model.state_dict(), path)  # Save the model's state dictionary
    print(f"Model saved to {path}")

# Save predictions, actuals, and other logs to a CSV file
def save_logs(predictions, actuals, dates, file_name=log_file_name):
    import pandas as pd
    # Inverse transform predictions and actuals to original scale
    original_features = feature_scaler.inverse_transform(X_test[:, -1, :])  # Last step of each sequence
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

# Training loop for the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    model.to(device)  # Ensure model is on the correct device
    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to device

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(X_batch)  # Forward pass
            loss = criterion(outputs, y_batch)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            train_loss += loss.item()  # Accumulate training loss

        train_loss /= len(train_loader)  # Average training loss

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)  # Average validation loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Log losses to TensorBoard
        tensorboard_writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        tensorboard_writer.add_scalar("Loss/Validation", val_loss, epoch + 1)

    save_model(model)  # Save the trained model

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)  # Start training

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
    save_logs(predictions, actuals, dates_test)  # Save predictions and actuals
    return predictions, actuals

# Evaluate the model on the test set
predictions, actuals = evaluate_model(model, test_loader)

# Close the TensorBoard writer
print("Closing TensorBoard writer...")
tensorboard_writer.close()
