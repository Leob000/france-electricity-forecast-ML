# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_pinball_loss

plt.rcParams["figure.figsize"] = [10, 6]

COLAB = False
# %%
device = torch.device("cpu")
if COLAB:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    df_full = pd.read_parquet("drive/MyDrive/data/full_treated.parquet")
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Colab GPU
else:
    df_full = pd.read_parquet("Data/full_treated.parquet")
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # MacOS GPU

print("Device used:", device)
# %%
# Load and sort the data (adjust file path as needed)
df_full_noscale = df_full.copy()
df_full["Year"] = df_full["Year"].astype(float)

# Séparation des datasets
df_train = df_full[df_full.index < "2022-09-02"]
df_test = df_full[df_full.index >= "2022-09-02"]

# Scaling
feature_to_scale = [
    "Load.1",
    # "Load.7",
    "Net_demand",
    "Temp",
    "Temp_s95",
    "Temp_s99",
    "Temp_s95_min",
    "Temp_s95_max",
    "Temp_s99_min",
    "Temp_s99_max",
    "Wind",
    "Wind_weighted",
    "Nebulosity",
    "Nebulosity_weighted",
    "Year",
    "Solar_power.1",
    # "Solar_power.7",
    "Wind_power.1",
    # "Wind_power.7",
    # "Net_demand.1",
    # "Net_demand.7",
]

# On scale dans df_train en gardant les scalers
scalers = {}
for col in feature_to_scale:
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_train.loc[:, col] = scaler.fit_transform(df_train[[col]])
    scalers[col] = scaler

# On scale df_test avec les scalers de df_train
for feature in scalers.keys():
    scaler = scalers[feature]
    df_test.loc[:, feature] = scaler.transform(df_test[[feature]])


# Fonction pour scale les variables lag comme sa variable classique correspondante
def lagged_value_scaler(df, from_feature, to_lagged_feature):
    min = scalers[from_feature].data_min_[0]
    max = scalers[from_feature].data_max_[0]
    df.loc[:, to_lagged_feature] = (df[to_lagged_feature] - min) / (max - min)


# Scale de certaines variables selon le scaling des autres
for df in [df_train, df_test]:
    lagged_value_scaler(df, "Load.1", "Load.7")
    lagged_value_scaler(df, "Solar_power.1", "Solar_power.7")
    lagged_value_scaler(df, "Wind_power.1", "Wind_power.7")
    lagged_value_scaler(df, "Net_demand", "Net_demand.1")
    lagged_value_scaler(df, "Net_demand", "Net_demand.7")

# Pour rescale:
# scalers["feature"].inverse_transform(df["feature"].to_numpy().reshape(-1,1))

# %%
feature_cols = df_full.columns.to_list()
feature_cols.remove("Net_demand")

# TODO test
feature_cols = [
    "Load.1",
    # "Load.7",
    "Net_demand",
    "Temp",
    "Temp_s95",
    "Temp_s99",
    "Temp_s95_min",
    "Temp_s95_max",
    "Temp_s99_min",
    "Temp_s99_max",
    "Wind",
    # "Wind_weighted",
    "Nebulosity",
    # "Nebulosity_weighted",
    # "BH_before",
    "BH",
    # "BH_after",
    "Year",
    "DLS",
    "Summer_break",
    "Christmas_break",
    "Holiday",
    # "Holiday_zone_a",
    # "Holiday_zone_b",
    # "Holiday_zone_c",
    "BH_Holiday",
    "Solar_power.1",
    # "Solar_power.7",
    "Wind_power.1",
    # "Wind_power.7",
    "Net_demand.1",
    "Net_demand.7",
    "sin_dayofweek",
    "cos_dayofweek",
    "sin_dayofyear",
    "cos_dayofyear",
    # "is_weekend",
]

target_col = "Net_demand"


# %%
class TimeSeriesDataset(Dataset):
    def __init__(self, data, feature_cols, target_col, seq_length):
        """
        Args:
            data (DataFrame): Time series data.
            feature_cols (list): List of feature column names.
            target_col (str): Name of the target column.
            seq_length (int): Number of time steps per sample.
        """
        self.data = data.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Extract a sequence of features: shape (seq_length, num_features)
        seq_x = self.data.loc[idx : idx + self.seq_length - 1, self.feature_cols].values
        # Target is the value in target_col at time step idx+seq_length
        seq_y = self.data.loc[idx + self.seq_length, self.target_col]
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(
            seq_y, dtype=torch.float32
        )


# %%
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        """
        Args:
            input_size (int): Number of features (e.g., 2 if using Net_demand and Net_demand.1).
            hidden_size (int): Number of hidden units.
            num_layers (int): Number of LSTM layers.
            output_size (int): Dimension of the forecast output.
        """
        super(LSTMForecast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer (batch_first=True so input shape is (batch, seq_length, input_size))
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        # Fully connected layer mapping hidden state to the output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Get LSTM outputs
        out, _ = self.lstm(x, (h0, c0))
        # Use the output from the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# %%
class PinballLoss(nn.Module):
    def __init__(self, quantile=0.8):
        super(PinballLoss, self).__init__()
        self.quantile = quantile

    def forward(self, predictions, targets):
        errors = targets - predictions
        loss = torch.max(self.quantile * errors, (self.quantile - 1) * errors)
        # loss = self.quantile * torch.max(targets - predictions, 0) + (
        #     1 - self.quantile
        # ) * torch.max(predictions - targets, 0)
        return torch.mean(loss)


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.view(-1, 1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


# %%
def forecast(model, input_seq, steps, device):
    """
    Forecast future target values using a sliding window with multiple features.

    Args:
        model: Trained LSTM model.
        input_seq (np.array): Starting sequence with shape (seq_length, num_features).
                              Assumes the first column is the target and the second is its lag.
        steps (int): Number of time steps to forecast.
        device: Device (CPU/GPU).

    Returns:
        predictions (list): Forecasted target values.
    """
    model.eval()
    predictions = []
    current_seq = input_seq.copy()  # shape: (seq_length, num_features)

    # For autoregressive updates, we assume:
    # - Column 0: current target value.
    # - Column 1: lag (one-day earlier target).
    # For the very first forecast, use the actual last value for the lag.
    with torch.no_grad():
        for step in range(steps):
            # Reshape to (1, seq_length, num_features)
            x = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(x)
            pred_value = pred.item()
            predictions.append(pred_value)

            # Create a new row for the forecasted time step.
            # For the target (column 0), use the predicted value.
            # For the lag (column 1), use the most recent known target.
            if step == 0:
                lag_value = current_seq[-1, 0]  # last actual value in the sequence
            else:
                lag_value = predictions[-2]  # previous forecasted value
            new_row = current_seq[
                -1
            ].copy()  # copy last row to retain any other features unchanged
            new_row[0] = pred_value
            new_row[1] = lag_value

            # Slide the window: remove the first row and append the new row
            current_seq = np.vstack((current_seq[1:], new_row))
    return predictions


# Set the sequence length (number of past days used as input)
seq_length = 60  # Meilleur avec 60 * 2

# Create the dataset and data loader.
dataset = TimeSeriesDataset(df_train, feature_cols, target_col, seq_length)
train_loader = DataLoader(dataset, batch_size=32 * 2, shuffle=True)

# %%
# Hyperparameters
input_size = len(feature_cols)  # e.g., 2 features
hidden_size = 100
num_layers = 4
output_size = 1
num_epochs = 25
learning_rate = 0.001
dropout = 0.1

# Initialize model, loss function, and optimizer.
model = LSTMForecast(input_size, hidden_size, num_layers, output_size, dropout).to(
    device
)
criterion = nn.MSELoss()
# criterion = PinballLoss(quantile=0.8)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %%
# Train the model.
train_model(model, train_loader, criterion, optimizer, num_epochs, device)

# %%
df_final = pd.concat([df_train, df_test])
idx = df_train.index.max()

scaler = scalers[target_col]
steps = 1

# Boucle pour sélectionner la séquence à input dans le LSTM
preds = []
for _ in range(395):
    last_seq = df_final.loc[df_final.index <= idx, feature_cols].values[-seq_length:]
    preds.append(forecast(model, last_seq, steps, device))
    idx += pd.Timedelta(days=1)

preds_array = np.array(preds)
preds_original = scaler.inverse_transform(preds_array)

# %%
truth = df_full_noscale.loc[df_full_noscale.index >= "2022-09-02", ["Net_demand"]]
truth["pred"] = preds_original
# %%
plt.plot(truth["Net_demand"])
plt.plot(truth["pred"])
plt.show()
# %%
err = mean_pinball_loss(truth["Net_demand"], truth["pred"], alpha=0.8)
print(err)
# %%
x_axis = range(-3500, 1500, 100)
err_li = []
for i in x_axis:
    err = mean_pinball_loss(truth["Net_demand"], truth["pred"] + i, alpha=0.8)
    err_li.append(err)

plt.plot(x_axis, err_li)
# %%
