# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_pinball_loss

COLAB = False

seq_length = 10  # 60 pas trop mal?
hidden_size = 50
num_layers = 4
output_size = 1
num_epochs = 2
learning_rate = 0.001
dropout = 0.1

plt.rcParams["figure.figsize"] = [10, 6]
# %%
device = torch.device("cpu")
if COLAB:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    df_full = pd.read_csv("drive/MyDrive/data/treated_data.csv")
    df_results = pd.read_csv("drive/MyDrive/data/test_better.csv")[-395:]
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Colab GPU
else:
    df_full = pd.read_csv("Data/treated_data.csv")
    df_results = pd.read_csv("Data/test_better.csv")[-395:]
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # MacOS GPU

print("Device used:", device)
# %%
df_full["Date"] = pd.to_datetime(df_full["Date"])
df_full = df_full.set_index("Date")

df_full = pd.get_dummies(data=df_full, columns=["WeekDays", "BH_Holiday"], dtype=int)
# %%
target_col = "Net_demand"
feature_cols = df_full.columns.to_list()
feature_cols.remove(target_col)
# %%
# Normalization des Net_demand
df_train_val = df_full[df_full.index <= "2022-09-01"]
f_mean = df_train_val["Net_demand"].mean()
f_std = df_train_val["Net_demand"].std()
for feature in ["Net_demand", "Net_demand.1", "Net_demand.7"]:
    df_full[feature] = (df_full[feature] - f_mean) / f_std

# %%
# TODO Créer un validation set, enlever cheat
df_train = df_full[df_full.index <= "2022-09-01"]
df_test = df_full[df_full.index >= "2022-09-02"]


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
# TODO Faire marcher pinball ou la supprimer
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


# Create the dataset and data loader.
dataset = TimeSeriesDataset(df_train, feature_cols, target_col, seq_length)
train_loader = DataLoader(dataset, batch_size=32 * 2, shuffle=True)

# %%
# Hyperparameters
input_size = len(feature_cols)  # e.g., 2 features

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
idx = df_train.index.max()
steps = 1

# Boucle pour sélectionner la séquence à input dans le LSTM
preds = []
for _ in range(395):
    last_seq = df_full.loc[df_full.index <= idx, feature_cols].values[-seq_length:]
    preds.append(forecast(model, last_seq, steps, device))
    idx += pd.Timedelta(days=1)

preds_array = np.array(preds)
preds_original = (preds_array * f_std) + f_mean
# %%
df_results["Date"] = pd.to_datetime(df_results["Date"])
df_results = df_results.set_index("Date")[["Net_demand"]]
df_results["preds"] = 0
df_results["preds"] = preds_original
# %%
plt.plot(df_results["Net_demand"])
plt.plot(df_results["preds"])
plt.show()
# %%
err = mean_pinball_loss(df_results["Net_demand"], df_results["preds"], alpha=0.8)
print("Pinball:", err)
# %%
x_axis = range(-1500, 1500, 100)
err_li = []
for i in x_axis:
    err = mean_pinball_loss(
        df_results["Net_demand"], df_results["preds"] + i, alpha=0.8
    )
    err_li.append(err)

plt.plot(x_axis, err_li)
# %%
