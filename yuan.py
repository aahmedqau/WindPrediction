
# This is the original source code from the article, placed here for reference
# PyTorch framework to build an Attention-LSTM time series model (https://mp.weixin.qq.com/s/Vs96H2hQy60qdRicO5ZRbg)

# Data Retrieval
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Stations, Daily
plt.rcParams['font.sans-serif'] = 'SimHei'  # Set Chinese font display
plt.rcParams['axes.unicode_minus'] = False
chongqing = (29.56301, 106.55156)

# Find weather stations near Chongqing
stations = Stations()
nearby_stations = stations.nearby(*chongqing).fetch(5)

# Get the ID of the nearest station
station_id = nearby_stations.index[0]

# Set time range
start = datetime(2008, 1, 1)
end = datetime(2024, 5, 25)

# Retrieve daily data
data = Daily(station_id, start, end)
data = data.fetch()
data.head()

# Data Splitting
import pandas as pd
df = pd.DataFrame()
df['tavg'] = data['tavg']

# Define split ratios
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# Calculate split indices
train_split = int(train_ratio * len(df))
val_split = int((train_ratio + val_ratio) * len(df))

# Split dataset
train_set = df.iloc[:train_split]
val_set = df.iloc[train_split:val_split]
test_set = df.iloc[val_split:]

plt.figure(figsize=(15, 10))
plt.subplot(3,1,1)
plt.plot(train_set, color='c', alpha=0.3)
plt.title('Train Sequence')

plt.subplot(3,1,2)
plt.plot(val_set, color='b', alpha=0.3)
plt.title('Validation Sequence')

plt.subplot(3,1,3)
plt.plot(test_set, color='r', alpha=0.3)
plt.title('Test Sequence')
plt.xticks(rotation=45)
plt.show()

# Data Normalization
from sklearn.preprocessing import MinMaxScaler

def normalize_dataframe(train_set, val_set, test_set):
    scaler = MinMaxScaler()
    scaler.fit(train_set)

    train = pd.DataFrame(scaler.transform(train_set), columns=train_set.columns, index=train_set.index)
    val = pd.DataFrame(scaler.transform(val_set), columns=val_set.columns, index=val_set.index)
    test = pd.DataFrame(scaler.transform(test_set), columns=test_set.columns, index=test_set.index)
    return train, val, test

train, val, test = normalize_dataframe(train_set, val_set, test_set)

# Standardization
mean = train_set.mean()
std = train_set.std()

train = (train_set - mean) / std
val = (val_set - mean) / std
test = (test_set - mean) / std

# Time Window Splitting
import numpy as np
def prepare_data(data, win_size):
    X = []
    y = []
    for i in range(len(data) - win_size):
        temp_x = data[i:i + win_size]
        temp_y = data[i + win_size]
        X.append(temp_x)
        y.append(temp_y)
    X = np.asarray(X)
    y = np.asarray(y)
    X = np.expand_dims(X, axis=-1)
    return X, y

win_size = 30
X_train, y_train = prepare_data(train['tavg'].values, win_size)
X_val, y_val = prepare_data(val['tavg'].values, win_size)
X_test, y_test = prepare_data(test['tavg'].values, win_size)

print("Train Shape:", X_train.shape, y_train.shape)
print("Validation Shape:", X_val.shape, y_val.shape)
print("Test Shape:", X_test.shape, y_test.shape)

# Convert to PyTorch Tensors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_validation_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_validation_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dataiter = iter(train_loader)
sample_x, sample_y = next(dataiter)
print('Sample input shape:', sample_x.shape)
print('Sample output shape:', sample_y.shape)

# Attention-LSTM Model
model_params = {
    'lstm': {
        'input_size': X_train.shape[2],
        'hidden_size': 256,
        'num_layers': 1,
        'output_size': 1
    },
    'attention': {
        'num_heads': 8
    }
}

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, lstm_output):
        lstm_output = lstm_output.permute(1, 0, 2)
        attn_output, attn_weights = self.attention(lstm_output, lstm_output, lstm_output)
        attn_output = self.dropout(attn_output)
        attn_output = attn_output.permute(1, 0, 2)
        return attn_output, attn_weights

class Attention_LSTM(nn.Module):
    def __init__(self, lstm_params, attention_params):
        super(Attention_LSTM, self).__init__()
        self.hidden_size = lstm_params['hidden_size']
        self.num_layers = lstm_params['num_layers']
        self.lstm = nn.LSTM(lstm_params['input_size'], lstm_params['hidden_size'], lstm_params['num_layers'], batch_first=True)
        self.attention = MultiHeadAttention(lstm_params['hidden_size'], attention_params['num_heads'])
        self.fc1 = nn.Linear(lstm_params['hidden_size'], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, lstm_params['output_size'])
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        attn_out, _ = self.attention(lstm_out)
        out = self.relu(attn_out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)
        return out

lstm_params = model_params['lstm']
attention_params = model_params['attention']
model = Attention_LSTM(lstm_params, attention_params).to(device)
print(model)

# Model Training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 150
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}')

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.show()

# Model Prediction and Evaluation
torch.save(model.state_dict(), 'Attention+lstm.pth')
lstm_model = Attention_LSTM(lstm_params, attention_params).to(device)
lstm_model.load_state_dict(torch.load('Attention+lstm.pth'))
lstm_model.eval()
predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = lstm_model(inputs)
        predictions.extend(outputs.cpu().numpy())
predictions = np.array(predictions)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, np.array([i for arr in predictions for i in arr]))
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, np.array([i for arr in predictions for i in arr]))
from sklearn.metrics import r2_score
r2 = r2_score(y_test, np.array([i for arr in predictions for i in arr]))
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R-squared:", r2)

df_max = np.max(train_set)
df_min = np.min(train_set)
print("df_max:", df_max, "df_min:", df_min)
print("Predictions shape:", predictions.shape)

plt.figure(figsize=(15,4), dpi=300)
plt.subplot(2,1,1)
plt.plot(train_set, color='c', label='Train Set')
plt.plot(val_set, color='r', label='Validation Set')
plt.plot(test_set, color='b', label='Test Set')
plt.plot(pd.date_range(start='2021-03-15', end='2024-05-25', freq='D'),
         predictions * (df_max - df_min) + df_min, color='y', label='Test Predictions')
plt.legend()

plt.subplot(2,1,2)
plt.plot(test_set, color='b', label='Test Set')
plt.plot(pd.date_range(start='2021-03-15', end='2024-05-25', freq='D'),
         predictions * (df_max - df_min) + df_min, color='y', label='Test Predictions')
plt.legend()
plt.show()
