#  这是文章上的程序源码，放在这里方便查验
#  Pythorch框架构建Attention-lstm时序模型（https://mp.weixin.qq.com/s/Vs96H2hQy60qdRicO5ZRbg）


#  数据调取
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Stations, Daily
plt.rcParams['font.sans-serif'] = 'SimHei' # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
chongqing = (29.56301, 106.55156)
# 以下是原数据读取方式
# # 查找重庆附近的气象站
# stations = Stations()
# nearby_stations = stations.nearby(*chongqing).fetch(5)
#
# # 获取最近的气象站的ID
# station_id = nearby_stations.index[0]
#
# # 设置时间范围
# start = datetime(2008, 1, 1)
# end = datetime(2024, 5, 25)
#
# # 获取每日数据
# data = Daily(station_id, start, end)
# data = data.fetch()
# data.head()


# 数据划分

import pandas as pd

# 我们的数据读取方式
data = pd.read_csv('./Data/day/Sensonal_Five_year_Origin.csv')


df = pd.DataFrame()
df['tavg'] = data['A']

# 定义划分比例
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# 计算划分的索引
train_split = int(train_ratio * len(df))
val_split = int((train_ratio + val_ratio) * len(df))

# 划分数据集
train_set = df.iloc[:train_split]
val_set = df.iloc[train_split:val_split]
test_set = df.iloc[val_split:]

plt.figure(figsize=(15, 10))
plt.subplot(3,1,1)
plt.plot(train_set, color='c',  alpha=0.3)
plt.title('train时序图')

plt.subplot(3,1,2)
plt.plot(val_set, color='b',  alpha=0.3)
plt.title('val时序图')

plt.subplot(3,1,3)
plt.plot(test_set, color='r',  alpha=0.3)
plt.title('test时序图')
plt.xticks(rotation=45)
plt.show()


#  数据归一化

from sklearn.preprocessing import MinMaxScaler


def normalize_dataframe(train_set, val_set, test_set):
    scaler = MinMaxScaler()
    scaler.fit(train_set)  # 在训练集上拟合归一化模型

    train = pd.DataFrame(scaler.transform(train_set), columns=train_set.columns, index=train_set.index)
    val = pd.DataFrame(scaler.transform(val_set), columns=val_set.columns, index=val_set.index)
    test = pd.DataFrame(scaler.transform(test_set), columns=test_set.columns, index=test_set.index)
    return train, val, test


train, val, test = normalize_dataframe(train_set, val_set, test_set)

# 数据标准化
# 计算均值和标准差
mean = train_set.mean()
std = train_set.std()

# 对数据进行标准化
train = (train_set - mean) / std
val = (val_set - mean) / std
test = (test_set - mean) / std


#  时间窗口划分
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

# 训练集
X_train, y_train= prepare_data(train['tavg'].values, win_size)

# 验证集
X_val, y_val= prepare_data(val['tavg'].values, win_size)

# 测试集
X_test, y_test = prepare_data(test['tavg'].values, win_size)

print("训练集形状:", X_train.shape, y_train.shape)
print("验证集形状:", X_val.shape, y_val.shape)
print("测试集形状:", X_test.shape, y_test.shape)


#  数据集转换为 PyTorch 张量
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset

#device 表示了一个用于存储和计算张量的设备。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# 检查是否有可用的 GPU
# 将NumPy数组转换为PyTorch张量
#将 numpy 数组 X_train_ts 转换为 PyTorch 的张量，并指定数据类型为 torch.float32，将张亮放置在指定的设备上进行存储和计算
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

X_validation_tensor=torch.tensor(X_val, dtype=torch.float32).to(device)
y_validation_tensor= torch.tensor(y_val,dtype=torch.float32).to(device)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 创建训练集、验证集和测试集数据集
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 定义批量大小
batch_size = 64 #批量大小，算力越强，可以设置越大，可自定义 ，常见的批量大小通常在32到256之间

# 创建数据加载器 shuffle=True 表示在每个 epoch 开始时将数据打乱
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 打印训练数据形状
dataiter = iter(train_loader)
sample_x, sample_y = next(dataiter)  # 修改这里，使用next方法手动获取一个批次的数据
print('Sample input shape: ', sample_x.shape)
print('Sample output shape: ', sample_y.shape)


#  Attention-lstm模型构建
# 定义模型参数字典
model_params = {
    'lstm': {
        'input_size': X_train.shape[2],  # 输入特征维度
        'hidden_size': 256,              # LSTM隐藏层维度
        'num_layers': 1,                 # LSTM层数
        'output_size': 1                 # 输出维度
    },
    'attention': {
        'num_heads': 8                   # 注意力头数
    }
}

# 定义多头注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        # 定义多头注意力层
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)

    def forward(self, lstm_output):
        # lstm_output 形状: (batch_size, seq_length, hidden_size)
        # MultiheadAttention 期望的输入形状: (seq_length, batch_size, hidden_size)
        lstm_output = lstm_output.permute(1, 0, 2)  # 转置维度
        attn_output, attn_weights = self.attention(lstm_output, lstm_output, lstm_output)
        attn_output = attn_output.permute(1, 0, 2)  # 转置回原来的维度
        return attn_output, attn_weights

# 定义 Attention_LSTM 模型
class Attention_LSTM(nn.Module):
    def __init__(self, lstm_params, attention_params):
        super(Attention_LSTM, self).__init__()
        self.hidden_size = lstm_params['hidden_size']
        self.num_layers = lstm_params['num_layers']
        # 定义LSTM层
        self.lstm = nn.LSTM(lstm_params['input_size'], lstm_params['hidden_size'], lstm_params['num_layers'], batch_first=True)
        # 定义多头注意力层
        self.attention = MultiHeadAttention(lstm_params['hidden_size'], attention_params['num_heads'])
        # 定义全连接层
        self.fc1 = nn.Linear(lstm_params['hidden_size'], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, lstm_params['output_size'])
        self.relu = nn.ReLU()  # 激活函数ReLU

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))

        # 应用多头注意力层
        attn_out, _ = self.attention(lstm_out)

        # 取最后一个时间步的输出
        out = self.relu(attn_out[:, -1, :])

        # 全连接层前向传播
        out = self.relu(self.fc1(out))  # 全连接层1
        out = self.relu(self.fc2(out))  # 全连接层2
        out = self.relu(self.fc3(out))  # 全连接层3
        out = self.relu(self.fc4(out))  # 全连接层4
        out = self.fc5(out)  # 输出层
        return out

# 模型参数
lstm_params = model_params['lstm']
attention_params = model_params['attention']
# 实例化模型
model = Attention_LSTM(lstm_params, attention_params).to(device)
print(model)


#  采用随机失活层Dropout()，避免使用Attention时产生过拟合
# 定义多头注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        # 定义多头注意力层
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.dropout = nn.Dropout(p=0.1)  # Dropout层，防止过拟合

    def forward(self, lstm_output):
        # lstm_output 形状: (batch_size, seq_length, hidden_size)
        # MultiheadAttention 期望的输入形状: (seq_length, batch_size, hidden_size)
        lstm_output = lstm_output.permute(1, 0, 2)  # 转置维度
        attn_output, attn_weights = self.attention(lstm_output, lstm_output, lstm_output)
        attn_output = self.dropout(attn_output)  # 应用Dropout
        attn_output = attn_output.permute(1, 0, 2)  # 转置回原来的维度
        return attn_output, attn_weights

#  模型训练
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
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

# 绘制损失曲线
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.show()


#模型预测评价

# 保存模型
torch.save(model.state_dict(), 'Attention+lstm.pth')
# 调用模型
lstm_model = Attention_LSTM(lstm_params, attention_params).to(device)
lstm_model.load_state_dict(torch.load('Attention+lstm.pth'))
lstm_model.eval()
# 在测试集上进行预测
predictions = []
lstm_model.eval()
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = lstm_model(inputs)
        predictions.extend(outputs.cpu().numpy())
predictions = np.array(predictions)
# print("predictions数据形状：",predictions.shape)
# print(predictions)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, np.array([i for arr in predictions for i in arr]))
rmse = np.sqrt(mse)
mae = metrics.mean_absolute_error(y_test, np.array([i for arr in predictions for i in arr]))
from sklearn.metrics import r2_score
r2 = r2_score(y_test, np.array([i for arr in predictions for i in arr]))
print("均方误差 (MSE):", mse)
print("均方根误差 (RMSE):", rmse)
print("平均绝对误差 (MAE):", mae)
print("拟合优度:", r2)

# 可视化
df_max = np.max(train_set)
df_min = np.min(train_set)
# df_max与df_min值异常，无法获取。手动重写
# df_max = 0.39569  #我改了
# df_min = -0.29564
print("df_max值：", df_min,"df_min值：", df_min)
print("predictions数据形状：", predictions.shape)
# print("predictions的长：", len(predictions))
# print(predictions)
# plt.figure(figsize=(15,4), dpi =300)
plt.subplot(2,1,1)
plt.rcParams.update({'font.size': 10})


print("train_set数据形状：", train_set.shape)
print(train_set)
print("val_set数据形状：", val_set.shape)
# print(val_set)
print("test_set数据形状：", test_set.shape)
# print(test_set)

plt.plot(train_set, color = 'c', label = '训练集')
plt.plot(val_set, color = 'r', label = '验证集')
plt.plot(test_set, color = 'b', label = '测试集')
# 下行代码中注意修改all_data值为数据量总数
all_data = 43824
plt.plot(np.arange(all_data-len(predictions), all_data)
         ,predictions*(df_max-df_min)+df_min, color = 'y', label = '测试集预测')

plt.legend()

plt.subplot(2,1,2)
plt.plot(test_set, color = 'b', label = '测试集')
plt.plot(np.arange(all_data-len(predictions), all_data)
         ,predictions*(df_max-df_min)+df_min, color = 'y', label = '测试集预测')

plt.legend()
plt.show()
# 输出预测数据csv
predictions = predictions*(df_max-df_min)+df_min
np.savetxt('predictions.csv', predictions, delimiter='\n')