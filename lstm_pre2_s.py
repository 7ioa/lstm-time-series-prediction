import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import time

# 设置风格，让图表更适合论文
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

# ===========================
# 1. 配置参数 (Configuration)
# ===========================
CONFIG = {
    'seq_len': 168,        # 输入过去 168 小时 (7天)
    'pred_len': 24,       # 预测未来 24 小时 (1天)
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 100,        # 设置大一点，配合 Early Stopping
    'hidden_size': 64,
    'num_layers': 2,
    'patience': 10,       # Early Stopping 耐心轮数
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'seed': 42
}

# 固定随机种子，保证毕设结果可复现
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG['seed'])

# ===========================
# 2. 数据处理与加载
# ===========================
print("正在加载数据...")
# 模拟 PJME 数据 (如果你有真实 csv，请替换这里的读取逻辑)
try:
    df = pd.read_csv('PJME_hourly.csv')
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    data = df['PJME_MW'].values.reshape(-1, 1)
except:
    print("Warning: 未找到文件，生成模拟的正弦波+噪声数据用于演示。")
    dates = pd.date_range(start='2010-01-01', periods=20000, freq='H')
    # 模拟明显的日周期和周周期
    values = (np.sin(np.linspace(0, 400, 20000)) * 5000 + 
              np.sin(np.linspace(0, 400/7, 20000)) * 2000 +
              30000 + np.random.normal(0, 500, 20000))
    data = values.reshape(-1, 1)

# 归一化（标准化，Z-score）
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 数据集切分 (70% Train, 15% Val, 15% Test)
n = len(data_scaled)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_data = data_scaled[:train_end]
val_data = data_scaled[train_end:val_end]
test_data = data_scaled[val_end:]

# 滑动窗口函数
def create_sequences(data, seq_len, pred_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len - pred_len):
        x = data[i:(i + seq_len)]
        y = data[(i + seq_len):(i + seq_len + pred_len)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 创建 Dataset
X_train, y_train = create_sequences(train_data, CONFIG['seq_len'], CONFIG['pred_len'])
X_val, y_val = create_sequences(val_data, CONFIG['seq_len'], CONFIG['pred_len'])
X_test, y_test = create_sequences(test_data, CONFIG['seq_len'], CONFIG['pred_len'])

# 转换为 Tensor
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

print(f"训练集样本数: {len(X_train)}, 验证集样本数: {len(X_val)}, 测试集样本数: {len(X_test)}")

# ===========================
# 3. 模型定义
# ===========================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, feature)
        # lstm_out shape: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out

model = LSTMModel(hidden_size=CONFIG['hidden_size'], 
                  num_layers=CONFIG['num_layers'], 
                  output_size=CONFIG['pred_len']).to(CONFIG['device'])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

# ===========================
# 4. 训练与早停机制
# ===========================
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0

print("开始训练...")
start_time = time.time()

for epoch in range(CONFIG['epochs']):
    # --- Training Phase ---
    model.train()
    batch_losses = []
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(CONFIG['device']), batch_y.to(CONFIG['device'])
        batch_y = batch_y.squeeze(-1) # 调整形状匹配 (batch, pred_len)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    
    avg_train_loss = np.mean(batch_losses)
    train_losses.append(avg_train_loss)
    
    # --- Validation Phase ---
    model.eval()
    val_batch_losses = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(CONFIG['device']), batch_y.to(CONFIG['device'])
            batch_y = batch_y.squeeze(-1)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_batch_losses.append(loss.item())
    
    avg_val_loss = np.mean(val_batch_losses)
    val_losses.append(avg_val_loss)
    
    # --- Early Stopping Check ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_lstm_model.pth') # 保存最佳模型
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} *")
    else:
        patience_counter += 1
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
    if patience_counter >= CONFIG['patience']:
        print("Early stopping triggered!")
        break

print(f"训练结束，总耗时: {time.time()-start_time:.2f}s")

# ===========================
# 5. 测试与结果可视化
# ===========================
# 加载最佳模型进行测试
model.load_state_dict(torch.load('best_lstm_model.pth', weights_only=True))
model.eval()

predictions = []
ground_truth = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(CONFIG['device'])
        outputs = model(batch_x)
        predictions.append(outputs.cpu().numpy())
        ground_truth.append(batch_y.squeeze(-1).numpy())

# 拼接，当前仍在 [0,1] 的归一化空间
pred_array = np.concatenate(predictions, axis=0)
real_array = np.concatenate(ground_truth, axis=0)

# 在归一化空间 [0,1] 下计算常规时序预测论文中的 MSE / MAE
mse_norm = np.mean((pred_array - real_array) ** 2)
mae_norm = np.mean(np.abs(pred_array - real_array))

print(f"\n===== 归一化空间下的时序预测指标 =====")
print(f"MSE (normalized, 0-1): {mse_norm:.6f}")
print(f"MAE (normalized, 0-1): {mae_norm:.6f}")

# 反归一化到原始 MW 尺度，再计算实体含义更强的 MAE / RMSE / MAPE
pred_inv = scaler.inverse_transform(pred_array)
real_inv = scaler.inverse_transform(real_array)

# 计算原始尺度下的指标
mae = np.mean(np.abs(pred_inv - real_inv))
rmse = np.sqrt(np.mean((pred_inv - real_inv)**2))
mape = np.mean(np.abs((real_inv - pred_inv) / real_inv)) * 100

print(f"\n===== 测试集评估（原始尺度） =====")
print(f"MAE: {mae:.2f} MW")
print(f"RMSE: {rmse:.2f} MW")
print(f"MAPE: {mape:.2f}%")

# ===========================
# 6. 绘图分析 (论文级图表)
# ===========================
plt.figure(figsize=(15, 12))

# --- 图1: 训练 Loss 曲线 ---
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

# --- 图2: 预测对比 (局部细节) ---
# 只取前 200 个样本的第一个预测步长进行展示，太长看不清
plt.subplot(2, 2, 2)
subset_len = 200
plt.plot(real_inv[:subset_len, 0], label='Ground Truth', color='black', alpha=0.7)
plt.plot(pred_inv[:subset_len, 0], label='LSTM Prediction', color='red', linestyle='--')
plt.title(f'Prediction Visualization (First {subset_len} hours)')
plt.xlabel('Time Steps (Hours)')
plt.ylabel('Load (MW)')
plt.legend()
plt.grid(True)

# --- 图3: 散点回归图 ---
# 理想情况下应该所有点都落在对角线上
plt.subplot(2, 2, 3)
plt.scatter(real_inv.flatten(), pred_inv.flatten(), alpha=0.1, s=1, color='blue')
# 画出 y=x 参考线
min_val = min(np.min(real_inv), np.min(pred_inv))
max_val = max(np.max(real_inv), np.max(pred_inv))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title('Scatter Plot: Actual vs Predicted')
plt.xlabel('Actual Load (MW)')
plt.ylabel('Predicted Load (MW)')
plt.grid(True)

# --- 图4: 误差分布直方图 ---
# 检查误差是否呈正态分布
plt.subplot(2, 2, 4)
errors = real_inv - pred_inv
sns.histplot(errors.flatten(), bins=50, kde=True, color='purple')
plt.title('Error Distribution (Residuals)')
plt.xlabel('Error (MW)')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()

# 额外：保存结果到 csv 方便后续画更漂亮的图
result_df = pd.DataFrame({
    'Real': real_inv[:, 0],
    'Pred': pred_inv[:, 0]
})
result_df.to_csv('lstm_baseline_results.csv', index=False)
print("结果已保存至 lstm_baseline_results.csv")