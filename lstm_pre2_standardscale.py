import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# 设置风格，让图表更适合论文
sns.set_theme(style='white', palette='muted', font_scale=1.2)
plt.rcParams.update({'axes.grid': False})

# ===========================
# 1. 配置参数 (Configuration)
# ===========================
CONFIG = {
    'seq_len': 168,        # 输入过去 168 小时 (7天)
    'pred_len': 24,       # 预测未来 24 小时 (1天)
    'batch_size': 128,
    'learning_rate': 0.001,
    'epochs': 100,        # 设置大一点，配合 Early Stopping
    'hidden_size': 64,
    'num_layers': 2,
    'patience': 10,       # Early Stopping 耐心轮数
    'use_standard_scaler': True,
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
# 1.1 输出目录 & 日志设置
# ===========================
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path("runs") / f"lstm_standardscale_{run_id}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log_path = OUTPUT_DIR / "train.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("lstm")

# 保存配置，便于复现实验
with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(CONFIG, f, ensure_ascii=False, indent=2, default=str)

# ===========================
# 2. 数据处理与加载
# ===========================
logger.info("正在加载数据...")
# 模拟 PJME 数据 (如果你有真实 csv，请替换这里的读取逻辑)
try:
    df = pd.read_csv('PJME_hourly.csv')
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    data = df['PJME_MW'].values.reshape(-1, 1)
except:
    logger.warning("未找到文件，生成模拟的正弦波+噪声数据用于演示。")
    dates = pd.date_range(start='2010-01-01', periods=20000, freq='H')
    # 模拟明显的日周期和周周期
    values = (np.sin(np.linspace(0, 400, 20000)) * 5000 + 
              np.sin(np.linspace(0, 400/7, 20000)) * 2000 +
              30000 + np.random.normal(0, 500, 20000))
    data = values.reshape(-1, 1)

# 数据集切分 (70% Train, 10% Val, 20% Test)
n = len(data)
train_end = int(n * 0.7)
val_end = int(n * 0.8)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# 滑动窗口函数
def create_sequences(data, seq_len, pred_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len - pred_len):
        x = data[i:(i + seq_len)]
        y = data[(i + seq_len):(i + seq_len + pred_len)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# 使用 StandardScaler 在训练集上 fit，并对 train/val/test 做全局变换
scaler = StandardScaler()
scaler.fit(train_data)
train_data_scaled = scaler.transform(train_data)
val_data_scaled = scaler.transform(val_data)
test_data_scaled = scaler.transform(test_data)

# 创建 Dataset（基于全局 StandardScaler 变换后的数据）
X_train, y_train = create_sequences(train_data_scaled, CONFIG['seq_len'], CONFIG['pred_len'])
X_val, y_val = create_sequences(val_data_scaled, CONFIG['seq_len'], CONFIG['pred_len'])
X_test, y_test = create_sequences(test_data_scaled, CONFIG['seq_len'], CONFIG['pred_len'])

# 转换为 Tensor
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

logger.info(f"训练集样本数: {len(X_train)}, 验证集样本数: {len(X_val)}, 测试集样本数: {len(X_test)}")
logger.info(f"输出目录: {OUTPUT_DIR.resolve()}")

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
val_loss_iters = []
train_iter_losses = []
best_val_loss = float('inf')
patience_counter = 0

logger.info("开始训练...")
start_time = time.time()

global_iter = 0
epoch_rows = []  # 用于保存每个 epoch 的训练记录
for epoch in range(CONFIG['epochs']):
    # --- Training Phase ---
    model.train()
    batch_losses = []
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(CONFIG['device']), batch_y.to(CONFIG['device'])
        # batch_x: (B, seq_len, 1), batch_y: (B, pred_len, 1) -- 使用 StandardScaler 已在数据创建阶段完成变换
        batch_y = batch_y.squeeze(-1) # (B, pred_len)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        global_iter += 1
        train_iter_losses.append(loss.item())
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
    val_loss_iters.append(global_iter)
    
    # --- Early Stopping Check ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_ckpt_path = OUTPUT_DIR / "best_lstm_model.pth"
        torch.save(model.state_dict(), best_ckpt_path) # 保存最佳模型
        logger.info(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} *")
    else:
        patience_counter += 1
        logger.info(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
    epoch_rows.append({
        "epoch": epoch + 1,
        "global_iter": global_iter,
        "train_loss": float(avg_train_loss),
        "val_loss": float(avg_val_loss),
        "best_val_loss_so_far": float(best_val_loss),
        "patience_counter": int(patience_counter),
        "is_best": bool(avg_val_loss <= best_val_loss + 1e-12),
    })

    if patience_counter >= CONFIG['patience']:
        logger.info("Early stopping triggered!")
        break

elapsed = time.time() - start_time
logger.info(f"训练结束，总耗时: {elapsed:.2f}s，总迭代次数: {global_iter}")

# 落盘训练曲线（epoch 级 / iteration 级）与训练日志表
pd.DataFrame(epoch_rows).to_csv(OUTPUT_DIR / "train_history_epoch.csv", index=False, encoding="utf-8-sig")
np.save(OUTPUT_DIR / "train_iter_losses.npy", np.array(train_iter_losses, dtype=np.float32))
np.save(OUTPUT_DIR / "val_losses.npy", np.array(val_losses, dtype=np.float32))
np.save(OUTPUT_DIR / "val_loss_iters.npy", np.array(val_loss_iters, dtype=np.int64))

# ===========================
# 5. 测试与结果可视化
# ===========================
# 加载最佳模型进行测试
best_ckpt_path = OUTPUT_DIR / "best_lstm_model.pth"
model.load_state_dict(torch.load(best_ckpt_path, map_location=CONFIG['device']))
model.eval()

predictions = []
ground_truth = []


with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(CONFIG['device'])
        batch_y = batch_y.to(CONFIG['device'])

        outputs = model(batch_x)  # (B, pred_len)
        predictions.append(outputs.cpu().numpy())
        ground_truth.append(batch_y.squeeze(-1).cpu().numpy())

# 拼接，当前仍在 scaler 变换后的空间
pred_array = np.concatenate(predictions, axis=0)
real_array = np.concatenate(ground_truth, axis=0)

# 归一化空间下的指标（基于 StandardScaler 变换后的数据）
mse_norm = np.mean((pred_array - real_array) ** 2)
mae_norm = np.mean(np.abs(pred_array - real_array))

print(f"\n===== 归一化空间下的时序预测指标 =====")
print(f"MSE (normalized): {mse_norm:.6f}")
print(f"MAE (normalized): {mae_norm:.6f}")

# 原始 MW 尺度：使用 StandardScaler 做反变换
pred_inv = scaler.inverse_transform(pred_array.reshape(-1, 1)).reshape(pred_array.shape)
real_inv = scaler.inverse_transform(real_array.reshape(-1, 1)).reshape(real_array.shape)

# 计算原始尺度下的指标
mae = np.mean(np.abs(pred_inv - real_inv))
rmse = np.sqrt(np.mean((pred_inv - real_inv)**2))
eps = 1e-8
mape = np.mean(np.abs((real_inv - pred_inv) / (np.abs(real_inv) + eps))) * 100

print(f"\n===== 测试集评估（原始尺度） =====")
print(f"MAE: {mae:.2f} MW")
print(f"RMSE: {rmse:.2f} MW")
print(f"MAPE: {mape:.2f}%")

# 保存评估指标，便于和其他模型（如 Time-LLM）做同尺度对比
metrics = {
    "run_id": run_id,
    "seq_len": CONFIG["seq_len"],
    "pred_len": CONFIG["pred_len"],
    "batch_size": CONFIG["batch_size"],
    "use_standard_scaler": True,
    "epochs_ran": int(len(train_losses)),
    "total_iters": int(global_iter),
    "train_time_sec": float(elapsed),
    "mse_normalized": float(mse_norm),
    "mae_normalized": float(mae_norm),
    "mae_MW": float(mae),
    "rmse_MW": float(rmse),
    "mape_percent": float(mape),
}
with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
logger.info(f"指标已保存: {(OUTPUT_DIR / 'metrics.json').resolve()}")

# ===========================
# 6. 绘图分析 (论文级图表)
# ===========================
plt.figure(figsize=(15, 12))

# --- 图1: 训练 Loss 曲线 ---
plt.subplot(2, 2, 1)
# 为了图像更清晰，对 iteration 进行下采样（最多显示约 1000 个点）
max_points = 1000
iters = np.arange(1, len(train_iter_losses) + 1)
train_iter_losses_np = np.array(train_iter_losses)
if len(train_iter_losses_np) > max_points:
    step = len(train_iter_losses_np) // max_points
    iters_vis = iters[::step]
    train_iter_vis = train_iter_losses_np[::step]
else:
    iters_vis = iters
    train_iter_vis = train_iter_losses_np

plt.plot(iters_vis, train_iter_vis, label='Training Loss (per iteration, downsampled)')
plt.plot(val_loss_iters, val_losses, label='Validation Loss (per epoch-end)', linewidth=2)
plt.title('Training and Validation Loss (Iteration Scale, Downsampled)')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(False)

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
plt.grid(False)

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
plt.grid(False)

# --- 图4: 误差分布直方图 ---
# 检查误差是否呈正态分布
plt.subplot(2, 2, 4)
errors = real_inv - pred_inv
sns.histplot(errors.flatten(), bins=50, kde=True, color='purple')
plt.title('Error Distribution (Residuals)')
plt.xlabel('Error (MW)')
plt.ylabel('Frequency')
plt.grid(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'lstm_training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 额外：保存结果到 csv 方便后续画更漂亮的图
result_df = pd.DataFrame({
    'Real': real_inv[:, 0],
    'Pred': pred_inv[:, 0]
})
result_df.to_csv(OUTPUT_DIR / 'lstm_baseline_results.csv', index=False, encoding="utf-8-sig")
logger.info(f"结果已保存至 {(OUTPUT_DIR / 'lstm_baseline_results.csv').resolve()}")