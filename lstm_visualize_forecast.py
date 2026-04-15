import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns


CONFIG = {
    "seq_len": 168,
    "batch_size": 24,
    "hidden_size": 64,
    "num_layers": 2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "seed": 42,
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out


def create_sequences(data: np.ndarray, seq_len: int, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(data) - seq_len - pred_len):
        x = data[i : (i + seq_len)]
        y = data[(i + seq_len) : (i + seq_len + pred_len)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def load_series(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df = df.set_index("Datetime")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df["PJME_MW"].values.reshape(-1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one LSTM forecast: seq_len history + pred_len forecast vs true future.")
    parser.add_argument("--csv", type=str, default="PJME_hourly.csv", help="Path to PJME_hourly.csv")
    parser.add_argument("--ckpt", type=str, default="best_lstm_model.pth", help="Path to saved model checkpoint")
    parser.add_argument("--sample_idx", type=int, default=0, help="Which test sample to visualize (0-based)")
    parser.add_argument("--seq_len", type=int, default=CONFIG["seq_len"], help="History length (for visualization only)")
    # vis_pred_len <= actual pred_len; if 0, use full pred_len from checkpoint
    parser.add_argument("--vis_pred_len", type=int, default=24, help="How many future steps to show (<= model pred_len). 0 means use full.")
    parser.add_argument("--fig_w", type=float, default=12.0, help="Figure width (inches)")
    parser.add_argument("--fig_h", type=float, default=9.0, help="Figure height (inches). Default is 4:3 ratio with fig_w=12.")
    args = parser.parse_args()

    set_seed(CONFIG["seed"])

    sns.set_theme(style="white", palette="muted", font_scale=1.2)
    plt.rcParams.update({"axes.grid": False})

    # 1) Load raw series
    t0 = time.time()
    try:
        data = load_series(args.csv)
    except Exception:
        raise FileNotFoundError(f"读取数据失败：请确认 `{args.csv}` 存在且包含 Datetime/PJME_MW 两列。")

    # 2) Same split logic as training script (70/10/20 by time)
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    train_raw = data[:train_end]
    test_raw = data[val_end - int(args.seq_len) :]

    # 3) Fit scaler on TRAIN only (same as training script)
    scaler = StandardScaler()
    scaler.fit(train_raw)
    test_scaled = scaler.transform(test_raw)

    # 5) Load model and run ONE sample prediction
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到模型文件：`{args.ckpt}`")

    state = torch.load(str(ckpt_path), map_location=CONFIG["device"])

    # Infer model's pred_len from checkpoint (fc output dim)
    if "fc.bias" not in state:
        raise RuntimeError("无法从 checkpoint 推断 pred_len：未找到 fc.bias。")
    pred_len = int(state["fc.bias"].shape[0])

    seq_len = int(args.seq_len)

    # 4) Build test dataset (scaled space) using seq_len/pred_len
    X_test, y_test = create_sequences(test_scaled, seq_len, pred_len)
    if len(X_test) == 0:
        raise RuntimeError("测试集样本数为 0：请检查数据长度/seq_len/pred_len 设置。")

    sample_idx = int(args.sample_idx)
    if sample_idx < 0 or sample_idx >= len(X_test):
        raise IndexError(f"sample_idx 越界：应在 [0, {len(X_test)-1}]，但你给的是 {sample_idx}。")

    model = LSTMModel(
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        output_size=pred_len,
    ).to(CONFIG["device"])

    model.load_state_dict(state)
    model.eval()

    x = torch.tensor(X_test[sample_idx : sample_idx + 1], dtype=torch.float32, device=CONFIG["device"])  # (1, seq_len, 1)
    y_true = y_test[sample_idx]  # (pred_len, 1) in scaled space

    with torch.no_grad():
        y_pred = model(x).cpu().numpy().reshape(-1, 1)  # (pred_len, 1) in scaled space

    # 6) Inverse transform to original MW scale
    x_hist_inv = scaler.inverse_transform(X_test[sample_idx].reshape(-1, 1)).reshape(seq_len)
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(pred_len)
    y_pred_inv = scaler.inverse_transform(y_pred).reshape(pred_len)

    # 7) Optionally limit how many forecast steps to visualize
    vis_pred_len = int(args.vis_pred_len) if args.vis_pred_len else pred_len
    if vis_pred_len <= 0 or vis_pred_len > pred_len:
        vis_pred_len = pred_len

    y_true_inv_vis = y_true_inv[:vis_pred_len]
    y_pred_inv_vis = y_pred_inv[:vis_pred_len]

    # Compose two lines on the SAME x-axis length = seq_len + vis_pred_len
    total_len = seq_len + vis_pred_len
    x_axis = np.arange(total_len)

    # Model line: history (real) + future (pred, possibly truncated)
    model_line = np.concatenate([x_hist_inv, y_pred_inv_vis], axis=0)

    # Actual line: only the future segment; pad history part with NaN so it aligns to same x-axis
    actual_line = np.concatenate([np.full(seq_len, np.nan), y_true_inv_vis], axis=0)

    # 8) Plot
    # Bright, paper-friendly palette
    pred_color = "#f28e2b"   # bright orange
    truth_color = "#4e79a7"  # bright blue

    # Default fig size is 4:3 ratio (12x9 inches)
    plt.figure(figsize=(args.fig_w, args.fig_h))
    plt.plot(x_axis, model_line, label="Prediction", color=pred_color, linewidth=2.2)
    plt.plot(x_axis, actual_line, label="Ground Truth", color=truth_color, linewidth=2.2)

    plt.title(f"LSTM Forecast Visualization (test sample_idx={sample_idx})")
    plt.xlabel(f"Time steps (seq_len + vis_pred_len = {total_len})")
    plt.ylabel("Load (MW)")
    plt.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#333333",
        framealpha=1.0,
        fancybox=False,
    )
    plt.grid(False)
    plt.tight_layout()

    # Save into ./Visualize folder with filename = sample_idx.png
    out_dir = Path(__file__).parent / "Visualize_168_24"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_idx}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"完成：已保存 `{out_path}`，seq_len={seq_len}, pred_len={pred_len}，耗时 {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()

