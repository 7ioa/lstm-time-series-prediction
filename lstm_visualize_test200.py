import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns


CONFIG = {
    "seq_len": 168,
    "batch_size": 128,
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


def load_series(csv_path: str, sort_by_time: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Datetime" not in df.columns or "PJME_MW" not in df.columns:
        raise RuntimeError(f"CSV 文件必须包含 'Datetime' 和 'PJME_MW' 两列：{csv_path}")
    df = df.set_index("Datetime")
    df.index = pd.to_datetime(df.index)
    if sort_by_time:
        df = df.sort_index()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize first-step predictions for the first N test samples (or starting at sample_idx)")
    parser.add_argument("--csv", type=str, default="PJME_hourly.csv", help="Path to PJME_hourly.csv")
    parser.add_argument("--ckpt", type=str, default="best_lstm_model.pth", help="Path to saved model checkpoint")
    parser.add_argument("--n", type=int, default=200, help="How many test samples to visualize (default 200)")
    parser.add_argument("--sample_idx", type=int, default=0, help="Start index within test samples (0-based)")
    parser.add_argument("--seq_len", type=int, default=CONFIG["seq_len"], help="Sequence history length")
    parser.add_argument("--sort", action="store_true", help="Sort time series by Datetime before processing (default: no)")
    parser.add_argument("--out_dir", type=str, default="Visualize_test200(16824)", help="Output folder for saved figure")
    parser.add_argument("--fig_w", type=float, default=12.0, help="Figure width (inches)")
    parser.add_argument("--fig_h", type=float, default=9.0, help="Figure height (inches)")
    args = parser.parse_args()

    set_seed(CONFIG["seed"])

    sns.set_theme(style="white", palette="muted", font_scale=1.1)
    plt.rcParams.update({"axes.grid": False})

    t0 = time.time()

    df = load_series(args.csv, sort_by_time=args.sort)
    data = df["PJME_MW"].values.reshape(-1, 1)

    # same split logic as other scripts: 70/10/20
    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    train_raw = data[:train_end]
    test_raw = data[val_end - int(args.seq_len) :]

    # scaler fitted on train only
    scaler = StandardScaler()
    scaler.fit(train_raw)
    test_scaled = scaler.transform(test_raw)

    # load checkpoint and infer pred_len
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到模型文件：{args.ckpt}")

    state = torch.load(str(ckpt_path), map_location=CONFIG["device"])
    if "fc.bias" not in state:
        raise RuntimeError("无法从 checkpoint 推断 pred_len：未找到 fc.bias。")
    pred_len = int(state["fc.bias"].shape[0])

    seq_len = int(args.seq_len)
    X_test, y_test = create_sequences(test_scaled, seq_len, pred_len)
    if len(X_test) == 0:
        raise RuntimeError("测试集样本数为 0：请检查数据长度/seq_len/pred_len 设置。")

    sample_idx = int(args.sample_idx)
    if sample_idx < 0 or sample_idx >= len(X_test):
        raise IndexError(f"sample_idx 越界：应在 [0, {len(X_test)-1}]，但你给的是 {sample_idx}。")

    # determine how many samples we can actually take
    n_want = int(args.n)
    max_available = len(X_test) - sample_idx
    n_take = min(n_want, max_available)
    if n_take <= 0:
        raise RuntimeError("没有可用的测试样本可用来绘图，请调整 sample_idx 或 n 参数。")

    model = LSTMModel(hidden_size=CONFIG["hidden_size"], num_layers=CONFIG["num_layers"], output_size=pred_len).to(CONFIG["device"])
    model.load_state_dict(state)
    model.eval()

    # Batch inference for the slice; only keep first-step prediction
    X_slice = torch.tensor(X_test[sample_idx : sample_idx + n_take], dtype=torch.float32, device=CONFIG["device"])  # (n_take, seq_len, 1)
    with torch.no_grad():
        preds = model(X_slice).cpu().numpy()  # (n_take, pred_len)

    # take first-step predictions
    preds_first_scaled = preds[:, 0].reshape(-1, 1)
    truths_first_scaled = y_test[sample_idx : sample_idx + n_take, 0, :].reshape(-1, 1)  # first step of true future

    preds_first = scaler.inverse_transform(preds_first_scaled).reshape(-1)
    truths_first = scaler.inverse_transform(truths_first_scaled).reshape(-1)

    # x axis: relative sample index (or actual time index from df if desired)
    x_axis = np.arange(n_take)

    pred_color = "#f28e2b"
    truth_color = "#4e79a7"

    plt.figure(figsize=(args.fig_w, args.fig_h))
    plt.plot(x_axis, preds_first, label="Prediction (1-step)", color=pred_color, linewidth=2.2)
    plt.plot(x_axis, truths_first, label="Ground Truth (t+1)", color=truth_color, linewidth=2.2)
    plt.xlabel(f"Sample index (start={sample_idx}, shown={n_take})")
    plt.ylabel("Load (MW)")
    plt.title(f"First-step predictions vs Ground Truth (n={n_take}, start={sample_idx})")
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#333333", framealpha=1.0, fancybox=False)
    plt.grid(False)
    plt.tight_layout()

    out_dir = Path(__file__).parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sample_{sample_idx}_n{n_take}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"完成：已保存 {out_path}, seq_len={seq_len}, pred_len={pred_len}, n_shown={n_take}, 耗时 {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
