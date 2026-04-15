# LSTM时间序列预测项目

基于LSTM神经网络的电力负荷时间序列预测项目，用于预测未来24小时的电力负荷。

## 📋 项目介绍

本项目使用LSTM（长短期记忆网络）对PJME电力负荷数据进行时间序列预测。项目包含完整的数据预处理、模型训练、评估和可视化流程。

## 📊 数据集

本项目使用PJME hourly load dataset（PJM Interconnection East负荷数据）。

**数据集说明：**
- 数据来源：PJM Interconnection East电力公司
- 时间范围：2002-01-01 至 2018-08-03
- 时间粒度：每小时数据
- 数据列：Datetime, PJME_MW（电力负荷，单位：兆瓦）
- 总样本数：约145,362条

**注意：** 由于数据集文件较大，已从仓库中移除。如需使用，请从以下来源获取：
- Kaggle: https://www.kaggle.com/datasets/robikscube/hourly-energy-statistics
- 或联系作者获取数据集

## 🛠️ 环境要求

### 系统要求
- Python 3.8+
- Windows/Linux/macOS

### 依赖包

```bash
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
torch>=1.9.0
```

### 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 📁 项目结构

```
毕设相关/
├── lstm_pre2_standardscale.py    # 主训练脚本
├── pjme_feature_viz.py           # 数据可视化脚本
├── visualize_pjme.py             # 可视化分析脚本
├── .gitignore                    # Git忽略配置
├── requirements.txt              # 依赖包列表
├── README.md                     # 项目说明文档
└── runs/                         # 训练输出目录（自动生成，已忽略）
    └── lstm_standardscale_*      # 每次训练的独立目录
        ├── config.json           # 训练配置
        ├── metrics.json          # 评估指标
        ├── train_history_epoch.csv
        ├── best_lstm_model.pth   # 最佳模型
        └── *.png                 # 可视化图表
```

## 🚀 使用方法

### 1. 数据准备

将`PJME_hourly.csv`文件放在项目根目录下，确保包含以下列：
- `Datetime`: 时间戳
- `PJME_MW`: 电力负荷值

### 2. 数据可视化分析

运行探索性数据分析脚本：

```bash
python pjme_feature_viz.py --csv PJME_hourly.csv --outdir runs/pjme_eda
```

参数说明：
- `--csv`: 输入CSV文件路径
- `--outdir`: 输出目录
- `--width`: 图形宽度（英寸）
- `--height`: 图形高度（英寸）
- `--dpi`: 图形分辨率
- `--heatmap-year`: 指定热力图年份

### 3. 模型训练

运行主训练脚本：

```bash
python lstm_pre2_standardscale.py
```

**训练配置说明：**
- 序列长度（seq_len）：168小时（7天）
- 预测长度（pred_len）：24小时（1天）
- 批大小（batch_size）：128
- 学习率（learning_rate）：0.001
- 隐藏层大小（hidden_size）：64
- LSTM层数（num_layers）：2
- 训练轮数（epochs）：100
- 早停耐心（patience）：10

### 4. 输出结果

训练完成后，结果保存在`runs/lstm_standardscale_*`目录下：

**配置文件：**
- `config.json`: 训练参数配置

**评估指标：**
- `metrics.json`: 测试集评估指标（MAE, RMSE, MAPE）
- `metrics.csv`: 评估指标CSV格式

**模型文件：**
- `best_lstm_model.pth`: 训练好的最佳模型

**训练历史：**
- `train_history_epoch.csv`: 每轮训练记录
- `train_iter_losses.npy`: 迭代损失
- `val_losses.npy`: 验证损失

**可视化图表：**
- `lstm_training_results.png`: 训练结果图
- `lstm_baseline_results.csv`: 预测结果数据

## 📈 模型性能指标

模型在测试集上的评估指标：
- **MAE (Mean Absolute Error)**: 平均绝对误差
- **RMSE (Root Mean Square Error)**: 均方根误差
- **MAPE (Mean Absolute Percentage Error)**: 平均绝对百分比误差

## 🔧 自定义配置

可在`lstm_pre2_standardscale.py`中修改CONFIG字典：

```python
CONFIG = {
    'seq_len': 168,        # 输入序列长度
    'pred_len': 24,       # 预测长度
    'batch_size': 128,
    'learning_rate': 0.001,
    'epochs': 100,
    'hidden_size': 64,
    'num_layers': 2,
    'patience': 10,
    'use_standard_scaler': True,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'seed': 42
}
```

## 📝 数据预处理

1. **时间索引处理**: 将Datetime列转换为datetime类型并设置为索引
2. **重复值处理**: 对相同时间戳的数据取平均
3. **数据划分**: 70%训练集，10%验证集，20%测试集
4. **标准化**: 使用StandardScaler对数据进行标准化
5. **滑动窗口**: 创建序列数据用于LSTM训练

## 🎯 项目特点

- ✅ 完整的时间序列特征分析
- ✅ 支持GPU加速训练
- ✅ 早停机制防止过拟合
- ✅ 详细的训练日志和可视化
- ✅ 可复现性：固定随机种子
- ✅ 模块化代码结构

## 📚 参考资料

- LSTM论文: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
- PyTorch官方文档: https://pytorch.org/
- 时间序列预测教程

## 🤝 作者

- **姓名**: [zxl]
- **学校**: [hust]
- **邮箱**: [3401769334@qq.com]

## 📄 许可证

MIT License

## 🙏 致谢

- PJM Interconnection提供电力数据
- 使用PyTorch框架实现
