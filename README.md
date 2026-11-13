# 滚动窗口 GJR-GARCH-skewt 银行 VaR

独立提取自已有银行风险研究的第50号对照实验。包括完整银行收益数据、原始行情归档、滚动估计代码、参考预测与可复现实验入口。无需原项目目录、PyTorch、API 密钥或联网下载数据。

## 快速运行

Python 3.9–3.12（本次验证环境为3.9.6；依赖版本见 requirements.txt）。在本仓库根目录运行：

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell 使用 .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python run.py --reference reference/predictions.npy
```

完整实验约需数分钟，取决于机器。简短检查：`python run.py --limit 6 --output outputs/smoke`。

每期重新估计最近1500条收益：

```bash
python run.py --refit-every 1 --output outputs/daily_refit
```

后者不是默认对照结果，不应与默认参考预测作逐项一致性断言。
自定义数据可使用 `--data data/bank_returns.csv`；CSV 第一列为日期，其余列为银行，数值必须是百分数对数收益。也支持含 y、dates、banks 的 NPZ。`--window`、`--warmup` 可以设置窗口与首次评价位置。

## 模型与预测顺序

r_t = mu + epsilon_t，epsilon_t = sqrt(h_t) z_t；
h_t = omega + alpha epsilon_(t-1)^2 + gamma I(epsilon_(t-1)<0) epsilon_(t-1)^2 + beta h_(t-1)。

z_t 采用 Hansen 标准化偏 Student-t，单位方差。使用 arch 的 Constant mean / GARCH(p=1,o=1,q=1) / skewt 极大似然数值估计及该库的参数约束。

默认窗口1500条保留记录，前2000条用于历史初始化；从索引2000开始每5条重新估计。每次仅使用目标日期以前的数据。先重估（若到期），再预测，再读取当期收益并更新方差。两次重估之间参数不变，方差每期递推。这是滚动重估版本，不是逐样本 Adam 参数更新版本，也不是每日全历史估计。

首次冷启动，后续用上次参数热启动（最多500次迭代）；不收敛时冷启动重试（最多1500次）。重试仍失败则保留该银行之前参数与当前预测方差，首次失败直接报错。summary.json 显式保存失败和重试。数值解不宣称全局最优。本次默认复算有1个银行重估失败并按原规则回退，详见 docs/PROVENANCE.md 和 reference/summary.json。

预测收益下尾 q_(tau,t) = mu + sqrt(h_t) F_skewt^(-1)(tau)，tau 为0.01和0.05。
`return_quantile` 为收益分位数（通常负）；`var_loss_log_pct=-return_quantile` 为损失口径的99%/95% VaR，单位是百分数对数收益，未转换成货币或简单收益，也不强行截断为非负。超越定义为 r_t < q_t。

## 数据与输出

14家银行，4575个保留日期，2007-10-31至2026-10-08；评价2575期，2016-02-03起。详见 [数据说明](data/README.md)。银行名称和顺序保存在数据中；无宏观特征依赖。

- `data/bank_returns.npz`：精确浮点计算输入。
- `data/bank_returns.csv`：同一数据的可读表格。
- `data/raw/`：归档收益宽表及14家银行后复权行情，用于来源核对。
- `reference/`：原实验预测和本次完整复算结果，包括完整预测 CSV、预测前参数、逐银行指标与重估诊断。
- `outputs/default/predictions.csv`：逐日、逐银行、逐尾部的 VaR、实际收益、超越和 pinball loss。
- `outputs/default/forecast_states.npz`：预测前7个参数和方差。
- `outputs/default/metrics_by_bank.csv` 与 `summary.json`：损失、超越率、运行时间及重估诊断。

原实验默认平均 pinball loss 约为0.042191 / 0.138102，超越率约0.865% / 4.924%。不同平台优化器数值可能有微小差异；严格参考检查失败应检查环境与诊断，不要修改参考预测。

该仓库计算银行边际 VaR，不计算 CoVaR、系统联合尾部或 ES。该历史区间已多轮查看，结果属于探索性历史评价。运行时间不是跨机器固定承诺。

## 上传 GitHub

本文件夹内容就是仓库根目录，可整体上传。已包含 .gitignore 与 GitHub Actions 检查。无需上传 outputs、虚拟环境或缓存。未创建远程仓库、未推送。

代码及数据来源、授权范围见 [来源说明](docs/PROVENANCE.md)；没有擅自为第三方市场数据赋予开源许可证。
