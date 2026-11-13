# 滚动窗口 GJR-GARCH-skewt 银行 VaR

使用14家中国上市银行日收益，滚动估计 GJR-GARCH(1,1) 与 Hansen 偏 Student-t 分布，预测99%和95% VaR。

## 仓库内容

```text
README.md    项目介绍与运行方法
data/        银行收益数据、原始行情和数据说明
docs/        方法与结论（含图表），以及对应结果数据
code/        VaR 计算代码、绘图代码与依赖清单
```

## 方法与结果

默认使用最近1500条收益，每5期重新估计参数，两次重估之间逐期递推方差。每期严格先预测、后揭示当期收益。前2000条记录初始化，后2575条评价。

| 指标 | 99% VaR | 95% VaR |
|---|---:|---:|
| 平均 pinball loss | 0.042191 | 0.138102 |
| 实际违背率 | 0.865% | 4.924% |
| 目标违背率 | 1% | 5% |

文档包含14家银行各自的 VaR 时序图，以及逐银行分位数损失、违背率和违背次数表。

[阅读完整方法、图表与结论](docs/方法与结论.md)。这些是探索性历史回测结果，不构成模型最优或系统性风险指标准确性的证明。

## 运行

在仓库根目录打开终端。Python 3.9–3.12；已验证环境为 Python 3.9.6。

```bash
python -m pip install -r code/requirements.txt
python code/run.py
```

无需安装本项目为 Python 包，也无需联网下载数据。默认结果保存至 `outputs/default/`（运行后生成）。包括逐日预测 CSV、预测前模型参数、逐银行损失与违背率，以及拟合诊断。

快速检查（只计算6个评价日期）：

```bash
python code/run.py --limit 6 --output outputs/smoke
```

完整复算并与本仓库参考预测比较：

```bash
python code/run.py --reference docs/results/predictions.npy
```

每期重新估计最近1500条收益：

```bash
python code/run.py --refit-every 1 --output outputs/daily_refit
```

每期重估是另一种配置，不与默认每5期重估结果混用。可用 `--window`、`--warmup` 设置窗口和首次评价位置；用 `--data` 指定符合格式的 CSV/NPZ。完整计算耗时取决于机器，本次复算约93秒。

重画文档中的图：

```bash
python -m pip install -r code/requirements-figures.txt
python code/plot_results.py
```

## 数据与口径

数据来自既有 BaoStock 行情归档。收益单位为百分数对数收益，-1表示-1%的对数收益；输出 `return_quantile` 为收益下尾分位数，`var_loss_log_pct` 为它的相反数。数据清洗、日期缺口和停牌处理见 [数据说明](data/README.md)。

此项目仅计算银行边际 VaR，不计算 CoVaR、系统联合风险或 ES。第三方数据再分发许可未在本次整理中独立核实；不对第三方数据赋予开源许可证。本项目也尚未指定代码开源许可证。
