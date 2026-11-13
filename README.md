# 银行 VaR 多模型对比

25个具体模型：17个传统统计/计量模型与8个机器学习模型；机器学习覆盖核方法、树模型、前馈网络和时序网络四类。使用同一数据与评价日期比较每家银行的单期VaR。

**这是一组滚动样本外预测的历史比较，尚不是独立保留测试集上的最终验证。** 过去研究已多次使用该评价段。银行股票自身VaR也不等同于系统性风险度量。

## 模型目录

| 类别 | 模型 |
| --- | --- |
| 历史模拟法 | [Historical Simulation](经典计量经济学模型/01_historical/hs/README.md) |
| 静态参数法 | [Normal](经典计量经济学模型/02_parametric/normal/README.md) |
| 静态参数法 | [Student-t](经典计量经济学模型/02_parametric/student/README.md) |
| 蒙特卡洛模拟 | [Monte Carlo Normal](经典计量经济学模型/03_monte_carlo/mc_normal/README.md) |
| 蒙特卡洛模拟 | [Monte Carlo Student-t](经典计量经济学模型/03_monte_carlo/mc_student/README.md) |
| 条件波动率模型 | [GARCH-Normal](经典计量经济学模型/04_conditional_volatility/garch_normal/README.md) |
| 条件波动率模型 | [GARCH-t](经典计量经济学模型/04_conditional_volatility/garch_t/README.md) |
| 条件波动率模型 | [GARCH-skewt](经典计量经济学模型/04_conditional_volatility/garch_skewt/README.md) |
| 条件波动率模型 | [GJR-GARCH-Normal](经典计量经济学模型/04_conditional_volatility/gjr_normal/README.md) |
| 条件波动率模型 | [GJR-GARCH-t](经典计量经济学模型/04_conditional_volatility/gjr_t/README.md) |
| 条件波动率模型 | [GJR-GARCH-skewt](经典计量经济学模型/04_conditional_volatility/gjr_skewt/README.md) |
| EVT与半参数方法 | [POT-GPD](经典计量经济学模型/05_evt_semiparametric/pot_gpd/README.md) |
| EVT与半参数方法 | [GARCH-t-EVT](经典计量经济学模型/05_evt_semiparametric/garch_t_evt/README.md) |
| EVT与半参数方法 | [GARCH-t-FHS](经典计量经济学模型/05_evt_semiparametric/garch_t_fhs/README.md) |
| 统计：条件分位数模型 | [CAViaR-SAV](经典计量经济学模型/06_conditional_quantile/caviar_sav/README.md) |
| 统计：条件分位数模型 | [CAViaR-AS](经典计量经济学模型/06_conditional_quantile/caviar_as/README.md) |
| 统计：条件分位数模型 | [Linear Quantile Regression](经典计量经济学模型/06_conditional_quantile/quantile_regression/README.md) |
| 机器学习：核方法 | [Nystrom-RBF Quantile](machine_learning/01_kernel/ml_kernel/README.md) |
| 机器学习：树模型 | [Quantile Regression Forest](machine_learning/02_trees/ml_qrf/README.md) |
| 机器学习：树模型 | [Quantile Histogram Boosting](machine_learning/02_trees/ml_hgbr/README.md) |
| 机器学习：前馈神经网络 | [Quantile MLP](machine_learning/03_feedforward/ml_mlp/README.md) |
| 机器学习：时序神经网络 | [Quantile LSTM](machine_learning/04_sequence/ml_lstm/README.md) |
| 机器学习：时序神经网络 | [Quantile GRU](machine_learning/04_sequence/ml_gru/README.md) |
| 机器学习：时序神经网络 | [Quantile TCN](machine_learning/04_sequence/ml_tcn/README.md) |
| 机器学习：时序神经网络 | [Quantile Transformer](machine_learning/04_sequence/ml_transformer/README.md) |

历史模拟只列一次；蒙特卡洛是分位数计算手段，这里的两个版本分别基于高斯和t分布。类别用于组织文件，并非互斥的统计理论分类。

## 总体比较

以下对2575个日期与14家银行等权平均。Pinball loss越低越好；违背率应接近名义1%/5%，并非越低越好。

| 模型                          | 99%损失      | 99%违背率  | 95%损失      | 95%违背率  | 计算耗时（秒） |
| --------------------------- | ---------- | ------- | ---------- | ------- | ------- |
| GARCH-Normal                | 0.04210622 | 1.1012% | 0.14095009 | 3.5201% | 36.96   |
| GJR-GARCH-skewt             | 0.04219116 | 0.8655% | 0.13810227 | 4.9237% | 119.46  |
| GARCH-skewt                 | 0.04219693 | 0.8488% | 0.13817646 | 4.7490% | 81.51   |
| GJR-GARCH-Normal            | 0.04221847 | 1.1207% | 0.14058721 | 3.6366% | 49.24   |
| GARCH-t-EVT                 | 0.04235135 | 0.8017% | 0.13837180 | 4.6352% | 105.66  |
| GARCH-t-FHS                 | 0.04247007 | 0.8738% | 0.13827760 | 4.6741% | 68.58   |
| CAViaR-SAV                  | 0.04290000 | 1.0097% | 0.13938384 | 5.0042% | 72.44   |
| CAViaR-AS                   | 0.04305703 | 1.0402% | 0.13974973 | 5.0763% | 87.80   |
| GJR-GARCH-t                 | 0.04312067 | 0.6241% | 0.13874738 | 4.2302% | 83.51   |
| GARCH-t                     | 0.04313827 | 0.6019% | 0.13895189 | 4.1415% | 53.72   |
| Linear Quantile Regression  | 0.04411922 | 1.0014% | 0.14069156 | 4.7157% | 31.85   |
| Nystrom-RBF Quantile        | 0.04420932 | 1.0763% | 0.14103175 | 5.1956% | 636.26  |
| Quantile GRU                | 0.04481531 | 1.2954% | 0.14392353 | 4.9570% | 607.43  |
| Quantile Histogram Boosting | 0.04549073 | 1.0680% | 0.14214705 | 4.9847% | 544.24  |
| Quantile LSTM               | 0.04638835 | 1.7698% | 0.14617310 | 5.4230% | 781.92  |
| Quantile Regression Forest  | 0.04741145 | 0.7074% | 0.14694957 | 4.1969% | 898.14  |
| Normal                      | 0.04823393 | 1.0069% | 0.16103969 | 2.7573% | 0.32    |
| Monte Carlo Normal          | 0.04826289 | 0.9931% | 0.16104527 | 2.7517% | 1.79    |
| Quantile Transformer        | 0.04914342 | 1.9972% | 0.14907623 | 5.5700% | 688.98  |
| POT-GPD                     | 0.05080048 | 0.6241% | 0.15311033 | 3.8141% | 29.37   |
| Historical Simulation       | 0.05121892 | 0.5853% | 0.15290826 | 3.8835% | 0.24    |
| Student-t                   | 0.05390059 | 0.4910% | 0.15462921 | 3.5118% | 121.82  |
| Monte Carlo Student-t       | 0.05392446 | 0.4910% | 0.15464800 | 3.5035% | 124.76  |
| Quantile TCN                | 0.07403105 | 5.9556% | 0.17054091 | 9.5728% | 282.46  |
| Quantile MLP                | 0.08140110 | 6.5381% | 0.17430110 | 9.8613% | 16.85   |

计算耗时为每个模型完成全部14家银行、两个分位水平的拟合、预测和状态更新的墙钟时间；不含数据读取、指标导出、绘图与报告生成。统计与机器学习分批运行，计算进程数、森林内部线程和编译开销不同；具体设置见summary.json，另有绘图进程。存在资源竞争；CAViaR可能含首次JIT编译。因此仅报告实测耗时，不据此给出严格速度排名。

当前样本中，99% VaR平均分位数损失最低的是 **GARCH-Normal**；95%最低的是 **GJR-GARCH-skewt**。这里只报告样本点估计排名，没有据此声称显著优于其他模型。

[逐银行违背率表](comparison/violation_rates.csv) · [逐银行分位数损失表](comparison/pinball_losses.csv) · [全部指标与覆盖检验](comparison/all_bank_metrics.csv) · [拟合诊断](comparison/fit_diagnostics.csv)

## 数据与回测口径

4575条共同日期、14家银行；评价2016-02-03至2026-10-08，共2575条。前2000条用于初始化/开发；训练窗口1500条；每5条重估参数/经验分布。递归模型在间隔内更新状态，静态模型维持阈值。所有预测在观察当期收益前产生。

[数据及来源说明](data/README.md) · [数学方法、估计细节与限制](docs/方法与评价.md) · [验证记录](docs/验证记录.md)

## 文件结构

```text
README.md          总体比较与入口
comparison/        所有模型的汇总表
data/              共享数据、原始行情与来源说明
经典计量经济学模型/  原统计模型六类
machine_learning/  机器学习四类
code/              完整计算、绘图、检查代码
docs/              统一方法和验证记录
```

每个模型目录包含 `predictions.csv`、`predictions.npy`、`raw_predictions.npy`、`metrics_by_bank.csv`、`summary.json` 和14套PNG/PDF/SVG图。CSV中q_01/q_05是收益分位数；VaR_99/VaR_95是其相反数。NPY维度为[日期,银行,分位水平]，分位顺序[0.01,0.05]；银行/日期顺序见共享NPZ。

## 复现

```bash
python -m pip install -r code/requirements-ml.txt
python code/run_all.py --workers 1
python code/build_report.py
python code/verify_results.py
python code/check_ml.py
```

可用 `--models hs garch_normal` 选择模型，或 `--limit 6 --output outputs/smoke` 做短运行。默认完整重算包含CAViaR与线性规划，需等待。`--refit-every 1` 改为每条重估；改配置后必须重新生成结果，不应继续引用此包的默认排名。Python 3.9–3.12建议使用虚拟环境。

## 机器学习设置

机器学习配置预先固定，未用评价期调参；这是单随机种子、有限训练预算的基准，不代表各算法的最佳可达精度。神经网络为每5条热启动小批量更新，森林/提升树冷启动重训，核模型用固定基点；训练机制不同。详见[机器学习方法](docs/机器学习方法与实验.md)与[机器学习验证](docs/机器学习验证.md)。蒙特卡洛使用固定种子和每次20000条情景，仅单种子结果，数值误差不代表新的模型优势。CAViaR非凸且pinball不光滑，有限多起点不保证全局最优；所有拟合异常和交叉均保留。覆盖检验是逐银行诊断，未做多重检验校正，未拒绝不能证明模型正确。运行耗时含并行竞争与编译影响，只作为运行记录，不能当严格速度基准。数据日期缺口和停牌填零说明见data目录。

目前只完成本地发布包；本次没有改动GitHub远端仓库。上传时将本文件所在文件夹内部内容放在仓库根目录。