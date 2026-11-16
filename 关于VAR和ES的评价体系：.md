

# 总结

1. 无法直接衡量预测 VaR、预测 ES 与真实 VaR、真实 ES 的距离。
2. 评价 VaR 效果——期望分位数损失。
3. 评价 VaR 和 ES 联合效果——FZ 损失。

---

## 1. 目前没有直接衡量预测距离的指标

直接告诉你“算出的 VaR 与真实 VaR 相差多少”，在真实金融数据中，没有这样的通用指标——因为**真实条件 VaR 不可观测**。

---

## 2. 衡量 VaR 预测表现的指标：期望分位数损失

使用分位数损失比较不同模型的 VaR 预测表现，具有理论依据。

**评价过程：**

1. **先给出预测。** 例如预测下一期损失的 99% 分位数，得到 $\widehat{q}_t$。放到金融风险中，这就是预测的 99% VaR。
2. **观察下一期真实结果。** 得到实际损失 $L_t$，注意它不是“真实 VaR”。
3. **根据实际结果给预测评分。**

$$
QL_t = \begin{cases} 
0.99(L_t-\widehat{q}_t), & L_t \ge \widehat{q}_t, \\ 
0.01(\widehat{q}_t-L_t), & L_t < \widehat{q}_t. 
\end{cases}
$$

4. **重复很多期，比较平均得分。**

$$
\overline{QL} = \frac{1}{T}\sum_{t=1}^{T} QL_t.
$$

在相同预测任务下，平均损失更低的预测，在这个准则下表现更好。

---

## 3. 分位数损失为什么能够评价 VaR

> **核心结论：** 真实条件 VaR 能使期望分位数损失最小，因此可以用样本外平均分位数损失评价 VaR 的预测表现。

### 3.1 定义

采用右尾损失口径。设实际损失为 $L$，预测 VaR 为 $q$，置信水平为 $\alpha$。假设损失具有连续分布函数 $F$ 和密度 $f$，且 $E|L| < \infty$。

真实 VaR 是损失分布的 $\alpha$ 分位数：

$$
q_{\alpha} = F^{-1}(\alpha), \qquad F(q_{\alpha}) = \alpha.
$$

分位数损失定义为：

$$
QL_\alpha(L,q) = \begin{cases} 
(1-\alpha)(q-L), & L < q, \\ 
\alpha(L-q), & L \ge q. 
\end{cases}
$$

它同时惩罚风险低估和过度保守；例如 $\alpha = 0.99$ 时，两种情形的权重分别为 $0.99$ 和 $0.01$。

### 3.2 证明：真实分位数使期望损失最小

记期望损失为 $R(q) = E[QL_\alpha(L,q)]$。按实际损失是否超过预测值分开积分：

$$
R(q) = (1-\alpha)\int_{-\infty}^{q}(q-l)f(l)\,dl + \alpha\int_q^{+\infty}(l-q)f(l)\,dl.
$$

对 $q$ 求导。积分边界处均有 $q-l=0$，因此边界项为零，得到：

$$
\begin{aligned}
R'(q) &= (1-\alpha)\int_{-\infty}^{q}f(l)\,dl - \alpha\int_q^{+\infty}f(l)\,dl \\
&= (1-\alpha)F(q) - \alpha[1-F(q)] \\
&= F(q) - \alpha.
\end{aligned}
$$

由于 $F(q)$ 单调不减，$R'(q)$ 也单调不减，故 $R(q)$ 为凸函数：
- 当 $F(q) < \alpha$ 时，$R'(q) < 0$，提高预测值会降低期望损失。
- 当 $F(q) > \alpha$ 时，$R'(q) > 0$，降低预测值会降低期望损失。
- 当 $F(q) = \alpha$ 时，期望损失达到全局最小。

而真实分位数满足 $F(q_{\alpha}) = \alpha$，因此：

$$
q_{\alpha} \in \arg\min_q E[QL_\alpha(L,q)]
$$

如果目标分位数唯一，则最优预测也唯一。上述证明采用连续分布；一般分布下，最小点条件为 $F(q-) \le \alpha \le F(q)$，仍然恰好对应分位数。

### 3.3 如何用于逐日 VaR 评价

金融预测使用截至昨天的信息 $\mathcal{F}_{t-1}$。将上面的 $F$ 换成损失的条件分布，同样得到：

$$
q_{\alpha,t} \in \arg\min_q E[QL_\alpha(L_t,q) \mid \mathcal{F}_{t-1}]
$$

即：**真实条件 VaR 最小化条件期望分位数损失**。

实际评价时，先保存事前预测 $\widehat{q}_t$，再用随后实现的损失 $L_t$ 计算每日得分，并取样本外平均：

$$
\overline{QL} = \frac{1}{T}\sum_{t=1}^{T} QL_\alpha(L_t, \widehat{q}_t).
$$

计算得分只需要预测 VaR 和实际损失，不需要知道真实 VaR。在适当统计条件下，平均得分能够反映期望评分表现；比较模型时，应使用相同日期、置信水平和损失单位，并考虑配对损失差的不确定性。

> **注意：** 较低的样本外平均 QL 支持更好的 VaR 预测表现，但不等于直接测量预测 VaR 与真实 VaR 的距离，也不能仅凭单日得分下结论。

> **依据：** Gneiting, T. (2011). *Quantiles as optimal point forecasts*, 27(2), 197–207. 公式 (2) 定义分位数损失，定理 2.3 给出分位数的最优性。 [原文 DOI](https://doi.org/10.1016/j.ijforecast.2009.12.015)。以上为其普通分位数损失情形的推导。

---

## 4. FZ 损失为什么能够评价 ES

FZ 的理论依据是：**真实 VaR 与真实 ES 的组合能使期望 FZ 损失最小**。因此，它可以用于评价包含 ES 在内的联合风险预测，而不需要事先观测真实 ES。

下面证明一种常用的 FZ0 形式。采用与前文一致的右尾损失口径，$\alpha$ 表示置信水平，$p = 1-\alpha$ 表示尾部概率。

### 4.1 定义与适用条件

设实际损失为 $L$，其分布 $F$ 连续、目标分位数唯一，且 $E|L| < \infty$。真实 VaR 和 ES 分别为：

$$
q_{\alpha} = F^{-1}(\alpha), \qquad e_{\alpha} = E[L \mid L > q_{\alpha}].
$$

因为 $P(L > q_{\alpha}) = p$，所以 ES 也可以表示为：

$$
\begin{aligned}
e_{\alpha} &= q_{\alpha} + E[L - q_{\alpha} \mid L > q_{\alpha}] \\
&= q_{\alpha} + \frac{E[(L - q_{\alpha})_+]}{p}.
\end{aligned}
$$

设预测 VaR 为 $q$、预测 ES 为 $e$。本文使用的右尾 FZ0 损失为：

$$
S_\alpha(L,q,e) = \frac{I\{L > q\}(L-q)}{pe} + \frac{q}{e} + \log e - 1
$$

**这一具体形式要求 $e > 0$，并假设真实 $e_{\alpha} > 0$。** 风险预测还应满足 $e \ge q$。正 ES 的条件针对本例的对数形式，不是所有 FZ 评分的共同定义域要求。

### 4.2 证明：真实 VaR 与 ES 联合最优

定义：

$$
A(q) = q + \frac{E[(L-q)_+]}{p}.
$$

由于 $I\{L > q\}(L-q) = (L-q)_+$，期望 FZ 损失可写为：

$$
R(q,e) = E[S_\alpha(L,q,e)] = \frac{A(q)}{e} + \log e - 1.
$$

**第一步：确定最优 VaR**

有：

$$
E[(L-q)_+] = \int_q^{+\infty}[1-F(l)]\,dl.
$$

因此：

$$
\begin{aligned}
A'(q) &= 1 - \frac{1-F(q)}{p} \\
&= \frac{F(q) - \alpha}{p}.
\end{aligned}
$$

这与前文分位数损失的证明具有相同结构。由于 $F$ 单调不减，$A$ 为凸函数；其最小点为 $q_{\alpha}$，且：

$$
A(q) \ge A(q_{\alpha}) = e_{\alpha}.
$$

对任意固定 $e > 0$，都有 $R(q,e) \ge R(q_{\alpha},e)$。即：固定 ES 预测时，真实 VaR 能使期望 FZ 最小。

**第二步：确定最优 ES**

将 $q = q_{\alpha}$ 代入：

$$
R(q_{\alpha},e) = \frac{e_{\alpha}}{e} + \log e - 1.
$$

对 $e$ 求导：

$$
\frac{\partial R(q_{\alpha},e)}{\partial e} = -\frac{e_{\alpha}}{e^2} + \frac{1}{e} = \frac{e-e_{\alpha}}{e^2}.
$$

由于 $e_{\alpha} > 0$：
- 当 $0 < e < e_{\alpha}$ 时，导数为负，提高 ES 预测会降低期望损失。
- 当 $e > e_{\alpha}$ 时，导数为正，降低 ES 预测会降低期望损失。
- 当 $e = e_{\alpha}$ 时，期望损失达到唯一最小值。

将两步合并：

$$
R(q,e) \ge R(q_{\alpha},e) \ge R(q_{\alpha},e_{\alpha}).
$$

所以：

$$
(q_{\alpha}, e_{\alpha}) = \arg\min_{q,e>0} E[S_\alpha(L,q,e)]
$$

在所述条件下最优组合唯一，且满足 $e_{\alpha} \ge q_{\alpha}$，因此将预测限制为风险上合理的 $e \ge q$ 也不改变这个最优组合。

**这说明 FZ 对 ES 的评价有明确目标：当 VaR 正确时，真实 ES 恰好是期望 FZ 最优的 ES 预测；同时，FZ 的联合最优点就是真实 VaR–ES 组合。**

### 4.3 为什么必须结合 VaR 评价 ES

如果预测 VaR $q$ 不正确，固定该 $q$ 后，期望 FZ 关于 $e$ 的最优点是：

$$
e_{\mathrm{opt}}(q) = A(q) = q + \frac{E[(L-q)_+]}{p},
$$

它不一定等于真实 $e_{\alpha}$。原因是 ES 所对应的尾部范围必须由正确的 VaR 阈值确定。

因此，比较两个模型时，**更低的平均 FZ 支持 VaR–ES 联合预测表现更好，不能直接解释为 ES 单独更准确**；更不能把它当作预测 ES 与真实 ES 的数值距离。只有在双方使用相同、正确的 VaR 等受控条件下，才可以更直接地将差异解释为 ES 预测的评分表现差异。

### 4.4 如何用于逐日预测评价

将以上分布换成给定历史信息 $\mathcal{F}_{t-1}$ 的条件分布，就有：

$$
(q_{\alpha,t}, e_{\alpha,t}) \in \arg\min_{q,e} E[S_\alpha(L_t,q,e) \mid \mathcal{F}_{t-1}],
$$

其中各期需满足相应条件，预测必须在当期损失实现前给出。

实际计算仅需要事前预测 $(\widehat{q}_t, \widehat{e}_t)$ 和随后实现的损失 $L_t$：

$$
\overline{FZ} = \frac{1}{T}\sum_{t=1}^{T} S_\alpha(L_t, \widehat{q}_t, \widehat{e}_t).
$$

在相同日期、置信水平、损失单位和评分形式下，较低的样本外平均 FZ 为更好的联合预测表现提供证据。样本平均反映期望表现需要适当统计条件，有限样本比较还应考虑配对损失差的不确定性。

FZ 得分可以为负，应比较平均得分或配对差值；不宜把普通百分比变化直接当作准确率提升。计算过程中若出现 $e \le 0$，本节公式不适用，不能事后随意截断后仍声称评价的是原预测。

> **文献依据：** 
> - [Fissler、Ziegel 与 Gneiting：ES 与 VaR 的联合可激励性](https://arxiv.org/abs/1507.00244)
> - [Patton、Ziegel 与 Chen（2019），第 2.1 节：VaR–ES 一致评分](https://public.econ.duke.edu/~ap172/Patton_Ziegel_Chen_JoE_2019.pdf)
> - 上述为右尾损失口径下 FZ0 的推导；原文中的收益左尾记号需作相应符号转换。
