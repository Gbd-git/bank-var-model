"""Build Chinese reports, comparison tables and one full time-series per bank/model.

Figure contract: show whether each bank's realized returns cross its model's
two predicted thresholds. All dates and banks retained; no uncertainty bars
because these are realized series, not replicate means. One axis per figure;
multi-panel alignment is not applicable. Adapted from the project's own plot.
"""
import argparse,json,hashlib,os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from models import CATALOG, model_relative_path
from run_all import ROOT,tables
NAMES=['Ping An','Ningbo','Shanghai Pudong','Huaxia','Minsheng','China Merchants',
       'Nanjing','Industrial','Beijing','Bank of Communications','ICBC',
       'China Construction','Bank of China','CITIC']
CATEGORIES={'01_historical':'历史模拟法','02_parametric':'静态参数法','03_monte_carlo':'蒙特卡洛模拟',
    '04_conditional_volatility':'条件波动率模型','05_evt_semiparametric':'EVT与半参数方法','06_conditional_quantile':'统计：条件分位数模型',
    'machine_learning/01_kernel':'机器学习：核方法','machine_learning/02_trees':'机器学习：树模型',
    'machine_learning/03_feedforward':'机器学习：前馈神经网络','machine_learning/04_sequence':'机器学习：时序神经网络'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.8,
    'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})

def md_table(df):
    # No optional tabulate dependency.
    cols=list(df.columns);lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in df.itertuples(index=False,name=None):lines.append('| '+' | '.join(str(x) for x in row)+' |')
    return '\n'.join(lines)

def plot(folder,q,y,dates,label):
    folder.mkdir(exist_ok=True)
    for b,name in enumerate(NAMES):
        fig,ax=plt.subplots(figsize=(7.2,3.5));fig.subplots_adjust(left=.11,right=.98,bottom=.17,top=.77)
        ax.plot(dates,y[:,b],color='#AAAAAA',linewidth=.4,label='Observed return',zorder=1)
        ax.plot(dates,q[:,b,1],color='#D18A36',linewidth=.8,linestyle='--',label='95% VaR threshold',zorder=2)
        ax.plot(dates,q[:,b,0],color='#285C86',linewidth=.9,label='99% VaR threshold',zorder=3)
        ax.set_ylabel('Log return / threshold (%)');ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(mdates.YearLocator(2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.margins(x=.01);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        fig.suptitle(f'{name}: {label} VaR',y=.98,fontsize=11)
        fig.legend(loc='upper center',bbox_to_anchor=(.54,.91),ncol=3,fontsize=8)
        # Single plot-area: no multi-panel alignment constraint.
        fig.savefig(folder/f'var_{b+1:02d}.png',dpi=600)
        fig.savefig(folder/f'var_{b+1:02d}.pdf');fig.savefig(folder/f'var_{b+1:02d}.svg');plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-plots',action='store_true');parser.add_argument('--plot-missing',action='store_true');args=parser.parse_args()
    d=np.load(ROOT/'data/bank_returns.npz',allow_pickle=False);banks=d['banks']
    methods=json.loads((ROOT/'code/methods.json').read_text());allmetrics=[];agg=[];statuses=[]
    protocols=[]
    for m,(cat,_) in CATALOG.items():
        s=json.loads((ROOT/model_relative_path(m)/'summary.json').read_text())
        protocols.append((s['window'],s['warmup'],s['refit_every'],s['start'],s['end'],s['data_sha256']))
    if len(set(protocols))!=1:raise ValueError('Model protocols differ; do not merge unmatched evaluations')
    for m,(cat,label) in CATALOG.items():
        folder=ROOT/model_relative_path(m);s=json.loads((folder/'summary.json').read_text());start=s['warmup']
        q=np.load(folder/'predictions.npy');raw=np.load(folder/'raw_predictions.npy');y=d['y'][start:];dates=d['dates'][start:].astype('datetime64[D]')
        assert q.shape==(*y.shape,2) and np.isfinite(q).all() and np.all(q[:,:,0]<=q[:,:,1])
        frame,ar=tables(y,q,banks,m);frame.to_csv(folder/'metrics_by_bank.csv',index=False);allmetrics.append(frame)
        for row in ar:row['elapsed_seconds']=s['elapsed_seconds']
        agg+=ar
        events=s['diagnostics']['events'];count={}
        for event in events:
            kind=event.get('kind','carry_previous_fit');count[kind]=count.get(kind,0)+1
        crossings=int((raw[:,:,0]>raw[:,:,1]).sum());assert crossings==s['diagnostics']['crossings']
        statuses.append(dict(model=m,crossings=crossings,events=len(events),details=json.dumps(count,ensure_ascii=False)))
        pred=pd.read_csv(folder/'predictions.csv');pred['VaR_99']=-pred.q_01;pred['VaR_95']=-pred.q_05;pred.to_csv(folder/'predictions.csv',index=False)
        if not args.no_plots and (not args.plot_missing or len(list((folder/'figures').glob('*.pdf')))!=14):plot(folder/'figures',q,y,dates,label)
        tab=frame[['bank','confidence','violations','violation_rate','pinball_loss']].copy()
        tab.columns=['银行','置信度','违背次数','违背率','Pinball loss']
        tab['违背率']=tab['违背率'].map(lambda v:f'{v:.4%}');tab['置信度']=tab['置信度'].map(lambda v:f'{v:.0%}')
        tab['Pinball loss']=tab['Pinball loss'].map(lambda v:f'{v:.10f}')
        content=f'# {label}\n\n## 方法\n\n{methods[m]}\n\n'
        doc='机器学习方法与实验.md' if m.startswith('ml_') else '方法与评价.md'
        rel=os.path.relpath(ROOT/'docs'/doc,folder)
        content+=f'统一窗口、预测时序、指标公式和限制见[方法总说明]({rel})。\n\n'
        content+=f'## 结果\n\n完整评价 {len(q)} 个日期、14家银行。原始分位数交叉 {crossings} 个银行日期；评分与绘图使用按分位水平排序后的预测。拟合事件：`{count}`。详见 [summary.json](summary.json)。\n\n'
        content+=f'| 计算范围 | 本次计算耗时（秒） |\n| --- | ---: |\n| 全部14家银行、两个分位水平、完整评价期的拟合与预测 | {s["elapsed_seconds"]:.2f} |\n\n耗时不含数据读取、指标表导出、绘图和报告生成；包含递归更新和可能的首次JIT编译。多进程运行存在资源竞争，不作严格速度排名；没有把总耗时均分后伪装成逐银行实测耗时。\n\n'
        content+=md_table(tab)+'\n\n[完整指标及覆盖检验](metrics_by_bank.csv) · [逐日预测与收益](predictions.csv)\n\n'
        content+='## 每家银行的 VaR 时序图\n\n灰线为实际收益；蓝线和橙色虚线为99%与95% VaR对应的收益下尾阈值，损失VaR是其相反数。所有评价日期均保留，不绘制总体违背率图。\n\n'
        for b,bank in enumerate(banks):content+=f'### {bank}\n\n![{bank} VaR](figures/var_{b+1:02d}.png)\n\n[PDF](figures/var_{b+1:02d}.pdf) · [SVG](figures/var_{b+1:02d}.svg)\n\n'
        (folder/'README.md').write_text(content)
        print('Report/plots completed:',m,flush=True)
    aggregate=pd.DataFrame(agg);metrics=pd.concat(allmetrics,ignore_index=True)
    dest=ROOT/'comparison';dest.mkdir(exist_ok=True)
    aggregate.to_csv(dest/'overall_metrics.csv',index=False);metrics.to_csv(dest/'all_bank_metrics.csv',index=False)
    aggregate[aggregate.model.str.startswith('ml_')].to_csv(dest/'machine_learning_metrics.csv',index=False)
    aggregate[~aggregate.model.str.startswith('ml_')].to_csv(dest/'statistical_metrics.csv',index=False)
    pd.DataFrame(statuses).to_csv(dest/'fit_diagnostics.csv',index=False)
    metrics.pivot(index=['bank','confidence'],columns='model',values='violation_rate').to_csv(dest/'violation_rates.csv')
    metrics.pivot(index=['bank','confidence'],columns='model',values='pinball_loss').to_csv(dest/'pinball_losses.csv')
    parts=['# 银行 VaR 多模型对比\n','25个具体模型：17个传统统计/计量模型与8个机器学习模型；机器学习覆盖核方法、树模型、前馈网络和时序网络四类。使用同一数据与评价日期比较每家银行的单期VaR。\n',
      '**这是一组滚动样本外预测的历史比较，尚不是独立保留测试集上的最终验证。** 过去研究已多次使用该评价段。银行股票自身VaR也不等同于系统性风险度量。\n',
      '## 模型目录\n']
    listing=[]
    for m,(cat,label) in CATALOG.items():listing.append({'类别':CATEGORIES[cat],'模型':f'[{label}]({model_relative_path(m)}/README.md)'})
    parts+=[md_table(pd.DataFrame(listing)),'\n历史模拟只列一次；蒙特卡洛是分位数计算手段，这里的两个版本分别基于高斯和t分布。类别用于组织文件，并非互斥的统计理论分类。\n',
      '## 总体比较\n','以下对2575个日期与14家银行等权平均。Pinball loss越低越好；违背率应接近名义1%/5%，并非越低越好。\n']
    wide=aggregate.pivot(index='model',columns='confidence',values=['pinball_loss','violation_rate'])
    result=[]
    for m in wide.sort_values(('pinball_loss',.99)).index:
        result.append({'模型':CATALOG[m][1],'99%损失':f'{wide.loc[m,("pinball_loss",.99)]:.8f}',
        '99%违背率':f'{wide.loc[m,("violation_rate",.99)]:.4%}','95%损失':f'{wide.loc[m,("pinball_loss",.95)]:.8f}',
        '95%违背率':f'{wide.loc[m,("violation_rate",.95)]:.4%}',
        '计算耗时（秒）':f'{aggregate.loc[aggregate.model==m,"elapsed_seconds"].iloc[0]:.2f}'})
    parts+=[md_table(pd.DataFrame(result)),'\n计算耗时为每个模型完成全部14家银行、两个分位水平的拟合、预测和状态更新的墙钟时间；不含数据读取、指标导出、绘图与报告生成。统计与机器学习分批运行，计算进程数、森林内部线程和编译开销不同；具体设置见summary.json，另有绘图进程。存在资源竞争；CAViaR可能含首次JIT编译。因此仅报告实测耗时，不据此给出严格速度排名。\n']
    best99=aggregate[aggregate.confidence==.99].sort_values('pinball_loss').iloc[0]
    best95=aggregate[aggregate.confidence==.95].sort_values('pinball_loss').iloc[0]
    parts += [f'当前样本中，99% VaR平均分位数损失最低的是 **{CATALOG[best99.model][1]}**；95%最低的是 **{CATALOG[best95.model][1]}**。这里只报告样本点估计排名，没有据此声称显著优于其他模型。\n',
      '[逐银行违背率表](comparison/violation_rates.csv) · [逐银行分位数损失表](comparison/pinball_losses.csv) · [全部指标与覆盖检验](comparison/all_bank_metrics.csv) · [拟合诊断](comparison/fit_diagnostics.csv)\n',
      '## 数据与回测口径\n','4575条共同日期、14家银行；评价2016-02-03至2026-10-08，共2575条。前2000条用于初始化/开发；训练窗口1500条；每5条重估参数/经验分布。递归模型在间隔内更新状态，静态模型维持阈值。所有预测在观察当期收益前产生。\n',
      '[数据及来源说明](data/README.md) · [数学方法、估计细节与限制](docs/方法与评价.md) · [验证记录](docs/验证记录.md)\n',
      '## 文件结构\n','```text\nREADME.md          总体比较与入口\ncomparison/        所有模型的汇总表\ndata/              共享数据、原始行情与来源说明\n经典计量经济学模型/  原统计模型六类\nmachine_learning/  机器学习四类\ncode/              完整计算、绘图、检查代码\ndocs/              统一方法和验证记录\n```\n',
      '每个模型目录包含 `predictions.csv`、`predictions.npy`、`raw_predictions.npy`、`metrics_by_bank.csv`、`summary.json` 和14套PNG/PDF/SVG图。CSV中q_01/q_05是收益分位数；VaR_99/VaR_95是其相反数。NPY维度为[日期,银行,分位水平]，分位顺序[0.01,0.05]；银行/日期顺序见共享NPZ。\n',
      '## 复现\n','```bash\npython -m pip install -r code/requirements-ml.txt\npython code/run_all.py --workers 1\npython code/build_report.py\npython code/verify_results.py\npython code/check_ml.py\n```\n',
      '可用 `--models hs garch_normal` 选择模型，或 `--limit 6 --output outputs/smoke` 做短运行。默认完整重算包含CAViaR与线性规划，需等待。`--refit-every 1` 改为每条重估；改配置后必须重新生成结果，不应继续引用此包的默认排名。Python 3.9–3.12建议使用虚拟环境。\n',
      '## 机器学习设置\n','机器学习配置预先固定，未用评价期调参；这是单随机种子、有限训练预算的基准，不代表各算法的最佳可达精度。神经网络为每5条热启动小批量更新，森林/提升树冷启动重训，核模型用固定基点；训练机制不同。详见[机器学习方法](docs/机器学习方法与实验.md)与[机器学习验证](docs/机器学习验证.md)。蒙特卡洛使用固定种子和每次20000条情景，仅单种子结果，数值误差不代表新的模型优势。CAViaR非凸且pinball不光滑，有限多起点不保证全局最优；所有拟合异常和交叉均保留。覆盖检验是逐银行诊断，未做多重检验校正，未拒绝不能证明模型正确。运行耗时含并行竞争与编译影响，只作为运行记录，不能当严格速度基准。数据日期缺口和停牌填零说明见data目录。\n',
      '目前只完成本地发布包；本次没有改动GitHub远端仓库。上传时将本文件所在文件夹内部内容放在仓库根目录。']
    (ROOT/'README.md').write_text('\n'.join(parts))
    ml_names={label for m,(_,label) in CATALOG.items() if m.startswith('ml_')}
    ml_rows=[row for row in result if row['模型'] in ml_names]
    findings=[]
    for confidence in (.99,.95):
        group=aggregate[np.isclose(aggregate.confidence,confidence)]
        best=group[group.model.str.startswith('ml_')].sort_values('pinball_loss').iloc[0]
        baseline=group[group.model=='gjr_skewt'].iloc[0]
        change=best.pinball_loss/baseline.pinball_loss-1
        direction='高' if change>=0 else '低'
        findings.append(f'{confidence:.0%} VaR：本次机器学习中平均pinball loss最低的是{CATALOG[best.model][1]}（{best.pinball_loss:.8f}），相对GJR-GARCH-skewt（{baseline.pinball_loss:.8f}）{direction}{abs(change):.2%}；其违背率为{best.violation_rate:.4%}，名义水平为{1-confidence:.0%}。')
    (dest/'机器学习模型对比.md').write_text('# 机器学习模型对比\n\n'+md_table(pd.DataFrame(ml_rows))+'\n\n'+'\n\n'.join(findings)+'\n\n以上仅是相同银行日期上的点估计比较，不是统计显著性结论。固定配置、单种子历史比较；耗时包含并发竞争，不是严格速度排名。方法与训练设置见[说明](../docs/机器学习方法与实验.md)。\n')
    (dest/'模型总体对比.md').write_text('\n'.join(parts[0:2])+ '\n\n'+md_table(pd.DataFrame(result))+'\n\n耗时为全部14家银行拟合与预测的运行记录，不含绘图；不同批次并行负载与JIT开销不同，不是严格速度基准。详见[项目说明](../README.md)。\n')
if __name__=='__main__':main()
