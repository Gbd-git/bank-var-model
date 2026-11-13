"""One complete VaR time series per bank; no subsampling or uncertainty claims.
Observed returns and negative VaR (return-tail thresholds) use the same log-% unit.
All 2575 evaluation dates and all 14 banks are retained. Single-axis figures.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
ROOT=Path(__file__).resolve().parents[1]
NAMES=['Ping An','Ningbo','Shanghai Pudong','Huaxia','Minsheng','China Merchants',
       'Nanjing','Industrial','Beijing','Bank of Communications','ICBC',
       'China Construction','Bank of China','CITIC']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':9,'axes.spines.right':False,'axes.spines.top':False,
    'axes.linewidth':0.8,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})

def main():
    import json
    with np.load(ROOT/'data/bank_returns.npz',allow_pickle=False) as data:
        y=data['y'];dates=data['dates'].astype('datetime64[D]');banks=data['banks']
    summary=json.loads((ROOT/'docs/results/summary.json').read_text())
    start=summary['warmup'];q=np.load(ROOT/'docs/results/predictions.npy')
    y=y[start:];dates=dates[start:]
    if q.shape!=(len(y),len(banks),2) or len(banks)!=14: raise ValueError('Data mismatch')
    folder=ROOT/'docs/figures';folder.mkdir(parents=True,exist_ok=True)
    for b,name in enumerate(NAMES):
        fig,ax=plt.subplots(figsize=(7.2,3.5))
        fig.subplots_adjust(left=.11,right=.98,bottom=.17,top=.77)
        ax.plot(dates,y[:,b],color='#AAAAAA',linewidth=.4,label='Observed return',zorder=1)
        ax.plot(dates,q[:,b,1],color='#D18A36',linewidth=.8,linestyle='--',label='95% VaR threshold',zorder=2)
        ax.plot(dates,q[:,b,0],color='#285C86',linewidth=.9,label='99% VaR threshold',zorder=3)
        ax.set_ylabel('Log return / threshold (%)');ax.set_xlabel('Date')
        ax.xaxis.set_major_locator(mdates.YearLocator(2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.margins(x=.01);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        fig.suptitle(f'{name}: rolling GJR-GARCH-skewt VaR',y=.98,fontsize=11)
        fig.legend(loc='upper center',bbox_to_anchor=(.54,.91),ncol=3,fontsize=8)
        # Single plot-area: multi-panel alignment gate is not applicable.
        fig.savefig(folder/f'var_{b+1:02d}.png',dpi=600)
        fig.savefig(folder/f'var_{b+1:02d}.pdf')
        fig.savefig(folder/f'var_{b+1:02d}.svg')
        plt.close(fig)
if __name__=='__main__': main()
