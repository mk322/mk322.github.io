"""Reproduce E2S plots. Author-confirmed offline scaling sweep; provenance is preserved in scaling-data.json."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC=Path(__file__).resolve().parent
OUT=SRC.parent
INK='#20252c'; MUTED='#525a65'; PURPLE='#6356a5'; TEAL='#2b8b88'; OCHRE='#ba8b51'; HAIR='#d7dce5'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica Neue','Arial','DejaVu Sans'],'font.size':11,'text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'svg.fonttype':'none','axes.unicode_minus':False})
scaling=json.loads((SRC/'scaling-data.json').read_text())
def style(ax):
 for side in ['top','right']:ax.spines[side].set_visible(False)
 for side in ['left','bottom']:ax.spines[side].set_color(HAIR);ax.spines[side].set_linewidth(.65)
 ax.tick_params(length=0,pad=7)
 ax.set_axisbelow(True)
def save(fig,name,title,desc):
 fig.savefig(OUT/(name+'.svg'),metadata={'Title':title,'Description':desc})
 fig.savefig('/tmp/'+name+'.png',dpi=130)
 path=OUT/(name+'.svg');path.write_text('\n'.join(l.rstrip() for l in path.read_text().splitlines())+'\n')
 plt.close(fig)

def scaling_plot(mobile=False):
 fig,ax=plt.subplots(figsize=(3.9,4.4) if mobile else (8,4.4),dpi=100)
 fig.subplots_adjust(left=.18 if mobile else .11,right=.94,top=.73,bottom=.27)
 style(ax)
 x=[p['samples_per_example'] for p in scaling['points']];y=[p['math_avg_pct'] for p in scaling['points']]
 ax.set_xscale('log',base=2)
 ax.plot(x,y,color=PURPLE,lw=1.7,ls='-',marker='o',markersize=5,markerfacecolor='white',markeredgewidth=1.7)
 for xx,yy in zip(x,y):ax.annotate(f'{yy:.1f}',(xx,yy),xytext=(0,11),textcoords='offset points',ha='center',fontsize=9 if mobile else 11,color=PURPLE)
 ax.set_xlim(.82,4.9);ax.set_ylim(54.5,58);ax.set_yticks([55,56,57,58]);ax.grid(axis='y',color='#edf0f4',lw=.65)
 ax.set_xticks(x);ax.set_xticklabels([f'{k}× per example\n{scaling["expert_examples"]*k:,} responses' for k in x],fontsize=8.2 if mobile else 11,linespacing=1.6)
 ax.set_ylabel('Math avg. (%)',fontsize=9 if mobile else 11)
 ax.tick_params(axis='y',labelsize=9 if mobile else 10)
 fig.text(.06 if mobile else .05,.947,'One corpus, more useful draws',fontsize=13.5 if mobile else 17,fontweight='medium')
 fig.text(.06 if mobile else .05,.866,'1 / 2 / 4 responses per expert example',fontsize=10 if mobile else 12,color=MUTED)
 fig.text(.06 if mobile else .05,.067,'E2S-Offline · 8,230 expert examples held fixed',fontsize=9 if mobile else 11,color=MUTED)
 save(fig,'e2s-offline-scaling'+('-mobile' if mobile else ''),'Offline scaling: author-confirmed values','Author-confirmed offline scaling: Math averages 55.0, 56.3 and 57.2 at 8230, 16460 and 32920 training responses. See source/scaling-data.json for confirmation and provenance.')


for mobile in [False,True]: scaling_plot(mobile)
