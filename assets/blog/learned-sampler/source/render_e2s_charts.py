"""Reproduce E2S plots. Measured records and hypothetical scaling stay separate."""
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
results=json.loads((SRC/'results-data.json').read_text())
rows={r['key']:r for r in results['rows']}
curve=json.loads((SRC/'online-curve-data.json').read_text())
scaling=json.loads((SRC/'scaling-illustration-data.json').read_text())
def fmt(v):return str(Decimal(str(v)).quantize(Decimal('.1'),rounding=ROUND_HALF_UP))
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

def bars(mobile=False):
 keys=results['displayed_methods'];labels=[{'offline':'E2S-Offline','online':'E2S-Online'}.get(k,rows[k]['name']) for k in keys]
 colors=[{'base':'#b2bac5','mcmc':OCHRE,'offline':'#a297c6','online':PURPLE}.get(k,'#c7ced7') for k in keys]
 fig=plt.figure(figsize=(3.9,8.3) if mobile else (8,4.7),dpi=100)
 boxes=[[.34,.55,.56,.28],[.34,.10,.56,.30]] if mobile else [[.22,.18,.35,.56],[.66,.18,.28,.56]]
 for panel,(metric,title,lim) in enumerate([('math_average','New tasks · Math avg.',65),('prior_average','Prior tasks · Prior avg.',47)]):
  ax=fig.add_axes(boxes[panel]);style(ax)
  vals=[rows[k][metric] for k in keys]; y=np.arange(len(keys))
  ax.barh(y,vals,height=.55,color=colors,zorder=2)
  ax.set_yticks(y);ax.set_yticklabels(labels if mobile or panel==0 else [],fontsize=9.5 if mobile else 10.8)
  ax.invert_yaxis();ax.set_xlim(0,lim)
  ax.set_xticks([0,20,40,60] if panel==0 else [0,20,40]);ax.tick_params(axis='x',labelsize=9)
  ax.grid(axis='x',color='#edf0f4',lw=.65)
  ax.spines['left'].set_visible(False)
  for i,v in enumerate(vals):ax.text(v+.7,i,fmt(v),va='center',fontsize=9 if mobile else 10.5,color=PURPLE if keys[i] in ['offline','online'] else INK)
  ax.set_xlabel('Accuracy (%)',fontsize=10,labelpad=9)
  ax.set_title(title,loc='left',fontsize=11 if mobile else 12.5,fontweight='medium',pad=25)
  if panel==1:
   ax.axvline(rows['base'][metric],color=TEAL,lw=1,ls=(0,(3,3)),zorder=1)
   ax.text(.0,1.04,'Dashed: base = 42.2%',transform=ax.transAxes,color=TEAL,fontsize=8.8 if mobile else 9.5)
 fig.text(.035 if mobile else .035,.95,'Learn more. Retain prior skills.',fontsize=14 if mobile else 17,fontweight='medium')
 fig.text(.035,.91 if mobile else .865,'Qwen2.5-3B · same seven-benchmark evaluation',fontsize=9.4 if mobile else 11,color=MUTED)
 fig.text(.035,.025 if mobile else .035,'Math gain vs. MCMC + SFT: offline +2.1 pp · online +6.0 pp',fontsize=8 if mobile else 10.8,color=PURPLE)
 save(fig,'e2s-results'+('-mobile' if mobile else ''),'E2S accuracy and prior-task retention','Published baseline values and author-confirmed E2S results. Both bar axes start at zero. Dashed line is base prior accuracy, not uncertainty.')

def online(mobile=False):
 fig,ax=plt.subplots(figsize=(3.9,3.8) if mobile else (8,4.1),dpi=100)
 fig.subplots_adjust(left=.16 if mobile else .10,right=.95,top=.79,bottom=.19)
 style(ax)
 x=[p['progress_pct'] for p in curve['points']];y=[p['math_avg'] for p in curve['points']]
 assert np.isclose(y[-1],rows['online']['math_average'])
 ax.axhline(rows['mcmc']['math_average'],color=OCHRE,lw=1.1,ls=(0,(4,3)))
 ax.axhline(rows['offline']['math_average'],color=TEAL,lw=1.1,ls=(0,(4,3)))
 ax.plot(x,y,color=PURPLE,lw=1.65,zorder=3)
 ax.scatter([x[-1]],[y[-1]],s=20,color=PURPLE,zorder=4)
 ax.text(2,56.35,'E2S-Offline · 55.5%',color=TEAL,fontsize=9 if mobile else 10.5,bbox={'facecolor':'white','edgecolor':'none','pad':1})
 ax.text(99,51.2,'MCMC + SFT · 53.4%',ha='right',color=OCHRE,fontsize=9 if mobile else 10.5,bbox={'facecolor':'white','edgecolor':'none','pad':1})
 ax.text(99,62.6,'E2S-Online · 59.4%',ha='right',color=PURPLE,fontsize=9 if mobile else 11)
 ax.set(xlim=(0,102),ylim=(29,65),xticks=[0,25,50,75,100],yticks=[30,40,50,60])
 ax.set_xlabel('Training progress (%)',labelpad=10,fontsize=10 if mobile else 11)
 ax.set_ylabel('Math avg. (%)',fontsize=10 if mobile else 11)
 ax.tick_params(labelsize=9 if mobile else 10);ax.grid(axis='y',color='#edf0f4',lw=.65)
 fig.text(.06 if mobile else .10,.94,'Learning with refreshed data',fontsize=14 if mobile else 17,fontweight='medium')
 fig.text(.06 if mobile else .10,.875,'Dashed lines: final scores of comparison methods',fontsize=8.8 if mobile else 11,color=MUTED)
 save(fig,'e2s-online-progress'+('-mobile' if mobile else ''),'Online learning with MCMC and offline references','Stored author-confirmed trajectory; horizontal lines are final scores, not matched-compute learning curves.')

def scaling_plot(mobile=False):
 fig,ax=plt.subplots(figsize=(3.9,4.4) if mobile else (8,4.4),dpi=100)
 fig.subplots_adjust(left=.18 if mobile else .11,right=.94,top=.73,bottom=.27)
 style(ax)
 x=[p['samples_per_example'] for p in scaling['points']];y=[p['estimated_math_avg_pct'] for p in scaling['points']]
 ax.set_xscale('log',base=2)
 ax.plot(x,y,color=PURPLE,lw=1.7,ls=(0,(4,3)),marker='o',markersize=5,markerfacecolor='white',markeredgewidth=1.7)
 for xx,yy in zip(x,y):ax.annotate(f'{yy:.1f} est.',(xx,yy),xytext=(0,11),textcoords='offset points',ha='center',fontsize=9 if mobile else 11,color=PURPLE)
 ax.set_xlim(.82,4.9);ax.set_ylim(54.5,58);ax.set_yticks([55,56,57,58]);ax.grid(axis='y',color='#edf0f4',lw=.65)
 ax.set_xticks(x);ax.set_xticklabels([f'{k}× per example\n{scaling["expert_examples"]*k:,} responses' for k in x],fontsize=8.2 if mobile else 11,linespacing=1.6)
 ax.set_ylabel('Illustrative Math avg. (%)',fontsize=9 if mobile else 11)
 ax.tick_params(axis='y',labelsize=9 if mobile else 10)
 fig.text(.06 if mobile else .05,.947,'One corpus, more useful draws',fontsize=13.5 if mobile else 17,fontweight='medium')
 fig.text(.06 if mobile else .05,.866,'ILLUSTRATIVE ESTIMATES · NOT MEASURED',fontsize=8.5 if mobile else 11,color='#89632f',bbox={'boxstyle':'round,pad=.5','facecolor':'#fbf6ec','edgecolor':'#e8d7bb','linewidth':.6})
 fig.text(.06 if mobile else .05,.067,'E2S-Offline · 8,230 expert examples held fixed',fontsize=9 if mobile else 11,color=MUTED)
 save(fig,'e2s-offline-scaling'+('-mobile' if mobile else ''),'Illustrative offline scaling hypothesis — not measured','All three accuracy values are hypothetical estimates, including the 1x point. Dataset sizes are 8230, 16460 and 32920 accepted responses. This is not evidence of diversity or absence of collapse.')

for mobile in [False,True]:bars(mobile);online(mobile);scaling_plot(mobile)
print('Rendered six plots from fixed result records and a separately labeled illustrative estimate record.')
