"""Render a single illustrative learning curve; no measured observations.

The x-axis is normalized progress, not training steps or runtime. Synthetic noise
is a visual placeholder, not estimated run variance or a confidence interval.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
results=json.loads(Path(__file__).with_name('results-data.json').read_text())
rows={r['key']:r for r in results['rows']}
x=np.linspace(0,100,121)
base=rows['base']['math_average']
end=rows['online']['math_average']
rng=np.random.default_rng(20261004)
trend=base+(end-base)*(1-np.exp(-x/24))/(1-np.exp(-100/24))
noise=np.zeros_like(x)
for i in range(1,len(x)):
 noise[i]=.43*noise[i-1]+rng.normal(0,.72)*(.65+.35*np.exp(-x[i]/45))
y=trend+noise
# Pin the two illustrative levels without forcing intermediate monotonicity.
y[0]=base;y[-1]=end
metadata={'status':'Illustrative, not measured','seed':20261004,
 'x_axis':'Normalized training progress (%); no actual step counts or runtime',
 'y_axis':'Illustrative student Math avg. (%)',
 'construction':'Saturating trend plus correlated synthetic noise; no observations or estimated uncertainty',
 'start':float(base),'end':float(end),
 'points':[{'progress_pct':float(a),'math_avg':float(b)} for a,b in zip(x,y)]}
Path(__file__).with_name('online-curve-data.json').write_text(json.dumps(metadata,indent=2)+'\n')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica Neue','Arial','DejaVu Sans'],
 'svg.fonttype':'none','axes.unicode_minus':False})
for name,figsize,fs in [('online-training-curve',(8,3.7),11),('online-training-curve-mobile',(3.8,3.1),10)]:
 fig,ax=plt.subplots(figsize=figsize,dpi=100)
 fig.patch.set_facecolor('white');ax.set_facecolor('white')
 fig.subplots_adjust(left=.095 if 'mobile' not in name else .16,right=.975 if 'mobile' not in name else .945,top=.81,bottom=.20)
 ax.plot(x,y,color='#6356a5',lw=1.9,solid_capstyle='round')
 ax.set_xlim(0,100);ax.set_ylim(29,64)
 ax.set_xticks([0,25,50,75,100]);ax.set_xticklabels(['0','25','50','75','100'])
 ax.set_yticks([30,40,50,60])
 ax.set_xlabel('Training progress (%)',fontsize=fs,color='#525a65',labelpad=9)
 ax.set_ylabel('Math avg. (%)',fontsize=fs,color='#525a65',labelpad=7)
 ax.tick_params(axis='both',labelsize=fs-1,colors='#6c7786',length=0,pad=7)
 for side in ['top','right']:ax.spines[side].set_visible(False)
 for side in ['left','bottom']:ax.spines[side].set_color('#cfd5df');ax.spines[side].set_linewidth(.7)
 ax.set_axisbelow(True);ax.grid(axis='y',color='#e9ecf2',lw=.65)
 fig.text(.095 if 'mobile' not in name else .16,.93,'Online post-training',fontsize=fs+1,weight='medium',color='#26313d')
 fig.text(.095 if 'mobile' not in name else .16,.865,'Illustrative trajectory · not measured',fontsize=fs-1,color='#7a8491')
 fig.savefig(OUT/(name+'.svg'),metadata={'Title':'Illustrative online learning curve — not measured','Description':metadata['construction']})
 svg=OUT/(name+'.svg')
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
 if 'mobile' not in name:fig.savefig(OUT/(name+'.png'),dpi=220)
 plt.close(fig)
print('Rendered one synthetic trajectory in desktop and mobile layouts.')
