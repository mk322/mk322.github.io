"""Conceptual probability densities only; not experimental measurements.

Horizontal position is a schematic ordering of responses. A contiguous valid
region is chosen for legibility, not as an assumption about real response spaces.
The constrained target is computed by masking and numerical renormalization.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]
BLUE='#3e73a8'; OCHRE='#ba8b51'; PURPLE='#6356a5'; TEAL='#2b8b88'; MUTED='#525a65'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica Neue','Arial','DejaVu Sans'],'font.size':11,'text.color':'#20252c','axes.labelcolor':MUTED,'svg.fonttype':'none'})
x=np.linspace(0,1,4001)
def normal(mu,sigma):return np.exp(-.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
p=.6*normal(.23,.075)+.25*normal(.64,.055)+.15*normal(.85,.043)
p/=np.trapezoid(p,x)
valid=x>=.48
expert=normal(.85,.048)*valid;expert/=np.trapezoid(expert,x)
q=p*valid;q/=np.trapezoid(q,x)
assert abs(np.trapezoid(q,x)-1)<1e-12 and np.all(q[~valid]==0)
assert np.ptp(q[valid]/p[valid])<1e-12
for mobile in [False,True]:
 fig,axes=plt.subplots(2,1,figsize=(3.9,6.5)) if mobile else plt.subplots(1,2,figsize=(8,3.6))
 fig.subplots_adjust(left=.14 if mobile else .075,right=.97,top=.85 if mobile else .73,bottom=.09 if mobile else .19,hspace=.64,wspace=.18)
 for i,ax in enumerate(axes):
  ax.axvspan(.48,1,color='#edf6f3',zorder=0)
  ax.axvline(.48,color=TEAL,lw=.8,ls=(0,(3,3)),alpha=.7)
  ax.text(.735,8.7,'Valid responses',ha='center',fontsize=10,color=TEAL)
  ax.text(.235,8.7,'Outside constraint',ha='center',fontsize=9,color='#87909b')
  ax.set_xlim(0,1);ax.set_ylim(0,9.5);ax.set_xticks([]);ax.set_yticks([])
  for edge in ['top','right','left']:ax.spines[edge].set_visible(False)
  ax.spines['bottom'].set_color('#cdd3dd');ax.spines['bottom'].set_linewidth(.8)
  ax.set_xlabel('Possible responses',fontsize=10,labelpad=9)
  if mobile or i==0:ax.set_ylabel('Probability density',fontsize=10,labelpad=6)
  ax.plot(x,p,color=BLUE,lw=1.5,ls='-' if i==0 else (0,(3,2)),alpha=1 if i==0 else .6,label='Student')
  if i==0:
   ax.fill_between(x,p,color=BLUE,alpha=.06)
   ax.plot(x,expert,color=OCHRE,lw=1.9,label='Expert')
   ax.set_title('1  Expert–student mismatch',fontsize=12,loc='left',pad=42,fontweight='medium')
  else:
   ax.fill_between(x,q,color=PURPLE,alpha=.10)
   ax.plot(x,q,color=PURPLE,lw=1.9,label='Constrained target')
   ax.set_title('2  Keep valid student responses',fontsize=12,loc='left',pad=42,fontweight='medium')
  ax.legend(loc='lower left',bbox_to_anchor=(0,1.04),ncol=2,frameon=False,fontsize=9.5,handlelength=2,columnspacing=1.4,borderaxespad=0)
 name='constrained-distribution'+('-mobile' if mobile else '')
 fig.savefig(OUT/(name+'.svg'),metadata={'Title':'Mismatch and the expert constraint','Description':'Conceptual illustration. Left: expert and student distributions differ, while expert responses satisfy the shaded constraint. Right: zero out student probability outside the valid set and renormalize, preserving relative probabilities inside. Not experimental data.'})
 fig.savefig('/tmp/'+name+'.png',dpi=150)
 path=OUT/(name+'.svg');path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
 plt.close(fig)
