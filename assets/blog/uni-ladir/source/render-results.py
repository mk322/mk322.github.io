"""Measured blog charts. Data: paper Tables 1–2 and Table 7, in results.json.
Uses fixed 0–100 scales, named comparators, and the paper's subdued palette.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent
font='/System/Library/Fonts/HelveticaNeue.ttc'
if Path(font).exists(): font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':['Helvetica Neue','Arial','DejaVu Sans'],'font.size':13,'text.color':'#20252C','axes.edgecolor':'#D7DCE5','xtick.color':'#525A65','ytick.color':'#525A65','svg.fonttype':'path','svg.hashsalt':'uni-ladir-results','savefig.facecolor':'white'})
data=json.loads((ROOT/'results.json').read_text())
colors=['#B9C1CB','#BA8B51','#6356A5']

def style(ax,n):
 ax.set_xlim(0,108);ax.set_xticks([0,25,50,75,100]);ax.set_yticks([])
 ax.set_ylim(n-.45,-.95)
 ax.tick_params(axis='x',length=0,pad=8,labelsize=12)
 ax.grid(axis='x',color='#E8EBEF',linewidth=.8,zorder=0)
 ax.spines[['top','left','right']].set_visible(False)

def save(fig,name):
 fig.savefig(OUT/(name+'.svg'),metadata={'Date':None},bbox_inches='tight',pad_inches=.12)
 fig.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight',pad_inches=.12)
 svg=OUT/(name+'.svg')
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
 plt.close(fig)

def main(mobile=False):
 fig,axes=plt.subplots(3 if mobile else 1,1 if mobile else 3,figsize=(4.6,11.6) if mobile else (15,4.7))
 for ax,p in zip(axes.flat,data['main']):
  style(ax,3)
  ax.barh(range(3),p['values'],height=.33,color=colors,zorder=3)
  for y,(name,role,value) in enumerate(zip(p['labels'],p['roles'],p['values'])):
   label=name.replace(' (3.3B)','').replace(' (7B)','').replace('LaST₀',r'LaST$_0$')
   kind={'Base model':'base','Direct policy':'direct','Modality-specific':'separate','Unified':'unified'}[role]
   ax.text(0,y-.24,label+' ('+kind+')',fontsize=14 if mobile else 15,va='bottom')
   ax.text(value+1.3,y,f'{value:.1f}',fontsize=14,va='center')
  ax.set_title(p['title'],loc='left',fontsize=19,fontweight='medium',pad=42)
  ax.text(0,1.055,p['metric'],transform=ax.transAxes,fontsize=14,color='#525A65')
 fig.subplots_adjust(left=.025,right=.99,bottom=.10,top=.78,wspace=.15) if not mobile else fig.subplots_adjust(left=.04,right=.96,bottom=.035,top=.93,hspace=.65)
 save(fig,'main-mobile' if mobile else 'main')

def interventions(mobile=False):
 d=data['interventions'];names=['Intact chain','Zeroed contents','Blocks shuffled',"Another example’s chain"]
 fig,axes=plt.subplots(2 if mobile else 1,1 if mobile else 2,figsize=(4.6,9.5) if mobile else (12,5.2))
 for ax,(key,metric) in zip(axes.flat,[('V*','Answer accuracy (%)'),('RLBench','Task success (%)')]):
  style(ax,4)
  values=d[key]
  ax.barh(range(4),values,height=.35,color=['#6356A5']+['#B9C1CB']*3,zorder=3)
  for y,(label,value) in enumerate(zip(names,values)):
   ax.text(0,y-.27,label,fontsize=13,va='bottom')
   ax.text(value+1.3,y,f'{value:.1f}',fontsize=13,va='center')
  ax.set_title(key,loc='left',fontsize=19,fontweight='medium',pad=43)
  ax.text(0,1.06,metric,transform=ax.transAxes,fontsize=13,color='#525A65')
 fig.subplots_adjust(left=.03,right=.99,bottom=.10,top=.80,wspace=.18) if not mobile else fig.subplots_adjust(left=.04,right=.96,bottom=.04,top=.92,hspace=.54)
 save(fig,'interventions-mobile' if mobile else 'interventions')
for mobile in [False,True]: main(mobile);interventions(mobile)
