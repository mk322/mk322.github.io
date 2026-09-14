"""Render original paper vectors and measured charts. Requires pymupdf, matplotlib.
Run from any directory: python assets/blog/uni-ladir/source/render.py
"""
from pathlib import Path
import json
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.titlesize':17,'axes.labelsize':14,'svg.fonttype':'none','text.color':'#20252C','axes.labelcolor':'#525A65','xtick.color':'#525A65','ytick.color':'#525A65','axes.edgecolor':'#D7DCE5','savefig.facecolor':'white'})
for name in ['fig1','fig2']:
 doc=fitz.open(ROOT/(name+'.pdf'));p=doc[0]
 (OUT/(name+'.svg')).write_text(p.get_svg_image(text_as_path=True))
 p.get_pixmap(matrix=fitz.Matrix(1800/p.rect.width,1800/p.rect.width),alpha=False).save(OUT/(name+'.png'))
# All reported values taken from the supplied manuscript; no fitted or simulated results.
data={'main':[{'title':'Visual reasoning','metric':'8-benchmark mean accuracy (%)','labels':['Qwen2.5-VL\nmatched base','Uni-LaDiR'],'values':[63.55,75.55],'gain':'+18.9% relative'},{'title':'Visual math & logic','metric':'4-benchmark mean accuracy (%)','labels':['ILVR\nreported baseline','Uni-LaDiR'],'values':[45.68,49.03],'gain':'+7.3% relative'},{'title':'Robot manipulation','metric':'RLBench mean success (%)','labels':['LaST₀','Uni-LaDiR'],'values':[82,87],'gain':'+6.1% relative'}], 'sharing':[{'title':'Text + image teachers','metric':'4-benchmark mean accuracy (%)','labels':['Separate\nspaces','Text + image\nshared'],'values':[62.97,67.66],'gain':'+7.4% relative'},{'title':'Visual + 3D + state teachers','metric':'LIBERO / RLBench mean success (%)','labels':['Separate','Visual\n+ 3D','Visual\n+ state','3D\n+ state','All three'],'values':[90.05,90.60,90.25,91.25,93.08],'gain':'+3.4% relative'}]}
(ROOT/'results.json').write_text(json.dumps(data,indent=2)+'\n')
def plot(key,stacked=False):
 panels=data[key];n=len(panels)
 if stacked:
  mobile(key)
  return
 else:
  fig,axes=plt.subplots(1,n,figsize=(14,4.8),gridspec_kw={'width_ratios':[1,1.8]} if key=='sharing' else None)
 for ax,p in zip(axes,panels):
  xs=list(range(len(p['values']))); colors=['#B4AFC7']*(len(xs)-1)+['#6356A5']
  ax.bar(xs,p['values'],width=.57,color=colors,zorder=3)
  for x,y in zip(xs,p['values']):ax.text(x,y+2,f'{y:.2f}',ha='center',va='bottom',fontsize=14,fontweight='bold')
  ax.set_ylim(0,120);ax.set_yticks([0,25,50,75,100]);ax.set_xticks(xs,p['labels']);ax.tick_params(axis='x',length=0,pad=10);ax.tick_params(axis='y',length=0)
  ax.grid(axis='y',color='#E7E9EF',zorder=0);ax.spines[['top','right','left']].set_visible(False)
  ax.set_title(p['title'],loc='left',fontweight='bold',pad=40)
  ax.text(0,1.06,p['metric'],transform=ax.transAxes,fontsize=13,color='#525A65')
  ax.text(.98,.94,p['gain'],transform=ax.transAxes,ha='right',fontsize=14,color='#6356A5',fontweight='bold')
 fig.subplots_adjust(left=.055 if not stacked else .09,right=.99,top=.79 if not stacked else .91,bottom=.16 if not stacked else .07,wspace=.27,hspace=.65)
 suffix='-mobile' if stacked else ''
 fig.savefig(OUT/(key+suffix+'.svg'));fig.savefig(OUT/(key+suffix+'.png'),dpi=150);plt.close(fig)
def mobile(key):
 panels=data[key];n=len(panels)
 fig,axes=plt.subplots(n,1,figsize=(4.5,n*4.3),squeeze=False)
 for ax,p in zip(axes[:,0],panels):
  labels=[x.replace('\n',' ') for x in p['labels']]
  if key=='main':labels=[{'Qwen2.5-VL matched base':'Matched base','ILVR reported baseline':'ILVR (reported)'}.get(x,x) for x in labels]
  values=p['values'];ys=list(range(len(values)))
  ax.barh(ys,values,color=['#B4AFC7']*(len(values)-1)+['#6356A5'],height=.42,zorder=3)
  for y,v,label in zip(ys,values,labels):
   ax.text(0,y-.30,label,fontsize=13,color='#20252C')
   ax.text(v+1.5,y,f'{v:.2f}',fontsize=12,va='center',fontweight='bold')
  ax.set_xlim(0,120);ax.set_xticks([0,50,100]);ax.tick_params(axis='x',length=0,labelsize=12);ax.set_yticks([])
  ax.set_ylim(len(values)-.4,-.85);ax.grid(axis='x',color='#E7E9EF',zorder=0)
  ax.spines[['top','right','left']].set_visible(False)
  title=p['title'].replace('Visual + 3D + state teachers','Visual + 3D + state').replace('Text + image teachers','Text + image')
  ax.set_title(title,loc='left',fontsize=16,fontweight='bold',pad=44)
  metric=p['metric'].replace('4-benchmark mean accuracy (%)','Mean accuracy · 4 tasks (%)').replace('8-benchmark mean accuracy (%)','Mean accuracy · 8 tasks (%)').replace('LIBERO / RLBench mean success (%)','LIBERO / RLBench mean (%)')
  ax.text(0,1.13,metric,transform=ax.transAxes,fontsize=11,color='#525A65')
  ax.text(0,1.03,p['gain'],transform=ax.transAxes,fontsize=13,fontweight='bold',color='#6356A5')
 fig.subplots_adjust(left=.06,right=.95,top=.86 if n==2 else .91,bottom=.035,hspace=.7)
 fig.savefig(OUT/(key+'-mobile.svg'));fig.savefig(OUT/(key+'-mobile.png'),dpi=150);plt.close(fig)
for key in ['main','sharing']:
 plot(key);plot(key,True)
