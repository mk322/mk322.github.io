"""Editable, conceptual blog diagrams; paper typography/palette, no measured data."""
from pathlib import Path
from html import escape
import re
def mobile_type(body):
 return re.sub(r'font-size="([0-9.]+)"',lambda m:f'font-size="{float(m[1])*1.25:g}"',body)
ROOT=Path(__file__).resolve().parent.parent
INK='#20252C'; MUTED='#525A65'; PURPLE='#6356A5'; BORDER='#D7DCE5'
COLORS=['#BA8B51','#2B8B88','#81739E'];PALES=['#FCF7EE','#F1F8F7','#F5F2F9']
def text(x,y,label,size=21,bold=False,color=INK,anchor='middle'):
 return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}">{escape(label)}</text>'
def box(x,y,w,h,label='',stroke=BORDER,fill='white',size=20,bold=False):
 return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'+ (text(x+w/2,y+h/2+size*.34,label,size,bold) if label else '')
def arrow(x1,y1,x2,y2,color=MUTED):
 return f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#arrow-{color[1:]})"/>'
def tokenblock(x,y,color,fill):
 return ''.join(box(x+i*21,y,16,29,stroke=color,fill=fill) for i in range(4))
def svg(w,h,body,title,desc):
 defs=''.join(f'<marker id="arrow-{c[1:]}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7Z" fill="{c}"/></marker>' for c in [MUTED,PURPLE,*COLORS])
 return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><defs>{defs}</defs><rect width="{w}" height="{h}" fill="white"/><g font-family="Helvetica Neue, Helvetica, Arial, sans-serif">{body}</g></svg>'
def decision_panel(unified):
 body=box(0,0,466,435,stroke='#CBC3DF' if unified else BORDER,fill='#F5F2FA' if unified else '#FAFBFD')
 body+=text(22,35,'Uni-LaDiR' if unified else 'Modality-specific thoughts',25,True,anchor='start')
 body+=text(22,64,'One learned thought space' if unified else 'A format for each modality',20,color=PURPLE if unified else MUTED,anchor='start')
 for i,(lab,col,pale) in enumerate(zip(['Image','3D','State'],COLORS,PALES)):
  x=23+i*147
  body+=box(x,91,126,42,lab,col,'white',21,True)
  body+=arrow(x+63,136,x+63,155,col)
  body+=box(x,160,126,42,'Features',col,pale,20)
  if unified:body+=arrow(x+63,205,x+63,229,col)
  else:
   body+=arrow(x+63,205,x+63,278,col)
   body+=tokenblock(x+24,286,col,pale)
   body+=text(x+63,343,'Step '+str(i+1),19,True,col)
 if unified:
  body+=box(23,235,420,42,'Shared encoder',stroke='#AEA1CC',fill='#E6E1F1',size=22,bold=True)
  body+=arrow(233,281,233,299,PURPLE)
  for i in range(3):
   x=23+i*147
   body+=tokenblock(x+24,309,PURPLE,'#E6E1F1')
   body+=text(x+63,365,'Step '+str(i+1),19,True,PURPLE)
   if i<2:body+=arrow(x+108,323,x+157,323,PURPLE)
  body+=text(233,407,'Learn contents for what comes next',20,True,PURPLE)
 else:
  for i in range(2):body+=arrow(23+i*147+108,300,23+i*147+157,300)
  body+=text(233,407,'Carry reasoning across formats',20,True,MUTED)
 return body
for mobile in [False,True]:
 w=490 if mobile else 1000; h=1015 if mobile else 565
 body=text(w/2,31,'Different teachers. What format should thoughts use?',22,True) if not mobile else text(w/2,30,'Different teachers. One thought space?',23,True)
 body+=text(w/2,61,'Illustrated with image, 3D, and state teacher steps',18,color=MUTED)
 body+=f'<g transform="translate(12,85)">{decision_panel(False)}</g>'
 body+=f'<g transform="translate({12 if mobile else 522},{544 if mobile else 85})">{decision_panel(True)}</g>'
 body+=text(w/2,h-13,'Color denotes the thought format, not decoded semantic content.',17,color=MUTED)
 if mobile:
  body=body.replace('Different teachers. One thought space?','One space for every thought?').replace('Illustrated with image, 3D, and state teacher steps','Same teachers; a different interface.')
  body=body.replace(text(233,407,'Carry reasoning across formats',20,True,MUTED),text(233,393,'Carry reasoning',20,True,MUTED)+text(233,422,'across formats',20,True,MUTED))
  body=body.replace(text(233,407,'Learn contents for what comes next',20,True,PURPLE),text(233,393,'Learn contents',20,True,PURPLE)+text(233,422,'for what comes next',20,True,PURPLE))
  body=body.replace('Color denotes the thought format, not decoded semantic content.','Color shows format, not meaning.')
  body=mobile_type(body)
 (ROOT/('decision-mobile.svg'  if mobile else 'decision.svg')).write_text(svg(w,h,body,'Different teachers, one learned thought space','Left: image, 3D, and state features retain modality-specific thought formats. Right: a shared encoder maps their features into one latent format learned for continuation. Teacher encoding is training-only.'))
def handoff_panel(continuation):
 body=box(0,0,466,460,stroke='#CBC3DF' if continuation else BORDER,fill='#F5F2FA' if continuation else '#FAFBFD')
 body+=text(233,38,'Continuation prediction' if continuation else 'Reconstruction',25,True,color=PURPLE if continuation else INK)
 body+=box(93,67,280,48,'Teacher step i',COLORS[0],PALES[0],22,True)
 body+=arrow(233,119,233,149)
 body+=text(276,141,'Encode',18,anchor='start',color=MUTED)
 body+=box(93,156,280,48,'Thought block zᵢ',PURPLE,'#E6E1F1',22,True)
 body+=arrow(233,208,233,294,PURPLE if continuation else MUTED)
 body+=text(254,252,'Predict',19,anchor='start',color=MUTED)
 if continuation:
  body+=box(18,224,153,51,stroke=BORDER,fill='white')+text(94,244,'Input + earlier',17)+text(94,264,'thought blocks',17)
  body+=arrow(174,250,230,250)
 body+=box(93,301,280,48,'Next teacher step i + 1' if continuation else 'Same teacher step i',COLORS[1] if continuation else COLORS[0],PALES[1] if continuation else PALES[0],21,True)
 body+=text(233,391,'Keep what helps the next step' if continuation else 'Keep what reproduces the source',21,True,color=PURPLE if continuation else MUTED)
 body+=text(233,425,'Supervision looks forward' if continuation else 'Supervision looks back',20,color=MUTED)
 return body
for mobile in [False,True]:
 w=490 if mobile else 1000;h=1065 if mobile else 600
 body=text(w/2,32,'Same teacher step. A different learning objective.',22,True) if not mobile else text(w/2,32,'What should train a thought?',24,True)
 body+=text(w/2,62,'The target determines which information must survive.',18,color=MUTED)
 body+=f'<g transform="translate(12,85)">{handoff_panel(False)}</g>'
 body+=f'<g transform="translate({12 if mobile else 522},{570 if mobile else 85})">{handoff_panel(True)}</g>'
 body+=text(w/2,h-15,'Final answer / action supervision also trains the complete chain.',17,color=MUTED)
 if mobile:
  body=body.replace('The target determines which information must survive.','Same evidence. Different targets.')
  body=body.replace('Final answer / action supervision also trains the complete chain.','Final output also supervises the chain.')
  body=mobile_type(body)
 (ROOT/('handoff-mobile.svg'  if mobile else 'handoff.svg')).write_text(svg(w,h,body,'Reconstruction versus continuation prediction','Reconstruction predicts the same teacher step from its thought. Continuation uses the thought with the task input and earlier blocks to predict the next teacher step. Final output supervision is omitted from the schematic.'))
if __name__=='__main__':
 import fitz
 for f in ['decision','decision-mobile','handoff','handoff-mobile']:
  d=fitz.open(ROOT/(f+'.svg'));p=d.convert_to_pdf();pdf=fitz.open('pdf',p);pdf[0].get_pixmap(matrix=fitz.Matrix(1,1)).save('/tmp/uni-'+f+'.png')
