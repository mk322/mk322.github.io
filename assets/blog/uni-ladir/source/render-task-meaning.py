"""Conceptual task comparison; not decoded thoughts or a task-conditioned encoder."""
from pathlib import Path
from html import escape
ROOT=Path(__file__).resolve().parent.parent
I='#20252C';M='#525A65';P='#6356A5';T='#2B8B88';B='#D7DCE5'
def txt(x,y,s,size=22,bold=False,c=I,anchor='start'):
 return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{c}" text-anchor="{anchor}">{escape(s)}</text>'
def rect(x,y,w,h,fill='white',stroke=B):return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
def scene(x,y,scale=1):
 s='<path d="M15 151 L248 151 L270 174 L0 174Z" fill="#F2ECE3" stroke="#B9A58C" stroke-width="2"/>'
 for xx,yy,col in [(35,56,'#BF656A'),(150,70,'#3E73A8')]:
  s+=f'<path d="M{xx+58} {yy+20} C{xx+94} {yy+10},{xx+94} {yy+64},{xx+57} {yy+58}" fill="none" stroke="{col}" stroke-width="10"/>'
  s+=f'<path d="M{xx} {yy} L{xx+64} {yy} L{xx+60} {yy+81} Q{xx+32} {yy+96},{xx+4} {yy+81}Z" fill="{col}" stroke="#525A65" stroke-width="1.5"/>'
  s+=f'<ellipse cx="{xx+32}" cy="{yy}" rx="32" ry="10" fill="white" fill-opacity=".75" stroke="{col}" stroke-width="3"/>'
 return f'<g transform="translate({x},{y}) scale({scale})">{s}</g>'
def panel(x,y,w,task,lines,nextlines,mobile=False):
 h=180 if mobile else 155
 s=rect(x,y,w,h,'#F7F5FA','#CBC3DF')+txt(x+20,y+34,task,24,True,c=P)
 s+=txt(x+20,y+70,'What matters for this task',19,True,c=M)
 for i,l in enumerate(lines):s+=txt(x+20,y+99+i*27,l,22)
 for i,l in enumerate(nextlines):s+=txt(x+20,y+h-22+i*25,l,21,True,c=T)
 return s
for mobile in [False,True]:
 w=420 if mobile else 1000;h=900 if mobile else 565
 body=txt(w/2,34,'One image. Two tasks.' if mobile else 'Same image, different reasoning needs',24,True,anchor='middle')
 body+=txt(w/2,64,'Conceptual example',19,c=M,anchor='middle')
 if mobile:
  body+=rect(12,88,396,240,'#FAFBFD')+txt(210,122,'The same observation',23,True,anchor='middle')+scene(76,129,1)
  body+=panel(12,356,396,'Task A: grasp the red mug',['Handle location and','space for the gripper'],['Next: plan the approach.'],True)
  body+=panel(12,558,396,'Task B: sort mugs by color',['Which mug is red;','which mug is blue'],['Next: choose each group.'],True)
  body+=rect(12,763,396,120,'#E6E1F1','#AEA1CC')
  for i,l in enumerate(['Continuation prediction teaches','the shared encoder which','information to retain.']):body+=txt(210,798+i*29,l,23,True,c=P,anchor='middle')
 else:
  body+=rect(20,109,290,310,'#FAFBFD')+txt(165,144,'The same observation',22,True,anchor='middle')+scene(30,168,1)
  body+=txt(165,392,'One image, two tasks',20,c=M,anchor='middle')
  body+=panel(370,101,610,'Task A: grasp the red mug',['Handle location + space for the gripper'],['Next: plan the approach.'])
  body+=panel(370,276,610,'Task B: sort mugs by color',['Which mug is red; which mug is blue'],['Next: choose each group.'])
  body+='<path d="M310 263 H336 V180 H358 M336 263 V355 H358" fill="none" stroke="#525A65" stroke-width="2"/>'
  for y in [180,355]:body+=f'<path d="M358 {y-5} L366 {y} L358 {y+5}Z" fill="#525A65"/>'
  body+=rect(20,463,960,86,'#E6E1F1','#AEA1CC')
  body+=txt(500,499,'The task determines which information must survive.',24,True,c=P,anchor='middle')
  body+=txt(500,530,'Continuation prediction trains the shared encoder to preserve it.',22,c=P,anchor='middle')
 out=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="task-title task-desc"><title id="task-title">Same image, different reasoning needs</title><desc id="task-desc">An illustrative image shows a red mug and a blue mug. Grasping requires handle location and gripper clearance; sorting requires color. These task demands supply different continuation supervision during training. This is not a measurement of latent contents.</desc><rect width="{w}" height="{h}" fill="white"/><g font-family="Helvetica Neue, Helvetica, Arial, sans-serif">{body}</g></svg>'
 (ROOT/('task-meaning-mobile.svg' if mobile else 'task-meaning.svg')).write_text(out)
