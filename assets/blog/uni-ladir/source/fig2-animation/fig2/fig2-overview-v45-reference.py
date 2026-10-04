#!/usr/bin/env python3
"""Fig. 2: masked batched construction and compact token-sequence supervision.
Uses the approved Fig. 1/Fig. 2 vector helper definitions without executing them.
"""
from pathlib import Path
import ast,math,io
from reportlab.lib.colors import Color,HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
import fitz
from PIL import Image,ImageOps
import matplotlib
matplotlib.rcParams["savefig.transparent"]=True
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent
W,H=1800,1215
OUT=ROOT/'fig2-overview-v45-reference.pdf'
GRAPHITE=HexColor('#20252C');TEXT_2=HexColor('#4D535C');HAIR=HexColor('#CFD4DC');CARD=HexColor('#FFFFFF')
HN='HelveticaNeue';HN_M='HelveticaNeue-Medium';HN_B='HelveticaNeue-Bold';SLATE=HexColor('#718198')
C={'s':('#EDF5F4','#397DBB'),'l':('#F7F2E8','#AF803F'),'z':('#F2EFF8','#6356A5'),'n':('#EDF3FB','#CB884D'),'x':('#F1F3F6','#69778B'),'y':('#F1F3F6','#69778B')}
C['q']=C['l']
C['z']=('#F1F3F6','#788391')
def helpers(path,names):
 for node in ast.parse(path.read_text()).body:
  if isinstance(node,ast.FunctionDef) and node.name in names:
   exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),globals())
helpers(ROOT.parent/'teaser/teaser-v87-rectangular-latent.py',{'register_fonts','by','rounded_rect','line','text','text_mid','arrowhead','open_head','open_straight','icon_visual','icon_speech','icon_cube'})
register_fonts()
pdfmetrics.registerFont(TTFont('TimesNRBI','/System/Library/Fonts/Supplemental/Times New Roman Bold Italic.ttf'))
helpers(ROOT/'fig2-fig1-style-v7.py',{'label','symbol','tok','arr','stop_marker'})
def stop_marker(x,y):
 # Open six-arm ice-blue snowflake: short branches stay visibly separate.
 rounded_rect(x-15,y-15,30,30,7,CARD,HexColor('#D5E7F7'),.9,False)
 blue=HexColor('#2585CC')
 c.saveState();c.setLineCap(1);c.setLineJoin(1)
 for k in range(6):
  a=math.pi/6+math.pi/3*k
  dx,dy=math.cos(a),math.sin(a)
  line(x,y,x+11*dx,y+11*dy,blue,1.9)
  bx,by_=x+6.5*dx,y+6.5*dy
  for side in (-1,1):
   ang=a+side*math.pi/3
   line(bx,by_,bx+3*math.cos(ang),by_+3*math.sin(ang),blue,1.9)
 c.restoreState()
def symbol(spec,x,y,size=38,color=GRAPHITE):
 # TeX-style math layout handles script size, kerning and collision clearance.
 base,idx,sup=spec
 if base=='q':size*=.92
 idx=idx.replace('−','-')
 tex=(r'\mathbfit{'+base+'}') if base in ('s','q') else base
 if idx:tex+='_{'+(r'\,' if sup=='t' else '')+idx+'}'
 if sup:tex+='^{'+(r'\,' if sup=='t' else '')+sup+'}'
 eq('$'+tex+'$',x,y,32 if size==34 else 40 if size==38 else 33 if size==30 else size,color.hexval().replace('0x','#'))
base_tok=tok
def tok(spec,x,y,kind=None,w=96,h=60):
 if spec[0] not in ('x','y'):
  return base_tok(spec,x,y,kind,w,h)
 word='input' if spec[0]=='x' else 'output'
 width=124 if spec[0]=='x' else 136
 fill,stroke=C[spec[0]]
 rounded_rect(x-width/2,y-h/2,width,h,10,HexColor(fill),HexColor(stroke),1.8,False)
 tw=pdfmetrics.stringWidth(word,HN_M,30);mw=pdfmetrics.stringWidth(spec[0],'TimesNRI',38)
 start=x-(tw+8+mw)/2
 label(word,start,y,30)
 symbol(spec,start+tw+8+mw/2,y,38)
c=canvas.Canvas(str(OUT),pagesize=(W,H));c.setTitle('Batched latent construction and token-sequence training views')
maths=[]
def eq(tex,x,y,size=36,color='#20252C'):
 maths.append((tex,x,y,size,color))
def title(letter,name,y):
 label(letter,36,y,32,True,color=HexColor('#6356A5'));label(name,85,y,34,True)
def model(x,y,w=226):
 rounded_rect(x-w/2,y-31,w,62,10,HexColor('#EEF2F8'),HexColor('#95A4BC'),1.6,False)
 label('Shared LLM',x,y,32,anchor='center')
def hat(spec,x,y,kind):
 fill,stroke=C[kind]
 rounded_rect(x-48,y-30,96,60,10,HexColor(fill),HexColor(stroke),1.8,False)
 eq(r'$\hat{z}_{i}$',x,y,38)
def clean(i,x,y,detached=False):
 tok(('z',str(i),'*'),x,y)
 if detached: stop_marker(x+46,y-29)
X=('x','','');Y=('y','','')
def dots(x,y):
 c.saveState();c.setFillColor(TEXT_2)
 for d in (-8,0,8):c.circle(x+d,H-y,2.2,fill=1,stroke=0)
 c.restoreState()
def supervised(x,y,w=96,h=60):
 # Neutral outer boundary denotes a supervised/readout position, after Coconut.
 rounded_rect(x-w/2-6,y-h/2-6,w+12,h+12,12,CARD,GRAPHITE,2.2,False)
def noisy(x,y,small=False):
 w,h=(44,42) if small else (96,60)
 rounded_rect(x-w/2,y-h/2,w,h,10,HexColor('#EDF3FB'),HexColor('#CB884D'),1.8,False)
 # Repeatable vector stippling, visible in grayscale, beneath a clear glyph area.
 c.saveState();c.setFillColor(HexColor('#7DABD8'))
 for ix in range(7):
  for iy in range(4):
   xx=x-w/2+8+ix*(w-16)/6;yy=y-h/2+6+iy*(h-12)/3
   if small:
    if ix not in (0,6):continue
    xx=x+(-17 if ix==0 else 17)
   # Keep the entire central glyph column free of noise dots.
   if abs(xx-x)<(16 if small else 29):continue
   c.circle(xx,H-yy,1.25,fill=1,stroke=0)
 c.restoreState()
 symbol(('z','i','t'),x,y,34 if small else 38)

base_label_weight=label
def label(t,x,y,size=30,bold=True,color=GRAPHITE,anchor='left'):
 base_label_weight(t,x,y,size,bold,color,anchor)
def arr(x,y,x2,y2):
 open_straight(x,y,x2,y2,SLATE,5,15)

def gradient_arrow(points,color,width=3.2,head=13,dash=(9,6)):
 # Dashed reverse-flow path with soft corners and a solid directional head.
 p=c.beginPath();p.moveTo(points[0][0],by(points[0][1]))
 for i in range(1,len(points)-1):
  a,b,d=points[i-1],points[i],points[i+1]
  before=math.hypot(b[0]-a[0],b[1]-a[1]);after=math.hypot(d[0]-b[0],d[1]-b[1])
  radius=min(12,before*.35,after*.35)
  entry=(b[0]-(b[0]-a[0])*radius/before,b[1]-(b[1]-a[1])*radius/before)
  leave=(b[0]+(d[0]-b[0])*radius/after,b[1]+(d[1]-b[1])*radius/after)
  p.lineTo(entry[0],by(entry[1]));p.curveTo(b[0],by(b[1]),b[0],by(b[1]),leave[0],by(leave[1]))
 end=points[-1];prev=points[-2]
 length=math.hypot(end[0]-prev[0],end[1]-prev[1])
 ux,uy=(end[0]-prev[0])/length,(end[1]-prev[1])/length
 p.lineTo(end[0]-ux*head*.85,by(end[1]-uy*head*.85))
 # A narrow white clearance keeps crossings visually separate from forward flow.
 c.saveState();c.setStrokeColor(CARD);c.setLineWidth(width+4);c.setLineCap(0)
 c.drawPath(p,fill=0,stroke=1);c.restoreState()
 c.saveState();c.setStrokeColor(color);c.setLineWidth(width);c.setLineCap(1);c.setLineJoin(1);c.setDash(dash)
 c.drawPath(p,fill=0,stroke=1);c.restoreState()
 prev=points[-2];arrowhead(end[0],end[1],end[0]-prev[0],end[1]-prev[1],color,head)

def panel(x,y,w,h,title_,color):
 rounded_rect(x,y,w,h,14,CARD,HexColor(color),2.2,False)
 tint={'#9581BD':'#F4F0FA','#6A9DD3':'#EFF5FC','#D9A073':'#FCF3EB'}[color]
 rounded_rect(x+2,y+2,w-4,50,12,HexColor(tint),HexColor(tint),0,False)
 letter,name=title_.split(' ',1)
 label(letter,x+16,y+28,32,True,color=HexColor(color))
 label(name,x+65,y+28,34,True)
def box(x,y,w,h,lines,fill='#EEF2F8',stroke='#95A4BC',size=28):
 rounded_rect(x-w/2,y-h/2,w,h,9,HexColor(fill),HexColor(stroke),1.8,False)
 for k,t in enumerate(lines):label(t,x,y+(k-(len(lines)-1)/2)*32,size,anchor='center')
def llm(x,y,w=125):
 box(x,y,w,80,[], '#EEF2F8','#95A4BC')
 label('LLM',x,y,28,anchor='center')
def mini(sp,x,y,w=64):
 w=96 if sp[0] in ('x','y') else 64
 fill,stroke=C['n'] if sp[2]=='t' else C[sp[0]]
 rounded_rect(x-w/2,y-30,w,60,10,HexColor(fill),HexColor(stroke),1.8,False)
 if sp[0] in ('x','y'):
  word='input' if sp[0]=='x' else 'output'
  label(word+' '+sp[0],x,y,20,anchor='center')
 else:symbol(sp,x,y,29 if sp[0]=='s' and len(sp[1])>1 else 34 if len(sp[1])>1 else 38)
def bracket(l,r,y,txt):
 line(l,y,l,y-10,SLATE,1.5);line(l,y-10,r,y-10,SLATE,1.5);line(r,y-10,r,y,SLATE,1.5)
 label(txt,(l+r)/2,y-32,24,anchor='center')
panel(20,20,940,520,'(a) Latent Encoding','#9581BD')
# Affine placement retains the approved construction alignment exactly.
base_eq=eq
base_label=label
base_arr=arr
def arr(x,y,x2,y2):
 open_straight(x,y,x2,y2,SLATE,6.5,18)
def label(txt,x,y,size=30,*args,**kwargs):
 c.saveState();c.resetTransforms()
 base_label(txt,20+x*.67,85+y*.85,(25 if txt=='Attention mask' else 26 if 'encoder' in txt else 28),*args,**kwargs)
 c.restoreState()
def eq(tex,x,y,size=36,color='#20252C'):
 base_eq(tex,20+x*.67,85+y*.85,size*(.75 if size==36 else .94),color)
c.saveState();c.translate(20,H-85-.85*H);c.scale(.67,.85)
# (a) Batched construction; teacher-specific encoders feed one shared pass.

for x,name,icon in [(155,'Text CoT',icon_speech),(470,'Visual CoT',icon_visual),(785,'3D CoT',icon_cube)]:
 label_width=pdfmetrics.stringWidth(name.replace('Text CoT','text CoT').replace('Visual CoT','visual CoT').replace('3D CoT','3d CoT'),HN_B,42)
 group_left=x-(label_width+52)/2
 c.saveState();c.translate(group_left+18,H-22);c.scale(.8,.8);icon(0,H,SLATE);c.restoreState()
 label(name.replace('Text CoT','text CoT').replace('Visual CoT','visual CoT').replace('3D CoT','3d CoT'),group_left+52,22,30)
 arr(x,53,x,76)
 rounded_rect(x-144,82,288,48,9,CARD,HexColor('#ABB7C9'),1.4,False)
 label({'Text CoT':'text encoder','Visual CoT':'visual encoder','3D CoT':'3d encoder'}[name],x,106,30,anchor='center')
 arr(x,139,x,177)
for i,(sx,lx) in enumerate([(155,295),(470,610),(785,925)],1):
 index='i' if i==3 else str(i)
 tok(('s',index,''),sx,220,w=64/.67,h=60/.85);tok(('q','',''),lx,220,w=64/.67,h=60/.85)
 arr(sx,264,sx,301);arr(lx,264,lx,301)
rounded_rect(37,309,940,62,10,HexColor('#EEF2F8'),HexColor('#95A4BC'),1.6,False)
label('LLM backbone',507,340,32,anchor='center')
for i,x in enumerate((295,610,925),1):
 arr(x,379,x,427);tok(('z','i' if i==3 else str(i),'*'),x,470,w=64/.67,h=60/.85)
label('latents',40,470,30,color=TEXT_2)
dots(700,220);dots(767,470)
# Square mask cells: use the same x/y scale, independently of panel placement.
c.saveState();c.resetTransforms();c.translate(20,H-140-.67*H);c.scale(.67,.67)
encoding_eq=eq;encoding_label=label
def eq(tex,x,y,size=36,color='#20252C'):
 base_eq(tex,20+x*.67,140+y*.67,size*.75,color)
def label(txt,x,y,size=30,*args,**kwargs):
 c.saveState();c.resetTransforms()
 base_label(txt,20+x*.67,140+y*.67,25,*args,**kwargs)
 c.restoreState()
def mask_symbol(spec,x,y):
 # Same bold-italic math face and script sizing as all other s/q tokens.
 base,idx,_=spec
 sub=('_{'+idx+'}') if idx else ''
 eq(r'$\mathbfit{'+base+'}'+sub+'$',x,y,33 if base=='q' else 36,'#20252C')
# Full construction mask: only self and the immediately preceding s_i.
label('Attention mask',1224,90,30,anchor='center')
ms=[('s','1',''),('q','',''),('s','2',''),('q','',''),None,('s','i',''),('q','','')]
maskchecks=[]
for r,sp in enumerate(ms):
 cx=1098+(r+.5)*36;cy=170+(r+.5)*36
 if sp is None:
  dots(cx,137)
  for dy in (-8,0,8):
   c.setFillColor(TEXT_2);c.circle(1054,H-cy-dy,2.2,fill=1,stroke=0)
 else:
  mask_symbol(sp,cx,137);mask_symbol(sp,1054,cy)
 for j,ksp in enumerate(ms):
  xx=1098+(j+.5)*36;yy=170+(r+.5)*36
  if sp is None or ksp is None:
   if sp is None and ksp is not None:
    for dy in (-8,0,8):
     c.setFillColor(TEXT_2);c.circle(xx,H-yy-dy,2.2,fill=1,stroke=0)
   else:dots(xx,yy)
   continue
  allowed=(r,j) in ((1,0),(3,2),(6,5))
  fill='#52627B' if r==j or allowed else '#FFFFFF'
  c.setFillColor(HexColor(fill));c.setStrokeColor(HAIR);c.setLineWidth(.85)
  c.rect(1098+j*36,H-170-(r+1)*36,36,36,fill=1,stroke=1)
  maskchecks.append((xx,yy,fill))

# Binary mask key, consistent with the appendix row-attends-to-column convention.
rounded_rect(1130,453,20,20,0,HexColor('#52627B'),HexColor('#52627B'),1,False)
label('allowed',1170,463,25)
rounded_rect(1130,491,20,20,0,CARD,HexColor('#000000'),1.2,False)
label('blocked',1170,501,25)
c.restoreState();eq=encoding_eq;label=encoding_label
c.restoreState();eq=base_eq;label=base_label;arr=base_arr
# Grounding detail: the prefix and targets flank the same parameterized backbone.
panel(985,110,795,365,'(b) Continuation Prediction','#6A9DD3')
y=300
for x,sp in [(1055,X),(1160,('z','1','*')),(1270,('z','i','*'))]:mini(sp,x,y,60)
dots(1215,y)
bracket(1007,1302,245,'input sequence')
arr(1310,y,1342,y);llm(1400,y,100);arr(1458,y,1477,y)
for x,sp in [(1517,('s','i+1','')),(1600,('s','i+2','')),(1722,Y)]:
 mini(sp,x,y,64)
dots(1653,y)
bracket(1485,1770,245,'future steps + output')
line(1485,363,1770,363,SLATE,2.2)
line(1485,343,1485,363,SLATE,2.2)
line(1770,343,1770,363,SLATE,2.2)
eq(r'$\mathcal{L}_{\mathrm{cont}}$',1627.5,425,48,'#397DBB')
# Additional vertical breathing room at fixed ICLR width.
page_eq=eq
def eq(tex,x,y,size=36,color='#20252C'):
 page_eq(tex,x,y+80,size,color)
c.saveState();c.translate(0,-80)
# Overview: forward computation only; the snowflake gates the diffusion branch.
box(150,585,250,95,[])
# Document icon, matching the reference's teacher-step pictogram.
rounded_rect(44,558,37,49,3,CARD,SLATE,2,False)
for yy in (570,579,588,597):line(51,yy,73,yy,SLATE,2)
label('Teacher',176,568,29,anchor='center')
label('CoT Step',176,601,29,anchor='center')
arr(284,585,330,585)
box(460,585,240,95,['Latent','Encoding'],'#F2EFF8','#6356A5',30)
arr(590,574,658,574)
mini(('z','i','*'),700,585,105)
line(740,574,805,574,SLATE,5)
line(805,514,805,665,SLATE,5)
arr(805,514,910,514);arr(805,665,910,665)
# Center the slightly larger stop-gradient badge in the branch-to-module gap.
c.saveState();c.translate(862.5,H-665);c.scale(1.2,1.2)
stop_marker(0,H)
c.restoreState()
box(1040,525,240,80,['Continuation','Prediction'],'#EFF5FC','#397DBB',28)
box(1040,665,240,80,['Diffusion','Training'],'#FCF3EB','#CB884D',28)
arr(1168,514,1237,514);arr(1168,654,1237,654)
box(1340,525,190,72,[],'#FFFFFF','#6A9DD3');eq(r'$\mathcal{L}_{\mathrm{cont}}$',1340,525,44,'#397DBB')
box(1340,665,190,72,[],'#FFFFFF','#D9A073');eq(r'$\mathcal{L}_{\mathrm{diff}}$',1340,665,44,'#CB884D')
# Reverse gradient flow follows the forward computation one stage at a time.
# Separate arrows keep the path readable and expose every optimized module.
gradient_arrow([(1237,536),(1168,536)],HexColor('#397DBB'))
gradient_arrow([(910,536),(827,536),(827,596),(740,596)],HexColor('#397DBB'))
gradient_arrow([(658,596),(590,596)],HexColor('#397DBB'))
gradient_arrow([(1237,676),(1168,676)],HexColor('#CB884D'))
# Joint objective: the two loss branches merge, without implying gradient arrows.
green=HexColor('#397D3C')
def merge_path(y):
 c.saveState();c.setStrokeColor(green);c.setLineWidth(5);c.setLineCap(1)
 path=c.beginPath();path.moveTo(1445,H-y)
 direction=1 if y<595 else -1
 path.lineTo(1466,H-y)
 path.curveTo(1478,H-y,1488,H-(y+10*direction),1488,H-(y+22*direction))
 path.lineTo(1488,H-(595-22*direction))
 c.drawPath(path);c.restoreState()
merge_path(525);merge_path(665)
c.setStrokeColor(green);c.setFillColor(HexColor('#F2F8EF'));c.setLineWidth(2.4)
c.circle(1488,H-595,22,fill=1,stroke=1)
# Geometric cross is exactly centered, independent of font sidebearings.
line(1479,595,1497,595,green,3.6)
line(1488,586,1488,604,green,3.6)
open_straight(1515,595,1558,595,green,5,15)
box(1670,595,210,76,['Joint Training'],'#F2F8EF','#397D3C',27)
# Detail links are plain dashed connectors, with no gradient arrows.
line(460,538,600,460,HexColor('#9581BD'),1.5,[7,5])
line(1040,485,1100,395,HexColor('#6A9DD3'),1.5,[7,5])
line(1040,707,1080,735,HexColor('#D9A073'),1.5,[7,5])
# Diffusion detail.
panel(590,735,1190,260,'(c) Diffusion Training','#D9A073')
y=895
for x,sp in [(655,X),(755,('z','1','*')),(845,('z','2','*')),(1010,('z','i-1','*'))]:
 mini(sp,x,y)
 if sp[2]=='*':stop_marker(x+31,y-29)
dots(927,y)
bracket(607,1042,844,'conditioning context')
# No arrows above latent labels. Noise texture uses the approved glyph-safe pattern.
mini(('z','i','t'),1145,y)
# Peripheral stippling keeps the time and index unobstructed.
c.setFillColor(HexColor('#7DABD8'))
for dx in (-25,25):
 for dy in (-20,0,20):c.circle(1145+dx,H-y-dy,1.3,fill=1,stroke=0)
label('noisy latent',1145,836,24,anchor='center')
arr(1187,y,1235,y);llm(1310,y,130);arr(1384,y,1440,y)
mini(('z','i',''),1490,y,72)
# Replace the plain output glyph with a hat using the standard math renderer.
maths.pop();eq(r'$\hat z_i$',1490,y,38)
label('denoised latent',1490,836,24,anchor='center')
mini(('z','i','*'),1720,y,72);stop_marker(1751,y-29);label('gt',1720,836,24,anchor='center')
# Both operands feed the loss below, leaving the token row uncluttered.
line(1490,934,1490,952,SLATE,5)
line(1720,934,1720,952,SLATE,5)
arr(1490,952,1550,952);arr(1720,952,1660,952)
eq(r'$\mathcal{L}_{\mathrm{diff}}$',1605,952,44,'#CB884D')
# Compact legend in the reference's lower-left space.
gradient_arrow([(128,795),(55,795)],SLATE,3.2,13)
label('gradient flow',150,795,27)
stop_marker(90,850);label('stop gradient',150,850,28)
mini(('q','',''),90,915,55);label('learnable embedding',150,915,26)
# Inference: each generated latent is the output of denoising, with a hat.
line(20,1025,1780,1025,HAIR,1.5)
label('Inference',40,1080,32,True)
seq=[(300,X),(580,('z','1','')),(900,('z','2','')),(1390,('z','i','')),(1730,Y)]
for x,sp in seq:
 mini(sp,x,1080)
 if sp[0]=='z':
  maths.pop();eq(r'$\hat z_{'+sp[1]+'}$',x,1080,34)
arr(356,1080,540,1080);label('denoise',441,1054,24,anchor='center')
arr(622,1080,860,1080);label('denoise',741,1054,24,anchor='center')
arr(942,1080,1085,1080);dots(1120,1080);arr(1155,1080,1350,1080)
label('denoise',1252,1054,24,anchor='center')
arr(1432,1080,1674,1080);label('predict',1561,1054,24,anchor='center')
c.restoreState();eq=page_eq
c.save()
doc=fitz.open(OUT)
for tex,x,y,size,col in maths:
 tex=r'$\boldsymbol{'+tex[1:-1]+'}$'
 buf=io.BytesIO();math_to_image(tex,buf,format='pdf',color=col,prop=FontProperties(size=size,math_fontfamily='stix'))
 m=fitz.open(stream=buf.getvalue(),filetype='pdf');r=m[0].rect
 # Subtle vector fill+stroke makes small token notation read clearly in print.
 # Preserve the established math face, rather than swapping font families.
 if r'\mathcal' not in tex:
  rgb=tuple(int(col[j:j+2],16)/255 for j in (1,3,5))
  streams=m[0].get_contents()
  for stream in streams:
   content=m.xref_stream(stream)
   prefix=('q %.5f %.5f %.5f RG 0.38 w 2 Tr\n'%rgb).encode()
   m.update_stream(stream,prefix+content+b'\nQ')
 doc[0].show_pdf_page(fitz.Rect(x-r.width/2,y-r.height/2,x+r.width/2,y+r.height/2),m,0)
final=fitz.open();p=final.new_page(width=396,height=H*396/W);p.show_pdf_page(p.rect,doc,0)
final.save(OUT,garbage=4,deflate=True)
p.get_pixmap(matrix=fitz.Matrix(5,5),alpha=False).save(OUT.with_suffix('.png'))
ImageOps.grayscale(Image.open(OUT.with_suffix('.png'))).save(OUT.with_name(OUT.stem+'-gray.png'))
print(OUT)
