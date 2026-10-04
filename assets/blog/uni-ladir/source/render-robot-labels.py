"""Expand robot-state labels in blog copies; original paper PDFs stay intact."""
from pathlib import Path
import fitz
from fontTools.ttLib import TTCollection
from io import BytesIO
ROOT=Path(__file__).resolve().parent
for name in ['fig1','paper-sharing']:
 d=fitz.open(ROOT/(name+'.pdf'));p=d[0]
 font_ref=next(f[0] for f in p.get_fonts() if ('HelveticaNeue-Bold' if name=='fig1' else 'TimesNewRoman') in f[3])
 if name=='fig1':
  stream=BytesIO();TTCollection('/System/Library/Fonts/HelveticaNeue.ttc').fonts[1].save(stream);fontbuf=stream.getvalue()
 else:fontbuf=d.extract_font(font_ref)[3]
 font=fitz.Font(fontbuffer=fontbuf)
 edits=[]
 for b in p.get_text('dict')['blocks']:
  for line in b.get('lines',[]):
   for sp in line['spans']:
    old=sp['text']
    if old not in ['State','state enc.','Visual + State','3D + State']:continue
    r=fitz.Rect(sp['bbox']);edits.append((old,r,sp));p.add_redact_annot(r,fill=(1,1,1))
 p.apply_redactions(images=0,graphics=0)
 p.insert_font(fontname='BlogOriginal',fontbuffer=fontbuf)
 for old,r,sp in edits:
  col=tuple(((sp['color']>>shift)&255)/255 for shift in (16,8,0))
  cx=(r.x0+r.x1)/2
  if old=='State': lines=['Robot','state'];size=22;ys=[r.y0+13,r.y0+36]
  elif old=='state enc.':lines=['Robot state enc.'];size=23;ys=[sp['origin'][1]-1]
  else:lines=[old.split(' + ')[0]+' +','Robot state'];size=sp['size']*.94;ys=[sp['origin'][1]-6,sp['origin'][1]+10]
  for label,y in zip(lines,ys):
   x=cx-font.text_length(label,fontsize=size)/2
   p.insert_text((x,y),label,fontname='BlogOriginal',fontsize=size,color=col)
 d.save(ROOT/(name+'-blog.pdf'),garbage=4,deflate=True)
 d.close();d=fitz.open(ROOT/(name+'-blog.pdf'));p=d[0]
 p.get_pixmap(matrix=fitz.Matrix(1800/p.rect.width,1800/p.rect.width)).save(ROOT.parent/(name+'.png'))
 (ROOT.parent/(name+'.svg')).write_text(p.get_svg_image(text_as_path=True))
