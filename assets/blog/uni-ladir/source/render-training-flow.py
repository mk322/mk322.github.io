"""Exact forward / gradient routes for the blog's joint-training walkthrough."""
from pathlib import Path
from html import escape
ROOT=Path(__file__).resolve().parents[4]
I='#20252C';M='#525A65';P='#6356A5';G='#2B8B88'
def txt(x,y,s,size=19,bold=False,c=I):return f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" font-weight="{700 if bold else 400}" fill="{c}">{escape(s)}</text>'
def node(key,x,y,w,h,lines,learned=False):
 s=f'<g data-node="{key}" class="uni-flow-node"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{("#F2F0F8" if learned else "#FAFBFD")}" stroke="{(P if learned else "#D7DCE5")}" stroke-width="1.7"/>'
 for j,l in enumerate(lines):s+=txt(x+w/2,y+h/2+(j-(len(lines)-1)/2)*25+6,l,20,j==0,c=P if learned else I)
 return s+'</g>'
def edge(key,path,grad=False):return f'<path data-edge="{key}" class="uni-flow-edge {"uni-gradient" if grad else "uni-forward"}" d="{path}" fill="none" stroke="{G if grad else M}" stroke-width="2.4" marker-end="url(#uf-{"grad" if grad else "forward"})"/>'
b=txt(500,34,'One backbone, two training paths',27,True)+txt(500,66,'Solid arrows: forward computation     Dashed teal arrows: gradients',19,c=M)
b+=txt(175,107,'CONTINUATION + FINAL OUTPUT',18,True,c=P)
b+=node('teacher',20,163,140,76,['Teacher-step','features sᵢ'])
b+=node('encoder',205,163,155,76,['Encoder θ','local to step'],True)
b+=node('thought',409,163,144,76,['Thought','block zᵢ*'])
b+=node('continuation',610,163,172,76,['Continuation','predictor θ'],True)
b+=node('contloss',839,163,141,76,['Loss','Lcont'])
b+=node('task',610,94,172,48,['Task input x'])
b+=txt(910,120,'Later teacher steps',17)+txt(910,142,'+ final output',17)
b+=edge('encode','M160 187H200')+edge('encode','M360 187H404')
b+=edge('cont-forward','M553 187H605')+edge('cont-forward','M782 187H834')+edge('cont-forward','M696 142V158')+edge('cont-forward','M910 146V158')
b+=edge('cont-backward','M839 220H787',True)+edge('cont-backward','M610 220H558',True)+edge('cont-backward','M409 220H365',True)
b+=txt(698,269,'Uses the permitted thought prefix',17,c=M)
b+='<line x1="20" y1="291" x2="980" y2="291" stroke="#D7DCE5"/>'
b+=txt(108,326,'DIFFUSION',18,True,c=P)
b+=node('stop',384,310,196,46,['sg(zᵢ*) · fixed'])
b+=edge('detach','M481 239V305')
b+=node('noise',20,430,140,76,['Gaussian','noise εᵢ'])
b+=node('noisy',210,430,280,76,['Noisy block zᵢᵗ','(1 − t)εᵢ + t sg(zᵢ*)'])
b+=node('diffusion',558,430,188,76,['Velocity','predictor θ'],True)
b+=node('diffloss',839,430,141,76,['Loss','Ldiff'])
b+=node('prefix',558,364,188,52,['Task input x','+ sg(z<ᵢ*)'])
b+=node('velocity',773,321,207, 62,['Fixed target vᵢ*','sg(zᵢ*) − εᵢ'])
b+=edge('diff-forward','M160 455H205')+edge('diff-forward','M481 356V387H350V425')+edge('diff-forward','M580 333H768')
b+=edge('diff-forward','M490 455H553')+edge('diff-forward','M746 455H834')+edge('diff-forward','M652 416V425')+edge('diff-forward','M910 383V425')
b+=edge('diff-backward','M839 487H751',True)
# The dashed return reaches a stop at the fixed input, not the encoder.
b+=edge('diff-stop','M558 487H519',True)
b+='<g data-node="stop-bar"><path d="M515 473V500" stroke="#2B8B88" stroke-width="3"/>'+txt(455,534,'No gradient into fixed blocks',17,c=G)+'</g>'
b+=txt(500,576,'All three purple modules share the same weights θ.',22,True,c=P)
b+=txt(500,608,'Either loss updates θ. Diffusion gradients stop at the detached thought blocks.',19,c=M)
svg=f'<svg xmlns="http://www.w3.org/2000/svg" class="uni-training-svg" viewBox="0 0 1000 634" role="img" aria-labelledby="uf-title uf-desc"><title id="uf-title">Forward computation and gradients in joint training</title><desc id="uf-desc">Continuation loss backpropagates through predicted continuations, thought blocks, and the encoder. Diffusion loss updates the same backbone weights through velocity prediction. Clean thought targets and prefix are detached. Each teacher step is encoded locally.</desc><defs><marker id="uf-forward" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8Z" fill="{M}"/></marker><marker id="uf-grad" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8Z" fill="{G}"/></marker></defs><rect width="1000" height="634" fill="white"/><g font-family="Helvetica Neue, Helvetica, Arial, sans-serif">{b}</g></svg>'
(ROOT/'_includes/blog/uni-ladir/training-flow-svg.html').write_text(svg)
