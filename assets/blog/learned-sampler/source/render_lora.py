"""Shared-backbone sampler roles, original SVG with mobile composition."""
from render_figures import start,text,rect,line,save,PURPLE,TEAL,MUTED

def draw(mobile=False):
 w,h=(380,550) if mobile else (800,320)
 s=start(w,h,'One backbone, two roles','The expert-conditioned sampler enables a separate LoRA adapter. Prompt-only student scoring and SFT disable the adapter. The backbone evolves only during student SFT.')
 text(s,22,34,'One backbone. Two roles.',22 if mobile else 24,weight=600)
 rect(s,22,63,w-44,66,'#fafbfd')
 text(s,w/2,91,'Current student backbone',19,weight=600,anchor='middle')
 text(s,w/2,115,'Shared weights θ',16,MUTED,anchor='middle')
 coords=[(22,153,336),(22,333,336)] if mobile else [(22,168,340),(438,168,340)]
 for i,(x,y,pw) in enumerate(coords):
  rect(s,x,y,pw,143,'#f2f0f8' if i==0 else '#eff7f6')
  text(s,x+18,y+31,'Sampler · LoRA on' if i==0 else 'Student · LoRA off',19,PURPLE if i==0 else TEAL,600)
  text(s,x+18,y+65,'Input: prompt + expert trace' if i==0 else 'Input: prompt only',16,MUTED)
  text(s,x+18,y+95,'Generate verified SFT responses' if i==0 else 'Score responses / run SFT',16)
  text(s,x+18,y+123,'Sampler phase: update adapter φ' if i==0 else 'SFT phase: update backbone θ',15,MUTED)
 if not mobile:
  line(s,'M230 132V156');line(s,'M570 132V156')
 else:text(s,w/2,518,'Adapter stays separate from the task model',14,MUTED,anchor='middle')
 save(s,'lora-sampler'+('-mobile' if mobile else '')+'.svg')
for mobile in (False,True):draw(mobile)
