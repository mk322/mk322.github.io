"""Flat, editable research diagrams; independent desktop and mobile compositions."""
from pathlib import Path
from html import escape
OUT=Path(__file__).resolve().parents[1]
C={'expert':('#ba8b51','#fbf7f0'),'student':('#3e73a8','#f2f7fc'),'sampler':('#6356a5','#f5f3fa'),'data':('#2b8b88','#eff8f7'),'neutral':('#8a9099','#fafbfc')}
def start(w,h,title):
 return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img"><title>{escape(title)}</title><defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L8 4 L0 8" fill="none" stroke="#747b86" stroke-width="1.2"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Inter, Helvetica Neue, Arial, sans-serif" fill="#20252c">']
def txt(s,x,y,t,size=15,weight=400,color='#525a65',anchor='middle'):
 s.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(t)}</text>')
def box(s,x,y,w,h,title,sub='',role='neutral'):
 stroke,fill=C[role];s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}" stroke-opacity=".6" stroke-width="1.1"/>')
 txt(s,x+w/2,y+h/2+(0 if sub else 5),title,15,500,'#20252c')
 if sub:txt(s,x+w/2,y+h/2+20,sub,12)
def arrow(s,d,dash=False):s.append(f'<path d="{d}" fill="none" stroke="#747b86" stroke-width="1.2" marker-end="url(#arrow)"'+(' stroke-dasharray="4 4"' if dash else '')+'/>')
def save(s,name):OUT.joinpath(name+'.svg').write_text(''.join(s)+ '</g></svg>')
# Desktop comparison: repeated response search vs one learned set of parameters.
s=start(780,400,'MCMC searches per example; E2S reuses sampler parameters')
txt(s,185,28,'MCMC: search per example',19,600,'#20252c');txt(s,580,28,'E2S: amortize the search',19,600,'#20252c')
s.append('<path d="M390 12 V385" stroke="#e4e6eb"/>')
for j,y in enumerate([85,245],1):
 box(s,8,y,102,66,'Expert',f'example {j}','expert');box(s,137,y,106,66,'Candidate',f'for prompt {j}');box(s,270,y,102,66,'Training','response','data')
 arrow(s,f'M112 {y+33} H132');arrow(s,f'M245 {y+33} H265')
 arrow(s,f'M217 {y-3} V{y-24} H161 V{y-3}')
 txt(s,188,y-32,'edit · score · accept / reject',11)
txt(s,190,366,'New prompt, new response search',13)
box(s,412, sixty:=62,156,60,'Expert examples','many prompts','expert');box(s,603,62,156,60,'Student','scores responses','student')
box(s,506,161,160, sixty,'Train sampler','shared parameters','sampler')
arrow(s,'M490 124 V143 H552 V157');arrow(s,'M681 124 V143 H620 V157')
arrow(s,'M586 223 V265');txt(s,597,248,'reuse parameters',12,anchor='start')
box(s,423,270,172,68,'E2S sampler','new expert example','sampler');box(s,632,270,135,68,'E2S data','training responses','data')
arrow(s,'M597 304 H627');txt(s,589,366,'Learning carries across prompts',13)
save(s,'amortization')
# Mobile comparison preserves visible per-example repetition.
s=start(360,710,'MCMC repeats search; E2S reuses learned parameters')
txt(s,180,27,'MCMC: search per example',18,600,'#20252c')
for j,y in enumerate([83,205],1):
 box(s,7,y,99,60,'Expert',f'example {j}','expert');box(s,131,y,98,60,'Candidate');box(s,254,y,99,60,'Training','response','data');arrow(s,f'M108 {y+30} H127');arrow(s,f'M231 {y+30} H250');arrow(s,f'M204 {y-3} V{y-20} H155 V{y-3}');txt(s,180,y-28,'edit · score · accept / reject',12)
txt(s,180,297,'New prompt, new search',13)
s.append('<path d="M8 323 H352" stroke="#e4e6eb"/>');txt(s,180,357,'E2S: amortize the search',18,600,'#20252c')
box(s,8,385,158,60,'Expert examples','many prompts','expert');box(s,194,385,158,60,'Student','scores responses','student')
arrow(s,'M87 447 V468 H148 V490');arrow(s,'M273 447 V468 H212 V490');box(s,92,494,176,60,'Train sampler','shared parameters','sampler');arrow(s,'M180 556 V607');txt(s,190,585,'reuse parameters',12,anchor='start')
box(s,7,612,168,68,'E2S sampler','new expert example','sampler');box(s,209,612,144,68,'E2S data','training responses','data');arrow(s,'M177 646 H204');save(s,'amortization-mobile')
# Symmetric training schedules, with feedback exclusively in the online version.
for online in [False,True]:
 name='online' if online else 'offline'
 title='E2S-Online' if online else 'E2S-Offline'
 s=start(780,230 if online else 200,title)
 for x,num,heading,sub,role in [(12,'01','Fit sampler','Current student' if online else 'Fixed student','sampler'),(278,'02','Generate batch' if online else 'Generate dataset','E2S data','data'),(544,'03','SFT','Updated student','student')]:
  txt(s,x+112,27,num,12,500);box(s,x,48,224, ninety:=90,heading,sub,role)
 arrow(s,'M239 93 H272');arrow(s,'M505 93 H538')
 if online:arrow(s,'M656 141 V193 H124 V143');txt(s,390,184,'Refit against the updated student · repeat',14)
 else:txt(s,390,179,'Fit once. Generate a fixed dataset. Then fine-tune.',14)
 save(s,name)
 s=start(360,460 if online else 428,title)
 for i,(heading,sub,role) in enumerate([('Fit sampler','Current student' if online else 'Fixed student','sampler'),('Generate batch' if online else 'Generate dataset','E2S data','data'),('SFT','Updated student','student')]):
  y=24+i*134;box(s,32,y,264,90,heading,sub,role)
  if i<2:arrow(s,f'M164 {y+93} V{y+129}')
 if online:arrow(s,'M299 337 H337 V69 H300');txt(s,180,425,'Refit against the updated student',14);txt(s,180,445,'Repeat the cycle',13)
 else:txt(s,180,419,'Fit once, then use the fixed dataset',14)
 save(s,name+'-mobile')
# Optional implementation diagram.
for mobile in [False,True]:
 s=start(360 if mobile else 780,390 if mobile else 190,'One backbone, two modes')
 if mobile:
  box(s,20,35,320,100,'Adapter on: sampler','Prompt + expert trace → response','sampler');box(s,20,217,320,100,'Adapter off: student','Prompt → score response / SFT','student');txt(s,180,180,'Same backbone',14);txt(s,180,362,'Expert trace is not a student input',13)
 else:
  box(s,8,48,352,96,'Adapter on: sampler','Prompt + expert trace → response','sampler');box(s,420,48,352,96,'Adapter off: student','Prompt → score response / SFT','student');txt(s,390,24,'One backbone, two modes',17,600,'#20252c');txt(s,390,180,'Expert trace is not a student input',13)
 save(s,'lora'+('-mobile' if mobile else ''))
