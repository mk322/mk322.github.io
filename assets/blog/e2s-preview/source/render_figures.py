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
 stroke,fill=C[role]
 # Data are drawn as small document stacks; models remain single modules.
 if role in ('expert','data') and h>50:
  s.append(f'<rect x="{x+3}" y="{y-3}" width="{w}" height="{h}" rx="6" fill="white" stroke="{stroke}" stroke-opacity=".22" stroke-width="1"/>')
 s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-opacity=".42" stroke-width="1"/>')
 s.append(f'<path d="M{x+12} {y+12} H{x+32}" stroke="{stroke}" stroke-width="2" stroke-linecap="round"/>')
 txt(s,x+w/2,y+h/2+(1 if sub else 5),title,14,500,'#20252c')
 if sub:txt(s,x+w/2,y+h/2+19,sub,11.5)
def arrow(s,d,dash=False):s.append(f'<path d="{d}" fill="none" stroke="#747b86" stroke-width="1.2" marker-end="url(#arrow)"'+(' stroke-dasharray="4 4"' if dash else '')+'/>')
def save(s,name):OUT.joinpath(name+'.svg').write_text(''.join(s)+ '</g></svg>')
# Explicit data path: expert trace tau -> constrained response y -> student SFT.
s=start(780,535,'Off-policy expert traces become more on-policy responses for student SFT')
txt(s,12,27,'MCMC · search per example',18,500,'#20252c','start')
txt(s,12,51,'Each expert trace starts a new response search.',14,anchor='start')
for j,y in enumerate([76,184],1):
 sub='₁' if j==1 else '₂'
 box(s,12,y,150,72,'Expert trace τ'+sub,'off-policy','expert')
 box(s,214,y,150,72,'MCMC search','edit · score · accept')
 box(s,416,y,150,72,'Response y'+sub,'more on-policy','data')
 box(s,618,y,150,72,'Student SFT','train on (x'+sub+', y'+sub+')','student')
 for a,b in [(164,209),(366,411),(568,613)]:arrow(s,f'M{a} {y+36} H{b}')
s.append('<path d="M12 279 H768" stroke="#e4e6eb"/>')
txt(s,12,312,'E2S · learn across examples',18,500,'#20252c','start')
box(s,12,339,150,64,'Expert corpus','off-policy traces','expert')
box(s,214,339,190,64,'Fit sampler','student score + constraint','sampler')
arrow(s,'M164 371 H209');arrow(s,'M309 406 V454',True)
txt(s,425,365,'PARAMETERS → REUSE',11,500,'#6356a5','start')
txt(s,425,387,'Share parameters across expert examples',13,anchor='start')
box(s,12,459,150,64,'Expert trace τ','off-policy','expert')
box(s,214,459,150,64,'E2S sampler','reuse parameters','sampler')
box(s,416,459,150,64,'Response y','more on-policy','data')
box(s,618,459,150,64,'Student SFT','train on (x, y)','student')
for a,b in [(164,209),(366,411),(568,613)]:arrow(s,f'M{a} 491 H{b}')
save(s,'amortization')

s=start(360,1090,'Expert trace tau becomes response y, which is used for student SFT')
txt(s,12,27,'MCMC · search per example',18,500,'#20252c','start')
for i,(title,sub,role) in enumerate([('Expert trace τ','off-policy','expert'),('MCMC search','edit · score · accept','neutral'),('Response y','more on-policy','data'),('Student SFT','train on (x, y)','student')]):
 y=55+i*98;box(s,40,y,280,64,title,sub,role)
 if i<3:arrow(s,f'M180 {y+67} V{y+94}')
txt(s,180,442,'Repeat the search for every expert example.',13)
s.append('<path d="M12 466 H348" stroke="#e4e6eb"/>')
txt(s,12,502,'E2S · reuse a learned sampler',18,500,'#20252c','start')
box(s,8,529,153,72,'Expert corpus','off-policy traces','expert')
box(s,196,529,156,72,'Train sampler','scores + constraint','sampler')
arrow(s,'M163 565 H191')
# Parameter reuse enters the sampler from the right, separate from the data path.
arrow(s,'M275 604 V617 H338 V803 H304',True)
txt(s,12,636,'Training carries across examples.',13,anchor='start')
for i,(title,sub,role) in enumerate([('Expert trace τ','off-policy','expert'),('E2S sampler','reuse parameters','sampler'),('Response y','more on-policy','data'),('Student SFT','train on (x, y)','student')]):
 y=655+i*116;box(s,40,y,260,64,title,sub,role)
 if i<3:arrow(s,f'M170 {y+67} V{y+112}')
save(s,'amortization-mobile')

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

# Two ways to address mismatch: where the trajectory comes from, where supervision enters.
s=start(780,490,'Two routes: supervise student rollouts, or prepare student-targeted SFT data')
txt(s,12,28,'A   Move learning to the student',18,500,'#20252c','start')
txt(s,12,53,'Let the student write a response, then give feedback on that attempt.',14,anchor='start')
box(s,12,84,150, eighty:=80,'Student','samples responses','student')
box(s,206,84,150,80,'Student responses','generated answer','student')
box(s,400,84,174,80,'Attach feedback','reward / teacher feedback','expert')
box(s,618,84,150,80,'Update student','RL / distillation','student')
for a,b in [(164,201),(358,395),(576,613)]:arrow(s,f'M{a} 124 H{b}')
txt(s,390,195,'Give feedback on what the student actually writes.',14)
s.append('<path d="M12 221 H768" stroke="#e4e6eb"/>')
txt(s,12,256,'B   Move the data to the student',18,500,'#20252c','start')
txt(s,12,281,'Start with an expert constraint; prepare valid responses for student SFT.',14,anchor='start')
box(s,12,313,150,80,'Expert constraint','what must be preserved','expert')
box(s,206,313,174,80,'Sample valid data','MCMC / E2S','sampler')
box(s,424,313,150,80,'Training targets','more on-policy','data')
box(s,618,313,150,80,'Update student','SFT','student')
for a,b in [(164,201),(382,419),(576,613)]:arrow(s,f'M{a} 353 H{b}')
box(s,206,433,174,44,'Student policy',role='student');arrow(s,'M293 430 V398')
txt(s,401,450,'sets relative probabilities',13,anchor='start');txt(s,401,469,'among valid responses',13,anchor='start')
save(s,'two-routes')

s=start(360,958,'Two routes: feedback on student rollouts versus new SFT targets')
txt(s,12,26,'A   Move learning to the student',18,500,'#20252c','start')
txt(s,12,51,'Let the student write, then give feedback.',13,anchor='start')
box(s,40,78,280,76,'Student responses','Sample from the current student','student')
arrow(s,'M180 157 V188')
box(s,40,192,280,76,'Attach feedback','reward / teacher feedback','expert')
arrow(s,'M180 271 V302')
box(s,40,306,280,76,'Update student','RL / distillation','student')
txt(s,180,413,'Give feedback on generated answer.',13)
s.append('<path d="M12 443 H348" stroke="#e4e6eb"/>')
txt(s,12,479,'B   Move the data to the student',18,500,'#20252c','start')
txt(s,12,504,'Prepare new SFT targets before updating.',13,anchor='start')
box(s,8,535,166,72,'Expert constraint','what to preserve','expert')
box(s,186,535,166,72,'Student policy','relative probabilities','student')
arrow(s,'M91 610 V632 H142 V653');arrow(s,'M269 610 V632 H218 V653')
box(s,40,657,280,76,'Sample valid data','MCMC / E2S','sampler')
arrow(s,'M180 736 V767')
box(s,40,771,280, sixty:=64,'Training targets','more on-policy','data')
arrow(s,'M180 838 V869')
box(s,40,873,280, sixty,'Update student','SFT','student')
save(s,'two-routes-mobile')
