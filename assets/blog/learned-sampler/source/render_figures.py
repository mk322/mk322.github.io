"""Editable, original conceptual diagrams. Run with Python 3; no dependencies."""
from pathlib import Path
from html import escape
OUT = Path(__file__).resolve().parents[1]
INK, MUTED, PURPLE, BLUE, TEAL, RED = '#20252c', '#525a65', '#6356a5', '#3e73a8', '#2b8b88', '#bf656a'

def start(w,h,title,desc):
    return [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="{MUTED}"/></marker><marker id="purple-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{PURPLE}"/></marker></defs><rect width="{w}" height="{h}" rx="12" fill="#fff"/><g font-family="Helvetica Neue, Arial, sans-serif" fill="{INK}">''']
def text(s,x,y,t,size=17,color=INK,weight=400,anchor='start'):
    s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(t)}</text>')
def rect(s,x,y,w,h,fill='#fafbfd',stroke='#d7dce5',rx=10):
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
def line(s,d,purple=False,dash=False):
    attr = 'stroke-dasharray="5 5"' if dash else ''
    color = PURPLE if purple else MUTED
    marker = 'purple-arrow' if purple else 'arrow'
    s.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.7" stroke-linejoin="round" {attr} marker-end="url(#{marker})"/>')
def save(s,name):
    (OUT/name).write_text('\n'.join(s)+ '\n</g></svg>\n')
def panel(s,x,y,w,label,vals,condition=False):
    rect(s,x,y,w,230,'#f7f5fb' if condition else '#fafbfd','#ded8ec' if condition else '#d7dce5')
    text(s,x+20,y+32,label,20,weight=600)
    labels=['Valid A','Valid B','Invalid']; colors=[PURPLE,TEAL,RED]
    for i,(v,l,c) in enumerate(zip(vals,labels,colors)):
        yy=y+70+i*50
        text(s,x+20,yy,l,16)
        text(s,x+w-20,yy,('⅔' if condition and i==0 else '⅓' if condition and i==1 else f'{v:g}%'),16,anchor='end',weight=500)
        bx,by,bw=x+20,yy+10,w-40
        rect(s,bx,by,bw,9,'#e9eaf0','#e9eaf0',4)
        if v: rect(s,bx,by,bw*v/100,9,c,c,4)

def conditioning(mobile=False):
    w,h=(380,648) if mobile else (800,384)
    s=start(w,h,'Correctness changes the support, not the valid-solution ratio','Toy student probabilities of 6%, 3%, and 91% become two thirds, one third, and zero when conditioned on validity.')
    text(s,22,34,'Keep the routes. Remove the errors.',20 if mobile else 24,weight=600)
    text(s,22,60,'Illustrative distribution · one prompt',15,MUTED)
    if mobile:
        panel(s,22,82,336,'Student', [6,3,91]); line(s,'M190 323V365'); text(s,208,350,'Condition on validity',15,MUTED)
        panel(s,22,377,336,'Training-data target',[200/3,100/3,0],True)
        text(s,190,632,'A : B stays 2 : 1',16,PURPLE,600,'middle')
    else:
        panel(s,22,88,336,'Student',[6,3,91]); panel(s,442,88,336,'Training-data target',[200/3,100/3,0],True)
        line(s,'M373 202H426'); text(s,400,180,'Keep',15,MUTED,anchor='middle'); text(s,400,240,'valid',15,MUTED,anchor='middle')
        text(s,400,355,'A : B stays 2 : 1',18,PURPLE,600,'middle')
    save(s,'conditioning-mobile.svg' if mobile else 'conditioning.svg')
def node(s,x,y,w,h,label,sub,color=INK,fill='#fafbfd'):
    rect(s,x,y,w,h,fill,'#d9d3e8' if color==PURPLE else '#d7dce5')
    text(s,x+w/2,y+33,label,18,color,600,'middle')
    for i,t in enumerate(sub): text(s,x+w/2,y+59+i*23,t,15,MUTED,anchor='middle')
def dataflow(mobile=False):
    w,h=(380,776) if mobile else (800,420)
    s=start(w,h,'A sampler learns to create student-compatible SFT data','Expert information enters the sampler. Verification and student likelihood define the sampler training density. Verified sampled responses train the student with SFT.')
    text(s,22,34,'Two models, different jobs',22 if mobile else 24,weight=600)
    if mobile:
        node(s,54,64,272,87,'Prompt + expert',['Privileged input'],BLUE,'#f2f6fb')
        line(s,'M190 153V182')
        node(s,54,196,272,90,'Learned sampler',['Generate candidate responses'],PURPLE,'#f2f0f8')
        line(s,'M190 288V320')
        node(s,54,334,272,113,'Check + score',['Verifier: valid?', 'Student: how likely?'])
        line(s,'M190 449V485'); text(s,203,475,'Verified draws',15,TEAL)
        node(s,54,499,272,100,'Student',['Ordinary SFT on (prompt, response)'],TEAL,'#f0f7f7')
        line(s,'M327 550H349V390H328',dash=True)
        text(s,334,645,'Dashed: student scoring, prompt only',14,MUTED,anchor='end')
        line(s,'M53 389H27V241H52',True)
        text(s,22,694,'Purple feedback: train the sampler',15,PURPLE)
        text(s,22,721,'Density = student probability × validity',15,MUTED)
        text(s,22,749,'The expert trace is not the SFT target.',15,MUTED)
    else:
        node(s,22,88,148,112,'Prompt + expert',['Privileged input'],BLUE,'#f2f6fb')
        node(s,220,88,160,112,'Sampler',['Generate','candidates'],PURPLE,'#f2f0f8')
        node(s,428,88,160,112,'Check + score',['Valid?','How likely?'])
        node(s,638,88,140,112,'SFT data',['Verified','sampled traces'],TEAL,'#f0f7f7')
        line(s,'M172 145H207'); line(s,'M382 145H415'); line(s,'M590 145H625')
        line(s,'M506 202V240H300V202',True)
        text(s,400,268,'Train sampler: p(y | x) × validity',16,PURPLE,anchor='middle')
        node(s,428,305,160,85,'Student',['Prompt only'],TEAL,'#f0f7f7')
        line(s,'M708 202V348H590'); text(s,651,325,'SFT update',16,TEAL,anchor='middle')
        line(s,'M566 303V281H609V177H590',dash=True)
        text(s,22,343,'Expert → sampler',16,BLUE,600)
        text(s,22,368,'Student → density to match',16,TEAL,600)
        text(s,22,395,'Dashed line: scoring, without the expert trace',14,MUTED)
    save(s,'data-flow-mobile.svg' if mobile else 'data-flow.svg')
def online(mobile=False):
    w,h=(380,625) if mobile else (800,314)
    s=start(w,h,'The sampler tracks an evolving student','Fit the sampler while holding the student fixed, draw verified data, then update the student. Repeat against the new student density.')
    text(s,22,34,'Refresh the data as the student learns',19 if mobile else 24,weight=600)
    if mobile:
        node(s,52,67,276,101,'01   Fit sampler',['Hold the student fixed'],PURPLE,'#f2f0f8'); line(s,'M190 170V204')
        node(s,52,218,276,101,'02   Draw data',['Sample and verify responses']); line(s,'M190 321V355')
        node(s,52,369,276,101,'03   Update student',['Ordinary SFT'],TEAL,'#f0f7f7')
        line(s,'M190 472V511H26V118H50',True)
        text(s,190,550,'New student → new target density',16,PURPLE,600,'middle')
        text(s,190,581,'Repeat with the same expert examples',15,MUTED,anchor='middle')
    else:
        node(s,22,80,222,108,'01   Fit sampler',['Student stays fixed'],PURPLE,'#f2f0f8')
        node(s,289,80,222,108,'02   Draw data',['Sample + verify'])
        node(s,556,80,222,108,'03   Update student',['Ordinary SFT'],TEAL,'#f0f7f7')
        line(s,'M246 134H276'); line(s,'M513 134H543'); line(s,'M667 190V236H133V190',True)
        text(s,400,267,'New student → new target density',17,PURPLE,600,'middle')
        text(s,400,296,'Expert information stays available throughout',15,MUTED,anchor='middle')
    save(s,'online-loop-mobile.svg' if mobile else 'online-loop.svg')
def sampling_routes(mobile=False):
    w,h=(380,700) if mobile else (800,410)
    s=start(w,h,'One constrained target, two sampling methods','MCMC uses repeated per-example transitions; a conditional learned sampler shares its parameters across examples. Both aim at the same information-constrained student distribution.')
    text(s,22,34,'One target. Two ways to sample.',20 if mobile else 24,weight=600)
    tx,ty,tw,th=(22,75,336,143) if mobile else (220,67,360,119)
    rect(s,tx,ty,tw,th,'#f7f5fb','#ded8ec')
    text(s,tx+tw/2,ty+31,'Shared target: p(y | x, C)',17,PURPLE,600,'middle')
    bx,by=tx+34,ty+th-27
    s.append(f'<path d="M{bx} {by}H{tx+tw-34}" fill="none" stroke="#d7dce5" stroke-width="1.2"/>')
    # A symbolic two-mode density: no empirical geometry or measured curve.
    s.append(f'<path d="M{bx} {by} C{bx+22} {by} {bx+28} {by-52} {bx+60} {by-52} S{bx+96} {by} {bx+128} {by} C{bx+145} {by} {bx+159} {by-26} {bx+187} {by-26} S{bx+225} {by} {tx+tw-34} {by}" fill="none" stroke="{PURPLE}" stroke-width="2.5"/>')
    if mobile:
        coords=[(22,287,336,153),(22,497,336,153)]
        line(s,'M190 285V232',True)
        line(s,'M360 574H369V245H332V221',True)
        text(s,22,263,'Both target the same density',15,MUTED)
    else:
        coords=[(22,235,350,149),(428,235,350,149)]
        line(s,'M197 233V208H309V189',True)
        line(s,'M603 233V208H491V189',True)
    x,y,pw,ph=coords[0]
    rect(s,x,y,pw,ph)
    text(s,x+22,y+32,'Per-example MCMC',20,weight=600)
    text(s,x+22,y+77,'τ → y¹ → y² → …',23,BLUE)
    text(s,x+22,y+108,'Propose + accept / reject',16,MUTED)
    text(s,x+22,y+132,'Repeat the search for each example',15,MUTED)
    x,y,pw,ph=coords[1]
    rect(s,x,y,pw,ph,'#f2f0f8','#ded8ec')
    text(s,x+22,y+32,'Learned sampler',20,PURPLE,600)
    text(s,x+22,y+77,'(x, τ) → qφ → y',23,PURPLE)
    text(s,x+22,y+108,'Learn a sampler, then draw responses',16,MUTED)
    text(s,x+22,y+132,'Share the sampler across examples',15,MUTED)
    if mobile:text(s,190,682,'Conceptual routes · not measured convergence',13,MUTED,anchor='middle')
    save(s,'sampling-routes-mobile.svg' if mobile else 'sampling-routes.svg')

if __name__ == '__main__':
    for mobile in (False,True):
        conditioning(mobile); dataflow(mobile); online(mobile); sampling_routes(mobile)
