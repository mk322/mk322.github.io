"""Minimal comparison: repeated search versus one reusable generation policy."""
from render_figures import start, text, rect, line, save, PURPLE, MUTED, TEAL


def document(s, x, y):
    rect(s, x, y, 25, 31, '#f0f7f7', '#83b5b1', 3)
    for dy in (9, 15, 21):
        s.append(f'<path d="M{x+6} {y+dy}h13" stroke="{TEAL}" stroke-width="1.2"/>')


def draw(mobile=False):
    w, h = (380, 545) if mobile else (800, 300)
    s = start(w, h, 'Learn the search. Reuse the sampler.',
              'MCMC repeats a chain of proposals for each data example. '
              'Sampler training learns one policy reused to produce many verified responses.')
    text(s, 22, 35, 'Learn the search. Reuse the sampler.' if not mobile else 'Learn the search. Reuse it.', 25 if not mobile else 22, weight=600)
    pw = 336 if mobile else 366
    for i in range(2):
        x = 22 if mobile else 22+i*390
        y = 61 + (242*i if mobile else 0)
        rect(s, x, y, pw, 222, '#fafbfd' if i==0 else '#f8f6fc')
        text(s, x+18, y+30, 'MCMC' if i==0 else 'Amortized sampler', 20, weight=600)
        text(s, x+18, y+55, 'Search for each response' if i==0 else 'Train, then reuse', 15, MUTED)
        if i==0:
            for j in range(3):
                yy=y+81+38*j
                for k in range(3):
                    xx=x+24+k*65
                    rect(s,xx,yy,39,22,'#e9edf2','#c6ceda',5)
                    if k<2: line(s,f'M{xx+42} {yy+11}h19')
                line(s,f'M{x+196} {yy+11}H{x+pw-59}')
                document(s,x+pw-52,yy-4)
            text(s,x+24,y+202,'Propose · score · repeat',14,MUTED)
        else:
            rect(s,x+18,y+86,65,38,'#fff','#ded8ec')
            text(s,x+50.5,y+110,'Train',16,MUTED,500,'middle')
            line(s,f'M{x+85} {y+105}H{x+107}',True)
            rect(s,x+114,y+78,96,54,'#e6e1f1','#c1b7d9')
            text(s,x+162,y+111,'Sampler',17,PURPLE,600,'middle')
            for j in range(3):
                dx=x+41+105*j
                line(s,f'M{x+162} {y+135}V{y+151}H{dx+12.5}V{y+165}',True)
                document(s,dx,y+168)
    save(s,'amortized-comparison'+('-mobile' if mobile else '')+'.svg')


for mobile in (False, True):
    draw(mobile)
