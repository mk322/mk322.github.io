"""Response revision versus learned constructive generation, both followed by SFT.

Conceptual offline schematic, not a runtime measurement. Sampler fitting uses
student scores and validity feedback, not MCMC trajectories. The sampler constructs
a response autoregressively: a rollout is not a single neural-network forward call.
"""
from render_figures import start, text, rect, line, save, PURPLE, MUTED, TEAL, INK


def label(s, x, y, w, h, top, bottom=None, color=INK, fill='#fff', stroke='#d7dce5'):
    rect(s,x,y,w,h,fill,stroke,8)
    text(s,x+w/2,y+h/2+(0 if bottom else 6),top,16,color,500,'middle')
    if bottom: text(s,x+w/2,y+h/2+23,bottom,14,MUTED,anchor='middle')


def drafts(s,x,y):
    for i,lab in enumerate(['Draft 0','Draft 1','Draft k']):
        a=x+i*108
        rect(s,a,y,82,58,'#fff','#cad1dc',7)
        text(s,a+41,y-10,lab,15,MUTED,anchor='middle')
        for j,length in enumerate([50,42-i*5,49-i*7]):
            s.append(f'<path d="M{a+15} {y+16+j*13}h{length}" stroke="{"#3e73a8" if i and j==1 else "#aeb7c5"}" stroke-width="4" stroke-linecap="round"/>')
        if i<2: line(s,f'M{a+85} {y+29}H{a+100}')


def prefixes(s,x,y):
    # Increasing prefix lengths expose the constructive operation explicitly.
    for a,w,lab in [(x,32,'∅'),(x+52,39,'t₁'),(x+112,65,'t₁ t₂'),(x+201,96,'… EOS')]:
        label(s,a,y,w,42,lab,color=PURPLE,fill='#fff',stroke='#c6badf')
    for a,b in [(x+35,x+44),(x+94,x+104),(x+180,x+193)]:
        line(s,f'M{a} {y+21}H{b}',True)


def desktop():
    s=start(800,584,'Search each response, or reuse a learned generation policy?',
        'Top: MCMC revises a complete response with repeated propose, score, and accept/reject steps for each example. Bottom: first train a policy across examples using student scores and validity feedback, then reuse that policy to construct responses by appending tokens. Both routes verify outputs, collect training data, and update the student with SFT.')
    rect(s,12,12,776,250,'#f8f9fb','#e1e5eb',12)
    text(s,30,45,'MCMC',22,weight=600)
    text(s,127,45,'Search again for each example',18,MUTED)
    label(s,30,125,94,76,'Prompt','+ expert')
    line(s,'M127 163H138')
    rect(s,146,77,338,172,'#fff','#d7dce5',10)
    text(s,315,103,'Propose · score · accept/reject',16,MUTED,anchor='middle')
    drafts(s,162,136)
    line(s,'M460 196V211H203V196')
    text(s,315,239,'Revise the response; weights stay fixed',14,MUTED,anchor='middle')
    line(s,'M487 163H515')
    label(s,523,125,112,76,'Verified','SFT data',TEAL,'#edf7f5','#a4ccc6')
    line(s,'M638 163H666')
    label(s,674,125,94,76,'SFT','Train student')

    rect(s,12,278,776,294,'#faf8fd','#ddd5e9',12)
    text(s,30,312,'Amortized sampler',22,weight=600)
    rect(s,146,333,338,62,'#eee9f7','#b9abd7',9)
    text(s,315,358,'Train one policy across examples',17,PURPLE,500,'middle')
    text(s,315,381,'Student scores + validity → update weights',14,MUTED,anchor='middle')
    line(s,'M315 398V416',True)
    label(s,30,449,94,76,'Prompt','+ expert')
    line(s,'M127 487H138')
    rect(s,146,424,338,135,'#fff','#b9abd7',10)
    text(s,315,450,'Reuse learned weights',18,PURPLE,500,'middle')
    prefixes(s,166,470)
    text(s,315,543,'Append tokens to build a new response',15,MUTED,anchor='middle')
    line(s,'M487 487H515')
    label(s,523,449,112,76,'Verified','SFT data',TEAL,'#edf7f5','#a4ccc6')
    line(s,'M638 487H666')
    label(s,674,449,94,76,'SFT','Train student')
    save(s,'amortized-comparison.svg')


def mobile():
    s=start(380,922,'Search each response, or reuse a learned generation policy?',
        'MCMC repeatedly revises each response with fixed weights. Sampler training changes a policy across examples; data generation reuses it to append tokens and construct a new response. Both routes end in verified data and SFT of the student.')
    rect(s,10,10,360,409,'#f8f9fb','#e1e5eb',12)
    text(s,28,43,'MCMC',22,weight=600)
    text(s,28,70,'Search again for each example',17,MUTED)
    text(s,190,103,'Prompt + expert',16,anchor='middle')
    line(s,'M190 113V132')
    rect(s,28,140,324,172,'#fff','#d7dce5',10)
    text(s,190,165,'Propose · score · accept/reject',16,MUTED,anchor='middle')
    drafts(s,41,198)
    line(s,'M338 260V278H82V260')
    text(s,190,300,'Revise response; weights stay fixed',14,MUTED,anchor='middle')
    line(s,'M190 315V328H103V335')
    label(s,28,343,150,58,'Verified','SFT data',TEAL,'#edf7f5','#a4ccc6')
    line(s,'M181 372H194')
    label(s,202,343,150,58,'SFT','Train student')

    rect(s,10,435,360,477,'#faf8fd','#ddd5e9',12)
    text(s,28,471,'Amortized sampler',22,weight=600)
    rect(s,28,493,324,77,'#eee9f7','#b9abd7',9)
    text(s,190,519,'Train across examples',18,PURPLE,500,'middle')
    text(s,190,543,'Student scores + validity',15,MUTED,anchor='middle')
    text(s,190,561,'Update policy weights',14,PURPLE,anchor='middle')
    line(s,'M280 573V636',True)
    text(s,190,607,'Prompt + expert',16,anchor='middle')
    line(s,'M145 617V636')
    rect(s,28,644,324,139,'#fff','#b9abd7',10)
    text(s,190,672,'Reuse learned weights',18,PURPLE,500,'middle')
    prefixes(s,40,690)
    text(s,190,762,'Append tokens to build a response',15,MUTED,anchor='middle')
    line(s,'M190 786V799H103V809')
    label(s,28,817,150,64,'Verified','SFT data',TEAL,'#edf7f5','#a4ccc6')
    line(s,'M181 849H194')
    label(s,202,817,150,64,'SFT','Train student')
    save(s,'amortized-comparison-mobile.svg')


if __name__=='__main__':
    desktop()
    mobile()
