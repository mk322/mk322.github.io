"""Four original explanatory SVGs; all probabilities are illustrative."""
from render_figures import start,text,rect,line,save,PURPLE,TEAL,BLUE,MUTED

def distributions(name, mobile=False):
    target=name=='target-distribution'
    w=380 if mobile else 800; h=(620 if target else 500) if mobile else (360 if target else 320)
    s=start(w,h,'Keep valid routes in the student’s proportions' if target else 'Match the mixture, not just the best route','Illustrative categorical probabilities, not experimental results.')
    text(s,22,34,'Generate what filtering would keep' if target and not mobile else 'Generate the correct responses' if target else 'Correct answers, wrong mixture',20 if mobile else 24,weight=600)
    labels=['Student','SFT data target'] if target else ['Sampler before training','Desired sampler']
    vals=[[6,3,91],[200/3,100/3,0]] if target else [[90,10],[200/3,100/3]]
    tops=[78,350 if target else 280] if mobile else [82,82]; xs=[22,22] if mobile else [22,438]; pw=336 if mobile else 340
    for j,(label,values) in enumerate(zip(labels,vals)):
        x,y=xs[j],tops[j]; ph=218 if target else 178
        rect(s,x,y,pw,ph,'#fafbfd' if j==0 else '#f2f0f8')
        text(s,x+18,y+30,label,18,weight=600)
        for i,v in enumerate(values):
            yy=y+65+i*49; route=['Step-by-step (A)','Check a value (B)','Incorrect'][i]; color=[PURPLE,TEAL,'#bf656a'][i]
            text(s,x+18,yy,route,16)
            value=f'{v:.1f}%' if j==1 and i<2 else f'{v:g}%'
            text(s,x+pw-18,yy,value,17,weight=600,anchor='end')
            rect(s,x+18,yy+10,pw-36,10,'#e8ebef','#e8ebef',4)
            if v:rect(s,x+18,yy+10,(pw-36)*v/100,10,color,color,4)
    if mobile:
        line(s,f'M190 {tops[0]+(218 if target else 178)+8}V{tops[1]-12}')
        text(s,190,h-18,'6 correct + 3 correct → 9 retained' if target else 'A : B changes from 9 : 1 to 2 : 1',16,PURPLE,600,'middle')
    else:
        line(s,'M376 182H420'); text(s,400,h-18,'Keep 9 correct responses: 6 step-by-step, 3 check a value' if target else 'Less A, more B · keep both routes',18,PURPLE,600,'middle')
    save(s,name+('-mobile' if mobile else '')+'.svg')

def stages(online=False,mobile=False):
    name='online-cycle' if online else 'offline-stages';w=380 if mobile else 800;h=(630 if online else 590) if mobile else (330 if online else 270)
    s=start(w,h,'Online: refresh after each student update' if online else 'Offline: finish data creation before SFT','Online repeats three stages against the updated student.' if online else 'Three sequential phases. The student is fixed until the final phase.')
    text(s,22,34,'Online · repeat each round' if online else 'Offline · run once, in order',22 if mobile else 24,weight=600)
    titles=['1  Fit sampler LoRA','2  Generate data','3  SFT backbone'] if online else ['1  Fit sampler','2  Generate data','3  Run SFT']
    subs=[['Backbone fixed','LoRA updated'],['LoRA on + expert input','Verify the next batch'],['LoRA off, weights fixed','Backbone updated']] if online else [['Starting student fixed','Sampler updated'],['Sampler fixed','Save a verified dataset'],['Use the saved dataset','Student updated']]
    for i in range(3):
        x=48 if mobile else 22+i*263;y=78+i*157 if mobile else 82;pw=284 if mobile else 230
        rect(s,x,y,pw,116,['#f2f0f8','#fafbfd','#eff7f6'][i]);text(s,x+18,y+31,titles[i],19,weight=600)
        for k,t in enumerate(subs[i]):text(s,x+18,y+64+k*24,t,16,MUTED)
        if i<2:line(s,f'M190 {y+120}V{y+145}' if mobile else f'M{x+pw+3} 140H{x+pw+25}')
    if online:
        line(s,'M190 510V544H25V136H45' if mobile else 'M663 201V250H137V201',True)
        text(s,w/2,h-38,'Updated student → new target',17,PURPLE,600,'middle');text(s,w/2,h-13,'Then fit the sampler again',15,MUTED,anchor='middle')
    else:text(s,w/2,h-24,'No feedback from SFT to the sampler',16,MUTED,anchor='middle')
    save(s,name+('-mobile' if mobile else '')+'.svg')

for mobile in (False,True):
    distributions('target-distribution',mobile);distributions('ratio-matching',mobile);stages(False,mobile);stages(True,mobile)
