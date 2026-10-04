"""Rebuild traceable baseline/illustrative-estimate tables in the local blog draft.

Reported values: Finetuning with Sampling, arXiv:2610.02140v1 Table 1 math panel.
Author-requested draft targets: +2pp offline, +3–5pp online (midpoint +4pp).
All our prior-task numbers copy the base checkpoint as a no-degradation TARGET.
These are planning values, not predictions fitted to experimental data.
"""
from pathlib import Path
from html import escape
import json

ROOT=Path(__file__).resolve().parents[4]
POST=ROOT/'_posts/2026-10-03-learning-an-on-policy-data-sampler.md'
# math=[MATH,AMC,MATH500,GSM8K]; prior=[Chemistry,MMLU,GPQA,reported average]
rows=[
 dict(key='base',name='Base model',kind='reported',math=[31.5,13.3,24.5,57.9],prior=[28.3,65.1,33.3,42.2]),
 dict(key='sft',name='Expert-data SFT',kind='reported',math=[24.3,10.0,16.8,45.5],prior=[22.2,64.8,29.8,38.9]),
 dict(key='opsd',name='OPSD',kind='reported',math=[26.7,8.4,33.2,52.4],prior=[24.2,65.2,31.3,40.4]),
 dict(key='grpo',name='GRPO',kind='reported',math=[45.7,24.9,31.3,80.8],prior=[27.8,65.2,31.3,41.4]),
 dict(key='uft',name='UFT',kind='reported',math=[47.0,29.3,29.7,74.6],prior=[28.3,65.3,32.8,42.1]),
 dict(key='mcmc',name='MCMC + SFT',kind='reported',math=[49.5,27.7,58.2,78.2],prior=[26.6,65.1,34.3,42.0]),
 dict(key='mcmc_rl',name='MCMC + SFT + RL',kind='reported',math=[54.5,24.1,65.2,83.0],prior=[28.5,65.2,35.4,43.0]),
]
for key,name,gain in [('offline','Our offline sampler',2),('online','Our online sampler',4)]:
 rows.append(dict(key=key,name=name,kind='estimate',math=[round(v+gain,1) for v in rows[5]['math']],prior=rows[0]['prior'].copy()))
by={r['key']:r for r in rows}
source={'baseline_source':'https://arxiv.org/html/2610.02140v1#S5','units':'accuracy percentage; deltas are percentage points','estimate_basis':'User-specified +2pp offline / +3–5pp online, using +4pp midpoint for all four math columns. Prior-task estimates are base-level retention targets. No measured runs.','rows':rows}
Path(__file__).with_name('results-data.json').write_text(json.dumps(source,indent=2)+'\n')

def table(keys,group,ident,title,subtitle):
 prior=group=='prior'
 headers=['Method','Chemistry','MMLU','GPQA','Prior avg.','Δ vs. base'] if prior else ['Method','MATH','AMC','MATH500','GSM8K','Δ MATH']
 out=[f'<div class="sampler-table-card" id="{ident}">',
      f'<div class="sampler-table-heading"><span class="sampler-table-kicker">{"RETENTION" if prior else "GENERALIZATION"} · ACCURACY (%)</span><h4 id="{ident}-title">{title}</h4><p>{subtitle}</p></div>',
      f'<div class="sampler-table-scroll" role="region" tabindex="0" aria-labelledby="{ident}-title">',
      '<table class="sampler-results-table">',
      '<thead><tr>'+''.join(f'<th scope="col">{h}</th>' for h in headers)+'</tr></thead><tbody>']
 for key in keys:
  r=by[key];est=r['kind']=='estimate';mark='<sup>†</sup>' if est else ''
  classes='sampler-estimate-row' if est else ('sampler-reference-row' if key in ('mcmc','mcmc_rl') else '')
  badge=' <span class="sampler-estimate-badge">Est. †</span>' if est else ''
  out.append(f'<tr class="{classes}"><th scope="row">{escape(r["name"])}{badge}</th>')
  for j,v in enumerate(r[group]):
   cell=f'{v:.1f}{mark}'
   if prior and j<3:
    delta=v-by['base']['prior'][j]
    sign='−' if delta<-.05 else '+'
    cell+=f'<span class="sampler-cell-delta {"sampler-down" if delta<-.05 else ""}">{sign}{abs(delta):.1f} pp</span>'
   out.append(f'<td>{cell}</td>')
  delta=r['prior'][3]-42.2 if prior else r['math'][0]-49.5
  sign='−' if delta<-.05 else '+'
  out.append(f'<td class="sampler-delta {"sampler-down" if delta<-.05 else ""}">{sign}{abs(delta):.1f}{mark}</td></tr>')
 out+=['</tbody></table></div>',f'<p class="sampler-table-footnote">{"Small numbers show each task’s change from the base model. Δ uses the reported rounded averages." if prior else "Δ MATH is the percentage-point change from MCMC + SFT."} <strong>† {"Retention targets" if prior else "Draft estimates"}; not measured.</strong></p></div>']
 return '\n'.join(out)

offline=table(['base','sft','opsd','grpo','uft','mcmc','offline'],'math','offline-accuracy','Learning the new task','Reported baselines and the +2-point offline estimate.')+'\n\n'+table(['base','sft','opsd','grpo','uft','mcmc','offline'],'prior','offline-retention','Keep prior capabilities visible','Per-task changes reveal losses that an average can hide.')
online=table(['mcmc','mcmc_rl','offline','online'],'math','online-accuracy','Does refreshing help?','The online estimate is the +4-point midpoint; MCMC + SFT + RL is a stronger comparator.')+'\n\n'+table(['base','mcmc','mcmc_rl','offline','online'],'prior','online-retention','Retention through the online loop','Both estimated sampler rows target the starting checkpoint’s prior-task accuracy.')
s=POST.read_text()
for name,content in [('offline',offline),('online',online)]:
 start=f'<!-- sampler-{name}-tables:start -->';end=f'<!-- sampler-{name}-tables:end -->'
 a=s.index(start)+len(start);b=s.index(end,a)
 s=s[:a]+'\n'+content+'\n'+s[b:]
POST.write_text(s)
print('Updated four tables; reported and estimated rows are explicitly distinguished.')
