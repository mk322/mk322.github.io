"""Rebuild traceable baseline/illustrative-estimate tables in the local blog draft.

Reported values: Finetuning with Sampling, arXiv:2610.02140v1 Table 1 math panel.
Author-requested draft targets: about +2pp offline, +3–5pp online.
Illustrative correct counts define the percentages; none are observed outcomes.
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
# Planning protocol: one single-shot evaluation per item, not a seed average.
# FWS repository file counts: AMC=83, MATH500=500, GSM8K=1320, Chemistry=600.
# MATH test size comes from the paper. MMLU uses the full micro-average test set.
# The baseline does not specify its GPQA variant; Diamond is an explicit assumption.
math_n=[1024,83,500,1320]
prior_n=[600,14042,198]
for key,name,math_count,prior_count in [
 ('offline','Our offline sampler',[528,25,299,1058],[170,9128,67]),
 ('online','Our online sampler',[550,27,311,1078],[169,9150,66]),
]:
 math=[100*c/n for c,n in zip(math_count,math_n)]
 prior=[100*c/n for c,n in zip(prior_count,prior_n)]
 rows.append(dict(key=key,name=name,kind='estimate',math=math,
                  prior=prior+[sum(prior)/3],illustrative_correct_counts={
                  'math':math_count,'prior':prior_count}))
by={r['key']:r for r in rows}
source={
 'baseline_source':'https://arxiv.org/html/2610.02140v1#S5',
 'units':'accuracy percentage; deltas are percentage points',
 'estimate_basis':'Unmeasured planning placeholders: roughly +2pp offline / +3–5pp online versus published MCMC + SFT. Prior-task values illustrate small variation near the base, not proven retention.',
 'estimate_protocol':{
  'evaluations_per_item':1,'display_decimals':1,
  'calculation':'100 * illustrative integer correct count / evaluation size; round only for display',
  'math_evaluation_sizes':dict(zip(['MATH','AMC','MATH500','GSM8K'],math_n)),
  'prior_evaluation_sizes':dict(zip(['Chemistry','MMLU','GPQA Diamond (assumed)'],prior_n)),
  'prior_average':'unweighted mean of the three unrounded task percentages',
  'sources':[
   'https://github.com/aakaran/finetuning-with-sampling (ood_data/AMC.json, ood_data/MATH-TTT.json, ood_data/GSM8K.jsonl, sci_data/test_data.jsonl)',
   'https://datasets-server.huggingface.co/info?dataset=cais%2Fmmlu&config=all (14042 test examples)',
   'https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_mmlu.yaml (micro-average)',
   'https://arxiv.org/abs/2311.12022 (GPQA Diamond assumption: 198 items)'],
  'baseline_note':'Published baseline values and rounding are preserved. Their seed aggregation and GPQA variant are not inferred from their decimals.'},
 'rows':rows}
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
   if est and ((prior and j<3) or not prior):
    n=(prior_n if prior else math_n)[j]
    c=r['illustrative_correct_counts'][group][j]
    cell=f'<span title="Illustrative count: {c:,} / {n:,}; not measured">{cell}</span>'
   if prior and j<3:
    delta=round(v,1)-by['base']['prior'][j]
    sign='−' if delta<-.05 else '+'
    cell+=f'<span class="sampler-cell-delta {"sampler-down" if delta<-.05 else ""}">{sign}{abs(delta):.1f} pp</span>'
   out.append(f'<td>{cell}</td>')
  delta=round(r['prior'][3],1)-42.2 if prior else round(r['math'][0],1)-49.5
  sign='−' if delta<-.05 else '+'
  out.append(f'<td class="sampler-delta {"sampler-down" if delta<-.05 else ""}">{sign}{abs(delta):.1f}{mark}</td></tr>')
 out+=['</tbody></table></div>',f'<p class="sampler-table-footnote">{"Small numbers show each task’s change from the base model. Δ uses displayed rounded scores." if prior else "Δ MATH is the percentage-point change from MCMC + SFT."} <strong>† {"Retention estimates" if prior else "Draft estimates"}; not measured.</strong></p></div>']
 return '\n'.join(out)

offline=table(['base','sft','opsd','grpo','uft','mcmc','offline'],'math','offline-accuracy','Learning the new task','Reported baselines; offline estimates vary around a +2-point gain.')+'\n\n'+table(['base','sft','opsd','grpo','uft','mcmc','offline'],'prior','offline-retention','Keep prior capabilities visible','Per-task changes reveal losses that an average can hide.')
online=table(['mcmc','mcmc_rl','offline','online'],'math','online-accuracy','Does refreshing help?','Online estimates vary by task (+3–5 points); include the stronger RL pipeline.')+'\n\n'+table(['base','mcmc','mcmc_rl','offline','online'],'prior','online-retention','Retention through the online loop','Illustrative task-level variation near the base checkpoint, with losses shown explicitly.')
s=POST.read_text()
for name,content in [('offline',offline),('online',online)]:
 start=f'<!-- sampler-{name}-tables:start -->';end=f'<!-- sampler-{name}-tables:end -->'
 a=s.index(start)+len(start);b=s.index(end,a)
 s=s[:a]+'\n'+content+'\n'+s[b:]
POST.write_text(s)
print('Updated four tables; reported and estimated rows are explicitly distinguished.')
