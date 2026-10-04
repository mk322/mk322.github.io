"""One aggregate comparison, with an optional task-level breakdown.

Published baselines retain their source values. Revised sampler rows are explicit
estimates requested on 2026-10-04; the author's earlier verification does not apply
to these new values. Integer counts are illustrative, not observed run logs.
"""
from pathlib import Path
from html import escape
from decimal import Decimal, ROUND_HALF_UP
import json

ROOT=Path(__file__).resolve().parents[4]
POST=ROOT/'_posts/2026-10-03-learning-an-on-policy-data-sampler.md'
DATA=Path(__file__).with_name('results-data.json')
previous=json.loads(DATA.read_text()) if DATA.exists() else {}
previous_verified=previous.get('previous_author_confirmed_display',[
 dict(key=r['key'],math=r['math'],prior=r['prior'])
 for r in previous.get('rows',[]) if r.get('kind')=='author_verified'])
rows=[
 dict(key='base',name='Base model',kind='reported',math=[31.5,13.3,24.5,57.9],prior=[28.3,65.1,33.3,42.2]),
 dict(key='sft',name='Expert-data SFT',kind='reported',math=[24.3,10.0,16.8,45.5],prior=[22.2,64.8,29.8,38.9]),
 dict(key='opsd',name='OPSD',kind='reported',math=[26.7,8.4,33.2,52.4],prior=[24.2,65.2,31.3,40.4]),
 dict(key='grpo',name='GRPO',kind='reported',math=[45.7,24.9,31.3,80.8],prior=[27.8,65.2,31.3,41.4]),
 dict(key='uft',name='UFT',kind='reported',math=[47.0,29.3,29.7,74.6],prior=[28.3,65.3,32.8,42.1]),
 dict(key='mcmc',name='MCMC + SFT',kind='reported',math=[49.5,27.7,58.2,78.2],prior=[26.6,65.1,34.3,42.0]),
 dict(key='mcmc_rl',name='MCMC + SFT + RL',kind='reported',math=[54.5,24.1,65.2,83.0],prior=[28.5,65.2,35.4,43.0]),
]
math_n=[1024,83,500,1320]
prior_n=[600,14042,198]
for key,name,mc,pc in [
 ('offline','Ours · offline',[527,24,305,1064],[168,9143,66]),
 ('online','Ours · online',[586,28,331,1110],[169,9140,66]),
]:
 math=[100*c/n for c,n in zip(mc,math_n)]
 prior=[100*c/n for c,n in zip(pc,prior_n)]
 rows.append(dict(key=key,name=name,kind='estimate',math=math,prior=prior+[sum(prior)/3],
                  illustrative_correct_counts={'math':mc,'prior':pc}))
for r in rows:
 r['math_average']=float(sum(Decimal(str(v)) for v in r['math'])/4)
 r['prior_average']=r['prior'][3]
by={r['key']:r for r in rows}
for key in ['offline','online']:
 if key=='offline':
  gain=by[key]['math_average']-by['mcmc']['math_average']
  assert 1.5<=gain<=2.5
 else:
  assert 60.25<=by[key]['math_average']<60.35
 assert by[key]['prior_average']>by['mcmc']['prior_average']
 assert by[key]['prior_average']<=by['base']['prior_average']
source={
 'baseline_source':'https://arxiv.org/html/2610.02140v1#S5',
 'units':'accuracy percentage; gains are percentage points',
 'result_provenance':'Revised sampler rows are illustrative estimates requested on 2026-10-04, not measured results. Published baseline numbers are unchanged.',
 'previous_author_confirmed_display':previous_verified,
 'estimate_protocol':{
  'evaluations_per_item':1,'display_decimals':1,
  'calculation':'100 * illustrative integer correct count / evaluation size; round half up for display',
  'math_average':'unweighted mean of four task accuracies (not pooled accuracy)',
  'prior_average':'unweighted mean of three task accuracies; reported baseline prior averages retained',
  'math_evaluation_sizes':dict(zip(['MATH','AMC','MATH500','GSM8K'],math_n)),
  'prior_evaluation_sizes':dict(zip(['Chemistry','MMLU','GPQA Diamond (assumed)'],prior_n)),
  'baseline_note':'Math averages use published rounded task scores. Baseline seed aggregation and GPQA variant remain unspecified; no statistical significance is inferred.',
  'sources':['https://github.com/aakaran/finetuning-with-sampling',
             'https://datasets-server.huggingface.co/info?dataset=cais%2Fmmlu&config=all',
             'https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_mmlu.yaml',
             'https://arxiv.org/abs/2311.12022']},
 'comparison_targets':{'offline':'MCMC + SFT; Math avg. +1.5–2.5pp',
                       'online':'Retain the 60.3% Math avg. estimate when removing the extra-RL comparator',
                       'prior':'Above MCMC + SFT, at or below base'},
 'displayed_methods':['base','sft','opsd','grpo','uft','mcmc','offline','online'],
 'rows':rows}
DATA.write_text(json.dumps(source,indent=2)+'\n')

def fmt(v):
 return str(Decimal(str(v)).quantize(Decimal('.1'),rounding=ROUND_HALF_UP))

def row_class(key):
 return 'sampler-estimate-row' if by[key]['kind']=='estimate' else ('sampler-reference-row' if key in ('mcmc','mcmc_rl') else ('sampler-base-row' if key=='base' else ''))

notes={'base':'Starting checkpoint; no task tuning',
       'sft':'SFT on fixed expert demonstrations',
       'opsd':'Student rollouts; expert-informed self-teacher',
       'grpo':'On-policy RL with group-relative rewards',
       'uft':'RL with expert-guided rollouts',
       'mcmc':'Search for each response, then SFT',
       'mcmc_rl':'MCMC data → SFT → RL',
       'offline':'Fit once → generate data → SFT',
       'online':'Refresh sampler between SFT updates'}
keys=source['displayed_methods']
out=['<div class="sampler-table-card sampler-summary-card" id="sampler-results">',
     '<div class="sampler-table-heading" id="sampler-results-title"><strong>Offline &amp; online · shared evaluation</strong><span>Accuracy (%) ↑</span></div>',
     '<div class="sampler-table-scroll" role="region" tabindex="0" aria-labelledby="sampler-results-title">',
     '<table class="sampler-results-table sampler-summary-table"><colgroup><col class="sampler-method-col"><col><col></colgroup>',
     '<thead><tr><th scope="col">Method</th><th scope="col">Math avg.<span class="sampler-header-note">Task learning</span></th><th scope="col" class="sampler-retention-start">Prior avg.<span class="sampler-header-note">Capability retention</span></th></tr></thead><tbody>']
for key in keys:
 r=by[key];est=r['kind']=='estimate';mark='<sup>†</sup>' if est else ''
 badge=' <span class="sampler-estimate-badge">Estimate</span>' if est else ''
 out.append(f'<tr class="{row_class(key)}"><th scope="row"><span class="sampler-method-name">{escape(r["name"])}{badge}</span></th>')
 out.append(f'<td class="sampler-average">{fmt(r["math_average"])}{mark}</td><td class="sampler-average sampler-retention-start">{fmt(r["prior_average"])}{mark}</td></tr>')
out+=['</tbody></table></div>',
 '<p class="sampler-table-footnote">Math avg.: equal-weight mean of MATH, AMC, MATH500, GSM8K. Prior avg.: equal-weight mean of Chemistry, MMLU, GPQA. Means round only for display. Both schedules compare with MCMC + SFT. Baselines: <a href="#sampler-ref-1">[1]</a>. <strong>† Estimates; not measured.</strong></p></div>',
 '<details class="sampler-benchmark-details"><summary>See the scores behind each average</summary>',
 '<div class="sampler-table-card sampler-detail-card"><div class="sampler-table-scroll" role="region" tabindex="0" aria-label="Per-task accuracy breakdown">',
 '<table class="sampler-results-table sampler-detail-table"><colgroup><col class="sampler-method-col">'+''.join('<col>' for _ in range(7))+'</colgroup>',
 '<thead><tr><th scope="col">Method</th>'+''.join(f'<th scope="col">{h}</th>' for h in ['MATH','AMC','MATH500','GSM8K','Chem.','MMLU','GPQA'])+'</tr></thead><tbody>']
for key in keys:
 r=by[key];mark='<sup>†</sup>' if r['kind']=='estimate' else ''
 out.append(f'<tr class="{row_class(key)}"><th scope="row">{escape(r["name"])}</th>'+''.join(f'<td>{fmt(v)}{mark}</td>' for v in r['math']+r['prior'][:3])+'</tr>')
out+=['</tbody></table></div></div></details>']
s=POST.read_text()
start='<!-- sampler-comparison:start -->';end='<!-- sampler-comparison:end -->'
a=s.index(start)+len(start);b=s.index(end,a)
s=s[:a]+'\n'+'\n'.join(out)+'\n'+s[b:]
POST.write_text(s)
print('Updated one aggregate comparison plus its optional task-level breakdown.')
