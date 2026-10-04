"""Rebuild published-baseline and author-verified result tables.

Baseline source: Finetuning with Sampling, arXiv:2610.02140v1, Table 1.
The author confirmed all displayed scores on 2026-10-04. The integer counts
below preserve the previous draft's rounding arithmetic; they are not raw
experimental logs independently checked by the editor.
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
# Display arithmetic retained from the draft, not a reconstructed run log.
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
 rows.append(dict(key=key,name=name,kind='author_verified',math=math,
                  prior=prior+[sum(prior)/3],display_derivation_counts={
                  'math':math_count,'prior':prior_count}))
by={r['key']:r for r in rows}
source={
 'baseline_source':'https://arxiv.org/html/2610.02140v1#S5',
 'units':'accuracy percentage; deltas are percentage points',
 'result_provenance':'The author confirmed all currently displayed scores on 2026-10-04. No raw run logs, seed-level results, or runtime measurements were supplied.',
 'draft_history':'These values were originally planning placeholders. Publication status changed after explicit author verification; scores are unchanged.',
 'display_arithmetic':{
  'display_decimals':1,
  'count_provenance':'Integer counts used in the previous draft to obtain display-compatible decimals; not independently verified experimental counts.',
  'calculation':'Draft arithmetic: 100 * integer count / evaluation size, rounded for display',
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

def table(keys,ident,label):
 headers=['MATH','AMC','MATH500','GSM8K','Chem.','MMLU','GPQA','Avg.']
 out=[f'<div class="sampler-table-card" id="{ident}">',
      f'<div class="sampler-table-heading" id="{ident}-title"><strong>{label}</strong><span>Accuracy (%) ↑</span></div>',
      f'<div class="sampler-table-scroll" role="region" tabindex="0" aria-labelledby="{ident}-title">',
      '<table class="sampler-results-table">',
      '<colgroup><col class="sampler-method-col">'+''.join('<col>' for _ in headers)+'</colgroup>',
      '<thead><tr class="sampler-column-groups"><th scope="col" rowspan="2">Method</th><th scope="colgroup" colspan="4">Math reasoning</th><th scope="colgroup" colspan="4" class="sampler-retention-start">Prior capabilities</th></tr>',
      '<tr>'+''.join('<th scope="col" class="{}">{}</th>'.format('sampler-retention-start' if j==4 else '',h) for j,h in enumerate(headers))+'</tr></thead><tbody>']
 names={'offline':'Ours · offline','online':'Ours · online','sft':'Expert-data SFT'}
 for key in keys:
  r=by[key]
  classes='sampler-ours-row' if r['kind']=='author_verified' else ('sampler-reference-row' if key in ('mcmc','mcmc_rl') else ('sampler-base-row' if key=='base' else ''))
  out.append(f'<tr class="{classes}"><th scope="row">{escape(names.get(key,r["name"]))}</th>')
  for j,v in enumerate(r['math']+r['prior']):
   classes=[]
   if j==4:classes.append('sampler-retention-start')
   if j>=4 and round(v,1)<by['base']['prior'][j-4]-.05:classes.append('sampler-retention-loss')
   if j==7:classes.append('sampler-prior-avg')
   class_names=' '.join(classes)
   out.append(f'<td class="{class_names}">{v:.1f}</td>')
  out.append('</tr>')
 out+=['</tbody></table></div>',
       '<p class="sampler-table-footnote"><span class="sampler-loss-key">Red</span> = below the base on prior tasks. Avg. = unweighted prior-task mean. Baselines: <a href="#sampler-ref-1">[1]</a>.<span class="sampler-table-swipe">Swipe horizontally for all metrics.</span></p></div>']
 return '\n'.join(out)

offline=table(['base','sft','opsd','grpo','uft','mcmc','offline'],'offline-results','Offline · fit once, then SFT')
online=table(['base','mcmc','mcmc_rl','offline','online'],'online-results','Online · refresh between SFT updates')
s=POST.read_text()
for name,content in [('offline',offline),('online',online)]:
 start=f'<!-- sampler-{name}-tables:start -->';end=f'<!-- sampler-{name}-tables:end -->'
 a=s.index(start)+len(start);b=s.index(end,a)
 s=s[:a]+'\n'+content+'\n'+s[b:]
POST.write_text(s)
print('Updated two combined tables: accuracy and retention in each offline/online comparison.')
