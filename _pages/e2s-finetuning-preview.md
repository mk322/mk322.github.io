---
layout: blog-post
title: "E2S Finetuning: From Off-Policy Expert Data to On-Policy Training Data"
subtitle: "Learn a reusable sampler instead of searching again for every training example."
permalink: /blog/e2s-finetuning-preview/
date: 2026-10-05
sitemap: false
author_profile: false
likes: false
tldr: |
  - **SFT data is usually off-policy.** Expert trajectories can be far from the student’s policy, so fitting them can require a large distribution shift and contribute to forgetting.
  - **We formulate data generation as constrained distribution matching.** For valid responses \\(y\\), the target preserves the student’s relative probabilities: \\(q^*(y)\propto p_{\mathrm{student}}(y)\\). Invalid responses receive zero probability.
  - **E2S Finetuning amortizes the sampling problem.** MCMC searches separately for every example; E2S learns a GFlowNet sampler that is reused across examples. E2S-Offline reaches 55.5% Math avg.; E2S-Online reaches 59.4%, while prior-task averages remain near the base model.
---

<link rel="stylesheet" href="{{ '/assets/blog/learned-sampler/article.css' | relative_url }}">
<link rel="stylesheet" href="{{ '/assets/blog/e2s-preview/article.css' | relative_url }}">

<nav class="sampler-toc" aria-label="Article contents"><details><summary>On this page</summary><ol><li><a href="#sft-problem">Off-policy mismatch</a></li><li><a href="#amortization">Why amortize search?</a></li><li><a href="#target">The constrained target</a></li><li><a href="#gflownet">GFlowNet and group matching</a></li><li><a href="#versions">Offline and online</a></li><li><a href="#experiments">Experiments</a></li><li><a href="#comparison">RL, OPD, and MCMC</a></li><li><a href="#lookahead">Looking ahead</a></li><li><a href="#references">References</a></li><li><a href="#citation">Citation</a></li></ol></details></nav>
<script defer src="{{ "/assets/blog/learned-sampler/navigation.js" | relative_url }}"></script>
<script defer src="{{ "/assets/blog/e2s-preview/navigation.js" | relative_url }}"></script>

**Standard SFT trains on demonstrations produced by someone other than the student—typically a human or a stronger model. That makes the data off-policy.**

Even when every demonstration is correct, its reasoning path may be unlikely under the student. Matching those trajectories can require a larger policy shift than the task itself demands. This mismatch is one mechanism that can contribute to catastrophic forgetting: learning a new task at the expense of existing skills.

**Can we preserve the expert information without forcing the student to imitate the expert distribution?**

We formulate this as constrained distribution matching. Among response distributions that satisfy the expert constraint, we choose the one closest to the student. The solution is simple: the student conditioned on being valid.

MCMC can target this distribution, but it searches again for every example. **E2S Finetuning amortizes that search:** we train a reusable GFlowNet sampler across examples, then use it to generate more on-policy training data. E2S changes where the SFT targets come from, not the SFT loss itself.

## Off-policy SFT creates a distribution mismatch
{: #sft-problem}

For a fixed prompt, let \\(q_{\mathrm{data}}\\) be the distribution of expert responses and \\(p_\theta\\) the student’s response distribution. SFT minimizes the expected negative log probability of those responses:

<div class="sampler-math">
\[
\begin{aligned}
\mathcal L_{\mathrm{SFT}}
&=\mathbb E_{y\sim q_{\mathrm{data}}}[-\log p_\theta(y)]\\
&=H(q_{\mathrm{data}})+D_{\mathrm{KL}}(q_{\mathrm{data}}\|p_\theta).
\end{aligned}
\]
</div>

**SFT pulls the student toward the data distribution.** The entropy term is fixed, so minimizing the loss minimizes this KL divergence. In the population limit, a sufficiently expressive model can fit \\(p_\theta=q_{\mathrm{data}}\\). Fitting distant data means moving toward a distant policy.

Some policy change is necessary to learn a new task. Imitating an expert’s particular trajectory can demand additional change. Larger shifts have been associated with more forgetting, motivating training data closer to the student; the KL identity alone does not prove that other skills will be lost. [[1]](#sampler-ref-1) [[2]](#sampler-ref-2)

The same mismatch appears token by token: SFT trains on prefixes visited by the expert, not prefixes the student would typically visit itself.

<details class="sampler-technical" markdown="1">
<summary>Technical note: off-policy prefixes</summary>

For prompt \\(x\\) and expert response \\(y=(y_1,\ldots,y_T)\\), the sequence loss is \\(-\sum_{t=1}^{T}\log p_\theta(y_t\mid x,y_{<t})\\), including the termination token. The prefixes \\(y_{<t}\\) come from the demonstration distribution. They are not sampled from the current student.

</details>

<div class="sampler-key"><p>Off-policy SFT can force the student to move farther than the task itself requires.</p></div>

## Two ways to reduce the mismatch
{: #amortization}

Both routes address the same mismatch, but they change different parts of training: **where supervision is applied, or which responses become training targets.**

**Move learning to the student.** Let the current student generate a response, then provide feedback on that trajectory. RL supplies rewards; on-policy distillation (OPD) supplies teacher probabilities on student-generated prefixes. The student chooses the path, and supervision follows the states it visits. [[3]](#sampler-ref-3) [[4]](#sampler-ref-4)

**Move the data to the student.** Use the expert constraint to define which responses are valid, and the student policy to define their relative probabilities. Sample from that constrained distribution, then train the student on the resulting responses with SFT. Here, we change the training targets themselves. MCMC and E2S are two ways to obtain them. [[5]](#sampler-ref-5)

<figure id="figure-two-routes"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/e2s-preview/two-routes-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/e2s-preview/two-routes.svg' | relative_url }}" loading="lazy" alt="Top: the student generates trajectories, receives reward or teacher feedback, and is updated through RL or distillation. Bottom: the expert constraint and student policy guide MCMC or E2S sampling; the resulting valid responses become SFT targets."></picture><figcaption>RL and OPD bring feedback to student rollouts. MCMC and E2S prepare a constrained response distribution for SFT. Both ultimately update the student.</figcaption></figure>

Rejection sampling can also produce the constrained target: draw from the student and keep valid responses. It wastes most rollouts when success is rare. MCMC instead starts from an expert solution and repeatedly proposes, scores, and accepts or rejects edits—the route developed in *Finetuning with Sampling*. [[5]](#sampler-ref-5)

Write \\(x\\) for the prompt, \\(\tau\\) for its off-policy expert trace, and \\(y\\) for a new response that satisfies the expert constraint while following the student’s relative probabilities. **Both MCMC and E2S turn \\(\tau\\) into a more on-policy \\(y\\), then use \\((x,y)\\) for student SFT.** The expert trace guides data preparation; it is the new response \\(y\\) that becomes the SFT target.

E2S takes this second route. MCMC runs a new chain for every example. **That repeated search is the motivation for E2S.** Instead, we train a conditional sampler across prompts: work on one example changes the sampler used for the next.

<figure id="figure-amortization"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/e2s-preview/amortization-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/e2s-preview/amortization.svg' | relative_url }}" loading="lazy" alt="Both routes turn off-policy expert trace τ into more on-policy response y, then train the student on (x, y). MCMC searches anew for each example; E2S reuses learned sampler parameters."></picture><figcaption>Both routes turn off-policy expert trace τ into more on-policy response y, then train the student on (x, y). MCMC searches anew for each example; E2S reuses learned sampler parameters.</figcaption></figure>

MCMC updates a response state; E2S updates a reusable sampling policy. E2S moves part of the repeated search cost into sampler training. Whether that saves total compute depends on the training cost and how much useful data the sampler subsequently generates.

<div class="sampler-key"><p>MCMC searches again; E2S learns a sampling policy it can reuse across examples.</p></div>

## The target: the student conditioned on the expert constraint
{: #target}

For a prompt \\(x\\) and expert solution \\(\tau\\), let \\(C_\tau\\) be the set of valid responses that preserve the information we care about. For math, the simplest constraint is the correct final answer; requiring correct intermediate reasoning would need a stronger check. Write \\(p(y)=p_{\mathrm{ref}}(y\mid x)\\) for the frozen student’s probability of response \\(y\\).

Among normalized distributions \\(q\\) supported on this valid set, choose the one closest to the student: [[5]](#sampler-ref-5)

<div class="sampler-math">
\[
q^*=\underset{q:\,\operatorname{supp}(q)\subseteq C_\tau}{\arg\min}
D_{\mathrm{KL}}(q\|p).
\]
</div>

Let \\(Z_\tau=\sum_{y\in C_\tau}p(y)>0\\) be the student’s total probability of producing a valid response. The solution is:

<div class="sampler-math">
\[
q^*(y)=\frac{p(y)\mathbf 1[y\in C_\tau]}{Z_\tau}.
\]
</div>

**The solution is simply the student, conditioned on being valid.** For any feasible distribution with finite KL, substituting \\(p(y)=Z_\tau q^*(y)\\) on the valid set gives:

<div class="sampler-math">
\[
D_{\mathrm{KL}}(q\|p)=D_{\mathrm{KL}}(q\|q^*)-\log Z_\tau.
\]
</div>

The second term is constant; the first is minimized at zero. Therefore \\(q=q^*\\).

Suppose the student assigns 6% to one valid solution and 3% to another, with all remaining responses invalid. Conditioning gives them two-thirds and one-third of the training mass. We remove invalid responses without erasing the student’s 2:1 preference between the valid ones.

We call these responses **more on-policy** because they retain the student’s relative probabilities within the valid set. Formally, the target is the constrained student distribution, rather than its unconstrained policy.

<div class="sampler-key"><p>The expert decides what is valid; the student decides the relative probability of valid responses.</p></div>

## Amortize the target with GFlowNet
{: #gflownet}

We can score a sampled response, but computing \\(Z_\tau\\) would require summing over all valid responses. We need to match a distribution known only up to an unnormalized score. **We use GFlowNet as the training method for this amortized sampler.** [[6]](#sampler-ref-6)

Define the target score \\(R_\tau(y)=p(y)\mathbf 1[y\in C_\tau]\\). First consider a normalized sampler \\(q_\phi\\) supported on the valid set, where \\(\phi\\) denotes its learned parameters. Exact matching requires, for every valid response:

<div class="sampler-math">
\[
\begin{aligned}
q_\phi(y)&=R_\tau(y)/Z_\tau,\\
\log q_\phi(y)-\log R_\tau(y)&=-\log Z_\tau.
\end{aligned}
\]
</div>

The right-hand side does not depend on the response. **Matching the target means making this log gap the same everywhere.** Trajectory balance enforces that condition with a shared offset \\(z_\tau\\): [[7]](#sampler-ref-7)

<div class="sampler-math">
\[
\ell_{\mathrm{TB}}(y)=\left[z_\tau+\log q_\phi(y)-\log R_\tau(y)\right]^2.
\]
</div>

If the residual is zero across the valid set, normalization forces \\(z_\tau=\log Z_\tau\\). For autoregressive text generation, each response has one prefix path, so its trajectory probability is simply its sequence probability. Earlier GFlowNet work applies this connection to language and visual reasoning. [[8]](#sampler-ref-8) [[9]](#sampler-ref-9)

### Remove the normalizer with a group of responses

Draw \\(K\ge2\\) valid responses for the same prompt and expert trace. Define each log gap as \\(a_i=\log q_\phi(y_i\mid x,\tau)-\log p_{\mathrm{ref}}(y_i\mid x)\\), and let \\(\bar a\\) be their mean. The offset minimizing the group’s squared residuals is \\(z=-\bar a\\). Substitution gives:

<div class="sampler-math">
\[
\mathcal L_{\mathrm{sampler}}(\phi)
=\frac1K\sum_{i=1}^{K}\left[a_i(\phi)-\bar a\right]^2.
\]
</div>

Relative to the group, an above-average gap means a response is overrepresented; a below-average gap means it is underrepresented. The loss adjusts those relative probabilities without a separate normalizer network. Related group-relative objectives appear in GFlowNet reasoning methods. [[10]](#sampler-ref-10) [[7]](#sampler-ref-7)

<div class="sampler-key"><p>GFlowNet turns the constrained target into a trainable amortized sampler across examples.</p></div>

<details class="sampler-technical" markdown="1">
<summary>Implementation details: acceptance, gradients, and sequence scores</summary>

A raw generator can produce unacceptable outputs. In practice, the group-relative loss matches its **accepted-output distribution**. If its acceptance probability is \\(A_\phi>0\\), then for acceptable responses \\(q_\phi^+(y)=q_\phi(y)/A_\phi\\). Replacing raw log probabilities by accepted-output log probabilities subtracts the same \\(\log A_\phi\\) from every gap. Group-centering cancels this term, so the residual can be computed from raw sampler scores. This does not itself remove invalid mass; the acceptance check determines which responses enter SFT.

For a fixed batch, differentiating the centered squared loss gives the same gradient whether the mean is detached or differentiated: the residuals sum to zero. The algorithm treats sampled responses as fixed and takes one update per fresh group. It does not differentiate through sampling. Reusing old rollouts requires a suitable off-policy correction.

Use full sequence log probabilities, including EOS, under the same sampling distribution used to draw the responses. Length normalization, altered sampling temperature, or top-p truncation changes that distribution and must be handled explicitly. The empirical group offset is not an exact estimate of \\(\log Z_\tau\\) before convergence. Finally, matching a sampled group does not establish coverage of every acceptable reasoning path.

</details>

## E2S-Offline and E2S-Online
{: #versions}

The sampler sees the prompt and expert trace. The student scores responses from the prompt alone. Expert information guides data preparation; it is not an additional input the student can rely on at evaluation time.

### E2S-Offline: fit once, generate a dataset, then SFT
{: #offline}

E2S-Offline fits the sampler against a fixed starting student, generates a dataset, then fine-tunes on that fixed dataset. All generated targets refer to the same student checkpoint.

<figure id="figure-offline"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/e2s-preview/offline-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/e2s-preview/offline.svg' | relative_url }}" loading="lazy" alt="Offline E2S fits once for a fixed student, then generates a fixed training set."></picture><figcaption>Offline E2S fits once for a fixed student, then generates a fixed training set.</figcaption></figure>

### E2S-Online: refresh as the student changes
{: #online}

After a student update, the constrained target changes too. E2S-Online refits the sampler against the current student, generates a fresh batch, and runs SFT. Repeating this cycle lets data generation track the evolving policy.

<figure id="figure-online"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/e2s-preview/online-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/e2s-preview/online.svg' | relative_url }}" loading="lazy" alt="Online E2S refits after each student update, so data generation tracks the current policy."></picture><figcaption>Online E2S refits after each student update, so data generation tracks the current policy.</figcaption></figure>

<div class="sampler-key"><p>Offline fits one student; online keeps tracking the student as it changes.</p></div>

<details class="sampler-technical" markdown="1">
<summary>Implementation: one backbone, LoRA sampler</summary>

We implement the sampler using LoRA adapters on the student backbone. [[11]](#sampler-ref-11) With the adapters enabled, the model generates responses conditioned on the prompt and expert trace. With them disabled, the student scores those responses from the prompt alone or receives SFT updates.

<figure id="figure-lora"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/e2s-preview/lora-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/e2s-preview/lora.svg' | relative_url }}" loading="lazy" alt="The expert trace conditions the sampler; student scoring and SFT use only the original prompt."></picture><figcaption>The expert trace conditions the sampler; student scoring and SFT use only the original prompt.</figcaption></figure>

For a weight matrix \\(W_\theta\\), the sampler uses \\(W_\theta+sBA\\). Here \\(B\in\mathbb R^{d_{\mathrm{out}}\times r}\\), \\(A\in\mathbb R^{r\times d_{\mathrm{in}}}\\), and \\(s\\) scales an update of rank at most \\(r\\). During sampler fitting, the backbone stays frozen. During student SFT, the adapters are disabled.

</details>

<details class="sampler-technical" markdown="1">
<summary>Algorithms: offline and online</summary>

```text
E2S-Offline
  Freeze the starting student.
  Fit the sampler across expert examples:
    Draw a fresh group of K ≥ 2 valid responses.
    Score with sampler (prompt + expert) and student (prompt).
    Update sampler parameters using the group-relative loss.
  Freeze the sampler; generate the fixed SFT dataset.
  Disable adapters; fine-tune the student on that dataset.

E2S-Online
  Repeat:
    Freeze the current student for sampler fitting.
    Refit the sampler against this student using fresh groups.
    Generate a new batch of valid responses.
    Disable adapters; update the student with SFT on the batch.
```

</details>

## Experiments: does E2S learn more without forgetting more?
{: #experiments}

**E2S improves math while keeping the evaluated prior-task average near the starting model.** On Qwen2.5-3B, E2S-Offline reaches 55.5% Math avg., compared with 53.4% for MCMC + SFT. E2S-Online reaches 59.4%. Their Prior averages are 42.1% and 42.2%, compared with the base model’s 42.2%.

<div class="sampler-table-scroll">
<table class="e2s-summary"><caption>Accuracy (%) on new math tasks and prior tasks</caption><thead><tr><th scope="col">Method</th><th scope="col">Math avg.</th><th scope="col">Prior avg.</th></tr></thead><tbody>
<tr><th scope="row">Base</th><td>31.8</td><td>42.2</td></tr>
<tr><th scope="row">Expert SFT</th><td>24.2</td><td>38.9</td></tr>
<tr><th scope="row">MCMC + SFT</th><td>53.4</td><td>42.0</td></tr>
<tr class="e2s-row"><th scope="row">E2S-Offline</th><td>55.5</td><td>42.1</td></tr>
<tr class="e2s-row"><th scope="row">E2S-Online</th><td>59.4</td><td>42.2</td></tr>
</tbody></table>
</div>

### Setup

We follow the math setting in *Finetuning with Sampling*: Qwen2.5-3B, MATH levels 3–5, 8,230 training problems, and 1,024 held-out MATH problems. Math avg. is the equal-weight mean of MATH, AMC, MATH500, and GSM8K; Prior avg. averages Chemistry, MMLU, and GPQA. [[5]](#sampler-ref-5)

The baseline scores come from that study; the E2S rows are our project results. Offline and online use the same starting checkpoint, training corpus, and evaluation suite. The comparison measures accuracy and retention, rather than speed at matched total compute.

### Offline result

E2S-Offline improves Math avg. by 2.1 points over MCMC + SFT while keeping Prior avg. within 0.1 point of the base model. **At exact convergence, MCMC and E2S-Offline target the same constrained distribution.** The observed gap is therefore not evidence of a better asymptotic target: finite-budget differences can come from approximation quality, coverage, compute allocation, and reuse across examples.

### Online result

E2S-Online reaches 59.4% Math avg.—3.9 points above offline and 6.0 points above MCMC + SFT—while matching the base model’s displayed Prior avg. The trajectory below tracks online learning; the horizontal lines mark the other methods’ final scores.

<figure id="online-training-curve" class="sampler-chart"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/learned-sampler/e2s-online-progress-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/learned-sampler/e2s-online-progress.svg' | relative_url }}" width="800" height="410" loading="lazy" alt="Online Math average reaches 59.4 percent. Horizontal reference lines mark MCMC plus SFT at 53.4 and E2S-Offline at 55.5 percent."></picture><figcaption>Author-confirmed online trajectory. Horizontal lines show MCMC + SFT (53.4%) and E2S-Offline (55.5%), not their training trajectories. Online progress is normalized within its own run and does not align compute across methods. <a href="/assets/blog/learned-sampler/source/online-curve-data.json">Stored trajectory</a>.</figcaption></figure>

<details class="sampler-benchmark-details"><summary>See the scores behind each average</summary>
<p>Published comparison methods also include OPSD (Math 30.2%, Prior 40.4%), GRPO (45.7%, 41.4%), and UFT (45.2%, 42.1%). OPSD uses on-policy self-distillation; UFT combines supervised and reinforcement fine-tuning. <a href="#sampler-ref-12">[12]</a> <a href="#sampler-ref-13">[13]</a></p>
<div class="sampler-table-card sampler-detail-card"><div class="sampler-table-scroll" role="region" tabindex="0" aria-label="Per-task accuracy breakdown">
<table class="sampler-results-table sampler-detail-table"><colgroup><col class="sampler-method-col"><col><col><col><col><col><col><col></colgroup>
<thead><tr class="sampler-column-groups"><th scope="col" rowspan="2">Method</th><th scope="colgroup" colspan="4">New tasks</th><th scope="colgroup" colspan="3" class="sampler-retention-start">Prior tasks</th></tr><tr><th scope="col">MATH</th><th scope="col">AMC</th><th scope="col">MATH500</th><th scope="col">GSM8K</th><th scope="col" class="sampler-retention-start">Chem.</th><th scope="col">MMLU</th><th scope="col">GPQA</th></tr></thead><tbody>
<tr class="sampler-base-row"><th scope="row">Base model</th><td>31.5</td><td>13.3</td><td>24.5</td><td>57.9</td><td class="sampler-retention-start">28.3</td><td>65.1</td><td>33.3</td></tr>
<tr class=""><th scope="row">Expert-data SFT</th><td>24.3</td><td>10.0</td><td>16.8</td><td>45.5</td><td class="sampler-retention-start">22.2</td><td>64.8</td><td>29.8</td></tr>
<tr class=""><th scope="row">OPSD</th><td>26.7</td><td>8.4</td><td>33.2</td><td>52.4</td><td class="sampler-retention-start">24.2</td><td>65.2</td><td>31.3</td></tr>
<tr class=""><th scope="row">GRPO</th><td>45.7</td><td>24.9</td><td>31.3</td><td>80.8</td><td class="sampler-retention-start">27.8</td><td>65.2</td><td>31.3</td></tr>
<tr class=""><th scope="row">UFT</th><td>47.0</td><td>29.3</td><td>29.7</td><td>74.6</td><td class="sampler-retention-start">28.3</td><td>65.3</td><td>32.8</td></tr>
<tr class="sampler-reference-row"><th scope="row">MCMC + SFT</th><td>49.5</td><td>27.7</td><td>58.2</td><td>78.2</td><td class="sampler-retention-start">26.6</td><td>65.1</td><td>34.3</td></tr>
<tr class="sampler-result-row"><th scope="row">E2S-Offline</th><td>51.5</td><td>28.9</td><td>61.0</td><td>80.6</td><td class="sampler-retention-start">28.0</td><td>65.1</td><td>33.3</td></tr>
<tr class="sampler-result-row"><th scope="row">E2S-Online</th><td>56.3</td><td>32.5</td><td>65.4</td><td>83.2</td><td class="sampler-retention-start">28.2</td><td>65.1</td><td>33.3</td></tr>
</tbody></table></div></div></details>

These results support stronger math learning with little change in the evaluated prior-task average. They compare complete training schedules; isolating the effect of refreshing alone requires equal-compute frozen-versus-refreshed experiments. An average over three prior benchmarks also does not establish retention of every capability.

<div class="sampler-key sampler-key-teal"><p>E2S improves math while keeping prior-task performance near the starting model.</p></div>

## How E2S differs from RL, OPD, and MCMC
{: #comparison}

**RL starts from student trajectories and primarily optimizes reward.** GRPO increases probability on better-rewarded trajectories using policy constraints and clipping. E2S explicitly specifies a constrained target and trains a sampler to match it. This is a distinction between these objectives, not a claim that RL cannot be understood through distributions: KL-regularized RL can also have a reward-tilted target. [[3]](#sampler-ref-3)

**OPD brings teacher supervision to student-visited states.** It samples from the student and supplies dense teacher feedback on the resulting prefixes. E2S instead starts from expert information and constructs an SFT target distribution for the student. [[4]](#sampler-ref-4)

**MCMC and E2S move expert information toward the student.** Both target the student conditioned on validity. MCMC searches separately for each example; E2S stores reusable sampling behavior in learned parameters. [[5]](#sampler-ref-5)

This distinction matters for diversity. A binary correctness reward alone does not require every valid reasoning mode to remain represented. If the E2S target assigns mass to several modes, exact distribution matching requires preserving their relative mass. **Diversity is a property of the target distribution, not an extra diversity bonus.** A finite learned sampler still needs sufficient coverage to achieve that goal.

## Looking ahead
{: #lookahead}

**Offline E2S makes data preparation specific to the learner.** During mid-training or post-training, an expert corpus can be rewritten into valid trajectories that a particular student is more likely to produce. A learned sampler also makes each expert example reusable: drawing multiple valid solutions could provide additional useful supervision. Testing this at matched compute can separate useful diversity from simply doing more training.

**Online E2S makes data generation part of learning.** An expert-conditioned sampler can guide generation toward valid responses while tracking the student as it changes. The broader idea is to learn the data-generation policy itself: not only which information to teach, but how to express it for the model that will learn from it.

<div class="sampler-key"><p>Instead of storing one expert answer, learn a reusable distribution of ways to teach it.</p></div>

## References
{: #references}


<p id="sampler-ref-1"><strong>[1]</strong> Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. <a href="https://arxiv.org/abs/2509.04259v1">RL’s Razor: Why Online Reinforcement Learning Forgets Less</a>. arXiv:2509.04259v1, 2025. See §4–5 and Appendix A for the KL analysis and its assumptions.</p>

<p id="sampler-ref-2"><strong>[2]</strong> Howard Chen, Noam Razin, Karthik Narasimhan, and Danqi Chen. <a href="https://proceedings.mlr.press/v306/chen26do.html">Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting</a>. ICML, 2026. See §3–4 for distributional analysis and approximately on-policy SFT; Appendix A.5 discusses limits of KL as a predictor.</p>

<p id="sampler-ref-3"><strong>[3]</strong> Zhihong Shao et al. <a href="https://arxiv.org/abs/2402.03300">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>. arXiv:2402.03300, 2024. See §4.1 for GRPO.</p>

<p id="sampler-ref-4"><strong>[4]</strong> Kevin Lu and Thinking Machines Lab. <a href="https://thinkingmachines.ai/blog/on-policy-distillation/">On-Policy Distillation</a>. Thinking Machines Lab: Connectionism, 2025.</p>

<p id="sampler-ref-5"><strong>[5]</strong> Aayush Karan, Sitan Chen, and Yilun Du. <a href="https://arxiv.org/abs/2610.02140v1">Finetuning with Sampling: SFT Learns Better Than You Think</a>. arXiv:2610.02140v1, 2026.</p>

<p id="sampler-ref-6"><strong>[6]</strong> Yoshua Bengio. <a href="https://yoshuabengio.org/en/blog/generative-flow-networks">Generative Flow Networks</a>. 2022. Discusses learning sequential construction policies and contrasts them with MCMC sampling.</p>

<p id="sampler-ref-7"><strong>[7]</strong> Xuekai Zhu et al. <a href="https://arxiv.org/abs/2509.15207v3">FlowRL: Matching Reward Distributions for LLM Reasoning</a>. arXiv:2509.15207v3, 2025.</p>

<p id="sampler-ref-8"><strong>[8]</strong> Fangxu Yu, Lai Jiang, Haoqiang Kang, Shibo Hao, and Lianhui Qin. <a href="https://arxiv.org/abs/2406.05673v6">Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples</a>. ICML, 2025.</p>

<p id="sampler-ref-9"><strong>[9]</strong> Haoqiang Kang, Enna Sachdeva, Piyush Gupta, Sangjae Bae, and Kwonjoon Lee. <a href="https://arxiv.org/abs/2503.06514">GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks</a>. CVPR, 2025.</p>

<p id="sampler-ref-10"><strong>[10]</strong> Xiaodong Liu et al. <a href="https://arxiv.org/abs/2607.13394v1">GFlowRL: Scaling Distribution-Matching RL to Large Language Models</a>. arXiv:2607.13394v1, 2026.</p>

<p id="sampler-ref-11"><strong>[11]</strong> Edward J. Hu et al. <a href="https://arxiv.org/abs/2106.09685">LoRA: Low-Rank Adaptation of Large Language Models</a>. ICLR, 2022.</p>

<p id="sampler-ref-12"><strong>[12]</strong> Siyan Zhao et al. <a href="https://arxiv.org/abs/2601.18734">Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models</a>. arXiv:2601.18734, 2026.</p>

<p id="sampler-ref-13"><strong>[13]</strong> Mingyang Liu, Gabriele Farina, and Asuman Ozdaglar. <a href="https://arxiv.org/abs/2505.16984">UFT: Unifying Supervised and Reinforcement Fine-Tuning</a>. arXiv:2505.16984, 2025.</p>

## Citation
{: #citation}

If you found this post useful, please cite it as:

Murray Kang. “E2S Finetuning: From Off-Policy Expert Data to On-Policy Training Data.” October 2026.

{% raw %}
```bibtex
@misc{kang2026onpolicysft,
  author = {Kang, Murray},
  title = {{E2S Finetuning: From Off-Policy Expert Data to On-Policy Training Data}},
  year = {2026},
  month = oct,
  howpublished = {Research blog},
  url = {https://mk322.github.io/blog/e2s-finetuning-preview/}
}
```
{% endraw %}
