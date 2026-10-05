---
layout: blog-post
title: "E2S Finetuning: From Expert Demonstrations to On-Policy Learning"
subtitle: "Make SFT great again! Turn off-policy expert data into more on-policy training data."
permalink: /blog/learned-on-policy-sampler/
date: 2026-10-03
last_modified_at: 2026-10-05
author_profile: false
excerpt: "E2S learns an amortized GFlowNet sampler to turn off-policy expert responses into more on-policy SFT data, offline for dataset creation or online for post-training."
tldr: |
  **E2S Finetuning learns an amortized sampler with GFlowNet to turn off-policy expert data into more on-policy training targets for SFT.** E2S-Offline creates a dataset for a fixed student. E2S-Online is a post-training method that alternates generating training data and fine-tuning the student.
---

<link rel="stylesheet" href="{{ '/assets/blog/learned-sampler/article.css' | relative_url }}">
<link rel="stylesheet" href="{{ '/assets/blog/e2s-preview/article.css' | relative_url }}">

<nav class="sampler-toc" aria-label="Article contents"><details><summary>On this page</summary><ol><li><a href="#sft-problem">Off-policy mismatch</a></li><li><a href="#amortization">Existing approaches: RL, OPD, and MCMC</a></li><li><a href="#target">The target distribution</a></li><li><a href="#gflownet">GFlowNet and group matching</a></li><li><a href="#versions">Offline and online</a></li><li><a href="#experiments">Experiments</a></li><li><a href="#scaling">Diversity scaling</a></li><li><a href="#comparison">RL, OPD, and MCMC</a></li><li><a href="#lookahead">Looking ahead</a></li><li><a href="#references">References</a></li><li><a href="#citation">Citation</a></li></ol></details></nav>
<script defer src="{{ "/assets/blog/learned-sampler/navigation.js" | relative_url }}"></script>
<script defer src="{{ "/assets/blog/e2s-preview/navigation.js" | relative_url }}"></script>

Expert responses can contain valuable information yet follow reasoning paths the student model would rarely produce. Training it to imitate those paths can require a large change in its behavior.

**E2S keeps the required expert information while favoring answers the model is already more likely to produce.**

<figure id="figure-constrained-distribution" class="sampler-chart"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/e2s-preview/constrained-distribution-mobile.svg' | relative_url }}?v=db097d1a6f63"><img src="{{ '/assets/blog/e2s-preview/constrained-distribution.svg' | relative_url }}?v=389e70b1b7b0" width="800" height="360" loading="lazy" alt="Before: the student model and expert prefer different responses. After: training data keeps responses that meet the expert semantic constraints and preserves the student model’s relative preferences among them."></picture><figcaption>Shaded region: responses that meet the expert semantic constraints. Their relative probabilities follow the student model. Conceptual illustration.</figcaption></figure>

MCMC can generate such answers by searching separately for each example. E2S learns a reusable sampler across examples; the student still learns through SFT.

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

The same mismatch appears word by word: SFT asks the student to continue the expert’s reasoning, even when the student would have written something different up to that point.

<details class="sampler-technical" markdown="1">
<summary>Technical note: learning to continue an expert response</summary>

For prompt \\(x\\) and expert response \\(y=(y_1,\ldots,y_T)\\), the sequence loss is \\(-\sum_{t=1}^{T}\log p_\theta(y_t\mid x,y_{<t})\\), including the termination token. The prefixes \\(y_{<t}\\) come from the demonstration distribution. They are not sampled from the current student.

</details>

<div class="sampler-key"><div class="e2s-key-title">Key message</div><p>Off-policy SFT can force the student to move farther than the task itself requires.</p></div>

## Existing approaches to the distribution mismatch
{: #amortization}

Both routes address the same mismatch, but they change different parts of training: **where supervision is applied, or which responses become training targets.**

**Move learning to the student — RL / OPD.** Let the current student generate a response, then provide feedback on that trajectory. RL scores the student’s answers with rewards. In on-policy distillation (OPD), a teacher reads what the student has written so far and provides probabilities for the next token. Feedback is applied to the student’s own attempt. [[3]](#sampler-ref-3) [[4]](#sampler-ref-4)

**Move the data to the student — MCMC + SFT.** Use the expert constraint to define which responses are valid, and the student policy to define their relative probabilities. Sample from that constrained distribution, then train the student on the resulting responses with SFT. Here, we change the training targets themselves. MCMC and E2S are two ways to obtain them. [[5]](#sampler-ref-5)

<figure id="figure-two-routes"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/e2s-preview/two-routes-mobile.svg' | relative_url }}?v=eeb5a96589fc"><img src="{{ '/assets/blog/e2s-preview/two-routes.svg' | relative_url }}?v=e0d012879c33" loading="lazy" alt="RL / OPD apply feedback to student-generated responses. MCMC + SFT uses expert information and the student policy to prepare new training responses before SFT."></picture><figcaption>RL / OPD bring feedback to student responses. MCMC + SFT changes the training responses before updating the student.</figcaption></figure>

Rejection sampling can also produce the constrained target: draw from the student and keep valid responses. It wastes most rollouts when success is rare. MCMC instead starts from an expert solution and repeatedly proposes, scores, and accepts or rejects edits—the route developed in *Finetuning with Sampling*. [[5]](#sampler-ref-5)

Write \\(x\\) for the prompt, \\(\tau\\) for its off-policy expert trace, and \\(y\\) for a new response that satisfies the expert constraint while following the student’s relative probabilities. **Both MCMC and E2S turn \\(\tau\\) into a more on-policy \\(y\\), then use \\((x,y)\\) for student SFT.** The expert trace guides data preparation; it is the new response \\(y\\) that becomes the SFT target.

E2S takes this second route. MCMC runs a new chain for every example. **That repeated search is the motivation for E2S.** Instead, we train a conditional sampler across prompts: work on one example changes the sampler used for the next.

<figure id="figure-amortization"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/e2s-preview/amortization-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/e2s-preview/amortization.svg' | relative_url }}" loading="lazy" alt="Both routes turn off-policy expert trace τ into more on-policy response y, then train the student on (x, y). MCMC searches anew for each example; E2S reuses learned sampler parameters."></picture><figcaption>Both routes turn off-policy expert trace τ into more on-policy response y, then train the student on (x, y). MCMC searches anew for each example; E2S reuses learned sampler parameters.</figcaption></figure>

MCMC revises one response at a time; E2S improves a sampler that can generate responses for many examples. E2S moves part of the repeated search cost into sampler training. Whether that saves total compute depends on the training cost and how much useful data the sampler subsequently generates.

<div class="sampler-key"><div class="e2s-key-title">Key message</div><p>MCMC searches again; E2S learns a sampling policy it can reuse across examples.</p></div>

## The target distribution: preserve expert information with minimal policy change
{: #target}

**The expert semantic constraint requires responses to be semantically equivalent to the expert response.** Following *Finetuning with Sampling*, let \\(C_\tau\\) contain responses satisfying this constraint for expert trace \\(\tau\\). Math and science use correct outcomes; knowledge tasks require preserving key facts. [[5]](#sampler-ref-5)

For prompt \\(x\\), write \\(p(y)=p_{\mathrm{ref}}(y\mid x)\\) for the frozen student’s response probability. The expert requirement selects valid responses; the student determines their relative probabilities.

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

**The solution preserves the student’s relative probabilities among responses that satisfy the expert requirement.** The opening illustration keeps both valid modes in their original proportions and removes responses outside the requirement.

For any feasible distribution with finite KL, substituting \\(p(y)=Z_\tau q^*(y)\\) on the valid set gives:

<div class="sampler-math">
\[
D_{\mathrm{KL}}(q\|p)=D_{\mathrm{KL}}(q\|q^*)-\log Z_\tau.
\]
</div>

The second term is constant; the first is minimized at zero. Therefore \\(q=q^*\\).

Suppose the student assigns 6% to one valid solution and 3% to another, with all remaining responses invalid. Conditioning gives them two-thirds and one-third of the training mass. We remove invalid responses without erasing the student’s 2:1 preference between the valid ones.

We call these responses **more on-policy** because they retain the student’s relative probabilities within the valid set. Formally, the target is the constrained student distribution, rather than its unconstrained policy.

<div class="sampler-key"><div class="e2s-key-title">Key message</div><p>The expert decides what is valid; the student decides the relative probability of valid responses.</p></div>

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

If the residual is zero across the valid set, normalization forces \\(z_\tau=\log Z_\tau\\). For text generated one token at a time, a response follows a single sequence of token choices. Its probability is the product of those token probabilities, including the end-of-response token. Earlier GFlowNet work applies this connection to language and visual reasoning. [[8]](#sampler-ref-8) [[9]](#sampler-ref-9)

### Remove the normalizer with a group of responses

Draw \\(K\ge2\\) valid responses for the same prompt and expert trace. Define each log gap as \\(a_i=\log q_\phi(y_i\mid x,\tau)-\log p_{\mathrm{ref}}(y_i\mid x)\\), and let \\(\bar a\\) be their mean. The offset minimizing the group’s squared residuals is \\(z=-\bar a\\). Substitution gives:

<div class="sampler-math">
\[
\mathcal L_{\mathrm{sampler}}(\phi)
=\frac1K\sum_{i=1}^{K}\left[a_i(\phi)-\bar a\right]^2.
\]
</div>

Relative to the group, an above-average gap means a response is overrepresented; a below-average gap means it is underrepresented. The loss adjusts those relative probabilities without a separate normalizer network. Related group-relative objectives appear in GFlowNet reasoning methods. [[10]](#sampler-ref-10) [[7]](#sampler-ref-7)

<div class="sampler-key"><div class="e2s-key-title">Key message</div><p>GFlowNet turns the constrained target into a trainable amortized sampler across examples.</p></div>

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

### E2S-Online: generate data during training
{: #online}

After a student update, the constrained target changes too. E2S-Online refits the sampler against the current student, generates a fresh batch, and runs SFT. Repeating this cycle lets data generation track the evolving policy.

<figure id="figure-online"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/e2s-preview/online-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/e2s-preview/online.svg' | relative_url }}" loading="lazy" alt="Online E2S refits after each student update, so data generation tracks the current policy."></picture><figcaption>Online E2S refits after each student update, so data generation tracks the current policy.</figcaption></figure>

<div class="sampler-key"><div class="e2s-key-title">Key message</div><p>Offline fits one student; online keeps tracking the student as it changes.</p></div>

<details class="sampler-technical" markdown="1">
<summary>Implementation: one backbone, LoRA sampler</summary>

We implement the sampler using LoRA adapters on the student backbone. [[11]](#sampler-ref-11) With the adapters enabled, the model generates responses conditioned on the prompt and expert trace. With them disabled, the student scores those responses from the prompt alone or receives SFT updates.

<figure id="figure-lora"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/e2s-preview/lora-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/e2s-preview/lora.svg' | relative_url }}" loading="lazy" alt="The expert trace conditions the sampler; student scoring and SFT use only the original prompt."></picture><figcaption>The expert trace conditions the sampler; student scoring and SFT use only the original prompt.</figcaption></figure>

For a weight matrix \\(W_\theta\\), the sampler uses \\(W_\theta+sBA\\). Here \\(B\in\mathbb R^{d_{\mathrm{out}}\times r}\\), \\(A\in\mathbb R^{r\times d_{\mathrm{in}}}\\), and \\(s\\) scales an update of rank at most \\(r\\). During sampler fitting, the backbone stays frozen. During student SFT, the adapters are disabled.

</details>

<details class="sampler-technical" markdown="1">
<summary>Algorithms: sampler fitting, offline E2S, and online E2S</summary>

Let D = {(x, τ)} be the expert corpus, θ the student parameters, and φ the sampler adapter parameters. Valid(x, τ, y) checks the expert constraint. SFT(θ, D*) minimizes the negative log probability of the generated responses given their original prompts, with the sampler adapters disabled.

**Subroutine · FitSampler**

```text
Input:  expert batch B, frozen student θ,
        sampler φ, group size K ≥ 2,
        update budget U, learning rate ηφ
Output: updated sampler φ

for u = 1, …, U do
    Draw (x, τ) from B
    Sample y₁, …, yK ∼ qφ(· | x, τ),
        retaining only Valid(x, τ, yi) = true
    for i = 1, …, K do
        ai ← log qφ(yi | x, τ)
              − log pθ(yi | x)
    ā ← (1/K) Σi ai
    L ← (1/K) Σi (ai − stopgrad(ā))²
    φ ← φ − ηφ ∇φ L
return φ
```

Only φ is updated. The student scores and sampled responses are held fixed during each gradient step. Each update uses a fresh group; the validity check is applied during sampling.

**Algorithm 1 · E2S-Offline**

```text
Input:  expert corpus D, student θ₀,
        sampler initialization φ₀,
        sampler budget U, samples per example m
Output: fine-tuned student θ

φ ← FitSampler(D, θ₀, φ₀, K, U, ηφ)
D* ← ∅
for each (x, τ) in D do
    Draw m valid responses y ∼ qφ(· | x, τ)
    Add each pair (x, y) to D*
θ ← SFT(θ₀, D*)
return θ
```

The student remains θ₀ throughout sampler fitting and data generation. D* is fixed before student training begins.

**Algorithm 2 · E2S-Online**

```text
Input:  expert corpus D, student θ₀,
        sampler initialization φ₀, rounds T,
        per-round sampler budget U,
        samples per example m
Output: fine-tuned student θT

θ ← θ₀; φ ← φ₀
for t = 1, …, T do
    Select expert batch Bt from D
    φ ← FitSampler(Bt, θ, φ, K, U, ηφ)
    Dt* ← ∅
    for each (x, τ) in Bt do
        Draw m valid responses y ∼ qφ(· | x, τ)
        Add each pair (x, y) to Dt*
    θ ← SFT(θ, Dt*)
return θ
```

Each round freezes the current student while fitting the sampler and generating targets. SFT then updates θ; the next round fits against that updated student. K and ηφ are shared sampler hyperparameters in both algorithms.

</details>

## Experiments: does E2S learn more without forgetting more?
{: #experiments}

### Setup

We follow the math setting in *Finetuning with Sampling*: Qwen2.5-3B, MATH levels 3–5, 8,230 training problems, and 1,024 held-out MATH problems. Math avg. is the equal-weight mean of MATH, AMC, MATH500, and GSM8K; Prior avg. averages Chemistry, MMLU, and GPQA. [[5]](#sampler-ref-5)

The baseline scores come from that study; the E2S rows are our project results. Offline and online use the same starting checkpoint, training corpus, and evaluation suite. The comparison measures accuracy and retention, rather than speed at matched total compute.

### Results

<div class="sampler-table-scroll">
<table class="e2s-summary"><caption>Accuracy (%) on new math tasks and prior tasks</caption><thead><tr><th scope="col">Method</th><th scope="col">Math avg.</th><th scope="col">Prior avg.</th></tr></thead><tbody>
<tr><th scope="row">Base</th><td>31.8</td><td>42.2</td></tr>
<tr><th scope="row">Expert SFT</th><td>24.2</td><td>38.9</td></tr>
<tr><th scope="row">OPSD</th><td>30.2</td><td>40.4</td></tr>
<tr><th scope="row">GRPO</th><td>45.7</td><td>41.4</td></tr>
<tr><th scope="row">MCMC + SFT</th><td>53.4</td><td>42.0</td></tr>
<tr class="e2s-row"><th scope="row">E2S-Offline</th><td>55.5</td><td>42.1</td></tr>
<tr class="e2s-row"><th scope="row">E2S-Online</th><td>59.4</td><td>42.2</td></tr>
</tbody></table>
</div>

**Offline.** E2S-Offline improves Math avg. by **3.9% relative to MCMC + SFT**, while keeping the evaluated prior-task average slightly closer to the starting model. **It learns more from expert data while retaining prior skills at least as well in this comparison.**

MCMC starts a new search for each example. E2S learns a sampling strategy across examples and reuses it to generate training responses. The practical advantage is that **search experience becomes reusable knowledge in the sampler**, rather than remaining in a separate chain for each prompt.

**Online.** E2S-Online adds a **7.0% relative improvement over offline**, or **11.2% over MCMC + SFT**, while matching the starting model’s displayed Prior avg. The trajectory below tracks online learning; the horizontal lines mark the other methods’ final scores.

<figure id="online-training-curve" class="sampler-chart"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/learned-sampler/e2s-online-progress-mobile.svg' | relative_url }}?v=e6cebf6d2943"><img src="{{ '/assets/blog/learned-sampler/e2s-online-progress.svg' | relative_url }}?v=a08036ba2e94" width="800" height="410" loading="lazy" alt="Online Math average reaches 59.4 percent. Horizontal reference lines mark MCMC plus SFT at 53.4 and E2S-Offline at 55.5 percent."></picture><figcaption><strong>Online training curve.</strong> Horizontal lines mark the final scores of MCMC + SFT and E2S-Offline.</figcaption></figure>

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

### Conclusion

**These results support adapting the training-data distribution as a way to improve the learning–retention tradeoff.** Expert information can support new-task learning without a comparable decline in the evaluated prior-task average. The online result further supports treating data generation as part of learning, rather than only preparing targets once.

The comparison evaluates complete training schedules. A controlled comparison at equal compute is needed to isolate which parts of online training drive the gain; an average over three prior benchmarks also does not establish retention of every capability.

<div class="sampler-key sampler-key-teal"><div class="e2s-key-title">Key message</div><p>Adapting expert data to the student can improve the learning–retention tradeoff without changing the SFT objective.</p></div>

## Diversity scaling: more responses from the same expert examples
{: #scaling}

**One expert example can teach the student through more than one response.** E2S learns a distribution of valid responses, so we can draw again from the same expert example rather than simply copy one answer. The question is whether those additional draws keep improving SFT.

### Setup

We keep the 8,230 expert problems and the fitted offline sampler fixed, then generate 1, 2, or 4 valid responses per problem. That gives 8,230, 16,460, or 32,920 SFT responses without adding new expert problems. We evaluate Math avg. on the same four math benchmarks. This is a separate scaling sweep from the main method comparison above.

### Results

Math avg. rises from 55.0% with one response per example to 56.3% with two and 57.2% with four—a 2.2-point gain from 1× to 4×. The expert corpus stays the same; the sampler supplies additional training responses.

<figure id="offline-scaling-curve" class="sampler-chart"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/e2s-preview/e2s-offline-scaling-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/e2s-preview/e2s-offline-scaling.svg' | relative_url }}" width="800" height="440" loading="lazy" alt="Offline Math average increases from 55.0 to 56.3 to 57.2 percent with 1, 2 and 4 responses per expert example, yielding 8230, 16460 and 32920 SFT responses."></picture><figcaption>Offline scaling with a fixed expert corpus. Values confirmed by the author; <a href="{{ '/assets/blog/e2s-preview/source/scaling-data.json' | relative_url }}">result record</a>.</figcaption></figure>

### Conclusion

Additional samples improve math performance in this sweep. This is a useful scaling direction: we can expand the training responses without collecting more expert solutions. Separating the benefit of different responses from extra SFT compute requires a matched-compute duplication control; accuracy alone does not measure how many reasoning paths the sampler covers.

<div class="sampler-key sampler-key-teal"><div class="e2s-key-title">Key message</div><p>Sampling more responses from the same expert examples continues to improve offline SFT.</p></div>

## How E2S differs from RL, OPD, and MCMC
{: #comparison}

**RL starts from student trajectories and primarily optimizes reward.** GRPO increases probability on better-rewarded trajectories using policy constraints and clipping. E2S explicitly specifies a constrained target and trains a sampler to match it. This is a distinction between these objectives, not a claim that RL cannot be understood through distributions: KL-regularized RL can also have a reward-tilted target. [[3]](#sampler-ref-3)

**OPD gives teacher feedback on the student’s own writing.** The student generates a response, and the teacher provides next-token probabilities after each part the student has written. E2S instead starts from expert information and constructs an SFT target distribution for the student. [[4]](#sampler-ref-4)

**MCMC and E2S move expert information toward the student.** Both target the student conditioned on validity. MCMC searches separately for each example; E2S stores reusable sampling behavior in learned parameters. [[5]](#sampler-ref-5)

This distinction matters for diversity. A binary correctness reward alone does not require every valid reasoning mode to remain represented. If the E2S target assigns mass to several modes, exact distribution matching requires preserving their relative mass. **Diversity is a property of the target distribution, not an extra diversity bonus.** A finite learned sampler still needs sufficient coverage to achieve that goal.

## Looking ahead
{: #lookahead}

**Offline E2S makes data preparation specific to the learner.** During mid-training or post-training, an expert corpus can be rewritten into valid trajectories that a particular student is more likely to produce. A learned sampler also makes each expert example reusable: the scaling sweep shows gains from additional valid responses. A matched-compute study can test how much of that gain comes from useful diversity rather than additional training.

**Online E2S makes data generation part of learning.** An expert-conditioned sampler can guide generation toward valid responses while tracking the student as it changes. The broader idea is to learn the data-generation policy itself: not only which information to teach, but how to express it for the model that will learn from it.

<div class="sampler-key"><div class="e2s-key-title">Key message</div><p>Instead of storing one expert answer, learn a reusable distribution of ways to teach it.</p></div>

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

Murray Kang. “E2S Finetuning: From Expert Demonstrations to On-Policy Learning.” October 2026.

{% raw %}
```bibtex
@misc{kang2026onpolicysft,
  author = {Kang, Haoqiang},
  title = {{E2S Finetuning: From Expert Demonstrations to On-Policy Learning}},
  year = {2026},
  month = oct,
  howpublished = {Research blog},
  url = {https://mk322.github.io/blog/learned-on-policy-sampler/}
}
```
{% endraw %}
