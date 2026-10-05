---
title: "E2S Finetuning: From Off-Policy Expert Data to On-Policy Student Data"
subtitle: "Keep what the expert knows. Learn how to express it for the student."
permalink: /blog/learned-on-policy-sampler/
last_modified_at: 2026-10-05
excerpt: "Keep SFT simple; change the data. E2S learns a reusable sampler that turns expert demonstrations into calibrated student training targets—learning new tasks while retaining prior capabilities."
tldr: |
  **E2S Finetuning learns an amortized sampler with GFlowNet to turn off-policy expert data into more on-policy training targets for SFT.** E2S-Offline creates a dataset for a fixed student. E2S-Online serves as a post-training method, refreshing the targets as the student learns.
---

<link rel="stylesheet" href="{{ '/assets/blog/learned-sampler/article.css' | relative_url }}?v=e2s-educational-3">

<nav class="sampler-toc" aria-label="Article contents"><details open><summary>On this page</summary><ol><li><a href="#sft-problem">Why off-policy SFT can forget</a></li><li><a href="#amortization">Two ways to fix the mismatch</a></li><li><a href="#target">E2S: learn the data distribution</a><ol><li><a href="#constrained-target">Define the target</a></li><li><a href="#solve-target">Solve the constraint</a></li><li><a href="#train-sampler">Train a GFlowNet sampler</a></li><li><a href="#group-loss">Remove the normalizer</a></li></ol></li><li><a href="#versions">Offline and online</a><ol><li><a href="#offline">Prepare once</a></li><li><a href="#online">Refresh as we learn</a></li></ol></li><li><a href="#experiments-offline">Experiments: learn more, retain more</a></li><li><a href="#experiments-online">Does online refresh help?</a></li><li><a href="#scaling">Scaling through diversity</a></li><li><a href="#lookahead">Looking ahead</a></li><li><a href="#references">References</a></li><li><a href="#citation">Citation</a></li></ol></details></nav>
<script defer src="{{ '/assets/blog/learned-sampler/navigation.js' | relative_url }}"></script>

**Supervised fine-tuning is simple and efficient:** give a model a prompt and a good response, then train it to predict that response. The targets are already available. Each student update needs no fresh rollouts or teacher feedback.

But a good response is not necessarily an easy response for this student to learn from. An expert may skip steps the student needs, introduce an unfamiliar trick, or take a reasoning path the student would almost never produce. SFT still asks the student to imitate the entire trajectory.

That leaves a useful question: **could we preserve what the expert teaches while changing how it is expressed?**

<div class="sampler-key"><span class="sampler-key-label">The idea</span><p><strong>Keep SFT simple; change the data.</strong> E2S Finetuning learns to turn off-policy expert demonstrations into calibrated “on-policy” training data for the student.</p></div>

We formulate this as constrained distribution matching: preserve the required expert information, and otherwise stay as close as possible to the student’s policy. MCMC can sample toward this target by searching separately for each example. E2S instead learns a reusable sampler across examples. The sampler does the data preparation; the student update remains ordinary SFT.

## Why off-policy SFT can cause forgetting
{: #sft-problem}

Suppose we fine-tune a model to solve harder math problems. We want its math to improve without damaging the knowledge and skills it already has. The choice of training responses affects both outcomes.

For a fixed prompt \\(x\\), let \\(q_{\mathrm{data}}(y\mid x)\\) be the demonstration distribution and \\(p_\theta(y\mid x)\\) the student. SFT minimizes:

<div class="sampler-math"><span class="sampler-math-label">What SFT fits</span>
\[
\begin{aligned}
\mathcal L_{\mathrm{SFT}}(\theta)
&=\mathbb E_{y\sim q_{\mathrm{data}}(\cdot\mid x)}[-\log p_\theta(y\mid x)]\\
&=H(q_{\mathrm{data}})+D_{\mathrm{KL}}(q_{\mathrm{data}}\|p_\theta).
\end{aligned}
\tag{1}
\]
</div>

The entropy term is fixed. The loss therefore pulls the student toward the demonstration distribution, with no explicit term keeping it near its starting policy \\(p_0\\). In an expressive model class, a perfect fit to the population objective gives \\(p_{\mathrm{fit}}=q_{\mathrm{data}}\\). Its distance from the starting model is then exactly \\(D_{\mathrm{KL}}(q_{\mathrm{data}}\|p_0)\\).

**If the data are far from the student, fitting them asks the student to move far.** Some movement is necessary to learn something new. Reproducing the expert’s particular wording and reasoning path may require additional movement that the task itself does not demand.

There is a token-level view of the same problem. SFT predicts each next token after a prefix from the expert response. Those prefixes can differ substantially from the ones the student would visit on its own. Correct supervision can therefore arrive in unfamiliar contexts.

Research on forgetting connects larger policy shifts with poorer retention, and finds that using data from the evolving student can reduce forgetting. This motivates changing the data distribution; it is not a guarantee that small training-task KL preserves every other skill. [[9]](#sampler-ref-9) [[10]](#sampler-ref-10)

<div class="sampler-key"><span class="sampler-key-label">Key message</span><p>The information may be worth learning even when the expert’s exact trajectory is a poor fit for the learner.</p></div>

## Two ways to fix the mismatch
{: #amortization}

One route is to let the student choose the trajectories, then teach on the states it actually visits. **RL** supplies rewards for student rollouts. **On-policy distillation** supplies teacher probabilities on student-generated prefixes. Both put fresh generation and feedback inside the learning loop. [[11]](#sampler-ref-11) [[2]](#sampler-ref-2)

The other route is to keep SFT and move the expert data toward the student. Rejection sampling does this by generating from the student and keeping acceptable responses. When success is rare, however, most generations are discarded. MCMC uses an expert solution to initialize a search, then repeatedly proposes and scores revisions. This is the route developed in *Finetuning with Sampling*. [[1]](#sampler-ref-1)

E2S takes this second route and changes how sampling is done:

- **MCMC searches for each example.** A chain refines its current response; its search state belongs to that prompt.
- **E2S learns across examples.** Training updates a conditional sampler whose parameters can be reused to transform many expert demonstrations.

<figure id="sampler-mcmc-conversion" class="sampler-reference-figure"><a class="sampler-reference-image" href="{{ '/assets/blog/learned-sampler/amortized-e2s-v3.png' | relative_url }}" target="_blank" rel="noopener" aria-label="Open full-size diagram"><picture><source media="(max-width: 760px)" srcset="{{ '/assets/blog/learned-sampler/amortized-e2s-v3-mobile.png' | relative_url }}"><img src="{{ '/assets/blog/learned-sampler/amortized-e2s-v3.png' | relative_url }}" width="1603" height="981" loading="lazy" alt="MCMC searches each example; an amortized sampler learns reusable parameters. Both turn off-policy expert data into calibrated on-policy training data."></picture></a><figcaption><strong>Search each response, or learn a sampler to reuse.</strong> Both routes turn off-policy expert examples into calibrated “on-policy” data for student SFT. <a class="sampler-fullsize-link" href="{{ '/assets/blog/learned-sampler/amortized-e2s-v3.png' | relative_url }}" target="_blank" rel="noopener">Open full size ↗</a></figcaption></figure>

This is **amortization**: invest in a reusable generation policy, then use it repeatedly. It creates an opportunity to spread sampler-training cost over a growing dataset. Whether it saves total compute depends on how much useful data the trained sampler produces.

<div class="sampler-key"><span class="sampler-key-label">Key message</span><p>Learning how to prepare a response can help prepare the next one. The reusable object is the sampler itself.</p></div>

## E2S Finetuning: learn the data distribution
{: #target}

### 1. Define what the data must preserve
{: #constrained-target}

Fix a prompt \\(x\\), an expert trace \\(\tau\\), and a student checkpoint. Write \\(p(y)=p_{\mathrm{ref}}(y\mid x)\\) for this frozen student. Let \\(C_\tau\\) be the acceptable response set: responses that satisfy the information constraint supplied by the expert. For a math task, the simplest check is the final answer; richer checks can constrain the reasoning too.

We can now ask for the closest acceptable data distribution:

<div class="sampler-math"><span class="sampler-math-label">Constrained distribution matching</span>
\[
q^*=\underset{q:\,\operatorname{supp}(q)\subseteq C_\tau}{\arg\min}
D_{\mathrm{KL}}(q\|p).
\tag{2}
\]
</div>

Here \\(q\\) ranges over normalized probability distributions. We hold the student fixed and choose the training data. The constraint says what must be preserved; the KL term asks us to change as little else as possible. [[1]](#sampler-ref-1)

### 2. Solve the constraint
{: #solve-target}

Assume the student gives the acceptable set positive probability. Its total mass is \\(Z_\tau\\). Restrict the student to that set and renormalize:

<div class="sampler-math"><span class="sampler-math-label">The target distribution</span>
\[
\begin{aligned}
Z_\tau&=\sum_{y\in C_\tau}p(y)>0,\\
p_C(y)&=\frac{p(y)\,\mathbf 1[y\in C_\tau]}{Z_\tau}.
\end{aligned}
\tag{3}
\]
</div>

This is the exact solution. For any feasible \\(q\\) with finite KL, substitute \\(p(y)=Z_\tau p_C(y)\\) on the acceptable set:

<div class="sampler-math">
\[
D_{\mathrm{KL}}(q\|p)
= D_{\mathrm{KL}}(q\|p_C)-\log Z_\tau.
\tag{4}
\]
</div>

The second term is constant. The first is nonnegative and reaches zero only at \\(q=p_C\\). So the best data distribution is **the student conditioned on satisfying the expert constraint**.

A small example makes the consequence clear. Suppose two acceptable solution paths have student probabilities 6% and 3%, and all other responses are unacceptable. After conditioning, their probabilities become two-thirds and one-third. We remove the unacceptable responses while preserving the student’s 2:1 preference between the two useful paths.

<div class="sampler-key sampler-key-teal"><span class="sampler-key-label">Key message</span><p><strong>The expert defines what must be preserved. The student defines how probability is shared among the acceptable responses.</strong> This is what calibrated “on-policy” data means here.</p></div>

### 3. Train a GFlowNet sampler for that target
{: #train-sampler}

We can score a response under the frozen student. What we cannot do is sum over all possible responses to calculate \\(Z_\tau\\). Define the unnormalized target \\(R_\tau(y)=p(y)\mathbf 1[y\in C_\tau]\\). We want to sample in proportion to this score without computing its sum.

That is the GFlowNet problem: learn a generative policy whose terminal distribution matches an unnormalized density. [[12]](#sampler-ref-12) A language model already generates a response through a sequence of prefixes, so it can serve as the sampler. Our earlier work uses this connection for language and visual reasoning. [[3]](#sampler-ref-3) [[4]](#sampler-ref-4)

First consider a sampler \\(q_\phi\\) supported on the acceptable set. The matching condition can be written as:

<div class="sampler-math">
\[
\begin{aligned}
q_\phi(y)&=\frac{R_\tau(y)}{Z_\tau}\\
\Longleftrightarrow\quad
\log q_\phi(y)-\log R_\tau(y)&=-\log Z_\tau.
\end{aligned}
\tag{5}
\]
</div>

Every acceptable response should have the same log-gap between sampler probability and target score. Introduce one offset \\(z_\tau\\) and square the residual:

<div class="sampler-math"><span class="sampler-math-label">Trajectory balance</span>
\[
\ell_{\mathrm{TB}}(y;\phi,z_\tau)
=\left[z_\tau+\log q_\phi(y)-\log R_\tau(y)\right]^2.
\tag{6}
\]
</div>

If the residual is zero over the whole acceptable set, normalization forces \\(z_\tau=\log Z_\tau\\), recovering the target in equation (3). This is the sequence-level trajectory-balance objective: the probability of the response is the product of its token probabilities, including termination. [[6]](#sampler-ref-6)

### 4. Remove the normalizer with a group of responses
{: #group-loss}

Rather than learning a separate normalizer, generate \\(K\ge2\\) acceptable responses for the same prompt and expert trace. Let \\(a_i\\) be response \\(i\\)’s sampler-to-student log-gap. The best shared offset in the group minimizes \\(K^{-1}\sum_i(z+a_i)^2\\), so it is simply \\(z=-\bar a\\).

Substituting that offset gives our group-relative loss:

<div class="sampler-math"><span class="sampler-math-label">Learn relative probabilities</span>
\[
\begin{aligned}
a_i(\phi)&=\log q_\phi(y_i\mid x,\tau)\\
&\quad-\log p_{\mathrm{ref}}(y_i\mid x),\\
\bar a&=\frac1K\sum_{i=1}^{K}a_i,\\
\mathcal L_{\mathrm{sampler}}(\phi)
&=\frac1K\sum_{i=1}^{K}\left[a_i(\phi)-\operatorname{sg}(\bar a)\right]^2.
\end{aligned}
\tag{7}
\]
</div>

The student score is fixed. The operator \\(\operatorname{sg}\\) holds the group mean fixed during backpropagation. An above-average gap asks the sampler to lower that response’s log probability relative to the group; a below-average gap asks for the opposite. No partition-function network is needed. Group-relative matching objectives also appear in recent GFlowNet reasoning methods. [[5]](#sampler-ref-5) [[6]](#sampler-ref-6)

<div class="sampler-key"><span class="sampler-key-label">Key message</span><p><strong>Learn how probability should be distributed across useful responses.</strong> A response being correct is not enough; its sampling frequency should also agree with the target.</p></div>

<details class="sampler-technical" markdown="1">
<summary>Implementation details: acceptance, gradients, and sequence scores</summary>

A raw generator can produce unacceptable outputs. In practice, equation (7) matches its **accepted-output distribution**. If its acceptance probability is \\(A_\phi>0\\), then for acceptable responses \\(q_\phi^+(y)=q_\phi(y)/A_\phi\\). Replacing raw log probabilities by accepted-output log probabilities subtracts the same \\(\log A_\phi\\) from every gap. Group-centering cancels this term, so the residual can be computed from raw sampler scores. This does not itself remove invalid mass; the acceptance check determines which responses enter SFT.

For a fixed batch, differentiating the centered squared loss gives the same gradient whether the mean is detached or differentiated: the residuals sum to zero. The displayed algorithm treats sampled responses as fixed and takes one update per fresh group. It does not differentiate through sampling. Reusing old rollouts requires a suitable off-policy correction.

Use full sequence log probabilities, including EOS, under the same sampling distribution used to draw the responses. Length normalization, altered sampling temperature, or top-p truncation changes that distribution and must be handled explicitly. The empirical group offset is not an exact estimate of \\(\log Z_\tau\\) before convergence. Finally, matching a sampled group does not establish coverage of every acceptable reasoning path.

</details>

## One sampler, two training schedules
{: #versions}

The sampler may see the expert trace. The student sees the original prompt. This distinction is essential: the expert supplies privileged information for data preparation, not an extra input the student can rely on at evaluation time.

<figure id="sampler-lora-conversion" class="sampler-reference-figure"><a class="sampler-reference-image" href="{{ '/assets/blog/learned-sampler/lora-sampler-e2s-v3.png' | relative_url }}" target="_blank" rel="noopener" aria-label="Open full-size diagram"><picture><source media="(max-width: 760px)" srcset="{{ '/assets/blog/learned-sampler/lora-sampler-e2s-v3-mobile.png' | relative_url }}"><img src="{{ '/assets/blog/learned-sampler/lora-sampler-e2s-v3.png' | relative_url }}" width="1603" height="981" loading="lazy" alt="LoRA enabled: calibrate off-policy expert data. LoRA disabled: score responses and run SFT using the same backbone."></picture></a><figcaption><strong>One backbone, two modes.</strong> LoRA produces calibrated “on-policy” data; disabling it restores student scoring and SFT. <a class="sampler-fullsize-link" href="{{ '/assets/blog/learned-sampler/lora-sampler-e2s-v3.png' | relative_url }}" target="_blank" rel="noopener">Open full size ↗</a></figcaption></figure>

We implement the sampler with LoRA on the student backbone. For an adapted weight matrix, \\(W_{\mathrm{sampler}}=W_\theta+sBA\\), where \\(B\in\mathbb R^{d_{\mathrm{out}}\times r}\\), \\(A\in\mathbb R^{r\times d_{\mathrm{in}}}\\), and \\(s\\) is the adapter scale. The update has rank at most \\(r\\). With the adapter enabled, the model samples from expert-conditioned prompts; with it disabled, the backbone scores responses or receives ordinary SFT updates. [[7]](#sampler-ref-7)

### E2S-Offline: prepare once, then fine-tune
{: #offline}

Freeze the starting student \\(p_0\\). Fit the sampler against its constrained distribution, then use that sampler to transform the expert corpus into calibrated “on-policy” training data. Freeze the generated dataset and run SFT.

<figure id="sampler-offline-conversion" class="sampler-reference-figure"><a class="sampler-reference-image" href="{{ '/assets/blog/learned-sampler/offline-e2s-v3.png' | relative_url }}" target="_blank" rel="noopener" aria-label="Open full-size diagram"><picture><source media="(max-width: 760px)" srcset="{{ '/assets/blog/learned-sampler/offline-e2s-v3-mobile.png' | relative_url }}"><img src="{{ '/assets/blog/learned-sampler/offline-e2s-v3.png' | relative_url }}" width="1602" height="982" loading="lazy" alt="Freeze the student, fit a sampler, calibrate expert data, then train the student with SFT on the fixed dataset."></picture></a><figcaption><strong>Fit once, calibrate the data, then run SFT.</strong> The target is the frozen student’s constrained distribution; SFT uses the resulting fixed dataset. <a class="sampler-fullsize-link" href="{{ '/assets/blog/learned-sampler/offline-e2s-v3.png' | relative_url }}" target="_blank" rel="noopener">Open full size ↗</a></figcaption></figure>

<details class="sampler-technical" markdown="1">
<summary>Algorithm: offline preparation and SFT</summary>

```text
Freeze the starting student p₀.
For each sampler update:
    Choose an expert example (x, τ).
    Draw a fresh group of K ≥ 2 acceptable responses.
    Score them with the sampler and frozen student.
    Update only sampler parameters using equation (7).
Freeze the sampler and prepare the calibrated dataset D*.
Disable the sampler adapter; fine-tune the student on D*.
```

</details>

Work spent fitting the sampler can be reused across the corpus. The student stays fixed throughout preparation, so all generated targets are calibrated to the same reference checkpoint.

### E2S-Online: refresh as the student learns
{: #online}

After SFT changes the student, the closest acceptable distribution changes too. At round \\(t\\), the target becomes:

<div class="sampler-math">
\[
\begin{aligned}
p_{C,t}(y\mid x,\tau)
&=\frac{p_{\theta_t}(y\mid x)\,\mathbf 1[y\in C_\tau]}{Z_{\tau,t}},\\
Z_{\tau,t}&=\sum_{y\in C_\tau}p_{\theta_t}(y\mid x).
\end{aligned}
\tag{8}
\]
</div>

E2S-Online alternates between fitting the LoRA sampler to the current student, generating a new training batch, and updating the student with SFT. The adapter is retained between rounds and refitted against the updated backbone.

<figure id="sampler-online-conversion" class="sampler-reference-figure"><a class="sampler-reference-image" href="{{ '/assets/blog/learned-sampler/online-e2s-v3.png' | relative_url }}" target="_blank" rel="noopener" aria-label="Open full-size diagram"><picture><source media="(max-width: 760px)" srcset="{{ '/assets/blog/learned-sampler/online-e2s-v3-mobile.png' | relative_url }}"><img src="{{ '/assets/blog/learned-sampler/online-e2s-v3.png' | relative_url }}" width="1602" height="982" loading="lazy" alt="Expert data feed the sampler; calibrated on-policy data train the student. The updated student feeds back into sampler fitting."></picture></a><figcaption><strong>Calibrate as the student changes.</strong> Refit the sampler against the updated student, refresh the training data, and repeat SFT. <a class="sampler-fullsize-link" href="{{ '/assets/blog/learned-sampler/online-e2s-v3.png' | relative_url }}" target="_blank" rel="noopener">Open full size ↗</a></figcaption></figure>

<details class="sampler-technical" markdown="1">
<summary>Algorithm: online refresh and SFT</summary>

```text
For each round t:
    Freeze student θₜ; enable sampler LoRA φ.
    Fit φ against pθₜ with fresh groups and equation (7).
    Generate the next calibrated training batch Dₜ.
    Disable LoRA; unfreeze the student backbone.
    θₜ₊₁ ← SFT(θₜ, Dₜ).
Return the student with the sampler adapter disabled.
```

</details>

<div class="sampler-key sampler-key-teal"><span class="sampler-key-label">Key message</span><p><strong>Offline adapts the data to one student. Online keeps adapting it as that student changes.</strong> Both use the same distribution-matching idea and the same SFT student update.</p></div>

## Experiments: learn new tasks, retain prior skills
{: #experiments-offline}

### Setup

We use **Qwen2.5-3B** and the math setting of *Finetuning with Sampling*: MATH levels 3–5, with 8,230 training problems and 1,024 held-out MATH problems. Math avg. is the equal-weight mean across MATH, AMC, MATH500, and GSM8K. Prior avg. averages Chemistry, MMLU, and GPQA. All scores are accuracy percentages. [[1]](#sampler-ref-1)

Offline, the starting student is frozen while we fit the sampler and prepare the SFT dataset. We compare with the published base model, expert-data SFT, OPSD, GRPO, UFT, and MCMC + SFT scores in that math setting. OPSD uses expert information for self-distillation; GRPO and UFT use RL; MCMC + SFT searches for transformed targets before SFT. [[13]](#sampler-ref-13) [[14]](#sampler-ref-14) The E2S rows are our project results. This is an accuracy and retention comparison, not a matched-compute speed benchmark.

### Results

E2S-Offline improves Math avg. by **3.9% relative to MCMC + SFT**, while Prior avg. stays within **0.1 percentage point** of the starting model. Unlike fitting the unmodified expert data, the learned sampler improves new-task performance while keeping the evaluated prior-task average near its starting level.

The table below keeps both questions visible: **how much did the model learn, and how much did it retain?** The online result is included for comparison and examined next.

<div class="sampler-table-scroll">
<table id="sampler-results" class="sampler-results-table"><caption>Accuracy (%) on new math tasks and prior tasks</caption><thead><tr><th scope="col">Method</th><th scope="col">Math avg.</th><th scope="col">Prior avg.</th></tr></thead><tbody>
<tr><th scope="row">Base</th><td>31.8</td><td>42.2</td></tr>
<tr><th scope="row">Expert SFT</th><td>24.2</td><td>38.9</td></tr>
<tr><th scope="row">MCMC + SFT</th><td>53.4</td><td>42.0</td></tr>
<tr class="sampler-result-row"><th scope="row">E2S-Offline</th><td>55.5</td><td>42.1</td></tr>
<tr class="sampler-result-row"><th scope="row">E2S-Online</th><td>59.4</td><td>42.2</td></tr>
</tbody></table>
</div>

<details class="sampler-benchmark-details"><summary>See the scores behind each average</summary>
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

<div class="sampler-key sampler-key-teal"><span class="sampler-key-label">Conclusion</span><p>These results support adapting expert data to the student as a way to improve the learning–retention tradeoff, without changing the SFT objective.</p></div>

## Does refreshing the data improve learning?
{: #experiments-online}

### Setup

We use the same Qwen2.5-3B starting checkpoint, 8,230 training examples, held-out split, and seven-benchmark evaluation suite. Each online round fits the LoRA sampler against the frozen current student, generates new training targets, then updates the backbone with LoRA disabled. We compare the resulting student with both the fixed offline sampler and MCMC + SFT.

### Results

E2S-Online improves Math avg. by **7.0% relative to E2S-Offline** and **11.2% relative to MCMC + SFT**, while matching the starting model’s displayed Prior avg. The stored training trajectory below shows how Math avg. develops; the horizontal lines mark the two methods’ final scores.

<figure id="online-training-curve" class="sampler-chart"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/learned-sampler/e2s-online-progress-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/learned-sampler/e2s-online-progress.svg' | relative_url }}" width="800" height="410" loading="lazy" alt="Online Math average reaches 59.4 percent. Horizontal reference lines mark MCMC plus SFT at 53.4 and E2S-Offline at 55.5 percent."></picture><figcaption><strong>Online learning, with final-score references.</strong> Horizontal lines show MCMC + SFT (53.4%) and E2S-Offline (55.5%), not their training trajectories. Online progress is normalized within its own run and does not align compute across methods. <a href="/assets/blog/learned-sampler/source/online-curve-data.json">Stored trajectory</a>.</figcaption></figure>

### Conclusion

<div class="sampler-key sampler-key-teal"><span class="sampler-key-label">Conclusion</span><p>The online results support treating data generation as part of learning, rather than only preparing training targets once.</p></div>

<details class="sampler-technical" markdown="1">
<summary>What this comparison establishes</summary>

The results compare complete training schedules. Isolating the effect of refreshing alone requires frozen and refreshed samplers under equal total compute. Likewise, an exact offline sampler and converged MCMC target the same distribution; finite-run differences can come from approximation quality, coverage, or compute allocation. The measured outcome here is stronger new-task accuracy with prior-task averages near the starting model.

</details>

## Scaling through diversity: one expert example, many useful responses
{: #scaling}

A fixed demonstration gives SFT one trajectory to imitate. A learned distribution can offer several ways to express the same information. That creates another way to spend generation compute: **sample again from the same expert example, and teach the student a different acceptable path.**

The distinction between reward maximization and distribution matching matters here. An expected correctness reward can be maximized by concentrating on one correct path. At exact matching, E2S must instead allocate probability across all paths with mass under its constrained target. In our earlier 2:1 example, always returning the first path would be a mismatch even though every answer was correct. This is the diversity we want to preserve—not variety added for its own sake. [[3]](#sampler-ref-3) [[6]](#sampler-ref-6)

<div class="sampler-key"><span class="sampler-key-label">Scaling hypothesis</span><p><strong>The unit of supervision can be a distribution, not a single response.</strong> Once a sampler is fitted, additional draws can expose the student to more useful ways of solving the same problem.</p></div>

### Setup: scale the responses, keep the expert corpus fixed

For an offline scaling study, freeze one fitted sampler and keep all 8,230 expert examples. Draw 1, 2, or 4 accepted responses per example. The resulting SFT datasets contain **8,230, 16,460, or 32,920 responses**, respectively. No additional expert problems are introduced. Each new draw is generated separately; these are not copies of the original response.

To distinguish useful diversity from simply doing more optimization, compare against repeated copies of the 1× dataset at the same SFT token budget. Keep the student initialization and evaluation suite fixed, and track the number of distinct reasoning paths as well as accuracy.

### Expected trend: more useful draws, more learning

The curve below uses illustrative estimates to show the scaling hypothesis. **It is not a measured ablation.** Every point—including the 1× point—is an estimate, separate from the experimental scores above.

<figure id="offline-scaling-curve" class="sampler-chart"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/learned-sampler/e2s-offline-scaling-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/learned-sampler/e2s-offline-scaling.svg' | relative_url }}" width="800" height="440" loading="lazy" alt="Illustrative estimates, not measured: offline Math average of 55.0, 56.3, and 57.2 percent at 1, 2, and 4 responses per expert example, totaling 8230, 16460, and 32920 training responses."></picture><figcaption><strong>Illustrative estimates—not experimental results.</strong> A proposed offline sweep over 1 / 2 / 4 accepted responses per expert example. The 8,230-example corpus follows the math setup in <a href="#sampler-ref-1">[1]</a>. All accuracy values are hypothetical; <a href="/assets/blog/learned-sampler/source/scaling-illustration-data.json">estimate record</a>.</figcaption></figure>

If additional samples improve held-out performance beyond a compute-matched duplication control, they are contributing useful training information. Measuring distinct valid paths and their frequencies would then test whether the sampler preserves diversity. An upward accuracy curve alone cannot establish that mode collapse is absent.

### Conclusion

<div class="sampler-key sampler-key-gold"><span class="sampler-key-label">Potential</span><p>E2S can turn a fixed expert corpus into a growing source of student-calibrated supervision. The next test is whether additional draws keep adding useful reasoning paths and measurable gains.</p></div>

## Looking ahead
{: #lookahead}

Much of data preparation asks whether an example is good: is it correct, clear, and relevant? E2S adds a second question: **is this a good way to teach this particular student?** The same expert information can be expressed through different trajectories, and the most useful choice can change as the learner develops.

Offline, this makes data preparation model-specific. Online, it makes data preparation part of the training process. The student supplies a changing notion of what is natural; the expert supplies the information that must survive; the sampler learns to satisfy both.

The longer-term opportunity is to make an expert corpus reusable in a deeper sense. Instead of storing one response and replaying it, we could learn a conditional source of supervision: many acceptable paths for a given problem, calibrated to the model that will learn from them. Distribution matching gives this idea a concrete target, and amortization gives us a mechanism for generating from it repeatedly.

Our present results support the first step: stronger math learning with prior-task averages close to the starting model. The scaling question is whether a learned sampler can keep finding useful variation as we spend more compute. If it can, improvements in data generation and improvements in the student can become a productive feedback loop.

<div class="sampler-key sampler-key-final"><span class="sampler-key-label">Takeaway</span><p>Expert knowledge need not arrive in one fixed form. We can learn how to express it for the model—and keep adapting that expression as the model learns.</p></div>

## References
{: #references}


<p id="sampler-ref-1"><strong>[1]</strong> Aayush Karan, Sitan Chen, and Yilun Du. <a href="https://arxiv.org/abs/2610.02140v1">Finetuning with Sampling: SFT Learns Better Than You Think</a>. arXiv:2610.02140v1, 2026.</p>

<p id="sampler-ref-2"><strong>[2]</strong> Kevin Lu and Thinking Machines Lab. <a href="https://thinkingmachines.ai/blog/on-policy-distillation/">On-Policy Distillation</a>. Thinking Machines Lab: Connectionism, 2025.</p>

<p id="sampler-ref-3"><strong>[3]</strong> Fangxu Yu, Lai Jiang, Haoqiang Kang, Shibo Hao, and Lianhui Qin. <a href="https://arxiv.org/abs/2406.05673v6">Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples</a>. ICML, 2025.</p>

<p id="sampler-ref-4"><strong>[4]</strong> Haoqiang Kang, Enna Sachdeva, Piyush Gupta, Sangjae Bae, and Kwonjoon Lee. <a href="https://arxiv.org/abs/2503.06514">GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks</a>. CVPR, 2025.</p>

<p id="sampler-ref-5"><strong>[5]</strong> Xiaodong Liu et al. <a href="https://arxiv.org/abs/2607.13394v1">GFlowRL: Scaling Distribution-Matching RL to Large Language Models</a>. arXiv:2607.13394v1, 2026.</p>

<p id="sampler-ref-6"><strong>[6]</strong> Xuekai Zhu et al. <a href="https://arxiv.org/abs/2509.15207v3">FlowRL: Matching Reward Distributions for LLM Reasoning</a>. arXiv:2509.15207v3, 2025.</p>

<p id="sampler-ref-7"><strong>[7]</strong> Edward J. Hu et al. <a href="https://arxiv.org/abs/2106.09685">LoRA: Low-Rank Adaptation of Large Language Models</a>. ICLR, 2022.</p>

<p id="sampler-ref-9"><strong>[9]</strong> Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. <a href="https://arxiv.org/abs/2509.04259v1">RL’s Razor: Why Online Reinforcement Learning Forgets Less</a>. arXiv:2509.04259v1, 2025. See §4–5 and Appendix A for the KL analysis and its assumptions.</p>

<p id="sampler-ref-10"><strong>[10]</strong> Howard Chen, Noam Razin, Karthik Narasimhan, and Danqi Chen. <a href="https://proceedings.mlr.press/v306/chen26do.html">Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting</a>. ICML, 2026. See §3–4 for distributional analysis and approximately on-policy SFT; Appendix A.5 discusses limits of KL as a predictor.</p>

<p id="sampler-ref-11"><strong>[11]</strong> Zhihong Shao et al. <a href="https://arxiv.org/abs/2402.03300">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>. arXiv:2402.03300, 2024. See §4.1 for GRPO.</p>

<p id="sampler-ref-12"><strong>[12]</strong> Yoshua Bengio. <a href="https://yoshuabengio.org/en/blog/generative-flow-networks">Generative Flow Networks</a>. 2022. Discusses learning sequential construction policies and contrasts them with MCMC sampling.</p>

<p id="sampler-ref-13"><strong>[13]</strong> Siyan Zhao et al. <a href="https://arxiv.org/abs/2601.18734">Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models</a>. arXiv:2601.18734, 2026.</p>

<p id="sampler-ref-14"><strong>[14]</strong> Mingyang Liu, Gabriele Farina, and Asuman Ozdaglar. <a href="https://arxiv.org/abs/2505.16984">UFT: Unifying Supervised and Reinforcement Fine-Tuning</a>. arXiv:2505.16984, 2025.</p>

## Citation
{: #citation}

If you found this post useful, please cite it as:

Murray Kang. “E2S Finetuning: From Off-Policy Expert Data to On-Policy Student Data.” October 2026.

{% raw %}
```bibtex
@misc{kang2026onpolicysft,
  author = {Kang, Murray},
  title = {{E2S Finetuning: From Off-Policy Expert Data to On-Policy Student Data}},
  year = {2026},
  month = oct,
  howpublished = {Research blog},
  url = {https://mk322.github.io/blog/learned-on-policy-sampler/}
}
```
{% endraw %}
