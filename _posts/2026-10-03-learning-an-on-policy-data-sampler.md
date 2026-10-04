---
title: "From Off-Policy Data to On-Policy SFT"
subtitle: "Learning an amortized sampler for the information-constrained target distribution."
permalink: /blog/learned-on-policy-sampler/
last_modified_at: 2026-10-04
excerpt: "Can we keep the efficiency of SFT while reducing forgetting? We learn a sampler that brings expert data closer to the student’s policy, shifting repeated data-generation search into sampler training."
tldr: |
  - **Problem:** SFT is efficient, but fitting off-policy demonstrations can move a model far from its starting behavior and contribute to forgetting. Can we bring the data closer to the student while preserving the information it needs to learn?
  - **Our solution:** Train an expert-conditioned sampler to match the information-constrained student distribution. A group-relative GFlowNet loss removes the learned normalizer; sampler training absorbs work otherwise repeated during data generation.
  - **Two uses:** Offline, adapt an expert corpus for a chosen student. Online, refresh a LoRA sampler as that student learns, producing new targets for ordinary SFT.
  - **The real test:** Better accuracy at matched total compute, while retaining prior capabilities. A learned sampler addresses repeated search and stale data; verifier errors, incomplete coverage, and forgetting still need separate evaluation.
  - **Draft estimates, not measurements:** Offline about +2 percentage points and online +3–5 points over MCMC + SFT; no loss of prior-task accuracy is the retention target. These placeholders await experiments.

---

<link rel="stylesheet" href="{{ '/assets/blog/learned-sampler/article.css' | relative_url }}">

<nav class="sampler-toc" aria-label="Article contents"><details open><summary>On this page</summary><ol><li><a href="#sft-problem">SFT’s off-policy mismatch</a></li><li><a href="#target">What is the mismatch?</a></li><li><a href="#amortization">Amortize the search</a></li><li><a href="#train-sampler">From KL to the training loss</a></li><li><a href="#offline">Offline: sampler, then SFT</a></li><li><a href="#online">Online: the LoRA sampler</a></li><li><a href="#experiments-offline">Offline experiments</a></li><li><a href="#experiments-online">Online experiments</a></li><li><a href="#use-cases">Where this is useful</a></li></ol></details></nav>

<script defer src="{{ "/assets/blog/learned-sampler/navigation.js" | relative_url }}"></script>

<p class="sampler-status">Project note · Method implemented · Our numerical results below are draft estimates</p>

## SFT’s off-policy mismatch
{: #sft-problem}

Supervised fine-tuning (SFT) is attractive because the training loop is simple and fast. Given a dataset of responses, we train the model to predict their tokens. Those tokens are already available, so their losses can be computed in parallel. Each update needs no fresh autoregressive rollouts, reward-based advantage estimates, or teacher scoring of newly generated responses—the extra work involved in on-policy RL and distillation. [[2]](#sampler-ref-2) [[11]](#sampler-ref-11)

But learning a new task can damage abilities the model already had. A model may improve at math while becoming worse at following instructions or answering general questions. This is **catastrophic forgetting**. Recent comparisons find that SFT often forgets more than on-policy RL, even at similar performance on the new task. [[9]](#sampler-ref-9) [[10]](#sampler-ref-10)

One reason is the distribution SFT asks the model to learn. Its responses usually come from a human or another model, rather than the student's current policy. That makes the data **off-policy**. Even when every answer is correct, the demonstrations may use reasoning paths, wording, and intermediate steps that the student rarely generates. SFT asks it to reproduce that whole distribution.

Let \\(q_{\mathrm{data}}(y\mid x)\\) denote the demonstration distribution and \\(p_\theta(y\mid x)\\) the student. For a fixed prompt \\(x\\), ordinary SFT minimizes:

<div class="sampler-math">
\[
\begin{aligned}
\mathcal L_{\mathrm{SFT}}(\theta)
&=\mathbb E_{y\sim q_{\mathrm{data}}}[-\log p_\theta(y\mid x)]\\
&=H(q_{\mathrm{data}})+D_{\mathrm{KL}}(q_{\mathrm{data}}\|p_\theta).
\end{aligned}
\tag{1}
\]
</div>

The entropy is constant, so the objective pulls the student toward the demonstrations. There is no term here that keeps it near its starting policy \\(p_0\\). If the model can represent the data distribution and fits the population loss exactly, its fitted distribution is \\(p_{\mathrm{fit}}=q_{\mathrm{data}}\\). Its distance from the starting model is therefore:

<div class="sampler-math">
\[
D_{\mathrm{KL}}(p_{\mathrm{fit}}\|p_0)
=D_{\mathrm{KL}}(q_{\mathrm{data}}\|p_0).
\notag
\]
</div>

**Fitting distant data means moving toward a distant policy.** Some change is needed to learn the task; matching the expert's particular way of solving it can demand more. Because the same parameters support many skills, that movement can disrupt prior behavior. *RL’s Razor* connects larger distribution shifts with greater forgetting and analyzes a minimum-KL bias of on-policy learning in an idealized policy class. *Retaining by Doing* provides complementary evidence: refreshing SFT data from the evolving student reduces forgetting in its experiments. The connection is useful, but training-task KL alone does not guarantee retention on other tasks. [[9]](#sampler-ref-9) [[10]](#sampler-ref-10)

On-policy methods address the mismatch by changing where training trajectories come from:

- **RL, such as GRPO,** samples responses from a recent student policy, scores them, and updates the student using their relative rewards. The feedback is attached to behavior the student actually produces. [[11]](#sampler-ref-11)
- **On-policy distillation (OPD)** also samples from the student, then asks a teacher to score its tokens. This directly addresses the mismatch between teacher-written trajectories and student-visited contexts; Thinking Machines also shows how it can recover instruction-following behavior after further training. [[2]](#sampler-ref-2)

Both approaches keep generation and feedback inside the training loop. We ask whether we can keep ordinary SFT as the student update and handle the mismatch in the data instead: **can we transform off-policy demonstrations into a distribution closer to the student, preserve what they teach, and reduce forgetting?**

Finetuning with Sampling pursues this route with MCMC. [[1]](#sampler-ref-1) Our project learns an amortized sampler for the same distribution-matching objective, replacing repeated per-example search with a reusable generation policy. The next step is to define precisely what “closer to the student” should mean.

## What is the mismatch we want to remove?
{: #target}

Equation (1) tells us where SFT will try to move the student. We can choose that destination: replace the original demonstration distribution with one that preserves the required information while staying as close as possible to the student. This turns the motivation above into a data-distribution optimization problem.

For one prompt \\(x\\) and expert response \\(\tau\\), let \\(C_\tau\\) be the set of acceptable responses. In math, this can mean responses with the correct final answer; a stronger verifier can also check the reasoning. For the next few equations, write \\(p(y)=p_\theta(y\mid x)\\) for the fixed student and \\(q(y)\\) for the data distribution we are choosing. Our objective is:

<div class="sampler-math">
\[
\begin{aligned}
\min_q D_{\mathrm{KL}}(q\|p)
&=\min_q\sum_{y\in C_\tau}q(y)\log\frac{q(y)}{p(y)},\\
&\text{subject to }\operatorname{supp}(q)\subseteq C_\tau.
\end{aligned}
\tag{2}
\]
</div>

The constraint says that the training data must preserve the expert information. The KL says that, within this constraint, its distribution should stay close to the student. Unlike the SFT objective in equation (1), we now hold the student fixed and change the data distribution. This is the information-projection objective used to formalize student-compatible data: choose the closest distribution among those satisfying the task constraint. [[1]](#sampler-ref-1) [[9]](#sampler-ref-9)

A natural candidate is to keep the student's probabilities for acceptable responses and renormalize them. Let \\(Z\\) be their total probability:

<div class="sampler-math">
\[
Z=\sum_{y\in C_\tau}p(y)>0,
\qquad
p_C(y)=\frac{p(y)\,\mathbf 1[y\in C_\tau]}{Z}.
\tag{3}
\]
</div>

Why is this the closest distribution? On the valid set, \\(p(y)=Zp_C(y)\\). Substituting this into equation (2), for any admissible \\(q\\), gives:

<div class="sampler-math">
\[
\begin{aligned}
D_{\mathrm{KL}}(q\|p)
&=\sum_{y\in C_\tau}q(y)
  \left[\log\frac{q(y)}{p_C(y)}-\log Z\right]\\
&=D_{\mathrm{KL}}(q\|p_C)-\log Z.
\end{aligned}
\tag{4}
\]
</div>

The second term is constant. The first is nonnegative and becomes zero when \\(q=p_C\\), which proves that equation (3) minimizes our objective.

If the original demonstrations already satisfy the same constraint, they are one feasible choice of \\(q\\). The optimum therefore obeys \\(D_{\mathrm{KL}}(p_C\|p)\leq D_{\mathrm{KL}}(q_{\mathrm{data}}\|p)\\). This is the precise improvement we seek in the data distribution. Whether SFT on those data also retains more prior capability is what our experiments must establish.

We now know exactly what “more on-policy” means here: **generate from the student's distribution conditioned on preserving the expert information**. We discard unacceptable responses and retain the student's relative preferences among the acceptable ones. If those responses are rare, even this best possible match can be far from the original student; the constraint still has to be satisfied.

## Amortize the search into sampler training
{: #amortization}

Knowing the target distribution does not make it easy to sample. We could draw from the student and reject invalid responses, but that costs \\(1/Z\\) attempts per accepted response on average. This becomes impractical precisely when expert information is most useful: when the student rarely solves the problem on its own.

MCMC uses the expert trace to guide this search. For a given prompt, it proposes changes to the current response, scores them, and accepts or rejects each proposal. **The response changes; the model weights stay fixed during this search.** A new prompt requires another chain. [[1]](#sampler-ref-1)

**Amortized inference moves much of this repeated search cost into training an inference machine.** The reusable result is a set of sampler parameters: training on one batch can change how the sampler generates responses for later prompts. This is the training-for-inference tradeoff described by Bengio: invest computation in learning a reusable inference procedure, then use it to generate samples. [[8]](#sampler-ref-8)

In our case, the inference machine is a conditional sampler \\(q_\phi(y\mid x,\tau)\\). During fitting, it explores responses, receives student likelihoods and validity feedback, and learns to approximate \\(p_C\\) across expert examples. We train it directly from these scores, without MCMC-generated teaching examples.

At data-generation time, MCMC repeatedly revises a candidate response. The trained sampler instead starts with an empty response and appends tokens, using transition probabilities learned across examples. This constructive policy is the reusable object in a GFlowNet. [[12]](#sampler-ref-12) It still requires autoregressive decoding and verification. Both routes then use their verified outputs for SFT. Here, sampling-time inference refers to creating training data.

<figure><picture><source media="(max-width: 760px)" srcset="{{ '/assets/blog/learned-sampler/amortized-comparison-mobile.svg' | relative_url }}?v=3"><img src="{{ '/assets/blog/learned-sampler/amortized-comparison.svg' | relative_url }}?v=3" width="800" height="584" alt="MCMC repeats a chain of response revisions for each prompt and expert example. Amortized training learns policy weights across examples, then reuses those weights to construct new responses by appending tokens. Both routes produce verified data and train the student with SFT."></picture><figcaption><strong>Search per example versus a reusable generation policy.</strong> Fitting across examples turns scoring feedback into sampler weights. Those weights then construct responses token by token (∅ is the empty prefix). Both routes verify outputs, collect data, and update the student with SFT. This is the offline schedule; sampler fitting and student SFT are separate stages.</figcaption></figure>

The intended benefits follow from this reuse: lower cost per generated example after training, shared learning across related prompts, and a sampler that can be refreshed as the student changes. The training investment only pays off if enough useful data is generated. Total cost must therefore include sampler training, student scoring, verification, and rejected samples.

One separation is essential. The sampler sees the expert response to help find acceptable outputs. The student scores each candidate using **the prompt alone**. Thus expert information defines what to preserve, while the student defines the density we want to learn.

## From the KL target to a trainable loss
{: #train-sampler}

We can now derive the sampler's training rule. Equation (4) says that minimizing the constrained KL is equivalent to matching \\(p_C\\). Define its unnormalized density as \\(R(y)=p(y)\mathbf 1[y\in C_\tau]\\). Our target is therefore \\(q_\phi(y)=R(y)/Z\\).

The problem has become distribution matching: we can score any candidate with \\(R(y)\\), but cannot enumerate all responses to compute \\(Z\\). The following steps turn that target into a loss we can evaluate on sampled responses.

### Match probabilities up to one common scale

Assume the sampler can represent the target and assigns positive probability to its valid responses. For a valid response, rearrange the matching condition and take logs:

<div class="sampler-math">
\[
\begin{aligned}
q_\phi(y)&=\frac{R(y)}{Z}\\
\Longleftrightarrow\quad Zq_\phi(y)&=R(y)\\
\Longleftrightarrow\quad \log q_\phi(y)-\log R(y)&=-\log Z.
\end{aligned}
\tag{5}
\]
</div>

The unknown value on the right is the same for every valid response to this expert example. So the sampler is correct when **every response has the same sampler-to-target log gap**.

This is the trajectory-balance condition for an autoregressive GFlowNet. The connection does not require a different generator: the sampler still appends one token at a time, and its complete-response probability is the product of those token probabilities. Each response has one path through its prefixes. Matching its probability to \\(R(y)/Z\\) therefore matches the probability of the full generation trajectory. Our earlier work applies this flow-based view to language and visual reasoning. [[3]](#sampler-ref-3) [[4]](#sampler-ref-4)

Introduce a scalar \\(z\\) to represent the unknown \\(\log Z\\), and square the error in equation (5):

<div class="sampler-math">
\[
\ell_{\mathrm{TB}}(y;\phi,z)
=\left[z+\log q_\phi(y)-\log R(y)\right]^2.
\tag{6}
\]
</div>

This is the GFlowNet **trajectory-balance loss**. It is computable from the sampler's token log probabilities, the student's score, and one common offset. Both too much and too little probability create an error. If the error vanishes across the valid set, \\(q_\phi(y)=e^{-z}R(y)\\); requiring the probabilities to sum to one forces \\(e^z=Z\\). We recover \\(q_\phi=p_C\\), the minimizer of our original KL.

We have replaced an intractable normalized target with a sample-level training error. The two objectives have the same ideal solution; reaching it still requires exploring the valid responses, not merely fitting a few observed ones. [[6]](#sampler-ref-6)

### Eliminate the offset with a group of responses

Rather than learning \\(z\\) with another model, we can solve for it within each rollout group. Generate \\(K\ge2\\) valid candidates for the same prompt and expert demonstration. Write their log gaps as \\(a_i(\phi)=\log q_\phi(y_i)-\log R(y_i)\\), and let \\(\bar a=K^{-1}\sum_i a_i(\phi)\\) be the group mean before the update.

Averaging equation (6) gives \\(K^{-1}\sum_i(z+a_i)^2\\). Its derivative with respect to \\(z\\) is \\(2(z+\bar a)\\), which is zero at \\(z=-\bar a\\). In other words, the best common offset centers the group's log gaps. Substitute this offset into equation (6), average over responses, and we obtain:

<div class="sampler-math">
\[
\begin{aligned}
a_i(\phi)&=\log q_\phi(y_i)-\log R(y_i),\\
\mathcal L_{\mathrm{sampler}}(\phi)
&=\frac1K\sum_{i=1}^{K}
\left[a_i(\phi)-\operatorname{sg}(\bar a)\right]^2.
\end{aligned}
\tag{7}
\]
</div>

This is the **group-relative matching loss**: make the sampler-to-target log gap agree across responses. The operator \\(\operatorname{sg}\\) means that the measured group mean is held fixed during backpropagation. An above-average gap means a response is overrepresented relative to the group, so the loss calls for lowering its log probability. A below-average gap calls for the opposite adjustment. We learn the relative probabilities without learning the normalizer. [[3]](#sampler-ref-3) [[4]](#sampler-ref-4) [[5]](#sampler-ref-5)

Each expert example gets its own group mean because it has its own \\(Z\\). This mean is a batch offset, not an exact log-normalizer estimate before convergence. Equation (7) uses one update per fresh rollout group. Reusing samples for multiple updates additionally requires accounting for the changed sampling policy, for example with importance weighting. [[6]](#sampler-ref-6)

For a valid candidate, \\(\log R(y_i)=\log p_\theta(y_i\mid x)\\). Training therefore needs only sampler and student log probabilities: compute their difference, subtract the group mean, and minimize the squared residual. Use complete sequence log probabilities, including termination; length-normalized scores would define a different matching problem.

The derivation assumes a distribution supported on acceptable responses. In practice, the generator can produce invalid outputs, so we match its **accepted-output distribution**. Conditioning on acceptance subtracts the same log acceptance probability from every valid output's log probability. That constant cancels in the group-centered residual, making equation (7) computable with the raw sampler probabilities.

This cancellation does not train away invalid mass. Verification determines what enters SFT, while acceptance rate and coverage must be evaluated separately. The sampling policy must also agree with the probabilities in the loss; changing temperature or truncating the rollout distribution requires corresponding correction.

We have arrived at a trainable sampler without changing the student's SFT objective. Next, we can either fit that sampler once before SFT, or keep fitting it as the student evolves.

## Offline: train the sampler, then run SFT
{: #offline}

The offline recipe has three stages. **The student does not change while we train the sampler.**

<figure><picture><source media="(max-width: 760px)" srcset="{{ '/assets/blog/learned-sampler/offline-stages-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/learned-sampler/offline-stages.svg' | relative_url }}" width="800" height="270" alt="Three sequential stages: train the sampler against a frozen student, freeze the sampler and create data, then update the student with SFT."></picture><figcaption>Offline training runs left to right once. There is no student-to-sampler feedback after SFT starts.</figcaption></figure>

<div class="sampler-algorithm" markdown="1">
<p class="sampler-algorithm-label">Offline · fit once, then SFT</p>

```text
fit(sampler, fixed student)  # Eq. 7
data ← verified_samples(sampler)
SFT(student, data)
```

</div>

`fit` updates only the sampler using fresh rollout groups and equation (7). Both `fit` and `verified_samples` condition generation on the prompt and expert demonstration; the latter retains only verified responses and holds the fitted sampler fixed. SFT updates only the student, using prompts and sampled responses; the expert demonstration is not an extra student input.

This version replaces per-example MCMC data creation with a trained, reusable generator. It pays an up-front sampler-training cost, then shares that work across examples. Whether the reuse saves compute is an experimental question: training, student scoring, verification, and failed generations all belong in the cost.

The sampler continues to target the **starting checkpoint**, even after SFT changes the student. That is a deliberate property of this offline recipe.

## Online: a LoRA sampler on the evolving student
{: #online}

The offline sampler targets the starting checkpoint. Online training instead refreshes the data distribution after the student changes. We implement the sampler as a **LoRA adapter on the current student backbone**, rather than maintaining a separate full-size sampler model. [[7]](#sampler-ref-7)

For an adapted weight matrix, the sampler uses a low-rank update:

<div class="sampler-math">
\[
W_{\mathrm{sampler},t}=W_{\theta_t}+B_{\phi_t}A_{\phi_t},
\qquad \operatorname{rank}(B_{\phi_t}A_{\phi_t})\le r.
\tag{8}
\]
</div>

The same backbone has two roles. **Adapter enabled:** generate with the prompt and expert demonstration. **Adapter disabled:** score candidates under the student using only the prompt, or update the student with ordinary SFT. We keep the adapter separate from the final task model.

<figure><picture><source media="(max-width: 760px)" srcset="{{ '/assets/blog/learned-sampler/lora-sampler-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/learned-sampler/lora-sampler.svg' | relative_url }}" width="800" height="320" alt="One shared student backbone, two modes: enable the sampler LoRA and supply the expert trace to generate data; disable it to score prompt-only responses and perform SFT on the student."></picture><figcaption>The sampler is an adapter, not a second full model. Expert conditioning and adapter weights belong to data creation; the student scores and learns with the adapter disabled.</figcaption></figure>

This design has three practical advantages. The sampler update trains a small set of parameters, reducing its additional parameter and optimizer-state storage. It starts from the student's existing language and reasoning capabilities instead of learning a generator from scratch. And as SFT improves the shared backbone, that improvement is immediately available to the sampler. These are architectural reasons for LoRA; whether they reduce end-to-end compute or improve accuracy is measured separately.

Sharing the backbone also creates a dependency. Even with fixed adapter weights, updating \\(\theta_t\\) changes the sampler's distribution. We therefore refit the adapter against the current constrained target:

<div class="sampler-math">
\[
p_{C,t}(y\mid x,\tau)
=\frac{p_{\theta_t}(y\mid x)\,\mathbf 1[y\in C_\tau]}{Z_t(x,\tau)}.
\tag{9}
\]
</div>

<figure><picture><source media="(max-width: 760px)" srcset="{{ '/assets/blog/learned-sampler/online-cycle-mobile.svg' | relative_url }}"><img src="{{ '/assets/blog/learned-sampler/online-cycle.svg' | relative_url }}" width="800" height="330" alt="Freeze the current backbone and fit the sampler LoRA, generate verified responses, disable LoRA and update the backbone with SFT, then refresh the adapter against the updated student."></picture><figcaption>Only the online schedule feeds the updated student back into sampler fitting. Adapter parameters and backbone parameters are updated in separate phases.</figcaption></figure>

<div class="sampler-algorithm" markdown="1">
<p class="sampler-algorithm-label">Online · refresh as the student learns</p>

```text
sampler ← LoRA(student)
repeat:
    fit(sampler, fixed student)  # Eq. 7
    data ← verified_samples(sampler)
    SFT(student, data)           # LoRA off
```

</div>

`fit` updates only the adapter, scoring candidates with LoRA disabled. The SFT phase disables the adapter and updates the backbone. Each new round fits the adapter against that updated backbone.

The expert demonstrations remain fixed; the generated SFT targets can evolve. Refreshing the adapter helps address stale data, but the refresh frequency has a cost. Small adapter updates also have limited capacity. Both the update schedule and LoRA rank therefore belong in the online ablation, rather than being treated as automatic improvements.

This differs from on-policy distillation, which samples student trajectories and uses teacher probabilities as feedback. [[2]](#sampler-ref-2) Our expert-conditioned adapter generates data for the student's constrained distribution; the student receives ordinary SFT.

## Offline experiments
{: #experiments-offline}

### Setup

The offline comparison follows the math setting of Finetuning with Sampling: **Qwen2.5-3B**, MATH levels 3–5, with 8,230 training problems and 1,024 test problems. We evaluate single-shot accuracy on MATH, AMC, MATH500, and GSM8K, plus Chemistry, MMLU, and GPQA for prior-capability retention. [[1]](#sampler-ref-1)

The source SFT search uses 1–2 epochs, learning rates {5e−5, 1e−5, 5e−6}, and batch sizes {16, 32, 64}, with AdamW and a cosine schedule. Its MCMC baseline uses 10 transitions, block size 32, and maximum sequence length 1,856. [[1]](#sampler-ref-1) Our offline sampler is fitted to the starting student, then frozen to create a dataset before SFT begins.

For these placeholders, each percentage comes from an illustrative integer correct count in one evaluation pass, then rounds to one decimal. The sizes are MATH 1,024; AMC 83; MATH500 500; GSM8K 1,320; Chemistry 600, following the paper and its [evaluation files](https://github.com/aakaran/finetuning-with-sampling). For example, **25/83 rounds to 30.1%** on AMC; MATH500 moves in **0.2-point** increments.

MMLU uses the [full 14,042-item test set](https://huggingface.co/datasets/cais/mmlu/viewer/all/test) and a [micro-average](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_mmlu.yaml). GPQA temporarily assumes the 198-item [Diamond subset](https://arxiv.org/abs/2311.12022); the baseline's variant needs confirmation before a measured comparison. The prior-task average is the unweighted mean of the three task scores.

### Results: task accuracy and retention

<p class="sampler-results-note"><strong>Draft estimates, not experimental measurements.</strong> Baseline scores are reported in Table 1 of <a href="https://arxiv.org/html/2610.02140v1#S5">Finetuning with Sampling</a>, converted to percentages. Rows marked <strong>Est. †</strong> are unmeasured planning values: gains vary around +2 points offline, with slight task-level variation in retention. Published baselines retain their original rounding.</p>

<!-- sampler-offline-tables:start -->
<div class="sampler-table-card" id="offline-accuracy">
<div class="sampler-table-heading"><span class="sampler-table-kicker">GENERALIZATION · ACCURACY (%)</span><h4 id="offline-accuracy-title">Learning the new task</h4><p>Reported baselines; offline estimates vary around a +2-point gain.</p></div>
<div class="sampler-table-scroll" role="region" tabindex="0" aria-labelledby="offline-accuracy-title">
<table class="sampler-results-table">
<thead><tr><th scope="col">Method</th><th scope="col">MATH</th><th scope="col">AMC</th><th scope="col">MATH500</th><th scope="col">GSM8K</th><th scope="col">Δ MATH</th></tr></thead><tbody>
<tr class=""><th scope="row">Base model</th>
<td>31.5</td>
<td>13.3</td>
<td>24.5</td>
<td>57.9</td>
<td class="sampler-delta sampler-down">−18.0</td></tr>
<tr class=""><th scope="row">Expert-data SFT</th>
<td>24.3</td>
<td>10.0</td>
<td>16.8</td>
<td>45.5</td>
<td class="sampler-delta sampler-down">−25.2</td></tr>
<tr class=""><th scope="row">OPSD</th>
<td>26.7</td>
<td>8.4</td>
<td>33.2</td>
<td>52.4</td>
<td class="sampler-delta sampler-down">−22.8</td></tr>
<tr class=""><th scope="row">GRPO</th>
<td>45.7</td>
<td>24.9</td>
<td>31.3</td>
<td>80.8</td>
<td class="sampler-delta sampler-down">−3.8</td></tr>
<tr class=""><th scope="row">UFT</th>
<td>47.0</td>
<td>29.3</td>
<td>29.7</td>
<td>74.6</td>
<td class="sampler-delta sampler-down">−2.5</td></tr>
<tr class="sampler-reference-row"><th scope="row">MCMC + SFT</th>
<td>49.5</td>
<td>27.7</td>
<td>58.2</td>
<td>78.2</td>
<td class="sampler-delta ">+0.0</td></tr>
<tr class="sampler-estimate-row"><th scope="row">Our offline sampler <span class="sampler-estimate-badge">Est. †</span></th>
<td><span title="Illustrative count: 528 / 1,024; not measured">51.6<sup>†</sup></span></td>
<td><span title="Illustrative count: 25 / 83; not measured">30.1<sup>†</sup></span></td>
<td><span title="Illustrative count: 299 / 500; not measured">59.8<sup>†</sup></span></td>
<td><span title="Illustrative count: 1,058 / 1,320; not measured">80.2<sup>†</sup></span></td>
<td class="sampler-delta ">+2.1<sup>†</sup></td></tr>
</tbody></table></div>
<p class="sampler-table-footnote">Δ MATH is the percentage-point change from MCMC + SFT. <strong>† Draft estimates; not measured.</strong></p></div>

<div class="sampler-table-card" id="offline-retention">
<div class="sampler-table-heading"><span class="sampler-table-kicker">RETENTION · ACCURACY (%)</span><h4 id="offline-retention-title">Keep prior capabilities visible</h4><p>Per-task changes reveal losses that an average can hide.</p></div>
<div class="sampler-table-scroll" role="region" tabindex="0" aria-labelledby="offline-retention-title">
<table class="sampler-results-table">
<thead><tr><th scope="col">Method</th><th scope="col">Chemistry</th><th scope="col">MMLU</th><th scope="col">GPQA</th><th scope="col">Prior avg.</th><th scope="col">Δ vs. base</th></tr></thead><tbody>
<tr class=""><th scope="row">Base model</th>
<td>28.3<span class="sampler-cell-delta ">+0.0 pp</span></td>
<td>65.1<span class="sampler-cell-delta ">+0.0 pp</span></td>
<td>33.3<span class="sampler-cell-delta ">+0.0 pp</span></td>
<td>42.2</td>
<td class="sampler-delta ">+0.0</td></tr>
<tr class=""><th scope="row">Expert-data SFT</th>
<td>22.2<span class="sampler-cell-delta sampler-down">−6.1 pp</span></td>
<td>64.8<span class="sampler-cell-delta sampler-down">−0.3 pp</span></td>
<td>29.8<span class="sampler-cell-delta sampler-down">−3.5 pp</span></td>
<td>38.9</td>
<td class="sampler-delta sampler-down">−3.3</td></tr>
<tr class=""><th scope="row">OPSD</th>
<td>24.2<span class="sampler-cell-delta sampler-down">−4.1 pp</span></td>
<td>65.2<span class="sampler-cell-delta ">+0.1 pp</span></td>
<td>31.3<span class="sampler-cell-delta sampler-down">−2.0 pp</span></td>
<td>40.4</td>
<td class="sampler-delta sampler-down">−1.8</td></tr>
<tr class=""><th scope="row">GRPO</th>
<td>27.8<span class="sampler-cell-delta sampler-down">−0.5 pp</span></td>
<td>65.2<span class="sampler-cell-delta ">+0.1 pp</span></td>
<td>31.3<span class="sampler-cell-delta sampler-down">−2.0 pp</span></td>
<td>41.4</td>
<td class="sampler-delta sampler-down">−0.8</td></tr>
<tr class=""><th scope="row">UFT</th>
<td>28.3<span class="sampler-cell-delta ">+0.0 pp</span></td>
<td>65.3<span class="sampler-cell-delta ">+0.2 pp</span></td>
<td>32.8<span class="sampler-cell-delta sampler-down">−0.5 pp</span></td>
<td>42.1</td>
<td class="sampler-delta sampler-down">−0.1</td></tr>
<tr class="sampler-reference-row"><th scope="row">MCMC + SFT</th>
<td>26.6<span class="sampler-cell-delta sampler-down">−1.7 pp</span></td>
<td>65.1<span class="sampler-cell-delta ">+0.0 pp</span></td>
<td>34.3<span class="sampler-cell-delta ">+1.0 pp</span></td>
<td>42.0</td>
<td class="sampler-delta sampler-down">−0.2</td></tr>
<tr class="sampler-estimate-row"><th scope="row">Our offline sampler <span class="sampler-estimate-badge">Est. †</span></th>
<td><span title="Illustrative count: 170 / 600; not measured">28.3<sup>†</sup></span><span class="sampler-cell-delta ">+0.0 pp</span></td>
<td><span title="Illustrative count: 9,128 / 14,042; not measured">65.0<sup>†</sup></span><span class="sampler-cell-delta sampler-down">−0.1 pp</span></td>
<td><span title="Illustrative count: 67 / 198; not measured">33.8<sup>†</sup></span><span class="sampler-cell-delta ">+0.5 pp</span></td>
<td>42.4<sup>†</sup></td>
<td class="sampler-delta ">+0.2<sup>†</sup></td></tr>
</tbody></table></div>
<p class="sampler-table-footnote">Small numbers show each task’s change from the base model. Δ uses displayed rounded scores. <strong>† Retention estimates; not measured.</strong></p></div>
<!-- sampler-offline-tables:end -->

The offline placeholders give **51.6% MATH**, versus 49.5% for MCMC + SFT. Gains across the four math tasks range from 1.6 to 2.4 points. The prior-task average is **42.4%**, but MMLU still slips by 0.1 point in this scenario. Learning the new task and retaining prior skills must be checked separately.

**Forgetting needs a per-task check.** The reported MCMC baseline is only 0.2 points below the base model on the prior-task average, yet Chemistry drops from 28.3% to 26.6%. An average can hide that loss. We therefore report all three prior tasks and their change from the base checkpoint. [[1]](#sampler-ref-1) A supported retention claim requires repeated runs and per-task confidence intervals with a prespecified tolerance for degradation; the estimated row is a target, not evidence of no forgetting.

For efficiency, compare equal-size verified datasets and include sampler fitting, generation, student scoring, verification, rejected candidates, and SFT in total compute. A one-pass rewrite and rejection-sampling SFT are useful controls: they test whether learning a distribution buys more than cheaper data generation alone.

### Conclusion

The offline hypothesis is concrete: **a reusable data-preparation model should preserve MCMC's learning benefit while reducing the cost of producing enough training data.** The draft accuracy target is roughly +2 points while keeping each prior capability close to its starting level. It succeeds as an efficiency method only if those gains survive a comparison at matched total compute. For a small dataset, sampler training may cost more than the search it replaces.

## Online experiments
{: #experiments-online}

### Setup

Use the same starting checkpoint, expert split, and evaluation suite. The sampler is a LoRA adapter on the evolving student. Compare three schedules at matched total compute: a sampler fitted to the initial checkpoint, an adapter left frozen while the backbone changes, and an adapter refreshed between SFT updates. The second control matters because a frozen adapter still changes its outputs when its backbone changes.

Report LoRA rank, adapted modules, rollout group size, refresh interval, and both learning rates with the runs. Include **MCMC sampling + RL** as a stronger post-training comparator, in addition to MCMC + SFT. Its published scores are available in the same math setting. [[1]](#sampler-ref-1)

### Results: projected online improvement

<p class="sampler-results-note"><strong>Draft estimates.</strong> Gains vary from 3.5 to 4.8 points over MCMC + SFT, rather than adding one constant to every task. Retention values include small gains and losses near the base checkpoint. All † entries remain unmeasured.</p>

<!-- sampler-online-tables:start -->
<div class="sampler-table-card" id="online-accuracy">
<div class="sampler-table-heading"><span class="sampler-table-kicker">GENERALIZATION · ACCURACY (%)</span><h4 id="online-accuracy-title">Does refreshing help?</h4><p>Online estimates vary by task (+3–5 points); include the stronger RL pipeline.</p></div>
<div class="sampler-table-scroll" role="region" tabindex="0" aria-labelledby="online-accuracy-title">
<table class="sampler-results-table">
<thead><tr><th scope="col">Method</th><th scope="col">MATH</th><th scope="col">AMC</th><th scope="col">MATH500</th><th scope="col">GSM8K</th><th scope="col">Δ MATH</th></tr></thead><tbody>
<tr class="sampler-reference-row"><th scope="row">MCMC + SFT</th>
<td>49.5</td>
<td>27.7</td>
<td>58.2</td>
<td>78.2</td>
<td class="sampler-delta ">+0.0</td></tr>
<tr class="sampler-reference-row"><th scope="row">MCMC + SFT + RL</th>
<td>54.5</td>
<td>24.1</td>
<td>65.2</td>
<td>83.0</td>
<td class="sampler-delta ">+5.0</td></tr>
<tr class="sampler-estimate-row"><th scope="row">Our offline sampler <span class="sampler-estimate-badge">Est. †</span></th>
<td><span title="Illustrative count: 528 / 1,024; not measured">51.6<sup>†</sup></span></td>
<td><span title="Illustrative count: 25 / 83; not measured">30.1<sup>†</sup></span></td>
<td><span title="Illustrative count: 299 / 500; not measured">59.8<sup>†</sup></span></td>
<td><span title="Illustrative count: 1,058 / 1,320; not measured">80.2<sup>†</sup></span></td>
<td class="sampler-delta ">+2.1<sup>†</sup></td></tr>
<tr class="sampler-estimate-row"><th scope="row">Our online sampler <span class="sampler-estimate-badge">Est. †</span></th>
<td><span title="Illustrative count: 550 / 1,024; not measured">53.7<sup>†</sup></span></td>
<td><span title="Illustrative count: 27 / 83; not measured">32.5<sup>†</sup></span></td>
<td><span title="Illustrative count: 311 / 500; not measured">62.2<sup>†</sup></span></td>
<td><span title="Illustrative count: 1,078 / 1,320; not measured">81.7<sup>†</sup></span></td>
<td class="sampler-delta ">+4.2<sup>†</sup></td></tr>
</tbody></table></div>
<p class="sampler-table-footnote">Δ MATH is the percentage-point change from MCMC + SFT. <strong>† Draft estimates; not measured.</strong></p></div>

<div class="sampler-table-card" id="online-retention">
<div class="sampler-table-heading"><span class="sampler-table-kicker">RETENTION · ACCURACY (%)</span><h4 id="online-retention-title">Retention through the online loop</h4><p>Illustrative task-level variation near the base checkpoint, with losses shown explicitly.</p></div>
<div class="sampler-table-scroll" role="region" tabindex="0" aria-labelledby="online-retention-title">
<table class="sampler-results-table">
<thead><tr><th scope="col">Method</th><th scope="col">Chemistry</th><th scope="col">MMLU</th><th scope="col">GPQA</th><th scope="col">Prior avg.</th><th scope="col">Δ vs. base</th></tr></thead><tbody>
<tr class=""><th scope="row">Base model</th>
<td>28.3<span class="sampler-cell-delta ">+0.0 pp</span></td>
<td>65.1<span class="sampler-cell-delta ">+0.0 pp</span></td>
<td>33.3<span class="sampler-cell-delta ">+0.0 pp</span></td>
<td>42.2</td>
<td class="sampler-delta ">+0.0</td></tr>
<tr class="sampler-reference-row"><th scope="row">MCMC + SFT</th>
<td>26.6<span class="sampler-cell-delta sampler-down">−1.7 pp</span></td>
<td>65.1<span class="sampler-cell-delta ">+0.0 pp</span></td>
<td>34.3<span class="sampler-cell-delta ">+1.0 pp</span></td>
<td>42.0</td>
<td class="sampler-delta sampler-down">−0.2</td></tr>
<tr class="sampler-reference-row"><th scope="row">MCMC + SFT + RL</th>
<td>28.5<span class="sampler-cell-delta ">+0.2 pp</span></td>
<td>65.2<span class="sampler-cell-delta ">+0.1 pp</span></td>
<td>35.4<span class="sampler-cell-delta ">+2.1 pp</span></td>
<td>43.0</td>
<td class="sampler-delta ">+0.8</td></tr>
<tr class="sampler-estimate-row"><th scope="row">Our offline sampler <span class="sampler-estimate-badge">Est. †</span></th>
<td><span title="Illustrative count: 170 / 600; not measured">28.3<sup>†</sup></span><span class="sampler-cell-delta ">+0.0 pp</span></td>
<td><span title="Illustrative count: 9,128 / 14,042; not measured">65.0<sup>†</sup></span><span class="sampler-cell-delta sampler-down">−0.1 pp</span></td>
<td><span title="Illustrative count: 67 / 198; not measured">33.8<sup>†</sup></span><span class="sampler-cell-delta ">+0.5 pp</span></td>
<td>42.4<sup>†</sup></td>
<td class="sampler-delta ">+0.2<sup>†</sup></td></tr>
<tr class="sampler-estimate-row"><th scope="row">Our online sampler <span class="sampler-estimate-badge">Est. †</span></th>
<td><span title="Illustrative count: 169 / 600; not measured">28.2<sup>†</sup></span><span class="sampler-cell-delta sampler-down">−0.1 pp</span></td>
<td><span title="Illustrative count: 9,150 / 14,042; not measured">65.2<sup>†</sup></span><span class="sampler-cell-delta ">+0.1 pp</span></td>
<td><span title="Illustrative count: 66 / 198; not measured">33.3<sup>†</sup></span><span class="sampler-cell-delta ">+0.0 pp</span></td>
<td>42.2<sup>†</sup></td>
<td class="sampler-delta ">+0.0<sup>†</sup></td></tr>
</tbody></table></div>
<p class="sampler-table-footnote">Small numbers show each task’s change from the base model. Δ uses displayed rounded scores. <strong>† Retention estimates; not measured.</strong></p></div>
<!-- sampler-online-tables:end -->

The online placeholder is **53.7% MATH**, versus 51.6% offline: a 4.2-point gain over MCMC + SFT. It remains below the published **54.5% MCMC + SFT + RL** result. The retention average is 42.2%, yet Chemistry is 0.1 point lower than the base. Neither task gains nor a stable average establish superiority over the stronger pipeline or prove an absence of forgetting. Prior-task retention must be checked at each checkpoint, because several individually small updates can accumulate into forgetting.

### What would explain an online gain?

Three comparisons can turn an accuracy difference into an explanation:

- **Tracking the student.** Refreshing should help most when the student has moved away from the offline target. Compare refreshed and frozen samplers at equal compute, measuring downstream accuracy, accepted-output log-gap dispersion, and validity together.
- **Reusing the adapter.** Keeping the previous LoRA may reduce the fitting work needed after a student update. Compare retained and reset adapters against the same backbone; record updates and compute to reach comparable sampling quality.
- **Refreshing at the right frequency.** Frequent fitting may reduce mismatch while leaving less budget for SFT. Sweep the refresh interval under a fixed total budget, and track both new-task accuracy and prior-task retention.

These are proposed explanations to test, not observations from completed runs.

### Conclusion

The online method treats data generation as part of post-training: the student changes, so the sampler learns a new target. The projected +3–5-point gain motivates the experiment, but the decisive result is an accuracy–retention–compute improvement over a fixed sampler and the stronger MCMC + SFT + RL pipeline. LoRA makes repeated fitting practical in parameter and optimizer storage; it does not by itself guarantee lower runtime or prevent forgetting.

## Where this is useful
{: #use-cases}

**Offline, this is student-aware data preparation.** An existing expert corpus can be rewritten into verified responses that better match a chosen model before SFT begins. The useful operation is more specific than removing bad examples: preserve what makes an example worth learning, while changing how that information is expressed for the learner. A fitted sampler can then process more examples from the same domain. Moving to a different student or a substantially different domain may require refitting.

**Online, the same mechanism becomes a post-training method.** A LoRA sampler keeps generating expert-informed supervision as the student evolves. This is useful when a fixed synthetic dataset becomes stale, or when on-policy exploration rarely discovers successful responses without expert help. The data distribution adapts; the student's update remains ordinary SFT.

### What does learning the sampler actually fix?

Our central argument concerns two costs: searching again for every example, and rebuilding data when the learner changes. Amortization can reduce the first through reuse; the online schedule addresses the second through refreshes. The paper's repeated MCMC transitions and fixed-reference preprocessing motivate these questions. [[1]](#sampler-ref-1) Our assessment is that the project should be judged on those operational benefits, not merely on replacing one sampling algorithm with another.

There are three boundaries to that argument. First, **at exact matching, MCMC and our offline sampler produce the same target distribution**. An offline accuracy gain must therefore come from better approximation, coverage, or compute allocation in a finite-budget run. Second, a correct final answer does not establish that every reasoning step is sound; a learned sampler inherits the limitations of its verifier. Third, matching responses on the training prompts places no direct constraint on unrelated evaluation tasks. Preserving prior capabilities remains an empirical requirement.

This suggests a useful role for the project in the post-training stack: **learn how to prepare data for the model, and reuse that preparation as the model learns.** Offline, the output is a student-adapted training corpus. Online, it is a continually refreshed source of SFT targets. The value comes from making expert information economical to reuse without sacrificing its content or the student's existing abilities.

## References

<p id="sampler-ref-1"><strong>[1]</strong> Aayush Karan, Sitan Chen, and Yilun Du. <a href="https://arxiv.org/abs/2610.02140v1">Finetuning with Sampling: SFT Learns Better Than You Think</a>. arXiv:2610.02140v1, 2026.</p>

<p id="sampler-ref-2"><strong>[2]</strong> Kevin Lu and Thinking Machines Lab. <a href="https://thinkingmachines.ai/blog/on-policy-distillation/">On-Policy Distillation</a>. Thinking Machines Lab: Connectionism, 2025.</p>

<p id="sampler-ref-3"><strong>[3]</strong> Fangxu Yu, Lai Jiang, Haoqiang Kang, Shibo Hao, and Lianhui Qin. <a href="https://arxiv.org/abs/2406.05673v6">Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples</a>. ICML, 2025.</p>

<p id="sampler-ref-4"><strong>[4]</strong> Haoqiang Kang, Enna Sachdeva, Piyush Gupta, Sangjae Bae, and Kwonjoon Lee. <a href="https://arxiv.org/abs/2503.06514">GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks</a>. CVPR, 2025.</p>

<p id="sampler-ref-5"><strong>[5]</strong> Xiaodong Liu et al. <a href="https://arxiv.org/abs/2607.13394v1">GFlowRL: Scaling Distribution-Matching RL to Large Language Models</a>. arXiv:2607.13394v1, 2026.</p>

<p id="sampler-ref-6"><strong>[6]</strong> Xuekai Zhu et al. <a href="https://arxiv.org/abs/2509.15207v3">FlowRL: Matching Reward Distributions for LLM Reasoning</a>. arXiv:2509.15207v3, 2025.</p>

<p id="sampler-ref-7"><strong>[7]</strong> Edward J. Hu et al. <a href="https://arxiv.org/abs/2106.09685">LoRA: Low-Rank Adaptation of Large Language Models</a>. ICLR, 2022.</p>

<p id="sampler-ref-8"><strong>[8]</strong> Yoshua Bengio, with Edward J. Hu. <a href="https://yoshuabengio.org/en/blog/scaling-service-reasoning-model-based-ml">Scaling in the service of reasoning &amp; model-based ML</a>. 2023.</p>

<p id="sampler-ref-9"><strong>[9]</strong> Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. <a href="https://arxiv.org/abs/2509.04259v1">RL’s Razor: Why Online Reinforcement Learning Forgets Less</a>. arXiv:2509.04259v1, 2025. See §4–5 and Appendix A for the KL analysis and its assumptions.</p>

<p id="sampler-ref-10"><strong>[10]</strong> Howard Chen, Noam Razin, Karthik Narasimhan, and Danqi Chen. <a href="https://proceedings.mlr.press/v306/chen26do.html">Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting</a>. ICML, 2026. See §3–4 for distributional analysis and approximately on-policy SFT; Appendix A.5 discusses limits of KL as a predictor.</p>

<p id="sampler-ref-11"><strong>[11]</strong> Zhihong Shao et al. <a href="https://arxiv.org/abs/2402.03300">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>. arXiv:2402.03300, 2024. See §4.1 for GRPO.</p>

<p id="sampler-ref-12"><strong>[12]</strong> Yoshua Bengio. <a href="https://yoshuabengio.org/en/blog/generative-flow-networks">Generative Flow Networks</a>. 2022. Discusses learning sequential construction policies and contrasts them with MCMC sampling.</p>

## Citation
{: #citation}

Please cite this post as:

Murray Kang. “From Off-Policy Data to On-Policy SFT.” October 2026.

{% raw %}
```bibtex
@misc{kang2026onpolicysft,
  author = {Kang, Murray},
  title = {{From Off-Policy Data to On-Policy SFT}},
  year = {2026},
  month = oct,
  howpublished = {Research blog},
  url = {https://mk322.github.io/blog/learned-on-policy-sampler/}
}
```
{% endraw %}
