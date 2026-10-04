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
  - **Illustrative targets:** Math avg. of 55.5% offline (+2.1 points versus MCMC + SFT) and 60.3% online (+6.9 points versus MCMC + SFT). Prior avg. targets are 42.1% and 42.2%, close to the base model’s 42.2%. These values and the online curve are estimates pending measured runs.

---

<link rel="stylesheet" href="{{ '/assets/blog/learned-sampler/article.css' | relative_url }}?v=5">

<nav class="sampler-toc" aria-label="Article contents"><details open><summary>On this page</summary><ol><li><a href="#sft-problem">SFT’s off-policy mismatch</a></li><li><a href="#target">What is the mismatch?</a></li><li><a href="#amortization">Amortize the search</a></li><li><a href="#train-sampler">From KL to the training loss</a></li><li><a href="#offline">Offline: sampler, then SFT</a></li><li><a href="#online">Online: the LoRA sampler</a></li><li><a href="#experiments-offline">Offline experiments</a></li><li><a href="#experiments-online">Online experiments</a></li><li><a href="#use-cases">Where this is useful</a></li></ol></details></nav>

<script defer src="{{ "/assets/blog/learned-sampler/navigation.js" | relative_url }}"></script>

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
<p class="sampler-algorithm-label">Algorithm 1 · Offline sampler training and SFT</p>

```text
Input: expert examples D, student θ₀,
       sampler φ, group size K ≥ 2
Output: fine-tuned student θ

θ_ref ← frozen copy of θ₀
for each sampler-training step:
    (x, τ) ← sample an expert example from D
    Y ← K verified responses from q_φ(· | x, τ)
    a_i ← log q_φ(y_i | x, τ) − log p_θ_ref(y_i | x)
    L ← mean_i (a_i − stop_gradient(mean_j a_j))²
    φ ← φ − η_φ ∇_φ L  # Eq. 7

Freeze φ
D_SFT ← verified responses generated from D using q_φ
θ ← SFT(θ₀, D_SFT)  # Eq. 1
Return θ
```

</div>

Each group contains verified responses for one prompt and expert demonstration. We draw fresh groups for each update, using the same rollout policy whose sequence probabilities enter the loss. If fewer than two responses pass verification, we collect more candidates or skip that update. The final dataset contains prompt–response pairs; the expert demonstration is not an extra student input.

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
<p class="sampler-algorithm-label">Algorithm 2 · Online sampler training and SFT</p>

```text
Input: expert examples D, student θ₀, LoRA parameters φ,
       group size K ≥ 2, number of rounds T
Output: fine-tuned student θ_T

for round t = 0, …, T − 1:
    Freeze backbone θ_t; enable sampler LoRA φ
    for each adapter-training step:
        (x, τ) ← sample an expert example from D
        Y ← K verified responses from q_(θ_t,φ)(· | x, τ)
        a_i ← log q_(θ_t,φ)(y_i | x, τ) − log p_θ_t(y_i | x)
        L ← mean_i (a_i − stop_gradient(mean_j a_j))²
        φ ← φ − η_φ ∇_φ L  # Eq. 7

    Freeze φ; generate verified SFT batch D_t
    Disable LoRA; unfreeze backbone θ_t
    θ_(t+1) ← SFT(θ_t, D_t)  # Eq. 1
Return θ_T with sampler LoRA disabled
```

</div>

During adapter fitting, sampler log probabilities use LoRA and expert conditioning; student log probabilities use neither. Only the adapter receives the matching-loss gradient. The SFT phase updates the backbone with the adapter disabled. We retain the adapter parameters between rounds and refit them against the updated backbone. As offline, each update uses a fresh group with at least two verified responses.

The sampler already has a GFlowNet reward: the student probability of a response, masked by verification. This is the unnormalized density in equation (9), and equation (7) learns its relative probabilities. The reward trains the sampler; the student update is SFT. This is why the two parameter updates are separate in Algorithm 2.

The expert demonstrations remain fixed; the generated SFT targets can evolve. Refreshing the adapter helps address stale data, but the refresh frequency has a cost. Small adapter updates also have limited capacity. Both the update schedule and LoRA rank therefore belong in the online ablation, rather than being treated as automatic improvements.

This differs from on-policy distillation, which samples student trajectories and uses teacher probabilities as feedback. [[2]](#sampler-ref-2) Our expert-conditioned adapter generates data for the student's constrained distribution; the student receives ordinary SFT.

## Offline experiments
{: #experiments-offline}

### Setup

Both schedules use **Qwen2.5-3B** and MATH levels 3–5: 8,230 training problems and 1,024 test problems. Offline, we fit the sampler against the starting checkpoint, freeze it, and generate verified data for SFT. The baseline scores below come from the math setting of Finetuning with Sampling. [[1]](#sampler-ref-1)

The baselines use the expert data in different ways:

- **Expert-data SFT** trains directly on fixed demonstrations.
- **OPSD (on-policy self-distillation)** generates student rollouts, then learns from the same model acting as a teacher with the expert solution in its context. [[13]](#sampler-ref-13)
- **GRPO (Group Relative Policy Optimization)** learns from rewards on student rollouts. **UFT (Unified Fine-Tuning)** combines RL with supervised learning, using expert-solution hints that gradually shorten during training. [[11]](#sampler-ref-11) [[14]](#sampler-ref-14)
- **MCMC + SFT** searches for student-compatible responses before SFT. [[1]](#sampler-ref-1)

We report **Math avg.** over MATH, AMC, MATH500, and GSM8K, and **Prior avg.** over Chemistry, MMLU, and GPQA. Each task has equal weight; larger datasets do not dominate either average. The base row is the checkpoint before task-specific training.

<details class="sampler-experiment-details" markdown="1">
<summary>Evaluation sizes and reference settings</summary>

Single-shot accuracy is rounded to one decimal. Evaluation sizes follow the paper and its [files](https://github.com/aakaran/finetuning-with-sampling): MATH 1,024; AMC 83; MATH500 500; GSM8K 1,320; Chemistry 600. MMLU uses the [14,042-item test set](https://huggingface.co/datasets/cais/mmlu/viewer/all/test) with a [micro-average](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_mmlu.yaml). GPQA uses a 198-item [Diamond basis](https://arxiv.org/abs/2311.12022); the source paper does not specify its GPQA variant. Estimated sampler percentages are calculated from illustrative integer counts, rather than adding one fixed gain to every task.

The reference MCMC pipeline uses 10 transitions, block size 32, and maximum sequence length 1,856. Its SFT search covers 1–2 epochs, learning rates {5e−5, 1e−5, 5e−6}, and batch sizes {16, 32, 64}, with AdamW and a cosine schedule. [[1]](#sampler-ref-1)

</details>

### Results

<p class="sampler-results-note"><strong>Reading this draft.</strong> Baseline rows are published results. The two sampler rows are estimates; † marks these entries. The online curve below is also illustrative.</p>

The offline target is **55.5% Math avg.**, versus **53.4%** for MCMC + SFT: a **2.1-point average gain**. The four individual gains differ; the target concerns their mean. Prior avg. is **42.1%**, slightly above **42.0%** for MCMC + SFT and near the base model’s **42.2%**. Both schedules share the table below.

<!-- sampler-comparison:start -->
<div class="sampler-table-card sampler-summary-card" id="sampler-results">
<div class="sampler-table-heading" id="sampler-results-title"><strong>Offline &amp; online · shared evaluation</strong><span>Accuracy (%) ↑</span></div>
<div class="sampler-table-scroll" role="region" tabindex="0" aria-labelledby="sampler-results-title">
<table class="sampler-results-table sampler-summary-table"><colgroup><col class="sampler-method-col"><col><col></colgroup>
<thead><tr><th scope="col">Method</th><th scope="col">Math avg.<span class="sampler-header-note">Task learning</span></th><th scope="col" class="sampler-retention-start">Prior avg.<span class="sampler-header-note">Capability retention</span></th></tr></thead><tbody>
<tr class="sampler-base-row"><th scope="row"><span class="sampler-method-name">Base model</span></th>
<td class="sampler-average">31.8</td><td class="sampler-average sampler-retention-start">42.2</td></tr>
<tr class=""><th scope="row"><span class="sampler-method-name">Expert-data SFT</span></th>
<td class="sampler-average">24.2</td><td class="sampler-average sampler-retention-start">38.9</td></tr>
<tr class=""><th scope="row"><span class="sampler-method-name">OPSD</span></th>
<td class="sampler-average">30.2</td><td class="sampler-average sampler-retention-start">40.4</td></tr>
<tr class=""><th scope="row"><span class="sampler-method-name">GRPO</span></th>
<td class="sampler-average">45.7</td><td class="sampler-average sampler-retention-start">41.4</td></tr>
<tr class=""><th scope="row"><span class="sampler-method-name">UFT</span></th>
<td class="sampler-average">45.2</td><td class="sampler-average sampler-retention-start">42.1</td></tr>
<tr class="sampler-reference-row"><th scope="row"><span class="sampler-method-name">MCMC + SFT</span></th>
<td class="sampler-average">53.4</td><td class="sampler-average sampler-retention-start">42.0</td></tr>
<tr class="sampler-estimate-row"><th scope="row"><span class="sampler-method-name">Ours · offline <span class="sampler-estimate-badge">Estimate</span></span></th>
<td class="sampler-average">55.5<sup>†</sup></td><td class="sampler-average sampler-retention-start">42.1<sup>†</sup></td></tr>
<tr class="sampler-estimate-row"><th scope="row"><span class="sampler-method-name">Ours · online <span class="sampler-estimate-badge">Estimate</span></span></th>
<td class="sampler-average">60.3<sup>†</sup></td><td class="sampler-average sampler-retention-start">42.2<sup>†</sup></td></tr>
</tbody></table></div>
<p class="sampler-table-footnote">Math avg.: equal-weight mean of MATH, AMC, MATH500, GSM8K. Prior avg.: equal-weight mean of Chemistry, MMLU, GPQA. Means round only for display. Both schedules compare with MCMC + SFT. Baselines: <a href="#sampler-ref-1">[1]</a>. <strong>† Estimates; not measured.</strong></p></div>
<details class="sampler-benchmark-details"><summary>See the scores behind each average</summary>
<div class="sampler-table-card sampler-detail-card"><div class="sampler-table-scroll" role="region" tabindex="0" aria-label="Per-task accuracy breakdown">
<table class="sampler-results-table sampler-detail-table"><colgroup><col class="sampler-method-col"><col><col><col><col><col><col><col></colgroup>
<thead><tr><th scope="col">Method</th><th scope="col">MATH</th><th scope="col">AMC</th><th scope="col">MATH500</th><th scope="col">GSM8K</th><th scope="col">Chem.</th><th scope="col">MMLU</th><th scope="col">GPQA</th></tr></thead><tbody>
<tr class="sampler-base-row"><th scope="row">Base model</th><td>31.5</td><td>13.3</td><td>24.5</td><td>57.9</td><td>28.3</td><td>65.1</td><td>33.3</td></tr>
<tr class=""><th scope="row">Expert-data SFT</th><td>24.3</td><td>10.0</td><td>16.8</td><td>45.5</td><td>22.2</td><td>64.8</td><td>29.8</td></tr>
<tr class=""><th scope="row">OPSD</th><td>26.7</td><td>8.4</td><td>33.2</td><td>52.4</td><td>24.2</td><td>65.2</td><td>31.3</td></tr>
<tr class=""><th scope="row">GRPO</th><td>45.7</td><td>24.9</td><td>31.3</td><td>80.8</td><td>27.8</td><td>65.2</td><td>31.3</td></tr>
<tr class=""><th scope="row">UFT</th><td>47.0</td><td>29.3</td><td>29.7</td><td>74.6</td><td>28.3</td><td>65.3</td><td>32.8</td></tr>
<tr class="sampler-reference-row"><th scope="row">MCMC + SFT</th><td>49.5</td><td>27.7</td><td>58.2</td><td>78.2</td><td>26.6</td><td>65.1</td><td>34.3</td></tr>
<tr class="sampler-estimate-row"><th scope="row">Ours · offline</th><td>51.5<sup>†</sup></td><td>28.9<sup>†</sup></td><td>61.0<sup>†</sup></td><td>80.6<sup>†</sup></td><td>28.0<sup>†</sup></td><td>65.1<sup>†</sup></td><td>33.3<sup>†</sup></td></tr>
<tr class="sampler-estimate-row"><th scope="row">Ours · online</th><td>57.2<sup>†</sup></td><td>33.7<sup>†</sup></td><td>66.2<sup>†</sup></td><td>84.1<sup>†</sup></td><td>28.2<sup>†</sup></td><td>65.1<sup>†</sup></td><td>33.3<sup>†</sup></td></tr>
</tbody></table></div></div></details>
<!-- sampler-comparison:end -->

### Conclusion

The offline test asks whether a learned data-preparation policy can improve math performance while retaining prior skills. The estimates set that goal concretely: about two points of average math improvement while keeping the prior-task average close to the starting model. This average must be checked alongside the individual tasks; it cannot rule out forgetting on every ability. Measured runs must establish those gains; total compute must also include fitting, generation, scoring, verification, rejected outputs, and SFT.

## Online experiments
{: #experiments-online}

### Setup

The data split, starting checkpoint, and evaluation suite stay the same. The sampler is a **LoRA adapter on the current student**. Each round fits the adapter against the frozen student, generates verified responses, and updates the student with SFT while the adapter is disabled. The next round refreshes the sampler against that updated student.

We compare with **MCMC + SFT** and the fixed offline sampler in the [shared table](#sampler-results). Our online sampler uses the GFlowNet reward—student probability masked by verification—and supplies refreshed data for SFT. The student-update objective remains the same; the data-generation method and refresh schedule change. Matched-compute runs must include sampler fitting, verification, student scoring, and SFT.

### Results

The online target is **60.3% Math avg.**, versus **53.4%** for MCMC + SFT: a **6.9-point average gain**. Prior avg. is **42.2%**, close to the base model and slightly above plain MCMC + SFT’s **42.0%**. These are average targets, not claims that every benchmark improves by the same amount.

The curve below sketches how student Math avg. might evolve during online training: uneven gains followed by a plateau near the table’s target.

<figure id="online-training-curve"><picture><source media="(max-width: 600px)" srcset="{{ '/assets/blog/learned-sampler/online-training-curve-mobile.svg' | relative_url }}?v=2"><img src="{{ '/assets/blog/learned-sampler/online-training-curve.svg' | relative_url }}?v=2" width="800" height="370" alt="Illustrative single-line online learning curve: Math average rises with noisy fluctuations and later plateaus near the 60.3 percent target. Training progress is normalized; this is not a measured training trace."></picture><figcaption><strong>Illustrative online trajectory; not measured.</strong> Student Math avg. over normalized training progress. The noisy trajectory and plateau are schematic; the endpoint matches the 60.3% target in the table.</figcaption></figure>

### Conclusion

The online target combines higher math accuracy with a prior-task average near the base model. A measured curve and matched-compute runs must establish this comparison. To identify the contribution of sampler refreshes, the frozen and refreshed variants must use the same reward settings and student-update schedule.

<details class="sampler-experiment-details" markdown="1">
<summary>Next ablations: what produces the online gain?</summary>

- **Tracking the student.** Compare a sampler fitted to the initial student, a frozen adapter on the changing backbone, and a refreshed adapter at equal compute and with the same student SFT schedule. Measure accuracy, validity, and accepted-output log-gap dispersion.
- **Reusing the adapter.** Compare retained and reset LoRA weights against the same backbone; record the fitting work needed to reach comparable sampling quality.
- **Refresh frequency.** Sweep the interval under a fixed total budget, tracking math accuracy and prior-task retention.

</details>

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

<p id="sampler-ref-13"><strong>[13]</strong> Siyan Zhao et al. <a href="https://arxiv.org/abs/2601.18734">Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models</a>. arXiv:2601.18734, 2026.</p>

<p id="sampler-ref-14"><strong>[14]</strong> Mingyang Liu, Gabriele Farina, and Asuman Ozdaglar. <a href="https://arxiv.org/abs/2505.16984">UFT: Unifying Supervised and Reinforcement Fine-Tuning</a>. arXiv:2505.16984, 2025.</p>

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
