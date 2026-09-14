---
title: "Your post title"
excerpt: "A short summary of the question, the main idea, and why it matters."
tags: [Reasoning, Reinforcement Learning]
tldr: |
  Write a short **TL;DR** here. It appears in an expandable summary above the article.
# Optional: override the estimated reading time (useful for Chinese posts).
# reading_time: 10
# last_modified_at: 2026-09-13
---

<div class="blog-toc" markdown="1">
**Contents**

* Table of contents
{:toc}
</div>

## Overview

Start with the question this post explores and the main takeaway.

## Background

Introduce the concepts and related work readers need. Add inline citations
like [Author et al. (2026)](#references) and footnotes where useful.[^note]

## Main idea

Explain the intuition before the details. Inline math: $p(x)$. Display math:

$$
\mathcal{L}(\theta) = -\mathbb{E}_{x \sim p_{\mathrm{data}}} \log p_\theta(x).
$$

### Example

```python
def expected_reward(rewards):
    return sum(rewards) / len(rewards)
```

| Method | Key idea | Limitation |
| :--- | :--- | :--- |
| Baseline | Describe the approach | Describe the tradeoff |

<!-- Place images in /images/blog/your-post/ and uncomment:
<figure>
  <img src="{{ '/images/blog/your-post/overview.png' | relative_url }}" alt="Describe what the diagram shows">
  <figcaption>Figure 1. Explain the takeaway.</figcaption>
</figure>
-->

## Discussion

Discuss the evidence, limitations, and open questions.

## Takeaways

Summarize what the reader should remember.

## References

- Author et al. (2026). *Paper title*. Add the source link.

[^note]: Add a supporting detail or clarification here.
