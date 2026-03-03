# Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity

## Repository for the paper [Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity](https://www.arxiv.org/abs/2509.22641)
Abstract:
> $N$-gram novelty is widely used to evaluate language models' ability to generate text outside of their training data. More recently, it has also been adopted as a metric for measuring textual creativity. However, theoretical work on creativity suggests that this approach may be inadequate, as it does not account for creativity's dual nature: novelty (how original the text is) and appropriateness (how sensical and pragmatic it is). We investigate the relationship between this notion of creativity and $n$-gram novelty through 8,618 expert writer annotations of novelty, pragmaticality, and sensicality via \emph{close reading} of human- and AI-generated text. We find that while $n$-gram novelty is positively associated with expert writer-judged creativity, approximately $91$% of top-quartile $n$-gram novel expressions are not judged as creative, cautioning against relying on $n$-gram novelty alone. Furthermore, unlike in human-written text, higher $n$-gram novelty in open-source LLMs correlates with lower pragmaticality. In an exploratory study with frontier closed-source models, we additionally confirm that they are less likely to produce creative expressions than humans. Using our dataset, we test whether zero-shot, few-shot, and finetuned models are able to identify expressions perceived as novel by experts (a positive aspect of writing) or non-pragmatic (a negative aspect). Overall, frontier LLMs exhibit performance much higher than random but leave room for improvement, especially struggling to identify non-pragmatic expressions. We further find that LLM-as-a-Judge novelty ratings align with expert writer preferences in an out-of-distribution dataset, more so than an n-gram based metric.

## 📂 Project Structure

* **`data/`** Contains the password-protected raw data.  
  > 🔐 **Access Request:** Please [fill out this form](https://forms.gle/U58ahCPBbNuXykUJ6) to request access. Contact me if you do not hear back within **3 business days**.

* **`linear_models/`** Contains R notebooks for applying the linear modeling as described in the paper.

* **`llm_performance/`** Contains code to run the **LLM-as-a-Judge** evaluation for *novelty* and *pragmaticality*.

* **`ngram_novelty_scores/`** Code to compute n-gram novelty scores.  
  > ⚠️ **Note:** This section is currently under construction. Enjoy at your own risk.

# Citation

```
@inproceedings{
saakyan2026death,
title={Death of the Novel(ty): Beyond N-Gram Novelty as a Metric for Textual Creativity},
author={Arkadiy Saakyan and Najoung Kim and Smaranda Muresan and Tuhin Chakrabarty},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=z2idLjqzBe}
}
```

# Contact
Note this repo is still being updated. If you are interested in a particular part of the paper that has not been uploaded here yet, or for any other questions, please contact [Arkadiy Saakyan](mailto:a.saakyan@cs.columbia.edu).
