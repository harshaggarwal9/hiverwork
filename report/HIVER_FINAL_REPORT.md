# Hiver AI Support Agent — Final Report

## 1. Executive Summary

This project builds an evidence-grounded customer-support agent for the Hiver SDE Intern take-home assignment using the Customer Support on Twitter dataset.

The system focuses on one support brand, AppleSupport, and performs four core tasks:

1. reconstruct conversation context,
2. classify customer intent,
3. retrieve relevant historical support interactions,
4. decide between auto-handling and escalation while generating a historically grounded reply.

The final system uses a compact 10-intent taxonomy, a context-aware TF-IDF classifier, FAISS retrieval over historical support examples, and conservative evidence-gated automation.

The strongest product signal is selective automation rather than raw classifier accuracy. The final agent achieves:

- **31.0% intent accuracy**
- **30.5% action accuracy**
- **25.50% automation coverage**
- **98.04% automation precision**
- **25.00% safe automation rate**

The main technical weakness is intent classification. The classifier remains imperfect, so the system deliberately uses confidence, historical evidence, and sensitive-intent rules before allowing automated handling.

---

## 2. Problem Framing

A support agent should not simply produce plausible text. It should determine whether an incoming customer request can be safely handled automatically and ground any response in evidence from previous support interactions.

The system therefore separates:

- intent understanding,
- historical evidence retrieval,
- reply generation,
- automation versus escalation.

The automation decision is deliberately conservative. A confident prediction without sufficient historical support evidence is not treated as sufficient for automation.

The core principle is:

```text
Intent confidence
       +
Historical evidence
       +
Safe intent category
       |
       v
  Auto-handle

Otherwise
       |
       v
   Escalate
```

---

## 3. Dataset and Brand Selection

The project uses the Customer Support on Twitter dataset.

Dataset profiling produced:

- 2,811,774 tweets
- 702,777 unique authors
- 2,017,439 tweets with response links
- dates spanning May 2008 through December 2017

AppleSupport was selected because it provided a large and coherent support corpus.

AppleSupport data contained:

- 106,860 AppleSupport tweets
- 143,518 AppleSupport-related tweets
- 81,391 reconstructed valid threads
- 228,380 total thread messages

Immediate customer-to-support examples and context-aware examples were then constructed from these threads.

The raw dataset is not committed to Git because of its size and licensing/distribution considerations.

---

## 4. Conversation Reconstruction

Twitter support conversations were reconstructed using response relationships between tweets.

For context-aware examples, up to four previous messages were included.

This decision was intended to preserve enough conversational information to resolve short follow-ups such as:

- "Yes I am."
- "Will do."
- "Should I erase & restore?"
- "I had not reset my device."

These messages can be difficult to classify reliably without their preceding customer-support context.

The conversation reconstruction step also provides the historical context used by the retrieval and reply-generation stages.

---

## 5. Intent Taxonomy

The final taxonomy contains 10 data-derived intents:

1. `software_update`
2. `device_troubleshooting`
3. `battery_power`
4. `connectivity`
5. `apple_music_media`
6. `account_security`
7. `purchase_store_refund`
8. `app_service_issue`
9. `how_to_settings`
10. `general_or_insufficient_context`

The taxonomy was intentionally kept compact.

A general/insufficient-context fallback category was included because many support tweets are short, ambiguous, or insufficiently contextualized.

The taxonomy was derived from the support corpus rather than imposing a large generic customer-service ontology.

---

## 6. Evaluation Design

A chronological split was used to reduce temporal leakage.

### Training set

- 96,723 examples

### Development set

- 24,181 examples

### Date boundaries

- Training: 2016-03-04 to 2017-11-17
- Development: 2017-11-17 to 2017-12-03

A 200-example golden evaluation set was constructed separately.

A separate 40-case sample was used for reply-quality evaluation.

An important limitation is that the golden annotations and reply-quality ratings were AI-assisted rather than independently collected human annotations. Therefore, these results should be interpreted as assisted development/evaluation evidence rather than as a fully independent human benchmark.

The evaluation design also excluded golden examples and their same-thread examples from the training split to reduce direct leakage.

---

## 7. Baselines

Three intent-classification baselines were evaluated.

| Model | Macro-F1 |
|---|---:|
| Majority baseline | 0.0222 |
| TF-IDF + Logistic Regression | 0.0928 |
| Context-aware TF-IDF + Logistic Regression | 0.1403 |

Adding conversational context improves the text-only TF-IDF baseline by:

- **0.0475 absolute Macro-F1**
- **51.21% relative improvement**

This supports the engineering decision to preserve conversational context rather than classifying the latest tweet in isolation.

---

## 8. Retrieval System

Historical support examples are embedded using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

A FAISS inner-product index is constructed over normalized embeddings.

Final configuration:

```text
Indexed historical examples: 96,723
Embedding dimension:          384
Top-K retrieval:              5
Retrieval threshold:          0.70
Minimum evidence examples:    2
```

The retrieval layer provides historical evidence before automated handling.

The final evaluation reported a mean top-1 retrieval similarity of approximately:

```text
0.840
```

Retrieval similarity is treated as an evidence signal rather than a substitute for human relevance judgement.

The current evaluation therefore does not claim a true human-judged Recall@K benchmark.

---

## 9. Final Agent

The final context-aware agent uses:

```text
Minimum intent confidence:    0.50
Minimum retrieval similarity: 0.70
Minimum evidence examples:    2
Top-K retrieval:              5
```

Sensitive intents such as account/security and purchase/refund are handled conservatively.

The final decision policy combines:

1. predicted intent,
2. intent confidence,
3. historical retrieval evidence,
4. sensitivity of the predicted intent.

A case is auto-handled only when the system has sufficient confidence and historical evidence.

Otherwise it is escalated.

This design intentionally trades automation coverage for safety.

---

## 10. Final Results

### Intent Classification

| Model | Macro-F1 |
|---|---:|
| Majority baseline | 0.0222 |
| TF-IDF + Logistic Regression | 0.0928 |
| Context-aware TF-IDF + Logistic Regression | 0.1403 |

### Final Agent

| Metric | Result |
|---|---:|
| Intent accuracy | 31.0% |
| Action accuracy | 30.5% |
| Auto-handle cases | 51 / 200 |
| Escalated cases | 149 / 200 |
| Automation coverage | 25.50% |
| Automation precision | 98.04% |
| Safe automation rate | 25.00% |
| Reply coverage | 100.0% |

The final agent auto-handled only 51 of the 200 evaluation cases.

Among those selected for automation, 98.04% matched the expected action according to the current evaluation annotations.

The low overall intent accuracy remains the primary technical limitation and prevents the system from being presented as a high-accuracy general-purpose support classifier.

---

## 11. Escalation Results

The final action results were:

| Metric | Result |
|---|---:|
| Escalation precision | 7.38% |
| Escalation recall | 91.67% |
| Escalation F1 | 13.66% |

The high escalation recall is a consequence of the conservative policy.

The system prefers to route uncertain or sensitive cases to support rather than risk automatically handling them incorrectly.

The low escalation precision indicates that the current policy over-escalates many cases.

This is consistent with the failure analysis and is an important area for improvement.

---

## 12. Misleading Headline Number

The headline **98.04% automation precision** is incomplete without its coverage.

The system auto-handled only:

```text
51 / 200 cases
```

which corresponds to:

```text
25.50% automation coverage
```

Therefore, 98.04% precision applies only to the subset selected for automation. It should not be interpreted as:

- overall support accuracy,
- overall classifier accuracy,
- or the percentage of all customer messages that can be safely automated.

The more complete operational picture is:

| Metric | Result |
|---|---:|
| Intent accuracy | 31.00% |
| Automation coverage | 25.50% |
| Automation precision | 98.04% |
| Safe automation rate | 25.00% |

This is why coverage and safe automation rate are reported alongside automation precision.

---

## 13. Reply Quality

A 40-case reply-quality sample was evaluated across six dimensions.

| Dimension | Mean |
|---|---:|
| Groundedness | 4.40 / 5 |
| Relevance | 4.40 / 5 |
| Helpfulness | 3.25 / 5 |
| Actionability | 3.38 / 5 |
| Brand alignment | 4.70 / 5 |
| Overall quality | 3.50 / 5 |

Human-rated escalation accuracy in the reply-quality sample was:

```text
90.0%
```

The reply evaluation is useful for identifying response-quality patterns, but the ratings were AI-assisted rather than independently collected human ratings.

The strongest dimensions were groundedness, relevance, and brand alignment.

Helpfulness and actionability remain weaker areas, particularly for ambiguous or technically complex support requests.

---

## 14. Reviewer Agreement

A second-reviewer comparison was computed on the same 40-case sample.

### Overall agreement

| Metric | Result |
|---|---:|
| Overall exact agreement | 75.7% |
| Overall adjacent agreement | 95.6% |
| Mean absolute difference | 0.288 |
| Weighted kappa | 0.757 |

### Dimension-level weighted kappa

| Dimension | Weighted Kappa |
|---|---:|
| Groundedness | 0.752 |
| Relevance | 0.784 |
| Helpfulness | 0.482 |
| Actionability | 0.614 |
| Brand alignment | 0.877 |
| Overall quality | 0.651 |

### Escalation agreement

| Metric | Result |
|---|---:|
| Comparable pairs | 36 |
| Reviewer 1 accuracy | 88.9% |
| Reviewer 2 accuracy | 91.7% |
| Exact agreement | 97.2% |
| Cohen's kappa | 0.842 |

The escalation calculation contains 36 comparable pairs because four cases do not have comparable escalation labels.

The second-reviewer ratings were AI-assisted rather than independently collected human ratings. Therefore, these statistics should be interpreted as reviewer-consistency evidence rather than definitive human-vs-LLM agreement.

---

## 15. Golden Evaluation Set

The evaluation set contains:

```text
200 examples
```

The current annotations were AI-assisted rather than independently hand-labelled by humans.

They are useful for development and benchmarking, but this remains an important limitation of the evaluation methodology.

The evaluation set was kept separate from the temporal training data, and same-thread examples were excluded from training to reduce direct evaluation leakage.

The golden set includes a mixture of:

- direct support requests,
- conversational follow-ups,
- troubleshooting questions,
- ambiguous messages,
- update-related issues,
- battery complaints,
- connectivity problems,
- purchase/refund cases,
- media/service issues.

---

## 16. Failure Analysis

The final failure analysis identified:

| Failure mode | Count |
|---|---:|
| Intent misclassification | 166 |
| Over-escalation | 22 |
| Weak retrieval evidence | 2 |
| Unsafe automation | 1 |
| Low intent confidence | 1 |

### Failure Mode 1 — Intent Misclassification

This is the dominant failure mode.

The classifier struggles with:

- noisy customer language,
- short messages,
- multi-intent complaints,
- context-dependent follow-ups,
- overlapping troubleshooting categories.

This is consistent with the final intent accuracy of 31.0%.

### Failure Mode 2 — Over-escalation

The conservative thresholding policy causes some cases with useful historical evidence to be escalated.

This protects against unsafe automation but reduces coverage.

### Failure Mode 3 — Weak Retrieval Evidence

A small number of cases fail to retrieve sufficiently strong historical evidence.

This limits the system's ability to produce an evidence-backed automated response.

### Failure Mode 4 — Unsafe Automation

One case was identified where automation would not have been appropriate.

This supports the use of sensitive-intent rules and evidence gates.

### Failure Mode 5 — Low Intent Confidence

Some cases contain insufficient information for reliable classification.

The system responds by escalating instead of forcing an automated answer.

Detailed artifacts are available in:

```text
evaluation/failure_analysis.csv
evaluation/failure_analysis_summary.txt
```

---

## 17. Engineering Decisions

The main non-obvious engineering decisions are documented in:

```text
DECISIONS.md
```

The decision log contains 15 major decisions:

1. single-brand scope,
2. conversation-thread reconstruction,
3. four-message context window,
4. data-derived intent design,
5. compact 10-intent taxonomy,
6. general/insufficient-context fallback,
7. chronological train/dev split,
8. exclusion of golden threads to reduce leakage,
9. multiple baselines,
10. FAISS retrieval,
11. historical evidence before automation,
12. conservative thresholds,
13. sensitive-intent escalation,
14. retrieval-grounded replies,
15. operational metrics focused on safe automation.

These decisions prioritize reproducibility, evidence grounding, and operational safety over maximizing a single benchmark number.

---

## 18. Reproducibility

The repository includes a quick reproduction script:

```bash
python scripts/30_quick_reproduce.py
```

The script:

1. loads the committed final evaluation metrics,
2. runs the final context-aware agent,
3. evaluates the 200-case golden set,
4. prints the final agent results,
5. reports the action distribution.

The final reproduction completed successfully with:

```text
Cases evaluated: 200
Intent accuracy: 0.3100
Action accuracy: 0.3050
Auto-handle: 51
Escalate: 149
```

The final historical retrieval index contains:

```text
96,723 examples
```

The repository also contains the numbered scripts used throughout the pipeline.

---

## 19. Tests

The current automated test suite reports:

```text
5 passed
```

Run:

```bash
pytest -q
```

The tests cover the reusable project components that can be validated without requiring the full raw dataset.

---

## 20. Limitations

The current evaluation has several important limitations:

1. The 200-example golden set was AI-assisted rather than independently hand-labelled by humans.

2. The 40-case reply-quality ratings were AI-assisted rather than independently collected human ratings.

3. The second reviewer was also AI-assisted, so the reported agreement is not definitive human-vs-LLM agreement.

4. Retrieval quality is assessed primarily through embedding similarity and evidence thresholds rather than a fully independent human relevance benchmark.

5. The final classifier is weakly labelled and remains the main technical bottleneck.

6. Final intent accuracy is only 31.0%, so the system should not be presented as a high-accuracy general-purpose support classifier.

7. The final action policy is deliberately conservative and therefore produces relatively low automation coverage.

8. The dataset is historical and spans 2008–2017, so current product behavior cannot be assumed from the historical responses.

9. Some customer-support messages are ambiguous even when conversational context is available.

10. Reply quality is evaluated on a relatively small 40-case sample.

---

## 21. What I Would Improve Next

The next iteration should focus on:

1. improving intent classification with stronger contextual features or a lightweight transformer,

2. cleaning ambiguous and noisy intent labels,

3. improving multi-intent detection,

4. collecting a genuinely human-reviewed evaluation set,

5. improving reply actionability while preserving historical grounding,

6. retuning automation thresholds using human-reviewed data,

7. improving the escalation policy so that useful historical evidence can be exploited without sacrificing safety,

8. adding stronger human relevance judgements for retrieval evaluation,

9. evaluating calibration of intent confidence rather than using a single fixed threshold,

10. testing the system on additional support brands after validating the AppleSupport pipeline.

The highest-priority improvement is intent classification because it is responsible for the majority of observed failures.

---

## 22. Final Takeaway

The main design principle is:

> **Do not automate merely because a classifier is confident; automate when the intent, historical evidence, and response path are jointly strong enough.**

The final system demonstrates this principle through conservative, evidence-gated automation.

Although the classifier remains a clear weakness, the architecture provides a reproducible framework for combining:

- conversational context,
- intent classification,
- historical retrieval,
- evidence thresholds,
- sensitive-intent safeguards,
- grounded reply generation,
- and operational automation metrics.

The most important next step is to replace the assisted evaluation with genuinely human-reviewed labels and then improve the classifier and escalation policy using that cleaner benchmark.