# Hiver AI Support Agent — AppleSupport

## 1. Executive Summary

This project builds an evaluation-first AI customer-support agent using the Customer Support on Twitter dataset. AppleSupport was selected as the target support brand. The system reconstructs conversation context, predicts a small data-derived intent taxonomy, retrieves historically similar AppleSupport interactions, decides whether to auto-handle or escalate, and produces a historically grounded reply.

The engineering objective was not to maximize one headline metric, but to build a conservative automation pipeline that uses historical evidence and exposes uncertainty.

The prototype demonstrates the complete pipeline. Its strongest operational result is **76.47% automation precision at 25.5% automation coverage**, giving a **19.5% safe automation rate**. Reply evaluation showed relatively strong **groundedness (4.075/5)** and **brand alignment (4.375/5)**.

Intent classification remains the primary bottleneck. The context-aware classifier achieved **0.1403 Macro-F1**, while the final agent achieved **17.0% intent accuracy** on the 200-case evaluation set.

The results therefore demonstrate a complete working architecture while making its current weaknesses explicit.

---

## 2. Problem Definition

For every incoming customer-support message, the agent must:

1. Reconstruct enough conversation context to understand the current message.
2. Assign the message to a small set of support intents.
3. Retrieve relevant historical AppleSupport responses.
4. Estimate whether the available evidence is sufficient for automation.
5. Auto-handle safe cases or escalate uncertain/sensitive cases.
6. Produce a grounded customer-facing reply when automation is selected.

The final pipeline is:

```text
Incoming customer message
        |
        v
Thread / context reconstruction
        |
        v
Context-aware intent classification
        |
        v
Historical retrieval
        |
        v
Evidence + confidence
        |
        v
Auto-handle / Escalate
       /        \
      v          v
Grounded reply  Human support
```

---

## 3. Data and Brand Selection

The source dataset is the Customer Support on Twitter Kaggle dataset.

Initial profiling found:

- **2,811,774 tweets**
- **702,777 unique authors**
- **2,017,439 tweets with `in_response_to_tweet_id`**
- Date range: **May 2008 – December 2017**

AppleSupport was selected from the candidate support brands because it provided a large support corpus with useful customer-to-support interactions.

The AppleSupport subset contained:

- **106,860 AppleSupport tweets**
- **143,518 related tweets**
- **81,391 reconstructed valid threads**
- **228,380 total messages**

The raw dataset is intentionally excluded from the Git repository because of its size.

---

## 4. Historical Support Corpus

Immediate customer-to-support pairing produced approximately **104,108 examples**.

Context-aware construction used up to four previous messages together with the current customer message and produced approximately **121,520 examples**.

This context is important because short Twitter messages can be difficult to interpret without conversation history.

---

## 5. Intent Taxonomy

Candidate intents were first discovered using clustering and keyword analysis over a 20,000-message sample.

The final taxonomy contains ten support intents:

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

The taxonomy was intentionally kept small enough to support routing rather than creating many overlapping labels.

---

## 6. Evaluation Design

A **200-case golden evaluation set** was created and excluded from training and retrieval.

Examples belonging to golden threads were also removed from train/dev data to reduce thread-level leakage.

Temporal split:

- Training: **96,723 examples**
- Development: **24,181 examples**

The training period precedes the development period chronologically.

The golden annotations and reply-quality ratings were **AI-assisted**, not independently produced by multiple human reviewers. Results should therefore be treated as development/evaluation evidence rather than production-grade human benchmarks.

---

## 7. Baselines

### Majority baseline

The majority-class classifier predicts `device_troubleshooting`.

**Macro-F1: 0.0222**

### TF-IDF + Logistic Regression

A text-only TF-IDF classifier trained using weakly labelled historical examples achieved:

**Macro-F1: 0.0928**

### Context-aware TF-IDF + Logistic Regression

Adding conversation context improved performance to:

**Macro-F1: 0.1403**

This is a **0.0475 absolute improvement** over the text-only baseline.

The absolute score remains low, showing that intent classification is still the dominant technical limitation.

---

## 8. Historical Retrieval

A FAISS `IndexFlatIP` index was built over **96,723 historical training examples** using normalized embeddings from:

`sentence-transformers/all-MiniLM-L6-v2`

The retrieval query includes conversation context and the current customer message.

The final automation policy uses a retrieval similarity threshold of:

**0.70**

Retrieval is used to provide historical evidence for response drafting and for deciding whether enough precedent exists to automate.

---

## 9. Escalation Policy

The agent uses conservative escalation logic.

Final thresholds:

```text
Minimum intent confidence: 0.50
Minimum retrieval similarity: 0.70
Minimum evidence examples: 2
```

Sensitive intents such as account/security and purchase/refund receive additional conservative treatment.

Thresholds were selected using development-set score distributions rather than tuning directly on the golden set.

The design intentionally trades coverage for safer automation when classifier confidence or historical evidence is insufficient.

---

## 10. Final Agent Results

The final agent was evaluated on **200 golden cases**.

| Metric | Result |
|---|---:|
| Intent accuracy | 17.0% |
| Action accuracy | 33.0% |
| Auto-handled cases | 51 / 200 |
| Escalated cases | 149 / 200 |
| Automation coverage | 25.5% |
| Automation precision | 76.47% |
| Safe automation rate | 19.5% |
| Escalation precision | 18.12% |
| Escalation recall | 69.23% |
| Reply coverage | 100% |

### Interpreting the headline metric

The **76.47% automation precision** should not be interpreted as a standalone success metric.

It was achieved at only **25.5% automation coverage**.

A high precision number can therefore be misleading if the system automates only a small fraction of incoming cases.

The more useful operational view is the combination of:

- **76.47% automation precision**
- **25.5% automation coverage**
- **19.5% safe automation rate**

---

## 11. Reply-Quality Evaluation

Forty cases were sampled for reply-quality review.

Average scores:

| Dimension | Score / 5 |
|---|---:|
| Groundedness | 4.075 |
| Relevance | 3.775 |
| Helpfulness | 2.950 |
| Actionability | 3.700 |
| Brand alignment | 4.375 |
| Overall quality | 3.625 |

The strongest properties were groundedness and brand alignment.

Helpfulness was weaker, reflecting limitations in the current simple retrieval-grounded response generator.

The response generator was deliberately kept simple and retrieval-grounded because the project did not rely on a paid external LLM API.

---

## 12. Second-Judge Agreement

A second AI-assisted judge evaluated the same 40 reply-quality cases.

Across 240 quality-rating pairs:

- **Exact agreement: 67.9%**
- **Adjacent agreement: 90.4%**
- **Mean absolute difference: 0.487**
- **Weighted kappa: 0.675**

For escalation decisions:

- **Exact agreement: 100%**
- **Cohen's kappa: 1.000**

These are **second-judge agreement measurements**, not human-human agreement.

---

## 13. Failure Analysis

The dominant observed failure mode was intent misclassification.

| Failure mode | Count | Share |
|---|---:|---:|
| Intent misclassification | 166 | 83.0% |
| Over-escalation | 22 | 11.0% |
| Weak retrieval evidence | 2 | 1.0% |
| Unsafe automation | 1 | 0.5% |
| Low intent confidence | 1 | 0.5% |
| Low helpfulness | 1 | 0.5% |
| No major failure | 7 | 3.5% |

### Intent misclassification

Short and ambiguous customer messages remain difficult even with conversation context.

Many support interactions contain replies whose meaning depends heavily on preceding messages, which limits the effectiveness of the current lightweight classifier.

### Over-escalation

The conservative decision policy escalates cases when confidence or historical evidence is weak. This reduces unsafe automation but also lowers automation coverage.

### Weak retrieval evidence

Some examples have plausible intent predictions but do not have strong historical matches above the retrieval threshold.

### Unsafe automation

One reviewed case exposed a weakness in the current reply policy where the generated response did not provide a sufficiently actionable path to specialist support.

### Low-confidence classification

Some cases contain inherently ambiguous language, resulting in low confidence and escalation.

The failure distribution indicates that improving intent classification is likely to provide the largest overall benefit.

---

## 14. Product Finding

The evaluation suggests that automation quality depends more on routing quality than on response wording alone.

A more sophisticated response generator cannot fully compensate for:

- incorrect intent classification
- weak retrieval evidence
- insufficient conversation context

The highest-value next investment is therefore better intent understanding, context modelling, retrieval reranking, and confidence calibration before adding a more powerful generation layer.

---

## 15. Key Engineering Decisions

### Decision 1 — Use one brand

AppleSupport was selected to keep the problem focused and allow deeper historical analysis.

### Decision 2 — Use conversation context

Context was added because many support messages are too short to classify reliably in isolation.

### Decision 3 — Retrieve before deciding automation

Historical retrieval happens before the final escalation decision so that insufficient precedent can trigger safe abstention.

### Decision 4 — Prefer conservative automation

The system sacrifices coverage when confidence or historical evidence is insufficient.

### Decision 5 — Keep replies grounded

Replies are derived from historical support behavior rather than unconstrained generation.

---

## 16. Limitations

The main limitations are:

1. **AI-assisted evaluation**  
   The golden labels and reply-quality judgments were AI-assisted rather than fully independent human annotations.

2. **Weakly labelled training data**  
   The classification training data was constructed using heuristic/silver labels rather than a large manually labelled training set.

3. **Low intent performance**  
   The context-aware classifier achieved only **0.1403 Macro-F1**, while the final agent achieved **17.0% intent accuracy**.

4. **Simple response generation**  
   The current response generator is retrieval-grounded but is not a production-grade LLM generation system.

5. **Conservative escalation**  
   The system intentionally escalates many uncertain cases, limiting automation coverage.

6. **Twitter-specific language**  
   Twitter support messages contain abbreviations, very short replies, typos, and conversational references that may not generalize directly to other support channels.

---

## 17. Next-Week Plan

### 1. Improve intent classification

Create a larger manually labelled training set and evaluate stronger contextual classifiers.

### 2. Calibrate confidence

Use validation-based calibration rather than raw classifier probabilities to improve escalation decisions.

### 3. Improve retrieval

Add metadata-aware filtering and reranking so the system prefers examples with matching intent, product context, and conversation state.

### 4. Add a controlled LLM rewrite layer

Use an LLM only after retrieval has selected trusted historical evidence. The LLM should rewrite or combine evidence rather than invent unsupported troubleshooting steps.

### 5. Expand human evaluation

Have multiple independent human reviewers score a larger sample and explicitly measure unsafe-response rate.

---

## 18. Reproducibility

The repository contains the numbered scripts used to construct the dataset, prepare examples, discover intents, train baselines, build retrieval, evaluate the agent, perform reply-quality evaluation, analyze thresholds, perform failure analysis, and generate final metrics.

The raw dataset is intentionally excluded from Git.

Expected local dataset path:

```text
data/raw/twcs.csv
```

Generated evaluation artifacts are stored under:

```text
evaluation/
```

Configuration is stored under:

```text
configs/
```

The test suite can be run with:

```bash
pytest -q
```

---

## 19. Conclusion

The project demonstrates a complete evaluation-first support-agent workflow:

**context → intent → retrieval → confidence → escalation → grounded reply**

The strongest operational result is **76.47% automation precision at 25.5% coverage**, while intent classification remains the dominant bottleneck.

The prototype is therefore not presented as production-ready. The next priority is improving labelled intent data, contextual classification, retrieval reranking, confidence calibration, and independent human evaluation.