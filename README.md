
````markdown
# Hiver AI Support Agent — AppleSupport

AI customer-support agent built for the Hiver SDE Intern take-home assignment using the Customer Support on Twitter dataset.

## Problem

For an incoming customer message, the system:

1. Reconstructs conversation context.
2. Classifies the customer intent.
3. Retrieves similar historical AppleSupport interactions.
4. Decides whether to auto-handle or escalate.
5. Generates a historically grounded reply for auto-handled cases.

## Architecture

```text
Incoming Tweet
      |
      v
Thread Reconstruction
      |
      v
Context-aware Intent Classification
      |
      v
Historical Retrieval (FAISS)
      |
      v
Evidence + Confidence
      |
      v
Escalation Decision
    /       \
Auto-handle  Escalate
   |            |
   v            v
Grounded      Human
Reply         Support
````

## Dataset

Dataset: Customer Support on Twitter (Kaggle)

Selected brand: **AppleSupport**

Expected local dataset path:

```text
data/raw/twcs.csv
```

The raw dataset and large generated model artifacts are intentionally excluded from Git.

After profiling the dataset:

* 2,811,774 tweets
* 702,777 unique authors
* 2,017,439 tweets with response links
* 106,860 AppleSupport tweets
* 143,518 AppleSupport-related tweets
* 81,391 reconstructed valid threads
* 228,380 total AppleSupport thread messages

The historical corpus spans May 2008 through December 2017.

## Intent Taxonomy

The final taxonomy contains 10 data-derived intents:

* `software_update`
* `device_troubleshooting`
* `battery_power`
* `connectivity`
* `apple_music_media`
* `account_security`
* `purchase_store_refund`
* `app_service_issue`
* `how_to_settings`
* `general_or_insufficient_context`

## Setup

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Main Pipeline

The repository contains numbered scripts for the complete pipeline:

```text
01  Profile dataset
02  Identify support brands
03  Reconstruct AppleSupport threads
04  Build support examples
05  Discover candidate intents
06  Review intents
07  Build context-aware examples
08  Create golden evaluation set
09  Apply annotations
10  Prepare temporal train/dev split
11  Train TF-IDF baseline
12  Build FAISS retrieval index
13  Evaluate retrieval
17  Train context-aware classifier
18  Audit golden labels
19  Run majority baseline
20  Run final context-aware agent
21  Analyze escalation thresholds
22  Update thresholds
23  Set final balanced thresholds
24  Prepare reply evaluation
25  Calculate reply quality
26  Prepare second reviewer
27  Calculate reviewer agreement
28  Failure analysis
29  Final metric consolidation
```

After the required dataset and generated intermediate artifacts are available, final evaluation can be reproduced with:

```bash
python scripts/20_update_agent_context.py
python scripts/24_prepare_reply_evaluation.py
python scripts/25_calculate_reply_quality.py
python scripts/26_prepare_reviewer2.py
python scripts/27_calculate_reviewer_agreement.py
python scripts/28_failure_analysis.py
python scripts/29_finalize_metrics.py
```

The complete dataset-processing pipeline is documented by the numbered scripts under `scripts/`.

## Final Configuration

```text
Minimum intent confidence: 0.50
Minimum retrieval similarity: 0.70
Minimum evidence examples: 2
Top-K retrieval: 5
Embedding model: sentence-transformers/all-MiniLM-L6-v2
```

Sensitive intents such as account/security and purchase/refund are handled conservatively.

## Results

### Intent Classification

| Model                                      | Macro-F1 |
| ------------------------------------------ | -------: |
| Majority baseline                          |   0.0222 |
| TF-IDF + Logistic Regression               |   0.0928 |
| Context-aware TF-IDF + Logistic Regression |   0.1403 |

Adding conversational context improves the TF-IDF baseline by 0.0475 absolute Macro-F1.

### Final Agent

| Metric               |    Result |
| -------------------- | --------: |
| Intent accuracy      |     17.0% |
| Action accuracy      |     33.0% |
| Auto-handle cases    |  51 / 200 |
| Escalated cases      | 149 / 200 |
| Automation coverage  |    25.50% |
| Automation precision |    76.47% |
| Safe automation rate |    19.50% |

The low intent accuracy is intentionally reported rather than hidden. The main product signal is selective, evidence-gated automation rather than unrestricted automation.

### Misleading Headline Number

The headline **76.47% automation precision** is incomplete without its coverage.

The system auto-handled only 51 of 200 evaluation cases, corresponding to 25.50% automation coverage. Therefore, the 76.47% precision applies only to cases selected for automation and should not be interpreted as overall support accuracy.

Coverage and safe automation rate are reported alongside precision.

## Reply Quality

A 40-case reply-quality sample was evaluated across six dimensions.

| Dimension       |     Mean |
| --------------- | -------: |
| Groundedness    | 4.40 / 5 |
| Relevance       | 4.40 / 5 |
| Helpfulness     | 3.25 / 5 |
| Actionability   | 3.38 / 5 |
| Brand alignment | 4.70 / 5 |
| Overall quality | 3.50 / 5 |

The current ratings are AI-assisted evaluation annotations rather than independent human ratings.

## Reviewer Agreement

A second-reviewer comparison was computed on the same 40 cases.

| Metric                     | Result |
| -------------------------- | -----: |
| Overall exact agreement    |  45.8% |
| Overall adjacent agreement |  82.9% |
| Mean absolute difference   |   0.80 |
| Weighted kappa             |  0.410 |
| Escalation exact agreement |  80.0% |
| Escalation Cohen's kappa   |  0.231 |

The second-reviewer ratings were AI-assisted rather than independently collected human ratings. These figures should therefore be interpreted as reviewer-consistency evidence rather than definitive human-vs-LLM agreement.

## Golden Evaluation Set

The evaluation set contains **200 examples**.

The current annotations were AI-assisted rather than independently hand-labelled by humans. They are useful for development and benchmarking, but this remains an important limitation of the evaluation methodology.

## Failure Analysis

The final failure analysis identified:

| Failure mode             | Count |
| ------------------------ | ----: |
| Intent misclassification |   166 |
| Over-escalation          |    22 |
| Weak retrieval evidence  |     2 |
| Unsafe automation        |     1 |
| Low intent confidence    |     1 |

Common examples include post-update device failures, noisy multi-intent complaints, unrelated generated replies, and cases where the agent selected escalation despite potentially useful historical evidence.

See:

```text
evaluation/failure_analysis.csv
evaluation/failure_analysis_summary.txt
```

## Engineering Decisions

The main non-obvious engineering decisions are documented in:

```text
DECISIONS.md
```

The decision log covers:

* single-brand scope,
* conversation-thread reconstruction,
* four-message context window,
* data-derived intent design,
* compact 10-intent taxonomy,
* a general/insufficient-context fallback intent,
* chronological train/dev splitting,
* exclusion of golden threads to reduce leakage,
* multiple baselines,
* FAISS retrieval,
* historical evidence before automation,
* conservative thresholds,
* sensitive-intent escalation,
* retrieval-grounded replies,
* operational metrics focused on safe automation.

## Important Evaluation Artifacts

```text
evaluation/context_agent_results.csv
evaluation/reply_quality_metrics.csv
evaluation/reviewer_agreement_metrics.csv
evaluation/final_metrics_summary.csv
evaluation/final_metrics_summary.txt
evaluation/failure_analysis.csv
evaluation/failure_analysis_summary.txt
evaluation/golden_set.csv
```

## Tests

Run:

```bash
pytest -q
```

Current result:

```text
5 passed
```

## Repository Structure

```text
hiver/

├── configs/
├── data/
├── evaluation/
├── notebooks/
├── report/
├── scripts/
├── src/
├── tests/
├── DECISIONS.md
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

## Limitations

The current evaluation has several important limitations:

1. The 200-example golden set was AI-assisted rather than independently hand-labelled by humans.
2. The 40-case reply-quality ratings were AI-assisted rather than independently collected human ratings.
3. The second reviewer was also AI-assisted, so the reported agreement is not definitive human-vs-LLM agreement.
4. Retrieval quality is assessed primarily through embedding similarity and evidence thresholds rather than a fully independent human relevance benchmark.
5. The final classifier is weakly labelled and remains the main technical bottleneck.
6. Final intent accuracy is only 17.0%, so the system should not be presented as a high-accuracy general-purpose support classifier.

## Next Week

The next iteration should focus on:

1. improving intent classification with stronger contextual features or a lightweight transformer,
2. cleaning ambiguous and noisy intent labels,
3. improving multi-intent detection,
4. collecting a genuinely human-reviewed evaluation set,
5. improving reply actionability while preserving historical grounding,
6. retuning automation thresholds using human-reviewed data,
7. adding a lightweight fast-path evaluation command for rapid reproduction of headline metrics.

## Detailed Report

See:

```text
report/HIVER_FINAL_REPORT.md
```

## Final Takeaway

The main design principle is:

**Do not automate merely because a classifier is confident; automate when the intent, historical evidence, and response path are jointly strong enough.**

```

After pasting, click **Commit changes**.
```
