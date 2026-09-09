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
```

## Dataset

Dataset: Customer Support on Twitter (Kaggle)

Selected brand: **AppleSupport**

Expected local dataset path:

```text
data/raw/twcs.csv
```

The raw dataset and large generated model artifacts are intentionally excluded from Git.

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

## Reproduce the evaluation

The repository contains the numbered scripts used for the full pipeline.

The main stages are:

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
26  Prepare second judge
27  Calculate judge agreement
28  Failure analysis
29  Final metric consolidation
```

After the required dataset and generated intermediate artifacts are available, the final evaluation can be reproduced with:

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

## Final configuration

```text
Minimum intent confidence: 0.50
Minimum retrieval similarity: 0.70
Minimum evidence examples: 2
Top-K retrieval: 5
Embedding model: sentence-transformers/all-MiniLM-L6-v2
```

## Results

### Intent classification

| Model | Macro-F1 |
|---|---:|
| Majority baseline | 0.0222 |
| TF-IDF + Logistic Regression | 0.0928 |
| Context-aware TF-IDF + Logistic Regression | 0.1403 |

### Final agent

| Metric | Result |
|---|---:|
| Intent accuracy | 17.0% |
| Action accuracy | 33.0% |
| Auto-handle coverage | 25.5% |
| Automation precision | 76.47% |
| Safe automation rate | 19.5% |

The 76.47% automation precision must be interpreted together with its 25.5% coverage.

### Reply quality

| Dimension | Score / 5 |
|---|---:|
| Groundedness | 4.075 |
| Relevance | 3.775 |
| Helpfulness | 2.950 |
| Actionability | 3.700 |
| Brand alignment | 4.375 |
| Overall | 3.625 |

Second-judge agreement:

- Exact agreement: 67.9%
- Adjacent agreement: 90.4%
- Weighted kappa: 0.675

The judges were AI-assisted rather than independent human reviewers.

## Golden evaluation set

The evaluation set contains **200 examples**.

The annotations were AI-assisted rather than fully independent human labels, which is an important limitation of the current evaluation.

## Failure analysis

The main failure mode was intent misclassification.

Other observed failures:

- over-escalation
- weak retrieval evidence
- unsafe automation
- low intent confidence
- occasional low-helpfulness replies

See:

```text
evaluation/failure_analysis.csv
evaluation/failure_analysis_summary.txt
```

## Tests

Run:

```bash
pytest -q
```

Expected current result:

```text
5 passed
```

## Repository structure

```text
hiver/
├── configs/
├── evaluation/
├── notebooks/
├── report/
├── scripts/
├── src/
├── tests/
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

## Limitations

The current evaluation has two important limitations:

1. Golden annotations were AI-assisted rather than independently hand-labelled by humans.
2. The second judge was also AI-assisted, so the reported agreement is not human-vs-LLM agreement.

The current classifier is also weakly labelled and remains the main technical bottleneck.

## Detailed report

See:

`report/HIVER_FINAL_REPORT.md`