# Hiver AI Support Agent — Final Report

## 1. Executive Summary

This project builds an evidence-grounded customer-support agent for the Hiver SDE Intern take-home assignment using the Customer Support on Twitter dataset.

The system focuses on one support brand, AppleSupport, and performs four core tasks:

1. reconstruct conversation context,
2. classify customer intent,
3. retrieve relevant historical support interactions,
4. decide between auto-handling and escalation while generating a historically grounded reply.

The final system uses a compact 10-intent taxonomy, a context-aware TF-IDF classifier, FAISS retrieval over historical support examples, and conservative evidence-gated automation.

The strongest product signal is selective automation rather than raw classifier accuracy. The final agent reaches 25.50% automation coverage with 76.47% automation precision and a 19.50% safe automation rate.

The main weakness is intent classification: final intent accuracy is 17.0%. This is explicitly reported as a central limitation rather than hidden behind downstream metrics.

## 2. Problem Framing

A support agent should not simply produce plausible text. It should determine whether the incoming customer request can be safely handled automatically and ground any reply in evidence from previous support interactions.

The system therefore separates:

- intent understanding,
- historical evidence retrieval,
- reply generation,
- automation versus escalation.

The automation decision is deliberately conservative. A confident prediction without sufficient historical support evidence is not treated as sufficient for automation.

## 3. Dataset and Brand Selection

The project uses the Customer Support on Twitter dataset.

Dataset profiling produced:

- 2,811,774 tweets,
- 702,777 unique authors,
- 2,017,439 tweets with response links,
- dates spanning May 2008 through December 2017.

AppleSupport was selected because it provided a large and coherent support corpus.

AppleSupport data contained:

- 106,860 AppleSupport tweets,
- 143,518 related tweets,
- 81,391 reconstructed valid threads,
- 228,380 total thread messages.

Immediate customer-to-support examples and context-aware examples were then constructed from these threads.

## 4. Conversation Reconstruction

Twitter support conversations were reconstructed using response relationships between tweets.

For context-aware examples, up to four previous messages were included.

This decision was intended to preserve enough conversational information to resolve short follow-ups such as:

- "Yes I am."
- "Will do."
- "Should I erase & restore?"
- "I had not reset my device."

These messages can be impossible to classify reliably without their preceding customer-support context.

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

The final fallback category is important because many support tweets are short, ambiguous, or insufficiently contextualized.

## 6. Evaluation Design

A chronological split was used to reduce temporal leakage.

Training set:

- 96,723 examples

Development set:

- 24,181 examples

Date boundaries:

- training: 2016-03-04 to 2017-11-17
- development: 2017-11-17 to 2017-12-03

A 200-example golden evaluation set was also constructed.

A separate 40-case sample was used for reply-quality evaluation.

An important limitation is that the golden annotations and reply-quality ratings were AI-assisted rather than independently human-labelled. Therefore, these results should be interpreted as assisted development/evaluation evidence rather than as a fully independent human benchmark.

## 7. Baselines

Three intent-classification baselines were evaluated.

| Model | Macro-F1 |
|---|---:|
| Majority baseline | 0.0222 |
| TF-IDF + Logistic Regression | 0.0928 |
| Context-aware TF-IDF + Logistic Regression | 0.1403 |

The context-aware model improves on the text-only TF-IDF baseline by:

- 0.0475 absolute Macro-F1,
- 51.21% relative improvement.

This supports the engineering decision to preserve conversational context.

## 8. Retrieval System

Historical support examples are embedded with:

`sentence-transformers/all-MiniLM-L6-v2`

A FAISS inner-product index is constructed over normalized embeddings.

Configuration:

- indexed examples: 96,723
- embedding dimension: 384
- Top-K: 5
- automation retrieval threshold: 0.70

The retrieval layer is intended to provide historical evidence before automated handling.

Retrieval similarity is not treated as a substitute for independent human relevance judgement; it is used as an evidence signal for the agent.

## 9. Final Agent

The final context-aware agent uses:

- minimum intent confidence: 0.50,
- minimum retrieval similarity: 0.70,
- minimum historical evidence examples: 2,
- Top-K retrieval: 5.

Sensitive intents such as account/security and purchase/refund are escalated conservatively.

The basic decision principle is:

```text
Strong intent evidence
        +
Strong historical evidence
        +
Safe intent category
        |
        v
   Auto-handle

Otherwise
        |
        v
     Escalate