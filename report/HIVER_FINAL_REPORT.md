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