# Hiver AI Support Agent

AI support agent built for the Hiver SDE Intern take-home assignment using the Customer Support on Twitter dataset.

## What it does

The system processes an incoming customer message through:

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