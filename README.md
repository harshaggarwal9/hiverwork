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
