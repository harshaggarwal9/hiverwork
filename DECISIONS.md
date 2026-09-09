# Decision Log

## 1. Select one support brand

**Decision:** Use AppleSupport as the target brand.

**Why:** The dataset contains many brands, so focusing on one brand keeps the problem tractable while providing enough historical support interactions for retrieval and evaluation.

---

## 2. Use conversation threads instead of isolated tweets

**Decision:** Reconstruct conversation context from tweet relationships.

**Why:** Many customer messages are short or refer to previous messages. Context such as “still not working” or “yes” is difficult to classify without the preceding conversation.

---

## 3. Limit context length

**Decision:** Use up to four previous messages when constructing context-aware examples.

**Why:** This captures the recent conversational state while avoiding unnecessarily large inputs and excessive noise from very old messages.

---

## 4. Derive intents from the selected brand's data

**Decision:** Build the intent taxonomy from AppleSupport interactions rather than importing an external intent taxonomy.

**Why:** The assignment asks for a small set of intents defined from the data. Clustering and manual review were used to identify recurring support themes.

---

## 5. Keep the intent taxonomy small

**Decision:** Use ten final intents.

**Why:** A small taxonomy is easier to route, evaluate, and operate safely than dozens of highly overlapping labels.

---

## 6. Include a general/insufficient-context intent

**Decision:** Add `general_or_insufficient_context`.

**Why:** Real support conversations contain ambiguous, incomplete, and very short messages that do not always provide enough evidence for a specific intent.

---

## 7. Use a temporal train/dev split

**Decision:** Split the development data chronologically rather than randomly.

**Why:** Support language and product issues can change over time. A temporal split better approximates deployment on future conversations and reduces future-information leakage.

---

## 8. Exclude golden threads from training

**Decision:** Remove examples from golden evaluation threads from the training/development corpus.

**Why:** Removing only the exact golden messages could still allow nearby messages from the same conversation to leak information into training or retrieval. Thread-level exclusion provides a cleaner evaluation.

---

## 9. Use multiple baselines

**Decision:** Compare a majority baseline, text-only TF-IDF classifier, and context-aware TF-IDF classifier.

**Why:** The majority baseline provides a trivial reference point, while TF-IDF provides a lightweight classical ML baseline. The context-aware version tests whether conversation history provides measurable value.

---

## 10. Use FAISS for historical retrieval

**Decision:** Use FAISS with normalized sentence embeddings for similarity search.

**Why:** The historical support corpus is large enough that efficient vector search is useful. FAISS provides a simple and fast retrieval mechanism without requiring a complex external infrastructure.

---

## 11. Use historical evidence before automating

**Decision:** Retrieval is performed before the final auto-handle/escalate decision.

**Why:** Even when the intent classifier is confident, the system should not automate a case when there is insufficient historical precedent. Retrieval therefore acts as an additional evidence check.

---

## 12. Use conservative escalation thresholds

**Decision:** Require both intent confidence and historical retrieval evidence before auto-handling.

**Final thresholds:**

```text
Intent confidence >= 0.50
Retrieval similarity >= 0.70
Minimum evidence examples = 2

---

## 13. Escalate sensitive intents conservatively

**Decision:** Treat `account_security` and `purchase_store_refund` conservatively.

**Why:** These cases can involve sensitive account or transaction-specific situations where incorrect automated guidance can be more costly than escalation.

---

## 14. Keep reply generation retrieval-grounded

**Decision:** Generate responses from retrieved historical support behavior rather than unconstrained generation.

**Why:** Historical support responses provide evidence for how similar issues were handled and reduce the risk of inventing unsupported troubleshooting instructions.

---

## 15. Evaluate operational behavior, not only classification

**Decision:** Report automation precision, automation coverage, and safe automation rate in addition to intent metrics.

**Why:** A support agent ultimately makes an operational decision about whether a message can be safely handled automatically. Classification accuracy alone does not capture this trade-off.