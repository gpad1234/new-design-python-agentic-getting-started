# Design Spec — `LearningAgent`

**File:** `agents/learning_agent.py`  
**Type:** Concrete class, extends `BaseAgent`  
**Phase:** Foundation (Phase 1)

---

## 1. Purpose

`LearningAgent` is a supervised text-classification agent.  It learns from
labelled `(text, intent)` examples, builds a **TF-IDF + Logistic Regression**
sklearn pipeline, and then classifies new observations at inference time.

Primary use-case: **intent routing** — mapping free-text user messages to a
fixed set of business intents so downstream systems know what action to take.

---

## 2. Class Signature

```python
class LearningAgent(BaseAgent):
    def __init__(
        self,
        name: str = "IntentClassifier",
        goal: str = "Classify user intent and route to the correct action",
    ) -> None
```

---

## 3. Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Inherited from `BaseAgent`. |
| `goal` | `str` | Inherited from `BaseAgent`. |
| `memory` | `list[dict]` | Episodic store (inherited). |
| `_examples` | `list[tuple[str, str]]` | Raw `(text, label)` training pairs. |
| `_pipeline` | `Pipeline \| None` | Fitted sklearn pipeline; `None` until `train()` is called. |
| `_is_trained` | `bool` | `True` after a successful `train()` call; reset to `False` when new examples are added. |

---

## 4. Lifecycle

```
                ┌─────────────────────────────────────────────┐
                │              LearningAgent lifecycle         │
                └─────────────────────────────────────────────┘

  1. Instantiate
        LearningAgent()
            │
  2. Add labelled examples
        agent.add_examples(TRAINING_DATA)   ←── _is_trained = False
            │
  3. Train
        report = agent.train()              ←── _is_trained = True
            │
  4. Inference (repeatable)
        result = agent.run("user query")    ←── calls perceive→decide→act
            │
  5. (Optional) Add new examples & retrain
        agent.add_example(text, label)      ←── _is_trained = False (stale)
        agent.train()                       ←── retrain on all data
```

---

## 5. Method Specifications

### `add_example(text: str, label: str) → None`

Appends one `(text, label)` pair to `_examples`.  Marks the model as stale
(`_is_trained = False`) because the pipeline no longer reflects all data.

---

### `add_examples(examples: list[tuple[str, str]]) → None`

Batch version of `add_example`; iterates and delegates.

---

### `train(test_size: float = 0.2, random_state: int = 42) → dict`

Trains the sklearn pipeline and returns an evaluation report.

**Requirements:** at least 4 labelled examples (raises `ValueError` otherwise).

**Steps:**

1. Unzip `_examples` into parallel `texts` / `labels` lists.
2. `train_test_split` — 80 % train / 20 % test, stratified by label.
3. Build pipeline:
   ```
   TfidfVectorizer(ngram_range=(1,2), min_df=1)
       └─► LogisticRegression(max_iter=1000)
   ```
4. `pipeline.fit(X_train, y_train)`
5. `pipeline.predict(X_test)` → `classification_report(..., output_dict=True)`.
6. Set `_is_trained = True`; return report dict.

**Return value:** `dict` — sklearn `classification_report` with keys per
label plus `"accuracy"`, `"macro avg"`, `"weighted avg"`.

---

### `perceive(observation: Any) → str`

Strips whitespace from the raw observation.

```python
return str(observation).strip()
```

---

### `decide(perception: str) → str`

Predicts the label for the cleaned text.

```python
predictions = self._pipeline.predict([perception])
return str(predictions[0])
```

Raises `RuntimeError` if called before `train()`.

---

### `act(action: str) → dict`

Maps a predicted label string to a business response dict.

**Built-in intent → response map:**

| Intent | Response |
|--------|----------|
| `recommend_product` | "Based on your interest, we recommend our Premium Plan." |
| `escalate_to_human` | "Let me connect you with a specialist right away." |
| `provide_faq` | "Here is the FAQ article that should help: /faq" |
| `collect_feedback` | "We'd love to hear from you — please fill in our survey." |
| *(unknown)* | `"Handling intent: <label>"` |

**Return schema:**
```python
{
    "predicted_intent": str,   # raw label from decide()
    "response": str            # human-readable reply
}
```

---

### `predict_proba(text: str) → dict[str, float]`

Returns per-class probability scores for a given text (useful for confidence
thresholding or audit logging).

```python
{"recommend_product": 0.812, "escalate_to_human": 0.062, ...}
```

Raises `RuntimeError` if the model is not yet trained.

---

### `is_trained` (property)

Read-only boolean; `True` when `_pipeline` is fitted and no new examples
have been added since the last `train()` call.

---

## 6. Pipeline Architecture

```
Raw text input
   │
   ▼  TfidfVectorizer(ngram_range=(1,2), min_df=1)
Sparse TF-IDF matrix  (unigrams + bigrams)
   │
   ▼  LogisticRegression(max_iter=1000)
Predicted class label + probability vector
```

**Why TF-IDF + LR?**
- Fast to train on small datasets (< 10k examples).
- Interpretable weights — easy to debug misfires.
- Deterministic given `random_state` — reproducible tests.
- Upgradeable: swap `LogisticRegression` for `SGDClassifier` or a
  `sentence-transformers` embedder without changing the rest of the class.

---

## 7. Training Data Format

Defined in `data/sample_data.py` as `TRAINING_DATA`:

```python
TRAINING_DATA: list[tuple[str, str]] = [
    ("I'm looking for a product that suits my budget", "recommend_product"),
    ("I want to speak to a real person",               "escalate_to_human"),
    ...
]
```

Minimum: 4 examples total, at least 1 per class (stratified split requires ≥ 1 per class in the test set).

---

## 8. Dependencies

| Package | Used for |
|---------|----------|
| `numpy` | `np.round` in `predict_proba` |
| `scikit-learn` | `TfidfVectorizer`, `LogisticRegression`, `Pipeline`, `train_test_split`, `classification_report` |
| `.base_agent.BaseAgent` | Inheritance |

---

## 9. Extension Points

- **Replace the classifier:** swap `LogisticRegression` with any sklearn estimator that supports `predict` / `predict_proba`.
- **Replace the vectoriser:** use `sentence-transformers` or `openai.embeddings` as the first pipeline stage for richer semantic features.
- **Override `act()`:** plug in real routing logic (e.g., call an internal API, open a ticket in Jira) instead of returning a static string.
- **Persist the model:** `joblib.dump(self._pipeline, "model.pkl")` / `joblib.load` — add `save()` / `load()` methods.

---

## 10. Limitations

- Closed-label classification — cannot predict intents it was not trained on.
- Training is in-memory and in-process; not suitable for datasets > ~100k rows without batching.
- No active learning loop — manual re-labelling and retraining required.
- `_examples` are never deduplicated; adding the same example twice biases the model.

---

## 11. Usage Example

```python
from agents import LearningAgent
from data.sample_data import TRAINING_DATA

agent = LearningAgent()
agent.add_examples(TRAINING_DATA)
report = agent.train()
print(f"Accuracy: {report['accuracy']:.0%}")

result = agent.run("How do I reset my password?")
# {"predicted_intent": "provide_faq", "response": "Here is the FAQ article ..."}

scores = agent.predict_proba("I need to speak with someone urgently")
# {"escalate_to_human": 0.91, "provide_faq": 0.04, ...}
```
