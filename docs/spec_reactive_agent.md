# Design Spec — `ReactiveAgent`

**File:** `agents/reactive_agent.py`  
**Type:** Concrete class, extends `BaseAgent`  
**Phase:** Foundation (Phase 1)

---

## 1. Purpose

`ReactiveAgent` is the simplest possible agent: it holds a static table of
keyword → response pairs and fires the first match.  There is no learning,
no history, and no external service calls.  Response time is O(n) in the
number of rules and is effectively instantaneous.

Primary use-case: **first-contact customer-support triage** where a handful
of high-frequency questions can be deflected without human intervention or
an LLM call.

---

## 2. Class Signature

```python
class ReactiveAgent(BaseAgent):
    def __init__(
        self,
        name: str = "SupportBot",
        goal: str = "Answer customer queries instantly using predefined rules",
        rules: dict[str, str] | None = None,
        fallback: str = DEFAULT_FALLBACK,
    ) -> None
```

---

## 3. Module-Level Constants

| Constant | Type | Description |
|----------|------|-------------|
| `DEFAULT_RULES` | `dict[str, str]` | Five built-in keyword → response pairs (price, refund, delivery, password, hours). |
| `DEFAULT_FALLBACK` | `str` | Response returned when no keyword matches. |

### Default Rule Table

| Keyword | Response |
|---------|----------|
| `"price"` | Pricing page link |
| `"refund"` | Refund timeline + ticket offer |
| `"delivery"` | Delivery timeline |
| `"password"` | Password reset link |
| `"hours"` | Business hours |

---

## 4. Instance Attributes

| Attribute | Type | Source | Description |
|-----------|------|--------|-------------|
| `rules` | `dict[str, str]` | Constructor arg or `DEFAULT_RULES` | Keyword → canned response map. |
| `fallback` | `str` | Constructor arg or `DEFAULT_FALLBACK` | Response when no rule fires. |
| `name` | `str` | Constructor arg | Inherited from `BaseAgent`. |
| `goal` | `str` | Constructor arg | Inherited from `BaseAgent`. |
| `memory` | `list[dict]` | `BaseAgent` | Episodic store appended on every `run()` call. |

---

## 5. Method Specifications

### `perceive(observation: Any) → str`

Normalises input to lower-case, stripped plain text.

```
input: Any  →  str(observation).lower().strip()
```

- Guarantees `decide()` always receives a consistent lowercase string.
- Non-string inputs (int, dict, etc.) are coerced via `str()`.

---

### `decide(perception: str) → str`

Linear scan of `self.rules`; returns the first matching response.

```
for keyword, response in self.rules.items():
    if keyword in perception:
        return response
return self.fallback
```

**Matching semantics:** substring match (case-insensitive because `perceive`
already lowercased the input).  Rule order is insertion order (Python 3.7+
dict guarantee) — put higher-priority rules first.

**Edge cases:**

| Scenario | Behaviour |
|----------|-----------|
| Multiple keywords match | First match wins (dict insertion order). |
| No keyword matches | Returns `self.fallback`. |
| Empty input | Returns `self.fallback`. |

---

### `act(action: str) → str`

Identity pass-through — the chosen response string is returned unchanged.

```python
def act(self, action: str) -> str:
    return action
```

`act()` is the hook for side-effects in richer subclasses (e.g., logging
the response to a CRM, sending an SMS).  In `ReactiveAgent` it is
intentionally a no-op to keep the class stateless.

---

## 6. Data Flow

```
User message (Any)
   │
   ▼  perceive()
"lowercase stripped string"
   │
   ▼  decide()
"canned response string"   (first matching rule, or fallback)
   │
   ▼  act()
"canned response string"   (unchanged)
   │
   ▼
Caller receives response; memory entry appended
```

---

## 7. Dependencies

| Import | Used for |
|--------|----------|
| `typing.Any` | perceive signature |
| `.base_agent.BaseAgent` | Inheritance |

No third-party dependencies.

---

## 8. Configuration & Customisation

### Replacing the rule table

```python
my_rules = {
    "invoice":  "Invoices are emailed within 24 hours of purchase.",
    "trial":    "You can start a 14-day free trial at /trial.",
}
agent = ReactiveAgent(rules=my_rules)
```

### Custom fallback

```python
agent = ReactiveAgent(
    fallback="I'm still learning! Please email support@example.com."
)
```

### Extending at runtime

```python
agent.rules["cancel"] = "To cancel your subscription visit /account/cancel."
```

---

## 9. Extension Points

- **Subclass and override `act()`** to add logging, CRM writes, or analytics.
- **Override `decide()`** to support regex patterns instead of substring matching.
- **Chain agents:** use `ReactiveAgent` as a fast pre-filter; fall through to `LLMAgent` when the response equals `fallback`.

---

## 10. Performance Characteristics

| Metric | Value |
|--------|-------|
| Latency per `run()` | < 1 ms (no I/O) |
| Memory per instance | Negligible (dict + list) |
| Scalability | Linear in rule count — add thousands of rules with no practical impact |

---

## 11. Limitations

- Exact keyword substring matching only — no stemming, synonyms, or semantic similarity.
- Rules are evaluated in declaration order; complex priority logic requires manual ordering.
- Stateless between turns — cannot handle multi-turn conversations (e.g., "which product?" → "the one I mentioned earlier").
- Not suitable for open-ended queries; hand off to `LLMAgent` for those.

---

## 12. Usage Example

```python
from agents import ReactiveAgent

agent = ReactiveAgent()

print(agent.run("What is the price?"))
# "Our pricing page is at /pricing. Can I help with anything else?"

print(agent.run("I need a refund"))
# "Refunds are processed within 5–7 business days. Shall I open a ticket?"

print(agent.run("Tell me a joke"))
# "I'm not sure about that. Would you like me to connect you with a human agent?"
```
