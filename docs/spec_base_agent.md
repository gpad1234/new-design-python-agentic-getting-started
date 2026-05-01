# Design Spec — `BaseAgent`

**File:** `agents/base_agent.py`  
**Type:** Abstract base class (ABC)  
**Phase:** Foundation (Phase 1)

---

## 1. Purpose

`BaseAgent` is the common contract every agent in this system must fulfil.
It encodes the classical **perceive → decide → act** loop that underpins all
agentic AI architectures.  Concrete subclasses implement the three abstract
methods; the base class wires them together in `run()` and provides shared
episodic memory.

---

## 2. Class Signature

```python
class BaseAgent(ABC):
    def __init__(self, name: str, goal: str) -> None
```

---

## 3. Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Human-readable identifier shown in logs and `__repr__`. |
| `goal` | `str` | One-sentence statement of purpose. Used by `LLMAgent` as a system prompt. |
| `memory` | `list[dict]` | Append-only episodic memory. Each entry: `{"observation", "action", "result"}`. |

---

## 4. Abstract Methods

### `perceive(observation: Any) → Any`
Transform raw environment input into a normalised internal representation.

- **Contract:** Must accept `Any` input and return something `decide()` can consume.
- **Typical implementation:** Lowercase/strip a string; parse a JSON payload; decode sensor bytes.

### `decide(perception: Any) → Any`
Select an action given the current perception.

- **Contract:** Stateless w.r.t. the base class (subclasses may maintain state).
- **Typical implementation:** Keyword lookup, model inference, LLM prompt construction.

### `act(action: Any) → Any`
Execute the chosen action and produce an observable result.

- **Contract:** Returns a value that will be stored in `memory` and returned to the caller.
- **Typical implementation:** Return a response string; call an API; write a file.

---

## 5. Concrete Methods

### `run(observation: Any) → Any`

The full agentic loop:

```
observation
   │
   ▼ perceive()
perception
   │
   ▼ decide()
action
   │
   ▼ act()
result ──► appended to self.memory ──► returned to caller
```

Memory entry schema:
```python
{"observation": <raw input>, "action": <chosen action>, "result": <output>}
```

### `__repr__() → str`
Returns `"ClassName(name='...', goal='...')"` for debugging.

---

## 6. Data Flow

```
Caller
  │  observation
  ▼
BaseAgent.run()
  ├─► perceive()   →  internal perception
  ├─► decide()     →  action descriptor
  ├─► act()        →  result
  └─► memory.append({...})
  │
  ▼  result
Caller
```

---

## 7. Dependencies

| Import | Used for |
|--------|----------|
| `abc.ABC`, `abc.abstractmethod` | Enforce subclass implementation |
| `typing.Any` | Flexible type hints for polymorphic interface |

No third-party dependencies.

---

## 8. Extension Points

- **Override `run()`** to add pre/post-processing (e.g., logging, rate-limiting) without touching `perceive`/`decide`/`act`.
- **Extend `memory`** into a richer object (e.g., timestamped, vector-embedded) by subclassing and overriding the append logic.
- **Add async variants** (`aperceive`, `adecide`, `aact`) for I/O-bound agents — this is planned in Phase 6.

---

## 9. Constraints & Limitations

- `memory` grows unboundedly; production subclasses should cap or flush it.
- The base class is synchronous only.
- No built-in thread safety; external locking is required for concurrent access.

---

## 10. Usage Example

```python
# You cannot instantiate BaseAgent directly — it is abstract.
# Minimal concrete subclass:
class EchoAgent(BaseAgent):
    def perceive(self, obs): return str(obs)
    def decide(self, perception): return f"Echo: {perception}"
    def act(self, action): return action

agent = EchoAgent(name="Echo", goal="Repeat everything")
print(agent.run("hello"))   # "Echo: hello"
print(agent.memory)         # [{"observation": "hello", "action": "Echo: hello", "result": "Echo: hello"}]
```
