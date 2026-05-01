# Design Spec — `LLMAgent`

**File:** `agents/llm_agent.py`  
**Type:** Concrete class, extends `BaseAgent`  
**Phase:** LLM Backbone (Phase 2)

---

## 1. Purpose

`LLMAgent` wraps a real large language model and exposes it through the
standard `BaseAgent` interface.  It is **provider-agnostic**: the same class
drives OpenAI, Anthropic, and local Ollama by reading a single environment
variable.

Key capabilities beyond `ReactiveAgent` / `LearningAgent`:
- Open-ended, generative responses (not limited to a fixed label or rule set).
- Multi-turn conversation with a sliding-window history buffer.
- System prompt derived from the agent's `goal`, allowing persona / behaviour
  customisation without code changes.

---

## 2. Class Signature

```python
class LLMAgent(BaseAgent):
    def __init__(
        self,
        name: str = "LLMAgent",
        goal: str = "You are a helpful AI assistant.",
        provider: str | None = None,
        model: str | None = None,
        history_window: int = 10,
        temperature: float = 0.7,
    ) -> None
```

---

## 3. Module-Level Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_HISTORY_WINDOW` | `10` | Number of conversation *turns* (user + assistant pairs) kept in context. |

---

## 4. Instance Attributes

| Attribute | Type | Source | Description |
|-----------|------|--------|-------------|
| `name` | `str` | Constructor | Inherited. |
| `goal` | `str` | Constructor | Becomes the LLM system prompt. |
| `provider` | `str` | Constructor or `AGENT_PROVIDER` env var | `"openai"` \| `"anthropic"` \| `"ollama"`. |
| `model` | `str` | Constructor or `LLM_MODEL` env var | Model identifier (e.g. `"gpt-4o"`). |
| `history_window` | `int` | Constructor | Max conversation turns kept; older turns are dropped. |
| `temperature` | `float` | Constructor | Sampling temperature passed to the LLM API. |
| `_history` | `list[dict]` | Internal | `[{"role": "user"\|"assistant", "content": str}, ...]` |
| `_client` | `Any` | `_build_client()` | Provider SDK client (OpenAI / Anthropic / Ollama). |
| `memory` | `list[dict]` | `BaseAgent` | Episodic store appended on every `run()` call. |

---

## 5. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_PROVIDER` | `"openai"` | Selects the LLM provider. |
| `OPENAI_API_KEY` | *(required for openai)* | OpenAI secret key. |
| `ANTHROPIC_API_KEY` | *(required for anthropic)* | Anthropic secret key. |
| `LLM_MODEL` | Provider default (see below) | Model name override. |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API base URL. |
| `OLLAMA_MODEL` | `"llama3"` | Model name when using Ollama. |

### Provider Default Models

| Provider | Default model |
|----------|--------------|
| `openai` | `gpt-4o` |
| `anthropic` | `claude-3-5-sonnet-20241022` |
| `ollama` | value of `OLLAMA_MODEL` env var, else `llama3` |

---

## 6. Method Specifications

### `perceive(observation: Any) → str`

Strips the raw input to a clean string.

```python
return str(observation).strip()
```

---

### `decide(perception: str) → list[dict]`

Appends the user turn to `_history`, then builds the full messages list
to pass to the LLM.

**Sliding-window logic:**
```
kept_history = _history[-(history_window * 2):]
messages = [{"role": "system", "content": self.goal}] + kept_history
```

`history_window * 2` because each turn has two entries (user + assistant).

**Return value:** OpenAI-compatible messages list ready for the API call.

---

### `act(action: list[dict]) → str`

Calls the LLM via the appropriate provider and appends the assistant reply
to `_history`.

```python
reply = self._call_llm(action)
self._history.append({"role": "assistant", "content": reply})
return reply
```

---

### `reset() → None`

Clears `_history`, starting a fresh conversation while keeping the same
agent instance and system prompt.

---

### `history` (property)

Returns a shallow copy of `_history` as a read-only list.

---

## 7. Provider Dispatch Architecture

```
LLMAgent._build_client()
   ├─ "openai"    → OpenAI(api_key=OPENAI_API_KEY)
   ├─ "anthropic" → anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
   └─ "ollama"    → OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
                    (Ollama exposes an OpenAI-compatible REST API)

LLMAgent._call_llm(messages)
   ├─ provider in ("openai", "ollama") → _call_openai_compat()
   └─ provider == "anthropic"         → _call_anthropic()
```

### `_call_openai_compat(messages) → str`

```python
response = client.chat.completions.create(
    model=self.model,
    messages=messages,
    temperature=self.temperature,
)
return response.choices[0].message.content.strip()
```

### `_call_anthropic(messages) → str`

Anthropic's API separates the system prompt from user/assistant messages:

```python
system       = <first message with role "system">
user_messages = [m for m in messages if m["role"] != "system"]
response = client.messages.create(
    model=self.model,
    max_tokens=1024,
    system=system,
    messages=user_messages,
    temperature=self.temperature,
)
return response.content[0].text.strip()
```

---

## 8. Multi-Turn Conversation Data Flow

```
Turn 1
  User: "How do I reset my password?"
    perceive → "How do I reset my password?"
    decide   → appends user msg to _history
               builds: [system, {user: "How do I..."}]
    act      → calls LLM → "Visit /account/reset."
               appends assistant msg to _history

Turn 2
  User: "What if I forgot my email too?"
    perceive → "What if I forgot my email too?"
    decide   → appends user msg
               builds: [system, {user: T1}, {asst: T1}, {user: T2}]
    act      → calls LLM with full context → contextual reply
```

History is capped at `history_window * 2` entries before the system message
is prepended, keeping total token usage bounded.

---

## 9. Error Handling

| Condition | Exception | Message |
|-----------|-----------|---------|
| `openai` package missing | `ImportError` | Instructs `pip install openai` |
| `anthropic` package missing | `ImportError` | Instructs `pip install anthropic` |
| `OPENAI_API_KEY` not set | `EnvironmentError` | Variable name reminder |
| `ANTHROPIC_API_KEY` not set | `EnvironmentError` | Variable name reminder |
| Unknown provider string | `ValueError` | Lists valid providers |

---

## 10. Dependencies

| Package | Used for |
|---------|----------|
| `openai` | OpenAI + Ollama API clients |
| `anthropic` | Anthropic API client |
| `python-dotenv` | `.env` loading in `main.py` (not imported in this file) |
| `os` | Environment variable reads |
| `.base_agent.BaseAgent` | Inheritance |

Both `openai` and `anthropic` are optional at import time — `ImportError` is
raised lazily in `_build_client()` only if the missing provider is actually
requested.

---

## 11. Configuration via `.env`

```ini
# .env  (copy from .env.example)
AGENT_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o
```

---

## 12. Extension Points

- **Add a new provider:** implement `_build_<provider>_client()` and `_call_<provider>()`, then register in `_build_client()` / `_call_llm()`.
- **Stream responses:** override `act()` to use the streaming API and yield tokens.
- **Tool calling:** override `decide()` to inject tool definitions into the messages list; override `act()` to detect `tool_use` responses and dispatch them — the planned approach for Phase 3.
- **Structured output:** pass `response_format={"type": "json_object"}` in `_call_openai_compat()`.
- **Async I/O:** replace the sync client calls with `AsyncOpenAI` / `AsyncAnthropic` in an `async def arun()` override.

---

## 13. Limitations

- Synchronous only; one token at a time — not suitable for streaming UIs without modifications.
- `max_tokens=1024` is hardcoded in the Anthropic path; long responses may be truncated.
- History window trims by turn count, not by token count — very long turns can still exceed context limits.
- No retry logic or exponential back-off on API errors.

---

## 14. Usage Example

```python
import os
os.environ["AGENT_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "sk-..."

from agents import LLMAgent

agent = LLMAgent(goal="You are a concise technical support agent.")

print(agent.run("How do I install Python on macOS?"))
print(agent.run("What about Windows?"))   # references prior turn via history

agent.reset()   # start fresh
print(agent.history)  # []
```

### Ollama (local, no API key)

```python
import os
os.environ["AGENT_PROVIDER"] = "ollama"
os.environ["OLLAMA_MODEL"]   = "llama3"

from agents import LLMAgent
agent = LLMAgent()
print(agent.run("Explain closures in Python"))
```
