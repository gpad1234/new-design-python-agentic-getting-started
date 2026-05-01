# Technical Specification — Agentic AI System

## 1. Overview

This document specifies the architecture, components, and interactions of a modular agentic AI platform built in Python. The system enables autonomous agents that can perceive environments, make decisions, and take actions—with optional learning, LLM backing, and tool use.

---

## 2. Core Architecture

### 2.1 Agentic Loop

Every agent implements the **perception-decision-action (PDA) loop**:

```
Environment Input
      ↓
   PERCEIVE (normalize, encode)
      ↓
   DECIDE (choose action)
      ↓
   ACT (execute, return result)
      ↓
   Store in MEMORY
      ↓
Return Result to Caller
```

**Key principle:** Agents are black boxes that accept any input and return any output. The base class only mandates the interface, not the implementation.

### 2.2 Component Hierarchy

```
BaseAgent (abstract)
├── ReactiveAgent     (Phase 1 — rule table)
├── LearningAgent     (Phase 1 — sklearn classifier)
├── LLMAgent          (Phase 2 — LLM backbone)
├── ToolUsingAgent    (Phase 3 — ReAct loop)
└── Orchestrator      (Phase 4 — multi-agent coordinator)
```

---

## 3. Agent Types

### 3.1 ReactiveAgent

**Purpose:** Instant, rule-based responses. No learning.

| Property | Value |
|----------|-------|
| Memory | Brief memory of past interactions (not used for decisions) |
| Training | None — configured at init |
| Latency | ~1 ms |
| Decisions | Keyword matching → lookup table |
| Use case | Customer support chatbot, FAQ router |

**Interface:**
- `perceive(obs)` → normalized text
- `decide(perception)` → rule lookup
- `act(action)` → return matched response

### 3.2 LearningAgent

**Purpose:** Learn patterns from labelled data. Predict intent / category on new inputs.

| Property | Value |
|----------|-------|
| Model | TF-IDF + Logistic Regression (sklearn) |
| Training | `add_examples()` → `train()` with sklearn |
| Accuracy | 60%+ on balanced datasets |
| Latency | ~5 ms |
| Confidence | `predict_proba()` for uncertainty |
| Use case | Intent classification, routing |

**Training Pipeline:**
1. Collect labelled `(text, intent)` pairs
2. Vectorize text (TF-IDF bigrams)
3. Train classifier (Logistic Regression, max_iter=1000)
4. Split into train/test, evaluate with classification_report
5. Deploy to production

### 3.3 LLMAgent (Phase 2)

**Purpose:** Open-ended responses using a large language model.

| Property | Value |
|----------|-------|
| Providers | OpenAI, Anthropic, Ollama |
| Models | gpt-4o, claude-3-5-sonnet-20241022, llama3, etc. |
| Context | Sliding-window conversation history (default: 10 turns) |
| Latency | 300-2000 ms (API dependent) |
| System prompt | `goal` parameter injected each call |
| Use case | General-purpose chat, writing, reasoning |

**Configuration:**
- Environment variables: `AGENT_PROVIDER`, `OPENAI_API_KEY`, etc.
- Fallback to defaults: OpenAI/gpt-4o if not specified
- Graceful error handling: if API key missing, agent raises `EnvironmentError`

### 3.4 ToolUsingAgent (Phase 3)

**Purpose:** Agents that can call external tools (functions) to act on the world.

**ReAct loop:**
```
User Input
   ↓
LLM Thinks (reason about next action)
   ↓
Tool Call (e.g., web_search, write_file)
   ↓
Observe Result
   ↓
Loop until done or max iterations
```

**Built-in tools:**
- `web_search` — query the web
- `calculator` — safe expression evaluation
- `read_file` / `write_file` — file I/O (sandboxed)
- `http_get` — fetch URLs

### 3.5 Orchestrator (Phase 4)

**Purpose:** Route complex tasks to multiple specialised agents.

**Agents in the orchestra:**
- **Planner** — decompose goal into subtasks
- **Executor** — carry out subtasks using tools
- **Reviewer** — validate and request retries

**Message passing:** In-process queue with shared episodic memory.

---

## 4. Memory System (Phase 5)

### 4.1 Episodic Memory

Every agent has `memory: list[dict]` that stores:
```python
{
    "observation": original_input,
    "action": chosen_action,
    "result": execution_result
}
```

**Used by:** Agents can inspect their own history for context (e.g., LLMAgent's sliding window).

### 4.2 Vector Memory (Phase 5)

**Technology:** ChromaDB or FAISS for semantic search.

**Data stored:**
- Interaction history (user queries + responses)
- User feedback (thumbs up/down)
- Long-term facts / context

**Query:** `retrieve_relevant_context(query, k=3)` returns k most similar past interactions.

---

## 5. API Layer (Phase 6)

### 5.1 REST Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat` | POST | Send a message, get a response |
| `/sessions/{id}` | GET | Retrieve session history |
| `/feedback` | POST | Rate a response (1-5 stars, comment) |
| `/health` | GET | Liveness probe |
| `/docs` | GET | OpenAPI specification (auto-generated) |
| `/metrics` | GET | Prometheus metrics |

### 5.2 Request/Response Models

```python
# Request
{
    "session_id": "user-123",
    "message": "What's your refund policy?",
    "agent_type": "learning" | "llm" | "reactive"  # optional
}

# Response
{
    "session_id": "user-123",
    "response": "Refunds are processed within 5-7 business days.",
    "confidence": 0.92,
    "intent": "provide_faq",  # learning agent only
    "model": "gpt-4o",  # llm agent only
    "reasoning": "..."  # llm agent only
}
```

---

## 6. Observability (Phase 7)

### 6.1 Logging

**Format:** Structured JSON (structlog)
```json
{
    "timestamp": "2026-04-30T10:15:30Z",
    "session_id": "user-123",
    "agent": "LLMAgent",
    "input": "refund policy",
    "output": "...",
    "latency_ms": 452,
    "model": "gpt-4o",
    "tokens_used": 45
}
```

### 6.2 Metrics

**Prometheus endpoints:**
- `agent_request_count{agent_type}` — total requests per agent
- `agent_latency_seconds{agent_type}` — request latency histogram
- `agent_error_count{agent_type}` — errors per agent
- `intent_distribution{intent}` — intent histogram (learning agent)
- `model_token_usage_total{model}` — tokens consumed (LLM agent)

### 6.3 Model Tracking

**MLflow:**
- Log experiment runs (train/test split, accuracy, hyperparams)
- Track model versions
- Stage models: staging → production
- Trigger retraining on new feedback

---

## 7. Deployment (Phase 7)

### 7.1 Containerisation

**Dockerfile:**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7.2 Kubernetes

**Deployment manifest:**
- Replicas: 3
- Resource limits: 1 CPU, 512 MB memory per pod
- Liveness probe: `/health` every 10s
- Readiness probe: `/health?ready=true`

### 7.3 CI/CD (GitHub Actions)

| Step | Tool | Purpose |
|------|------|---------|
| Lint | `ruff` | Code style |
| Type | `mypy` | Type checking |
| Test | `pytest` | Unit + integration tests |
| Build | Docker | Build image |
| Push | Docker Hub / ECR | Push to registry |
| Deploy | kubectl | Deploy to k8s |

---

## 8. Data Flows

### 8.1 Reactive Agent Flow

```
User: "What is the price?"
       ↓
   perceive() → "what is the price?"
       ↓
   decide() → lookup "price" in rules → "Our pricing page is at /pricing"
       ↓
   act() → return response
       ↓
   memory.append({observation, action, result})
       ↓
Return: "Our pricing page is at /pricing"
```

### 8.2 Learning Agent Flow

```
Training Phase:
  add_examples([("refund", "escalate_to_human"), ...])
       ↓
  train() → TF-IDF + LogisticRegression.fit(X_train, y_train)
       ↓
  evaluate() → classification_report on test set
       ↓
  is_trained = True

Prediction Phase:
  User: "I need a refund"
       ↓
  perceive() → "i need a refund"
       ↓
  decide() → pipeline.predict_proba(...) → {escalate_to_human: 0.92, ...}
       ↓
  act() → return mapped response
       ↓
  memory.append(...)
       ↓
Return: "Let me connect you with a specialist"
```

### 8.3 LLM Agent Flow

```
User: "How do I reset my password?"
       ↓
  perceive() → "How do I reset my password?"
       ↓
  decide() → build_messages([system_prompt, history..., user_message])
       ↓
  act() → openai.chat.completions.create(messages=...) → "Go to /account/reset"
       ↓
  history.append({role: "user", content: ...})
  history.append({role: "assistant", content: ...})
       ↓
Return: "Go to /account/reset"
```

### 8.4 Tool-Using Agent Flow (Phase 3)

```
User: "What is the current Bitcoin price?"
       ↓
  perceive() → "What is the current Bitcoin price?"
       ↓
  decide() → LLM thinks "I should search for Bitcoin price"
       ↓
  act() → tool_registry.call("web_search", query="Bitcoin price")
       ↓
  observe() → search result
       ↓
  loop? → LLM thinks "I have enough info" → return answer
       ↓
Return: "Bitcoin is trading at $67,000..."
```

---

## 9. Error Handling

### 9.1 Graceful Degradation

**LLM API fails:**
- Try with fallback model
- If all fail, fall back to learning agent
- If learning agent not trained, fall back to reactive agent

**Tool call fails:**
- Log error with traceback
- Return error message to user
- Increment error counter

### 9.2 Retry Logic

**LLM calls:** Exponential backoff (3 retries, base delay 1s)
**Tools:** Max 3 attempts before escalation

---

## 10. Security Considerations

### 10.1 API Keys

- Store in `.env` or environment variables (never in code)
- Use `python-dotenv` to load from `.env`
- Validate at agent init, fail fast if missing

### 10.2 File I/O Sandboxing

- Restrict file operations to a `data/` subdirectory
- Whitelist allowed file extensions
- Log all file access

### 10.3 Tool Call Validation

- Validate tool names against whitelist
- Validate arguments match schema
- Log all tool calls with arguments

---

## 11. Testing Strategy

| Level | Tool | Examples |
|-------|------|----------|
| Unit | pytest | Test `perceive()`, `decide()`, `act()` in isolation |
| Integration | pytest | Test full PDA loop with mocked I/O |
| Contract | pytest | Verify API responses match schema |
| E2E | pytest + httpx | Test full HTTP flow |

**Mocking:** Use `unittest.mock.patch` to mock LLM calls, API responses, DB queries.

---

## 12. Configuration & Environment

### 12.1 Environment Variables

```bash
# Agent
AGENT_PROVIDER=openai|anthropic|ollama
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3

# API
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://...

# Monitoring
PROMETHEUS_ENABLED=true
MLFLOW_TRACKING_URI=http://mlflow:5000
```

### 12.2 Config Files

- `.env` — secrets (git-ignored)
- `.env.example` — template for secrets
- `config.py` — app config (committed)

---

## 13. Success Criteria

| Phase | Metric | Target |
|-------|--------|--------|
| 1 | Tests pass | 100% |
| 2 | LLM latency | <1000 ms |
| 3 | Tool success rate | >95% |
| 4 | Multi-agent accuracy | >80% |
| 5 | RAG retrieval quality | >0.9 NDCG@5 |
| 6 | API latency p99 | <2000 ms |
| 7 | Uptime SLA | 99.9% |

---

## 14. Future Extensions

- [ ] Streaming responses (SSE)
- [ ] Voice input/output (Whisper + TTS)
- [ ] Vision capabilities (image understanding)
- [ ] Fine-tuning on user feedback (LoRA)
- [ ] Multi-language support
- [ ] Agentic loop with human-in-the-loop approval gates
