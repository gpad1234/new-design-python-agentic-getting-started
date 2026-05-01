# Agentic AI — Project Plan

## Vision

A production-ready, modular agentic AI platform in Python that can handle real user interactions, use external tools, collaborate across multiple agents, and be deployed as a web service.

---

## Phase 1 — Foundation ✅ (complete)

**Goal:** Runnable getting-started project that demonstrates the core agentic loop.

| Deliverable | Status |
|-------------|--------|
| `BaseAgent` abstract class (perceive → decide → act) | ✅ |
| `ReactiveAgent` — rule-based, keyword routing | ✅ |
| `LearningAgent` — TF-IDF + Logistic Regression | ✅ |
| Sample labelled dataset | ✅ |
| CLI demo with session monitoring | ✅ |
| `requirements.txt`, `.gitignore`, `README.md` | ✅ |

---

## Phase 2 — LLM Backbone ✅ (complete)

**Goal:** Replace the sklearn classifier with a real language model so agents can handle open-ended queries.

| Deliverable | Status |
|-------------|--------|
| `LLMAgent` — OpenAI, Anthropic, Ollama | ✅ |
| System-prompt from `goal`, sliding-window history | ✅ |
| `python-dotenv` + `.env.example` | ✅ |
| `tests/test_llm_agent.py` — 8 mocked tests | ✅ |
| `main.py` demo with live / graceful-skip | ✅ |

---

## Phase 3 — Tool Use

**Goal:** Agents can call external tools (functions) to take real actions in the world.

### Tasks
- [ ] `ToolRegistry` — register, validate, and dispatch tools
- [ ] Built-in tools:
  - `web_search` — DuckDuckGo / Serper API
  - `calculator` — safe expression evaluator
  - `read_file` / `write_file` — sandboxed file I/O
  - `http_get` — fetch a URL
- [ ] `ToolUsingAgent` — ReAct-style loop (reason → act → observe → repeat)
- [ ] Tool call serialisation / deserialisation (JSON schema)
- [ ] Max iteration guard to prevent infinite loops
- [ ] Tool call logging for audit trail
- [ ] Add `tests/test_tools.py`

**New dependencies:** `duckduckgo-search`, `requests`

---

## Phase 4 — Multi-Agent System

**Goal:** Multiple specialised agents collaborate to handle complex tasks.

### Tasks
- [ ] `AgentOrchestrator` — routes tasks to the right agent
- [ ] Agent roles:
  - `PlannerAgent` — decomposes a goal into subtasks
  - `ExecutorAgent` — carries out individual subtasks using tools
  - `ReviewerAgent` — validates executor output and requests retries
- [ ] Inter-agent message passing (in-process queue)
- [ ] Shared short-term memory (dict) and long-term memory (vector store)
- [ ] Escalation path — route to human when confidence is low
- [ ] Add `tests/test_orchestrator.py`

**New dependencies:** `chromadb` or `faiss-cpu` (vector memory)

---

## Phase 5 — Persistence & Memory

**Goal:** Agents remember past interactions and learn from feedback.

### Tasks
- [ ] `VectorMemory` — embed and store interaction history in ChromaDB/FAISS
- [ ] `retrieve_relevant_context(query, k)` for RAG-style augmentation
- [ ] User session tracking (session ID → history)
- [ ] Feedback loop — thumbs up/down stored and used to re-rank responses
- [ ] Periodic model fine-tuning trigger (when N new feedback items arrive)
- [ ] Add `tests/test_memory.py`

**New dependencies:** `chromadb`, `sentence-transformers`

---

## Phase 6 — Web API

**Goal:** Expose agents as REST endpoints consumable by any frontend or service.

### Tasks
- [ ] FastAPI app (`api/app.py`)
- [ ] Endpoints:
  - `POST /chat` — send a message, get a response
  - `GET  /sessions/{id}` — retrieve session history
  - `POST /feedback` — submit rating for a response
  - `GET  /health` — liveness check
- [ ] Async request handling (all agents become async-compatible)
- [ ] Request validation with Pydantic models
- [ ] Rate limiting middleware
- [ ] OpenAPI docs auto-generated at `/docs`
- [ ] Add `tests/test_api.py` with `httpx` test client

**New dependencies:** `fastapi`, `uvicorn`, `httpx`, `pydantic`

---

## Phase 7 — Observability & Deployment

**Goal:** Production-ready: containerised, monitored, and continuously updated.

### Tasks
- [ ] `Dockerfile` + `docker-compose.yml`
- [ ] Structured JSON logging (`structlog`)
- [ ] Prometheus metrics endpoint (`/metrics`):
  - Request count, latency, error rate per agent
  - Intent distribution, confidence scores
- [ ] Grafana dashboard definition (JSON)
- [ ] MLflow experiment tracking for model versions
- [ ] GitHub Actions CI pipeline:
  - Lint (`ruff`), type-check (`mypy`), test (`pytest`)
  - Build and push Docker image
- [ ] Kubernetes `deployment.yaml` + `service.yaml` (optional)

**New dependencies:** `structlog`, `prometheus-fastapi-instrumentator`, `mlflow`

---

## Dependency roadmap

```
Phase 1  numpy, pandas, scikit-learn, matplotlib
Phase 2  + openai, anthropic, ollama, python-dotenv
Phase 3  + duckduckgo-search, requests
Phase 4  + chromadb (or faiss-cpu)
Phase 5  + sentence-transformers
Phase 6  + fastapi, uvicorn, httpx, pydantic
Phase 7  + structlog, prometheus-fastapi-instrumentator, mlflow
```

---

## Folder structure (target end-state)

```
getting-started/
├── agents/
│   ├── base_agent.py
│   ├── reactive_agent.py
│   ├── learning_agent.py
│   ├── llm_agent.py          # Phase 2
│   ├── tool_using_agent.py   # Phase 3
│   └── orchestrator.py       # Phase 4
├── tools/                    # Phase 3
│   ├── registry.py
│   ├── web_search.py
│   ├── calculator.py
│   └── file_io.py
├── memory/                   # Phase 5
│   └── vector_memory.py
├── api/                      # Phase 6
│   ├── app.py
│   └── models.py
├── data/
│   └── sample_data.py
├── tests/                    # grows each phase
├── main.py
├── requirements.txt
├── Dockerfile                # Phase 7
├── docker-compose.yml        # Phase 7
├── .env.example              # Phase 2
├── .gitignore
└── README.md
```
