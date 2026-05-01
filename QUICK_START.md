# Quick Start — 5 Minutes to Your First Agent

Get a working agentic AI system in 5 minutes.

---

## Prerequisites

- Python 3.12+
- `pip` (or `conda`)
- Terminal / Command line

---

## Step 1: Clone and Enter (30 seconds)

```bash
cd /Users/gp/claude-code/python-agentic/getting-started
```

---

## Step 2: Set Up Environment (1 minute)

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate          # macOS/Linux
# or
.venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Step 3: Run the Demo (2 minutes)

```bash
python main.py
```

You'll see:
1. **ReactiveAgent** — 5 rule-based responses (~1ms each)
2. **LearningAgent** — Intent classification demo (trained on 24 examples)
3. **LLMAgent** (optional) — Skipped if no API key, but you can try it
4. **Interactive CLI** — Type queries and see real-time responses

Try:
```
You: How do I reset my password?
Learning Agent: [provide_faq] Here is the FAQ article that should help: /faq

You: r: What is the price?
Reactive Agent: Our pricing page is at /pricing. Can I help with anything else?

You: quit
```

---

## Step 4 (Optional): Use a Real LLM (2 minutes)

To use OpenAI, Anthropic, or Ollama:

```bash
# Copy the template
cp .env.example .env

# Edit .env with your API key
# OPENAI_API_KEY=sk-...
# or ANTHROPIC_API_KEY=sk-ant-...
# or set AGENT_PROVIDER=ollama (requires local Ollama)

# Re-run
python main.py
```

In the CLI, use `l:` prefix:
```
You: l: What payment methods do you accept?
LLM Agent: We accept all major credit cards, PayPal, and bank transfers.
```

---

## Step 5: Run Tests (1 minute)

```bash
pytest tests/test_llm_agent.py -v
```

All tests pass without an API key (they use mocks).

---

## What Just Happened?

You ran three agent types:

| Agent | Speed | Use Case | Learn More |
|-------|-------|----------|-----------|
| **Reactive** | ~1ms | Fixed responses, FAQ routing | [agents/reactive_agent.py](agents/reactive_agent.py) |
| **Learning** | ~5ms | Intent classification, trained on data | [agents/learning_agent.py](agents/learning_agent.py) |
| **LLM** | 300–2000ms | Open-ended chat, reasoning | [agents/llm_agent.py](agents/llm_agent.py) |

---

## Next Steps

### Understand the Architecture
Read [ARCHITECTURE.md](ARCHITECTURE.md) for visual diagrams of the PDA loop, agent types, and full-stack design.

### Read the Tech Spec
[TECH_SPEC.md](TECH_SPEC.md) details component specs, data flows, security, and testing.

### See the Roadmap
[PROJECT_PLAN.md](PROJECT_PLAN.md) shows all 7 phases: where we are (Phases 1–2 done) and what's next.

### Customize the Demo
- Edit [data/sample_data.py](data/sample_data.py) to add training examples
- Edit [agents/reactive_agent.py](agents/reactive_agent.py) to change rules
- Change the LLM system prompt in [main.py](main.py) line 154

### Explore the Code
```
agents/
├── base_agent.py      # Abstract interface (perceive → decide → act)
├── reactive_agent.py  # Rule-based lookup
├── learning_agent.py  # sklearn classifier
└── llm_agent.py       # OpenAI / Anthropic / Ollama
```

---

## Troubleshooting

### ImportError: No module named 'openai'
Run `pip install -r requirements.txt` again.

### OPENAI_API_KEY error
Copy `.env.example` to `.env` and add your key, or just skip the LLM demo.

### Tests fail
Ensure you're in the virtual environment: `source .venv/bin/activate`

### Want to use Ollama?
1. Install Ollama: https://ollama.ai
2. Run a model: `ollama run llama3`
3. Set `AGENT_PROVIDER=ollama` in `.env`
4. Rerun `python main.py`

---

## What's Next?

Once you're comfortable with the basics:

- **Phase 3** — Add tool use (web search, file I/O) → ToolUsingAgent
- **Phase 4** — Multi-agent orchestration (planner → executor → reviewer)
- **Phase 5** — Vector memory (ChromaDB) for RAG
- **Phase 6** — REST API (FastAPI) for production
- **Phase 7** — Observability (Prometheus, MLflow) and deployment (Docker, K8s)

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full roadmap.

---

## Get Help

- **Code question?** Check [TECH_SPEC.md](TECH_SPEC.md) or docstrings in agent files
- **Architecture question?** See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Stuck?** Review [README.md](README.md) for project structure
