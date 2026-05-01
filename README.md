# Building Agentic AI Systems in Python — Getting Started

A hands-on starter project that follows the 7-step guide from  
**[Codewave: Building Agentic AI Systems in Python](https://codewave.com/insights/agentic-ai-systems-python-guide/)**.

## Documentation

| File | Purpose |
|------|---------|
| [QUICK_START.md](QUICK_START.md) | **Start here** — 5-minute setup guide (venv, install, run demo) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Visual architecture & flow diagrams (Mermaid) — PDA loop, agent types, full stack |
| [TECH_SPEC.md](TECH_SPEC.md) | Detailed technical specification — component design, data flows, security, testing |
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | 7-phase roadmap with deliverables and dependencies |

---

```
getting-started/
├── agents/
│   ├── base_agent.py      # Step 2 — abstract agent interface
│   ├── reactive_agent.py  # Step 2 — rule-based agent (no learning)
│   └── learning_agent.py  # Steps 3 & 4 — trains a scikit-learn model
├── data/
│   └── sample_data.py     # Step 3 — labelled training examples
├── main.py                # Steps 5–7 — integrate, test, monitor
└── requirements.txt
```

---

## The 7-step guide mapped to this code

| Step | Guide description | Where in code |
|------|-------------------|---------------|
| 1 | Set up Python environment | `requirements.txt` |
| 2 | Define agent purpose | `BaseAgent`, `ReactiveAgent` |
| 3 | Collect & prepare data | `data/sample_data.py`, `LearningAgent.add_examples()` |
| 4 | Train the model | `LearningAgent.train()` |
| 5 | Integrate into application | CLI loop in `main.py` |
| 6 | Test and improve | Assertions + confidence scores in `main.py` |
| 7 | Deploy and monitor | Session summary printed at exit |

---

## Quick start

```bash
# Step 1 — create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the demo
python main.py
```

---

## Agent types

### ReactiveAgent
Responds to user input using a keyword → response rule table.  
No training required. Fast, predictable, easy to extend.

```python
from agents import ReactiveAgent

bot = ReactiveAgent()
print(bot.run("What is the price?"))
```

### LearningAgent
Trains a TF-IDF + Logistic Regression pipeline on labelled text examples.  
Classifies new queries into intents and maps them to actions.

```python
from agents import LearningAgent
from data.sample_data import TRAINING_DATA

agent = LearningAgent()
agent.add_examples(TRAINING_DATA)
report = agent.train()          # returns sklearn classification_report dict
result = agent.run("I need a refund")
print(result["response"])
```

---

## Extending the project

- **Add more training data** — edit `data/sample_data.py` or load a CSV with `pandas`.
- **Swap the model** — replace `LogisticRegression` in `learning_agent.py` with
  `RandomForestClassifier`, `SGDClassifier`, or a neural model.
- **Add a web UI** — wrap `main.py` logic in a Flask or FastAPI endpoint.
- **Deploy** — containerise with Docker (`FROM python:3.12-slim`) and deploy to
  AWS, GCP, or Azure.
- **Monitor** — integrate Prometheus + Grafana to track intent distribution and
  response latency in production.
