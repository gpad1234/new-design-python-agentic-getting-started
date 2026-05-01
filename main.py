"""
Main demo — Building Your First Agentic AI System: Step by Step
================================================================
This script walks through every step from the Codewave guide:
  Step 1 — Set up your Python environment  (see requirements.txt)
  Step 2 — Define your AI agent's purpose
  Step 3 — Collect and prepare data
  Step 4 — Train your AI model
  Step 5 — Integrate your agent into an application (CLI loop here)
  Step 6 — Test and improve
  Step 7 — Deploy and monitor  (monitoring metrics printed at exit)

Run:
    python main.py
"""

import sys
import time
import os
from collections import Counter

from dotenv import load_dotenv
load_dotenv()   # load .env if present

from agents import ReactiveAgent, LearningAgent
from data.sample_data import TRAINING_DATA

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Define agent purposes
# ──────────────────────────────────────────────────────────────────────────────

section("Step 2 — Reactive Agent  (rule-based, no learning)")

reactive = ReactiveAgent()
print(f"Agent created: {reactive}")

sample_queries = [
    "What is the price of your product?",
    "I need a refund for my last order",
    "When will my package arrive?",
    "Where can I find your opening hours?",
    "Can you help me with something random?",
]

for query in sample_queries:
    response = reactive.run(query)
    print(f"\n  User  : {query}")
    print(f"  Agent : {response}")


# ──────────────────────────────────────────────────────────────────────────────
# Steps 3 & 4: Collect data → train the learning agent
# ──────────────────────────────────────────────────────────────────────────────

section("Steps 3 & 4 — Learning Agent  (collect data → train → validate)")

learner = LearningAgent()
print(f"Agent created: {learner}")

print(f"\n  Loading {len(TRAINING_DATA)} labelled examples …")
learner.add_examples(TRAINING_DATA)

print("  Training model …")
t0 = time.perf_counter()
report = learner.train()
elapsed = time.perf_counter() - t0

accuracy = report.get("accuracy", 0.0)
print(f"  Training complete in {elapsed:.3f}s  |  Test accuracy: {accuracy:.0%}")

# Show per-class precision / recall
print("\n  Per-class metrics:")
for label, metrics in report.items():
    if isinstance(metrics, dict):
        print(
            f"    {label:<25} precision={metrics['precision']:.2f}  "
            f"recall={metrics['recall']:.2f}  "
            f"f1={metrics['f1-score']:.2f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Steps 5 & 6: Integrate + test  (a few quick assertions)
# ──────────────────────────────────────────────────────────────────────────────

section("Steps 5 & 6 — Integrate and test the learning agent")

test_cases = [
    ("I want to buy something affordable",  "recommend_product"),
    ("Please transfer me to a human",       "escalate_to_human"),
    ("How do I reset my password?",         "provide_faq"),
    ("I have feedback about the service",   "collect_feedback"),
]

passed = 0
for text, expected in test_cases:
    result = learner.run(text)
    predicted = result["predicted_intent"]
    ok = "✓" if predicted == expected else "✗"
    print(f'  {ok}  [{expected}]  "{text}"')
    if predicted == expected:
        passed += 1

print(f"\n  {passed}/{len(test_cases)} tests passed")

# Confidence scores for an ambiguous query
ambiguous = "I'd like some advice"
probs = learner.predict_proba(ambiguous)
print(f'\n  Confidence scores for "{ambiguous}":')
for intent, prob in sorted(probs.items(), key=lambda x: -x[1]):
    bar = "█" * int(prob * 20)
    print(f"    {intent:<25} {prob:.4f}  {bar}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: LLM Agent demo  (skipped when no API key is configured)
# ──────────────────────────────────────────────────────────────────────────────

_llm_agent = None
_llm_available = bool(
    os.getenv("OPENAI_API_KEY")
    or os.getenv("ANTHROPIC_API_KEY")
    or os.getenv("AGENT_PROVIDER", "").lower() == "ollama"
)

if _llm_available:
    section("Phase 2 — LLM Agent  (live API call)")
    try:
        from agents import LLMAgent
        _llm_agent = LLMAgent(goal="You are a concise customer-support agent. Answer in one sentence.")
        print(f"  Agent created: {_llm_agent}")
        print(f"  Provider: {_llm_agent.provider}  |  Model: {_llm_agent.model}\n")

        llm_queries = [
            "What payment methods do you accept?",
            "Can I cancel my subscription at any time?",
        ]
        for q in llm_queries:
            reply = _llm_agent.run(q)
            print(f"  User  : {q}")
            print(f"  LLM   : {reply}\n")
    except Exception as exc:
        print(f"  [skipped — {exc}]\n")
else:
    section("Phase 2 — LLM Agent  (skipped — no API key in .env)")
    print("  Copy .env.example to .env, add your key, and re-run to try the LLM agent.\n")

# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Interactive CLI loop  (Ctrl-C or "quit" to exit)
# ──────────────────────────────────────────────────────────────────────────────

section("Step 5 — Interactive CLI  (type 'quit' to exit)")

_llm_hint = "  Prefix 'l:' to use the LLM agent." if _llm_available else ""
print("  Try queries like: 'I want a recommendation', 'talk to a human', 'how to reset password'")
print(f"  Prefix 'r:' to use the reactive agent.{_llm_hint}\n")

stats: Counter = Counter()

try:
    while True:
        try:
            user_input = input("  You: ").strip()
        except EOFError:
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            break

        if user_input.startswith("r:"):
            # Reactive agent path
            raw = user_input[2:].strip()
            response = reactive.run(raw)
            print(f"  ReactiveAgent: {response}\n")
            stats["reactive"] += 1
        elif user_input.startswith("l:") and _llm_agent:
            # LLM agent path
            raw = user_input[2:].strip()
            try:
                response = _llm_agent.run(raw)
                print(f"  LLMAgent: {response}\n")
                stats["llm"] += 1
            except Exception as exc:
                print(f"  LLMAgent error: {exc}\n")
        else:
            # Learning agent path
            result = learner.run(user_input)
            print(f"  LearningAgent [{result['predicted_intent']}]: {result['response']}\n")
            stats["learning"] += 1

except KeyboardInterrupt:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Step 7: Monitor — print a simple session summary
# ──────────────────────────────────────────────────────────────────────────────

section("Step 7 — Session monitoring summary")

total_reactive  = len(reactive.memory)
total_learning  = len(learner.memory)
intent_counts: Counter = Counter(
    entry["action"]["predicted_intent"]
    for entry in learner.memory
    if isinstance(entry.get("action"), dict)
)

print(f"  ReactiveAgent interactions : {total_reactive}")
print(f"  LearningAgent interactions : {total_learning}")
if intent_counts:
    print("\n  Intent distribution (learning agent):")
    for intent, count in intent_counts.most_common():
        print(f"    {intent:<25} {count}")

print("\n  Done. Next step: containerise with Docker and deploy to your cloud of choice.\n")
