"""
Learning Agent  (Steps 3 & 4 — collect data, train a model)
------------------------------------------------------------
This agent trains a scikit-learn classifier on labelled examples,
then uses it to predict the best action for new observations.

Example use-case: a product-recommendation / intent-classification agent.

Lifecycle
---------
1. Collect labelled (observation, label) pairs  → agent.add_example()
2. Train                                         → agent.train()
3. Predict / act on new observations             → agent.run(obs)
4. Optionally add new examples and retrain       → continuous improvement
"""

from typing import Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .base_agent import BaseAgent


class LearningAgent(BaseAgent):
    """A supervised-learning agent that classifies text observations."""

    def __init__(
        self,
        name: str = "IntentClassifier",
        goal: str = "Classify user intent and route to the correct action",
    ) -> None:
        super().__init__(name, goal)
        self._examples: list[tuple[str, str]] = []   # (text, label)
        self._pipeline: Pipeline | None = None
        self._is_trained = False

    # ------------------------------------------------------------------
    # Step 3: Collect & prepare data
    # ------------------------------------------------------------------

    def add_example(self, text: str, label: str) -> None:
        """Add a labelled training example."""
        self._examples.append((text, label))
        self._is_trained = False   # model is stale after new data

    def add_examples(self, examples: list[tuple[str, str]]) -> None:
        """Bulk-add labelled training examples."""
        for text, label in examples:
            self.add_example(text, label)

    # ------------------------------------------------------------------
    # Step 4: Train the model
    # ------------------------------------------------------------------

    def train(self, test_size: float = 0.2, random_state: int = 42) -> dict:
        """
        Train a TF-IDF + Logistic Regression pipeline.

        Returns a dict with a classification report so callers can
        evaluate model accuracy (Step 4 — validate the model).
        """
        if len(self._examples) < 4:
            raise ValueError(
                "Need at least 4 labelled examples to train.  "
                "Call add_examples() first."
            )

        texts, labels = zip(*self._examples)

        # Split into train / test (Step 4 — split data)
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        # Build and fit pipeline (Step 4 — choose a model & train)
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf",   LogisticRegression(max_iter=1000, random_state=random_state)),
        ])
        self._pipeline.fit(X_train, y_train)
        self._is_trained = True

        # Evaluate (Step 4 — validate the model)
        y_pred = self._pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        return report

    # ------------------------------------------------------------------
    # BaseAgent interface  (Steps 5 & 6 — integrate and test)
    # ------------------------------------------------------------------

    def perceive(self, observation: Any) -> str:
        """Normalise raw input to a clean string."""
        return str(observation).strip()

    def decide(self, perception: str) -> str:
        """Predict the label (action) for the given text."""
        if not self._is_trained or self._pipeline is None:
            raise RuntimeError("Agent is not trained yet. Call train() first.")
        predictions = self._pipeline.predict([perception])
        return str(predictions[0])

    def act(self, action: str) -> dict:
        """
        Map a predicted label to a concrete response.
        Override this method to plug in real business logic.
        """
        responses = {
            "recommend_product":  "Based on your interest, we recommend our Premium Plan.",
            "escalate_to_human":  "Let me connect you with a specialist right away.",
            "provide_faq":        "Here is the FAQ article that should help: /faq",
            "collect_feedback":   "We'd love to hear from you — please fill in our survey.",
        }
        return {
            "predicted_intent": action,
            "response": responses.get(action, f"Handling intent: {action}"),
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return confidence scores for each class."""
        if not self._is_trained or self._pipeline is None:
            raise RuntimeError("Agent is not trained yet.")
        classes = self._pipeline.classes_
        probs = self._pipeline.predict_proba([text])[0]
        return dict(zip(classes, np.round(probs, 4)))
