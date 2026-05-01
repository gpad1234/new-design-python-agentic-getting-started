"""
Step 2: Define Your AI Agent's Purpose
---------------------------------------
BaseAgent defines the common interface all agents must implement.
Every agent has:
  - a name / goal
  - a perceive() method  (sense the environment)
  - a decide()  method  (choose an action)
  - an act()    method  (execute that action)
  - a run()     method  (the full perception-decision-action loop)
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base class for all agentic AI systems."""

    def __init__(self, name: str, goal: str) -> None:
        self.name = name
        self.goal = goal
        self.memory: list[dict] = []   # simple episodic memory

    # ------------------------------------------------------------------
    # Core agentic loop
    # ------------------------------------------------------------------

    @abstractmethod
    def perceive(self, observation: Any) -> Any:
        """Process raw input from the environment."""

    @abstractmethod
    def decide(self, perception: Any) -> Any:
        """Choose an action given the current perception."""

    @abstractmethod
    def act(self, action: Any) -> Any:
        """Execute the chosen action and return a result."""

    def run(self, observation: Any) -> Any:
        """Full perception → decision → action loop."""
        perception = self.perceive(observation)
        action = self.decide(perception)
        result = self.act(action)
        # Store experience in memory for later learning
        self.memory.append(
            {"observation": observation, "action": action, "result": result}
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, goal={self.goal!r})"
