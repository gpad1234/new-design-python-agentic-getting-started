"""
Reactive Agent  (Step 2 — "Reactive agents: respond to immediate inputs")
-------------------------------------------------------------------------
This agent uses a rule table to respond instantly to inputs.
It does NOT learn over time — each response depends only on the
current observation, not on history.

Example use-case: a simple customer-support triage bot.
"""

from typing import Any
from .base_agent import BaseAgent


# Rule table: keyword → response
DEFAULT_RULES: dict[str, str] = {
    "price":     "Our pricing page is at /pricing. Can I help with anything else?",
    "refund":    "Refunds are processed within 5–7 business days. Shall I open a ticket?",
    "delivery":  "Standard delivery takes 3–5 days. Express options are available at checkout.",
    "password":  "You can reset your password at /account/reset.",
    "hours":     "We are open Monday–Friday, 9 AM to 6 PM (EST).",
}
DEFAULT_FALLBACK = "I'm not sure about that. Would you like me to connect you with a human agent?"


class ReactiveAgent(BaseAgent):
    """A rule-based reactive agent that maps keywords to canned responses."""

    def __init__(
        self,
        name: str = "SupportBot",
        goal: str = "Answer customer queries instantly using predefined rules",
        rules: dict[str, str] | None = None,
        fallback: str = DEFAULT_FALLBACK,
    ) -> None:
        super().__init__(name, goal)
        self.rules: dict[str, str] = rules or DEFAULT_RULES
        self.fallback = fallback

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def perceive(self, observation: Any) -> str:
        """Normalize the raw text input."""
        return str(observation).lower().strip()

    def decide(self, perception: str) -> str:
        """Match the first keyword found in the user message."""
        for keyword, response in self.rules.items():
            if keyword in perception:
                return response
        return self.fallback

    def act(self, action: str) -> str:
        """'Acting' here means returning the chosen response string."""
        return action
