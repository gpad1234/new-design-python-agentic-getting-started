"""
Phase 2 — LLM Agent
--------------------
Provider-agnostic agent backed by a real language model.
Supports OpenAI, Anthropic, and local Ollama out of the box.

Configuration (via .env or environment variables):
    AGENT_PROVIDER   = openai | anthropic | ollama   (default: openai)
    OPENAI_API_KEY   = sk-...
    ANTHROPIC_API_KEY= sk-ant-...
    OLLAMA_BASE_URL  = http://localhost:11434         (default)
    OLLAMA_MODEL     = llama3                         (default)
    LLM_MODEL        = gpt-4o | claude-3-5-sonnet-20241022 | ...

Usage:
    from agents import LLMAgent
    agent = LLMAgent(goal="You are a helpful customer support agent.")
    print(agent.run("How do I reset my password?"))
"""

import os
from typing import Any

from .base_agent import BaseAgent

# Maximum number of conversation turns kept in context
DEFAULT_HISTORY_WINDOW = 10


class LLMAgent(BaseAgent):
    """An agent backed by a large language model."""

    def __init__(
        self,
        name: str = "LLMAgent",
        goal: str = "You are a helpful AI assistant.",
        provider: str | None = None,
        model: str | None = None,
        history_window: int = DEFAULT_HISTORY_WINDOW,
        temperature: float = 0.7,
    ) -> None:
        super().__init__(name, goal)
        self.provider = (provider or os.getenv("AGENT_PROVIDER", "openai")).lower()
        self.model = model or os.getenv("LLM_MODEL") or self._default_model()
        self.history_window = history_window
        self.temperature = temperature
        self._history: list[dict[str, str]] = []   # {"role": ..., "content": ...}
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def perceive(self, observation: Any) -> str:
        return str(observation).strip()

    def decide(self, perception: str) -> list[dict]:
        """Build the messages list for the LLM call."""
        self._history.append({"role": "user", "content": perception})
        # Keep only the most recent turns to stay within context limits
        window = self._history[-(self.history_window * 2):]
        messages = [{"role": "system", "content": self.goal}] + window
        return messages

    def act(self, action: list[dict]) -> str:
        """Call the LLM and return the assistant reply."""
        reply = self._call_llm(action)
        self._history.append({"role": "assistant", "content": reply})
        return reply

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    def _default_model(self) -> str:
        defaults = {
            "openai":    "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
            "ollama":    os.getenv("OLLAMA_MODEL", "llama3"),
        }
        return defaults.get(self.provider, "gpt-4o")

    def _build_client(self) -> Any:
        if self.provider == "openai":
            return self._build_openai_client()
        if self.provider == "anthropic":
            return self._build_anthropic_client()
        if self.provider == "ollama":
            return self._build_ollama_client()
        raise ValueError(
            f"Unknown provider {self.provider!r}. "
            "Choose 'openai', 'anthropic', or 'ollama'."
        )

    def _build_openai_client(self) -> Any:
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openai package is required for the 'openai' provider. "
                "Run: pip install openai"
            ) from exc
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set."
            )
        return OpenAI(api_key=api_key)

    def _build_anthropic_client(self) -> Any:
        try:
            import anthropic  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for the 'anthropic' provider. "
                "Run: pip install anthropic"
            ) from exc
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set."
            )
        return anthropic.Anthropic(api_key=api_key)

    def _build_ollama_client(self) -> Any:
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openai package is required for the Ollama provider "
                "(Ollama exposes an OpenAI-compatible API). "
                "Run: pip install openai"
            ) from exc
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        # Ollama doesn't require a real key but the client needs a non-empty value
        return OpenAI(api_key="ollama", base_url=base_url)

    def _call_llm(self, messages: list[dict]) -> str:
        if self.provider in ("openai", "ollama"):
            return self._call_openai_compat(messages)
        if self.provider == "anthropic":
            return self._call_anthropic(messages)
        raise ValueError(f"Unsupported provider: {self.provider!r}")

    def _call_openai_compat(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def _call_anthropic(self, messages: list[dict]) -> str:
        # Anthropic separates the system prompt from the messages list
        system = next(
            (m["content"] for m in messages if m["role"] == "system"), self.goal
        )
        user_messages = [m for m in messages if m["role"] != "system"]
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=user_messages,
            temperature=self.temperature,
        )
        return response.content[0].text.strip()
