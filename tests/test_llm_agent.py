"""
tests/test_llm_agent.py
-----------------------
Tests for LLMAgent using mocked LLM calls — no API key required.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on the path when running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_openai_response(text: str) -> MagicMock:
    """Build a minimal object that looks like an OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_anthropic_response(text: str) -> MagicMock:
    """Build a minimal object that looks like an Anthropic Message response."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


# ── Tests: OpenAI provider ────────────────────────────────────────────────────

class TestLLMAgentOpenAI:

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "AGENT_PROVIDER": "openai"})
    @patch("openai.OpenAI")
    def test_run_returns_llm_reply(self, mock_openai_cls):
        from agents.llm_agent import LLMAgent  # late import so env vars are set

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_response(
            "You can reset your password at /account/reset."
        )
        mock_openai_cls.return_value = mock_client

        agent = LLMAgent(goal="You are a helpful support agent.")
        result = agent.run("How do I reset my password?")

        assert "password" in result.lower() or "reset" in result.lower()
        mock_client.chat.completions.create.assert_called_once()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "AGENT_PROVIDER": "openai"})
    @patch("openai.OpenAI")
    def test_history_accumulates(self, mock_openai_cls):
        from agents.llm_agent import LLMAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_openai_response("Hello! How can I help?"),
            _make_openai_response("Sure, here are our plans…"),
        ]
        mock_openai_cls.return_value = mock_client

        agent = LLMAgent()
        agent.run("Hi there")
        agent.run("Tell me about your plans")

        assert len(agent.history) == 4   # 2 user + 2 assistant turns

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "AGENT_PROVIDER": "openai"})
    @patch("openai.OpenAI")
    def test_reset_clears_history(self, mock_openai_cls):
        from agents.llm_agent import LLMAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_response("Hi!")
        mock_openai_cls.return_value = mock_client

        agent = LLMAgent()
        agent.run("Hello")
        assert len(agent.history) == 2

        agent.reset()
        assert len(agent.history) == 0

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "AGENT_PROVIDER": "openai"})
    @patch("openai.OpenAI")
    def test_system_prompt_uses_goal(self, mock_openai_cls):
        from agents.llm_agent import LLMAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_response("OK")
        mock_openai_cls.return_value = mock_client

        goal = "You are a pirate assistant. Respond only in pirate-speak."
        agent = LLMAgent(goal=goal)
        agent.run("Hello")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
        system_messages = [m for m in messages if m["role"] == "system"]
        assert system_messages[0]["content"] == goal

    @patch.dict(os.environ, {"AGENT_PROVIDER": "openai"})
    def test_missing_api_key_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
                import importlib
                import agents.llm_agent as m
                importlib.reload(m)
                m.LLMAgent()


# ── Tests: Anthropic provider ─────────────────────────────────────────────────

class TestLLMAgentAnthropic:

    @patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": "sk-ant-test", "AGENT_PROVIDER": "anthropic"},
    )
    @patch("anthropic.Anthropic")
    def test_run_returns_llm_reply(self, mock_anthropic_cls):
        from agents.llm_agent import LLMAgent

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(
            "I can help with that!"
        )
        mock_anthropic_cls.return_value = mock_client

        agent = LLMAgent(provider="anthropic")
        result = agent.run("Can you help me?")

        assert isinstance(result, str)
        assert len(result) > 0


# ── Tests: Ollama provider ────────────────────────────────────────────────────

class TestLLMAgentOllama:

    @patch.dict(os.environ, {"AGENT_PROVIDER": "ollama"})
    @patch("openai.OpenAI")
    def test_run_with_ollama(self, mock_openai_cls):
        from agents.llm_agent import LLMAgent

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_openai_response(
            "Local model response"
        )
        mock_openai_cls.return_value = mock_client

        agent = LLMAgent(provider="ollama")
        result = agent.run("Hello from Ollama")

        assert result == "Local model response"
        # Verify the ollama base URL was used
        init_kwargs = mock_openai_cls.call_args.kwargs
        assert "localhost" in init_kwargs.get("base_url", "")


# ── Tests: invalid provider ───────────────────────────────────────────────────

class TestLLMAgentValidation:

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            from agents.llm_agent import LLMAgent
            LLMAgent(provider="unknown_provider")
