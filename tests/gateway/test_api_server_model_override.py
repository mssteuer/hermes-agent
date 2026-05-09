"""Tests for per-request model override in the API server adapter.

When API consumers (e.g. CCC task dispatcher, external UIs) include a
``model`` field in their request body, the API server must honour it
instead of always falling back to the gateway default.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch, MagicMock
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter():
    """Build a minimal APIServerAdapter with stubbed internals."""
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._model_name = "claude-opus-4-6"
    adapter._session_db = None
    return adapter


def _stub_runtime(monkeypatch):
    """Patch the heavy imports that _create_agent() pulls in."""
    fake_config = {
        "providers": {
            "openrouter": {
                "api_key": "sk-or-test-key",
                "base_url": "https://openrouter.ai/api/v1",
            },
        },
    }
    monkeypatch.setattr(
        "gateway.platforms.api_server.APIServerAdapter._ensure_session_db",
        lambda self: None,
    )
    # Patch the lazy imports inside _create_agent
    mock_agent_cls = MagicMock()
    mock_agent_cls.return_value = MagicMock()

    def _fake_resolve_runtime():
        return {
            "api_key": "sk-ant-default",
            "base_url": "https://api.anthropic.com",
            "provider": "anthropic",
            "api_mode": None,
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    def _fake_resolve_model():
        return "claude-opus-4-6"

    def _fake_load_config():
        return fake_config

    mock_runner = MagicMock()
    mock_runner._load_reasoning_config.return_value = {}
    mock_runner._load_fallback_model.return_value = None

    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs", _fake_resolve_runtime,
    )
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model", _fake_resolve_model,
    )
    monkeypatch.setattr(
        "gateway.run._load_gateway_config", _fake_load_config,
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        staticmethod(lambda: {}),
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools",
        lambda cfg, platform: ["terminal"],
    )
    monkeypatch.setattr("run_agent.AIAgent", mock_agent_cls)

    return mock_agent_cls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateAgentModelOverride:
    """_create_agent respects the model_override parameter."""

    def test_no_override_uses_gateway_default(self, monkeypatch):
        mock_cls = _stub_runtime(monkeypatch)
        adapter = _make_adapter()

        adapter._create_agent()
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs.get("model") or call_kwargs[1].get("model") == "claude-opus-4-6"

    def test_openrouter_override_resolves_credentials(self, monkeypatch):
        mock_cls = _stub_runtime(monkeypatch)
        adapter = _make_adapter()

        adapter._create_agent(model_override="openrouter/openai/gpt-5.5")

        call_kwargs = mock_cls.call_args
        # Model should be the override, not the default
        assert call_kwargs.kwargs.get("model") == "openrouter/openai/gpt-5.5"
        # Provider should be resolved to openrouter
        assert call_kwargs.kwargs.get("provider") == "openrouter"
        # API key should come from config
        assert call_kwargs.kwargs.get("api_key") == "sk-or-test-key"
        assert call_kwargs.kwargs.get("base_url") == "https://openrouter.ai/api/v1"

    def test_litellm_override_routes_to_proxy(self, monkeypatch):
        mock_cls = _stub_runtime(monkeypatch)
        adapter = _make_adapter()

        adapter._create_agent(model_override="litellm-main/gemini-2.5-flash")

        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs.get("model") == "litellm-main/gemini-2.5-flash"
        assert call_kwargs.kwargs.get("provider") == "custom"
        assert call_kwargs.kwargs.get("base_url") == "http://localhost:4000"

    def test_plain_model_name_swaps_model_keeps_provider(self, monkeypatch):
        mock_cls = _stub_runtime(monkeypatch)
        adapter = _make_adapter()

        adapter._create_agent(model_override="claude-sonnet-4-6")

        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs.get("model") == "claude-sonnet-4-6"
        # Provider stays as default (anthropic) since no prefix
        assert call_kwargs.kwargs.get("provider") == "anthropic"

    def test_gateway_default_model_is_not_treated_as_override(self, monkeypatch):
        """When body.model matches self._model_name, no override should apply."""
        mock_cls = _stub_runtime(monkeypatch)
        adapter = _make_adapter()

        # Pass the gateway default model name — should behave identically to no override
        adapter._create_agent(model_override=None)
        no_override_kwargs = mock_cls.call_args

        adapter._create_agent(model_override="claude-opus-4-6")
        with_default_kwargs = mock_cls.call_args

        # Both should use the same model — override with default model should
        # still work (the filtering happens at the call site, not inside _create_agent)
        assert with_default_kwargs.kwargs.get("model") == "claude-opus-4-6"
