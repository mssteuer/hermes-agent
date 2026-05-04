"""Regression tests for API server model_override passthrough.

Bug: CCC sends model='openrouter/openai/gpt-5.5' in the request body,
but the API server's _run_agent() was not receiving or applying it —
the agent was created with the gateway default model (e.g. claude-opus-4-6)
instead of the requested model.  This caused CCC task dispatches to burn
Opus tokens on mechanical coding work.

Fix: _create_agent() now accepts model_override, which overrides the
gateway default and resolves provider credentials when the model name
contains a recognized prefix (openrouter/, litellm-).
"""

import asyncio
import json
import os
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_GATEWAY_CONFIG = {
    "model": {"default": "claude-opus-4-6"},
    "providers": {
        "openrouter": {
            "api_key": "sk-or-test-key",
            "base_url": "https://openrouter.ai/api/v1",
        },
        "anthropic": {
            "api_key": "sk-ant-test-key",
        },
    },
    "platform_toolsets": {
        "api_server": ["web", "terminal", "file"],
    },
}


def _make_adapter(api_key: str = "test-key") -> APIServerAdapter:
    """Create an adapter matching the pattern from test_api_server.py."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    adapter = APIServerAdapter(config)
    adapter._model_name = "claude-opus-4-6"
    return adapter


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Create the aiohttp app with routes."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    # Runs endpoint
    if hasattr(adapter, "_handle_runs"):
        app.router.add_post("/v1/runs", adapter._handle_runs)
        app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


# ---------------------------------------------------------------------------
# Tests: _create_agent model override
# ---------------------------------------------------------------------------


class TestCreateAgentModelOverride:
    """Test that _create_agent respects model_override parameter."""

    def test_no_override_uses_gateway_default(self, adapter):
        """When no model_override is passed, agent uses gateway default."""
        captured = {}

        def _fake_agent(*args, **kwargs):
            captured.update(kwargs)
            agent = MagicMock()
            agent.tools = []
            return agent

        with patch("run_agent.AIAgent", side_effect=_fake_agent, create=True), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "anthropic", "api_key": "sk-ant-test-key",
                 "base_url": "", "api_mode": "messages",
             }), \
             patch("gateway.run._resolve_gateway_model", return_value="claude-opus-4-6"), \
             patch("gateway.run._load_gateway_config", return_value=MOCK_GATEWAY_CONFIG), \
             patch("gateway.run.GatewayRunner._load_fallback_model", return_value=None):
            adapter._create_agent()

        assert captured["model"] == "claude-opus-4-6"

    def test_override_with_openrouter_prefix(self, adapter):
        """When model_override has openrouter/ prefix, agent uses that model
        and resolves OpenRouter credentials."""
        captured = {}

        def _fake_agent(*args, **kwargs):
            captured.update(kwargs)
            agent = MagicMock()
            agent.tools = []
            return agent

        with patch("run_agent.AIAgent", side_effect=_fake_agent, create=True), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "anthropic", "api_key": "sk-ant-test-key",
                 "base_url": "", "api_mode": "messages",
             }), \
             patch("gateway.run._resolve_gateway_model", return_value="claude-opus-4-6"), \
             patch("gateway.run._load_gateway_config", return_value=MOCK_GATEWAY_CONFIG), \
             patch("gateway.run.GatewayRunner._load_fallback_model", return_value=None):
            adapter._create_agent(model_override="openrouter/openai/gpt-5.5")

        assert captured["model"] == "openrouter/openai/gpt-5.5"
        assert captured["provider"] == "openrouter"
        assert captured["api_key"] == "sk-or-test-key"
        assert captured["base_url"] == "https://openrouter.ai/api/v1"

    def test_override_with_litellm_prefix(self, adapter):
        """When model_override has litellm- prefix, routes to LiteLLM proxy."""
        captured = {}

        def _fake_agent(*args, **kwargs):
            captured.update(kwargs)
            agent = MagicMock()
            agent.tools = []
            return agent

        with patch("run_agent.AIAgent", side_effect=_fake_agent, create=True), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "anthropic", "api_key": "sk-ant-test-key",
                 "base_url": "", "api_mode": "messages",
             }), \
             patch("gateway.run._resolve_gateway_model", return_value="claude-opus-4-6"), \
             patch("gateway.run._load_gateway_config", return_value=MOCK_GATEWAY_CONFIG), \
             patch("gateway.run.GatewayRunner._load_fallback_model", return_value=None):
            adapter._create_agent(model_override="litellm-gemini/gemini-2.5-flash")

        assert captured["model"] == "litellm-gemini/gemini-2.5-flash"
        assert captured["provider"] == "custom"

    def test_override_plain_model_name(self, adapter):
        """When model_override is a plain name (no provider prefix),
        just overrides the model, keeps existing provider."""
        captured = {}

        def _fake_agent(*args, **kwargs):
            captured.update(kwargs)
            agent = MagicMock()
            agent.tools = []
            return agent

        with patch("run_agent.AIAgent", side_effect=_fake_agent, create=True), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "anthropic", "api_key": "sk-ant-test-key",
                 "base_url": "", "api_mode": "messages",
             }), \
             patch("gateway.run._resolve_gateway_model", return_value="claude-opus-4-6"), \
             patch("gateway.run._load_gateway_config", return_value=MOCK_GATEWAY_CONFIG), \
             patch("gateway.run.GatewayRunner._load_fallback_model", return_value=None):
            adapter._create_agent(model_override="claude-sonnet-4-20250514")

        assert captured["model"] == "claude-sonnet-4-20250514"
        # Provider stays anthropic since no prefix routing triggered
        assert captured["provider"] == "anthropic"


# ---------------------------------------------------------------------------
# Tests: Chat Completions passes model to _run_agent
# ---------------------------------------------------------------------------


class TestChatCompletionsModelPassthrough:
    """Test that /v1/chat/completions passes the body 'model' field through
    as model_override to _run_agent."""

    @pytest.fixture
    def app(self, adapter):
        return _create_app(adapter)

    @pytest.mark.asyncio
    async def test_model_from_body_passed_as_override(self, adapter, app):
        """Body model != gateway default → passed as model_override."""
        captured_kwargs = {}

        async def _mock_run_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return (
                {"final_response": "ok", "messages": [], "api_calls": 1},
                {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )

        with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openrouter/openai/gpt-5.5",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    headers={"Authorization": "Bearer test-key"},
                )
                assert resp.status == 200

        assert captured_kwargs.get("model_override") == "openrouter/openai/gpt-5.5"

    @pytest.mark.asyncio
    async def test_model_same_as_gateway_default_not_overridden(self, adapter, app):
        """Body model == gateway default → no override (None)."""
        captured_kwargs = {}

        async def _mock_run_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return (
                {"final_response": "ok", "messages": [], "api_calls": 1},
                {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )

        with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "claude-opus-4-6",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    headers={"Authorization": "Bearer test-key"},
                )
                assert resp.status == 200

        # When model matches gateway default, it should be None (no override)
        assert captured_kwargs.get("model_override") is None


# ---------------------------------------------------------------------------
# Tests: Spawn/Run endpoint passes model to _create_agent
# ---------------------------------------------------------------------------


class TestSpawnRunModelPassthrough:
    """Test that POST /v1/runs with a model field passes it as model_override."""

    @pytest.fixture
    def app(self, adapter):
        return _create_app(adapter)

    @pytest.mark.asyncio
    async def test_run_passes_model_override(self, adapter, app):
        """POST /v1/runs with a model field uses it as override."""
        captured_override = {}

        def _spy_create_agent(**kwargs):
            captured_override["model_override"] = kwargs.get("model_override")
            # Return a fake agent
            agent = MagicMock()
            agent.tools = []
            agent.run_conversation = MagicMock(return_value={
                "final_response": "done",
                "messages": [],
                "api_calls": 1,
            })
            agent.session_prompt_tokens = 100
            agent.session_completion_tokens = 50
            agent.session_total_tokens = 150
            return agent

        with patch.object(adapter, "_create_agent", side_effect=_spy_create_agent):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/v1/runs",
                    json={
                        "model": "openrouter/openai/gpt-5.5",
                        "input": "Write hello world",
                    },
                    headers={"Authorization": "Bearer test-key"},
                )
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Wait for the background task to complete
                for _ in range(30):
                    status_resp = await client.get(
                        f"/v1/runs/{run_id}",
                        headers={"Authorization": "Bearer test-key"},
                    )
                    status_data = await status_resp.json()
                    if status_data.get("status") == "completed":
                        break
                    await asyncio.sleep(0.1)

        assert captured_override.get("model_override") == "openrouter/openai/gpt-5.5"
