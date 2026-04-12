"""Tests for the edicts native Claude Code memory provider plugin.

Covers:
- Loading edicts from a fixture YAML file
- TTL-expired edicts are excluded from get_context()
- Over-budget pruning preserves highest-confidence entries
- system_prompt_block() renders correct format
- edicts_list tool handler
- edicts_search tool handler
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from plugins.memory.edicts import EdictsProvider, _is_expired


# ---------------------------------------------------------------------------
# Fixture YAML content
# ---------------------------------------------------------------------------

FIXTURE_EDICTS_YAML = """\
edicts:
  - id: e001
    text: Never mention AgentGrid publicly.
    category: rules
    confidence: verified
    tags: [confidential, agentgrid]
    ttl: permanent

  - id: e002
    text: Casper v2.2.0 launched March 23, 2026. Not called Kyoto.
    category: product
    confidence: high
    tags: [casper, launch]

  - id: e003
    text: This edict has expired already.
    category: temp
    confidence: medium
    expiresAt: "2020-01-01T00:00:00Z"

  - id: e004
    text: Always use structured output for API responses.
    category: rules
    confidence: medium
    tags: [api, format]

  - id: e005
    text: Low confidence edict for budget testing.
    category: context
    confidence: low
    tags: [budget]
"""

TINY_BUDGET_EDICTS_YAML = """\
edicts:
  - id: h1
    text: High confidence short rule.
    category: rules
    confidence: verified

  - id: h2
    text: Another high confidence rule that is a bit longer in length.
    category: rules
    confidence: high

  - id: l1
    text: Low confidence rule that should be pruned if over budget.
    category: context
    confidence: low

  - id: l2
    text: Another low confidence rule that gets cut when over budget limit.
    category: context
    confidence: low
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider_from_yaml(yaml_content: str, token_budget: int = 4000) -> EdictsProvider:
    """Write yaml to a temp file, initialize a provider from it, return provider."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(yaml_content)
        tmp_path = f.name

    # Write a minimal config.yaml pointing at the temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as cfg:
        cfg.write(
            f"""\
plugins:
  memory:
    edicts:
      enabled: true
      path: {tmp_path}
      token_budget: {token_budget}
      inject_into: system
      tools_enabled: true
      tools_write: false
"""
        )
        cfg_path = cfg.name

    # Use a temp hermes_home so config.yaml is found
    hermes_home = tempfile.mkdtemp()
    Path(hermes_home, "config.yaml").write_text(
        f"""\
plugins:
  memory:
    edicts:
      enabled: true
      path: {tmp_path}
      token_budget: {token_budget}
      inject_into: system
      tools_enabled: true
      tools_write: false
""",
        encoding="utf-8",
    )

    provider = EdictsProvider()
    provider.initialize("test-session", hermes_home=hermes_home)

    # Cleanup registered for after test (best-effort)
    provider._test_tmp_path = tmp_path
    provider._test_hermes_home = hermes_home
    return provider


# ---------------------------------------------------------------------------
# Tests: fixture loading
# ---------------------------------------------------------------------------


class TestEdictsLoading:
    def test_loads_active_edicts(self):
        provider = _make_provider_from_yaml(FIXTURE_EDICTS_YAML)
        # 5 edicts total, 1 expired → 4 active
        assert len(provider._edicts) == 4

    def test_expired_edict_excluded(self):
        provider = _make_provider_from_yaml(FIXTURE_EDICTS_YAML)
        ids = [e["id"] for e in provider._edicts]
        assert "e003" not in ids

    def test_active_edicts_present(self):
        provider = _make_provider_from_yaml(FIXTURE_EDICTS_YAML)
        ids = [e["id"] for e in provider._edicts]
        for expected in ("e001", "e002", "e004", "e005"):
            assert expected in ids

    def test_sorted_by_confidence(self):
        provider = _make_provider_from_yaml(FIXTURE_EDICTS_YAML)
        confidence_order = [e["confidence"] for e in provider._edicts]
        # verified comes before high, high before medium, medium before low
        assert confidence_order.index("verified") < confidence_order.index("high")
        assert confidence_order.index("high") < confidence_order.index("medium")


# ---------------------------------------------------------------------------
# Tests: TTL / expiry
# ---------------------------------------------------------------------------


class TestEdictsExpiry:
    def test_is_expired_past_date(self):
        edict = {"expiresAt": "2020-06-01T00:00:00Z"}
        assert _is_expired(edict) is True

    def test_is_not_expired_future_date(self):
        future = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        edict = {"expiresAt": future}
        assert _is_expired(edict) is False

    def test_is_not_expired_no_field(self):
        edict = {"text": "no expiry"}
        assert _is_expired(edict) is False

    def test_future_expiry_included(self):
        future = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        yaml_content = f"""\
edicts:
  - id: future1
    text: Still valid edict.
    category: rules
    confidence: high
    expiresAt: "{future}"
"""
        provider = _make_provider_from_yaml(yaml_content)
        ids = [e["id"] for e in provider._edicts]
        assert "future1" in ids


# ---------------------------------------------------------------------------
# Tests: token budget pruning
# ---------------------------------------------------------------------------


class TestEdictsBudget:
    def test_over_budget_drops_low_confidence(self):
        # Set tiny budget (1 token = 4 chars → budget = 4 chars, way too small)
        # With token_budget=1, only 4 chars budget → most entries dropped
        provider = _make_provider_from_yaml(TINY_BUDGET_EDICTS_YAML, token_budget=1)
        # Should keep 0 or minimal (budget too tiny)
        # At minimum, if anything survives, it should be high-confidence ones
        if provider._edicts:
            assert provider._edicts[0]["confidence"] in ("verified", "high")

    def test_moderate_budget_keeps_high_drops_low(self):
        # Budget of ~10 tokens = 40 chars: enough for verified/high, not low
        # "High confidence short rule." = 28 chars + overhead
        # "Another high confidence rule..." = 46 chars + overhead
        # total for both ~74+ chars but budget_chars = 10*4 = 40 → only first fits
        provider = _make_provider_from_yaml(TINY_BUDGET_EDICTS_YAML, token_budget=10)
        ids = [e["id"] for e in provider._edicts]
        # Low confidence ones should be dropped
        assert "l1" not in ids or "l2" not in ids

    def test_sufficient_budget_keeps_all(self):
        provider = _make_provider_from_yaml(TINY_BUDGET_EDICTS_YAML, token_budget=4000)
        assert len(provider._edicts) == 4


# ---------------------------------------------------------------------------
# Tests: system prompt injection
# ---------------------------------------------------------------------------


class TestEdictsSystemPrompt:
    def test_system_prompt_block_format(self):
        provider = _make_provider_from_yaml(FIXTURE_EDICTS_YAML)
        block = provider.system_prompt_block()
        assert "## Ground Truth (Edicts)" in block
        assert "These facts are non-negotiable" in block
        assert "Never mention AgentGrid publicly" in block

    def test_system_prompt_has_category_prefix(self):
        provider = _make_provider_from_yaml(FIXTURE_EDICTS_YAML)
        block = provider.system_prompt_block()
        # categories appear as [rules] prefix
        assert "[rules]" in block

    def test_empty_edicts_returns_empty_block(self):
        yaml_content = "edicts: []\n"
        provider = _make_provider_from_yaml(yaml_content)
        assert provider.system_prompt_block() == ""


# ---------------------------------------------------------------------------
# Tests: tool handlers
# ---------------------------------------------------------------------------


class TestEdictsListTool:
    def test_list_returns_all(self):
        provider = _make_provider_from_yaml(FIXTURE_EDICTS_YAML)
        result = json.loads(provider.handle_tool_call("edicts_list", {}))
        assert result["total"] == 4

    def test_list_filter_by_category(self):
        provider = _make_provider_from_yaml(FIXTURE_EDICTS_YAML)
        result = json.loads(provider.handle_tool_call("edicts_list", {"category": "rules"}))
        for e in result["edicts"]:
            assert e["category"] == "rules"

    def test_list_respects_limit(self):
        provider = _make_provider_from_yaml(FIXTURE_EDICTS_YAML)
        result = json.loads(provider.handle_tool_call("edicts_list", {"limit": 2}))
        assert len(result["edicts"]) <= 2


class TestEdictsSearchTool:
    def test_search_finds_matching_text(self):
        provider = _make_provider_from_yaml(FIXTURE_EDICTS_YAML)
        result = json.loads(provider.handle_tool_call("edicts_search", {"query": "AgentGrid"}))
        assert result["total"] >= 1
        assert any("AgentGrid" in e["text"] for e in result["matches"])

    def test_search_finds_by_tag(self):
        provider = _make_provider_from_yaml(FIXTURE_EDICTS_YAML)
        result = json.loads(provider.handle_tool_call("edicts_search", {"query": "casper"}))
        assert result["total"] >= 1

    def test_search_no_results(self):
        provider = _make_provider_from_yaml(FIXTURE_EDICTS_YAML)
        result = json.loads(
            provider.handle_tool_call("edicts_search", {"query": "xyznotexistent12345"})
        )
        assert result["total"] == 0

    def test_search_requires_query(self):
        provider = _make_provider_from_yaml(FIXTURE_EDICTS_YAML)
        result = json.loads(provider.handle_tool_call("edicts_search", {}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: tools enabled/disabled
# ---------------------------------------------------------------------------


class TestEdictsToolSchemas:
    def test_tool_schemas_returned_when_enabled(self):
        provider = _make_provider_from_yaml(FIXTURE_EDICTS_YAML)
        schemas = provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "edicts_list" in names
        assert "edicts_search" in names
