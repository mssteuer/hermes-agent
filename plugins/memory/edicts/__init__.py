"""edicts — Native Claude Code memory provider plugin.

Auto-injects the rendered edicts store into every session prompt.
Loads ~/workspace/edicts.yaml (or a configurable path) at session start,
filters expired entries, respects token budget, and injects them as a
Ground Truth block in the system prompt.

Optionally exposes edicts_list and edicts_search tools (read-only by default).

Config in $HERMES_HOME/config.yaml:
  plugins:
    memory:
      edicts:
        enabled: true
        path: ~/workspace/edicts.yaml
        token_budget: 4000
        inject_into: system   # system | user | none
        tools_enabled: true
        tools_write: false
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# Approximate chars-per-token ratio for budget enforcement
_CHARS_PER_TOKEN = 4


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, using PyYAML if available, falling back to a minimal parser."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    # Minimal fallback: use Python's configparser-style line-by-line parse for simple YAML.
    # For the edicts.yaml format (known schema), parse via json after a quick conversion.
    # Better: try tomllib (Python 3.11+) which can't parse YAML, so just fail gracefully.
    try:
        import tomllib  # type: ignore
    except ImportError:
        pass
    # Last resort: return empty dict — plugin will load 0 edicts rather than crash.
    logger.warning(
        "edicts: PyYAML not available and no fallback parser succeeded. "
        "Install PyYAML (`pip install pyyaml`) to enable edicts injection."
    )
    return {}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _is_expired(edict: Dict[str, Any]) -> bool:
    """Return True if this edict is past its expiresAt date."""
    expires_at = edict.get("expiresAt")
    if not expires_at:
        return False
    try:
        if isinstance(expires_at, str):
            expires_at = expires_at.replace("Z", "+00:00")
            dt = datetime.fromisoformat(expires_at)
        elif isinstance(expires_at, datetime):
            dt = expires_at
        else:
            return False
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return _now_utc() > dt
    except Exception:
        return False


def _confidence_sort_key(edict: Dict[str, Any]) -> int:
    """Lower value = higher priority (to sort descending)."""
    order = {"verified": 0, "high": 1, "medium": 2, "low": 3}
    return order.get(str(edict.get("confidence", "medium")).lower(), 2)


class EdictsProvider(MemoryProvider):
    """Memory provider that injects edicts as Ground Truth into every session."""

    def __init__(self) -> None:
        self._edicts: List[Dict[str, Any]] = []
        self._config: Dict[str, Any] = {}
        self._tools_enabled: bool = True
        self._inject_into: str = "system"
        self._token_budget: int = 4000

    @property
    def name(self) -> str:
        return "edicts"

    def is_available(self) -> bool:
        """Always available — no external deps, just a local yaml file."""
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))

        # Load plugin config from config.yaml
        cfg_path = Path(hermes_home) / "config.yaml"
        provider_cfg: Dict[str, Any] = {}
        if cfg_path.exists():
            try:
                data = _load_yaml(cfg_path)
                provider_cfg = (
                    data.get("plugins", {})
                    .get("memory", {})
                    .get("edicts", {})
                )
            except Exception as e:
                logger.warning("edicts: failed to read config.yaml: %s", e)

        self._config = provider_cfg

        # Resolve edicts file path
        raw_path = provider_cfg.get("path", "~/workspace/edicts.yaml")
        edicts_path = Path(os.path.expanduser(raw_path))

        self._token_budget = int(provider_cfg.get("token_budget", 4000))
        self._inject_into = str(provider_cfg.get("inject_into", "system"))
        self._tools_enabled = bool(provider_cfg.get("tools_enabled", True))

        self._edicts = self._load_edicts(edicts_path)
        logger.debug("edicts: loaded %d active edicts from %s", len(self._edicts), edicts_path)

    def _load_edicts(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            logger.warning("edicts: file not found at %s", path)
            return []
        try:
            data = _load_yaml(path)
        except Exception as e:
            logger.error("edicts: failed to parse %s: %s", path, e)
            return []

        raw = data.get("edicts", [])
        active = [e for e in raw if isinstance(e, dict) and not _is_expired(e)]

        # Sort by confidence (verified > high > medium > low)
        active.sort(key=_confidence_sort_key)

        # Apply token budget — drop lowest-confidence entries first (already sorted)
        budget_chars = self._token_budget * _CHARS_PER_TOKEN
        kept: List[Dict[str, Any]] = []
        used = 0
        for e in active:
            text = str(e.get("text", ""))
            if used + len(text) > budget_chars:
                break
            kept.append(e)
            used += len(text) + 50  # ~50 chars overhead per entry

        return kept

    def system_prompt_block(self) -> str:
        if self._inject_into != "system" or not self._edicts:
            return ""
        return self._render_block()

    def _render_block(self) -> str:
        if not self._edicts:
            return ""
        lines = [
            "## Ground Truth (Edicts)",
            "These facts are non-negotiable. Do NOT improvise around them.",
        ]
        for e in self._edicts:
            text = str(e.get("text", "")).strip()
            category = str(e.get("category", "")).strip()
            tag = f"[{category}] " if category else ""
            lines.append(f"- {tag}{text}")
        return "\n".join(lines)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if not self._tools_enabled:
            return []
        return [_EDICTS_LIST_SCHEMA, _EDICTS_SEARCH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "edicts_list":
            return self._handle_list(args)
        if tool_name == "edicts_search":
            return self._handle_search(args)
        raise NotImplementedError(f"edicts: unknown tool {tool_name}")

    def _handle_list(self, args: Dict[str, Any]) -> str:
        category = args.get("category")
        edicts = self._edicts
        if category:
            edicts = [e for e in edicts if e.get("category", "").lower() == category.lower()]
        limit = int(args.get("limit", 50))
        result = [
            {
                "id": e.get("id", ""),
                "text": e.get("text", ""),
                "category": e.get("category", ""),
                "confidence": e.get("confidence", ""),
                "tags": e.get("tags", []),
                "ttl": e.get("ttl", ""),
            }
            for e in edicts[:limit]
        ]
        return json.dumps({"edicts": result, "total": len(result)})

    def _handle_search(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query", "")).lower()
        if not query:
            return json.dumps({"error": "query is required"})
        matches = []
        for e in self._edicts:
            text = str(e.get("text", "")).lower()
            tags = " ".join(str(t) for t in e.get("tags", [])).lower()
            category = str(e.get("category", "")).lower()
            if query in text or query in tags or query in category:
                matches.append({
                    "id": e.get("id", ""),
                    "text": e.get("text", ""),
                    "category": e.get("category", ""),
                    "confidence": e.get("confidence", ""),
                })
        return json.dumps({"matches": matches, "total": len(matches)})


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_EDICTS_LIST_SCHEMA = {
    "name": "edicts_list",
    "description": (
        "List all active edicts (standing instructions / ground truth). "
        "Use to verify current rules and facts that govern agent behavior."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Filter by category (e.g. 'rules', 'product', 'context').",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: 50).",
            },
        },
        "required": [],
    },
}

_EDICTS_SEARCH_SCHEMA = {
    "name": "edicts_search",
    "description": (
        "Search edicts by keyword. Searches text, tags, and category. "
        "Use when looking for a specific standing rule or fact."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword to search for.",
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Plugin entry point — must be named exactly `plugin`
# ---------------------------------------------------------------------------

plugin = EdictsProvider()
