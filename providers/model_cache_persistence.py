"""Disk persistence for the provider model-discovery cache.

The proxy keeps a per-process cache of model lists discovered from upstream
providers (NIM, OpenRouter, HuggingFace, ...). That cache is rebuilt from
scratch on every restart, which means each cold start hits every provider's
``/v1/models`` endpoint and adds 1-2 seconds of latency before Claude Code's
``/model`` picker can populate.

This module mirrors the in-memory cache to a JSON file under
``~/.cache/free-claude-code/models.json`` (overridable via
``FCC_MODEL_CACHE_PATH``) so a cold proxy can serve ``/v1/models`` instantly
while a background refresh runs to pull any new upstream models.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path

from loguru import logger

from providers.model_listing import ProviderModelInfo


_CACHE_PATH_ENV = "FCC_MODEL_CACHE_PATH"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "free-claude-code"
_DEFAULT_CACHE_FILE = "models.json"


def cache_path() -> Path:
    """Return the disk path for the persisted model cache."""
    override = os.environ.get(_CACHE_PATH_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return _DEFAULT_CACHE_DIR / _DEFAULT_CACHE_FILE


def load_cached_model_infos() -> tuple[
    dict[str, dict[str, ProviderModelInfo]], float | None
]:
    """Load the persisted cache from disk.

    Returns ``(per_provider_infos, saved_at_epoch)`` or ``({}, None)`` when the
    cache file is missing or unreadable.
    """
    path = cache_path()
    if not path.exists():
        return {}, None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "Model cache load failed: path={} exc_type={}",
            path,
            type(exc).__name__,
        )
        return {}, None

    if not isinstance(raw, Mapping):
        return {}, None
    providers = raw.get("providers")
    if not isinstance(providers, Mapping):
        return {}, None

    parsed: dict[str, dict[str, ProviderModelInfo]] = {}
    for provider_id, models in providers.items():
        if not isinstance(provider_id, str) or not isinstance(models, list):
            continue
        bucket: dict[str, ProviderModelInfo] = {}
        for entry in models:
            if not isinstance(entry, Mapping):
                continue
            model_id = entry.get("model_id")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            supports_raw = entry.get("supports_thinking")
            supports = supports_raw if isinstance(supports_raw, bool) else None
            bucket[model_id] = ProviderModelInfo(
                model_id=model_id, supports_thinking=supports
            )
        if bucket:
            parsed[provider_id] = bucket

    saved_at = raw.get("saved_at")
    saved_at_value = (
        float(saved_at) if isinstance(saved_at, (int, float)) else None
    )
    return parsed, saved_at_value


def save_model_infos(
    per_provider: Mapping[str, Mapping[str, ProviderModelInfo]],
) -> None:
    """Atomically persist the in-memory cache to disk."""
    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": time.time(),
            "providers": {
                provider_id: [
                    {
                        "model_id": info.model_id,
                        "supports_thinking": info.supports_thinking,
                    }
                    for info in models.values()
                ]
                for provider_id, models in per_provider.items()
            },
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        logger.warning(
            "Model cache save failed: path={} exc_type={}",
            path,
            type(exc).__name__,
        )


def clear_cache_file() -> bool:
    """Delete the on-disk cache. Returns True if a file was removed."""
    path = cache_path()
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except Exception as exc:
        logger.warning(
            "Model cache delete failed: path={} exc_type={}",
            path,
            type(exc).__name__,
        )
        return False
