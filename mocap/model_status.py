"""Model readiness as the GUI sees it.

Without remote workers this is mocap.models.ready/missing. With remote workers
configured (Linux), models are checked where they are used:

- models of stages that run on the workstation are checked *on the workstation*,
  from an inventory of its model files that the GUI fetches in the background
  (``python -m mocap.remote_workers --model-status``);
- models read on this machine - the body models used by the mesh preview,
  retargeting, correction and export, and the requirements of local stages -
  are checked here.

A local copy of a workstation-only model is not consulted; Model Manager uploads
downloads and imports to the workstation (see mocap.remote_workers). Worker-side
checks (worker.py, the backends) are unchanged: they run where the files are.
"""
from __future__ import annotations

import json
import os

from . import models
from .models import BY_KEY, STAGE_REQUIREMENTS

# Read on this machine even when inference runs remotely.
BODY = frozenset({'smplx', 'smplh'})
# Stages that run on this machine on Linux (everything else there is a remote stage).
LOCAL_STAGES = frozenset({'export', 'correct', 'flow_export'})

_inventory: dict | None = None  # {key: [size, mtime_ns, receipt text or None]} on the workstation
_error: str | None = None
_config: tuple | None = None  # (cache key, remote_workers settings)


def remote() -> dict | None:
    """The remote_workers settings when remote stages are active (Linux), else None."""
    global _config, _inventory, _error
    if os.name == 'nt':
        return None
    from .resource_safety import path
    try:
        stamp = path().stat().st_mtime_ns
    except OSError:
        stamp = None
    key = (os.environ.get('RGB_MOCAP_REMOTE'), stamp)
    if _config is None or _config[0] != key:
        from .remote_workers import settings
        config = settings()
        if _config is not None and config != _config[1]:
            _inventory, _error = None, None  # Another host or mirror: forget its inventory.
        _config = (key, config)
    config = _config[1]
    return config if config.get('host') else None


def remote_stage(stage: str, config) -> bool:
    from .remote_workers import remote_stages
    return stage in remote_stages(config)


def places(key: str, config) -> set[str]:
    """Where a model must be for the app to use it: 'local' and/or 'remote'."""
    from .remote_workers import remote_stages
    local = key in BODY or any(key in STAGE_REQUIREMENTS.get(stage, ()) for stage in LOCAL_STAGES
                               if not remote_stage(stage, config))
    if not local:
        return {'remote'}  # Includes files no stage lists, e.g. FlowHMR's metadata and HaMeR.
    remote_use = any(key in STAGE_REQUIREMENTS.get(stage, ()) for stage in remote_stages(config))
    return {'local', 'remote'} if remote_use else {'local'}


def set_inventory(value: dict) -> None:
    global _inventory, _error
    _inventory, _error = value, None


def set_error(message: str) -> None:
    global _error
    _error = message


def known() -> bool:
    return _inventory is not None


def error() -> str | None:
    return _error


def ready_from(model, size, mtime_ns, receipt_text) -> bool:
    """models.ready() for a file described by its size, mtime and receipt text."""
    valid = isinstance(size, int) and size >= model.minimum_bytes
    if valid and models.receipt_required(model):
        try:
            valid = models.receipt_matches(model, json.loads(receipt_text), size, mtime_ns)
        except (TypeError, ValueError, AttributeError):
            valid = False
    return valid


def remote_ready(model) -> bool:
    entry = (_inventory or {}).get(model.key)
    return bool(entry) and ready_from(model, *entry)


def ready(model) -> bool:
    config = remote()
    if config is None:
        return models.ready(model)
    where = places(model.key, config)
    return ('local' not in where or models.ready(model)) and ('remote' not in where or remote_ready(model))


def missing(stage: str, tool=None) -> list:
    config = remote()
    if config is None:
        return models.missing(stage, tool)
    if tool == 'flowhmr' and stage in ('correct', 'export', 'rig', 'mesh'):
        stage = 'flow_export'
    check = remote_ready if remote_stage(stage, config) else models.ready
    return [BY_KEY[key] for key in STAGE_REQUIREMENTS.get(stage, ()) if not check(BY_KEY[key])]


def describe(model) -> tuple[str, bool]:
    """(status text, ready) for Model Manager."""
    config = remote()
    if config is None:
        available = models.ready(model)
        return ('Ready' if available else 'Missing'), available
    where = places(model.key, config)
    here = models.ready(model) if 'local' in where else True
    if 'remote' not in where:
        return ('Ready' if here else 'Missing'), here
    if _inventory is None:
        return ('Workstation unreachable' if _error else 'Checking workstation...'), False
    there = remote_ready(model)
    if here and there:
        return ('Ready' if 'local' in where else 'Ready (workstation)'), True
    if 'local' in where and not here:
        return ('Missing' if not there else 'Missing here'), False
    return 'Missing on workstation', False
