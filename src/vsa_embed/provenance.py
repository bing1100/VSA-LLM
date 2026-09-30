"""Run-folder provenance: manifests, resolved configs and overwrite protection."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import yaml

MANIFEST_SCHEMA_VERSION = 2
_PACKAGES = ("torch", "transformers", "nltk", "PyYAML", "numpy", "safetensors", "vsa-embed")


def git_state(cwd: Path | None = None) -> dict[str, Any]:
    """Current commit and whether tracked source files differ from it."""
    root = cwd or Path(__file__).resolve().parent
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, check=True, capture_output=True, text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": sha, "git_dirty": None if status is None else bool(status)}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(canonical).hexdigest()


def prepare_output_dir(output_dir: Path) -> None:
    """Create a fresh run directory; refuse to write into one that already has files."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"run directory {output_dir} already contains files; choose a new run ID"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def apply_thread_setting(config: dict[str, Any]) -> None:
    """Honour `num_threads`: multithreaded CPU training is not bit-reproducible across processes."""
    if config.get("num_threads"):
        torch.set_num_threads(int(config["num_threads"]))


def require_clean_tree_for_promotion(promotion_eligible: bool) -> None:
    """Promotion-eligible runs must be reproducible from a commit."""
    if promotion_eligible and git_state().get("git_dirty") is not False:
        raise RuntimeError("promotion-eligible runs require a committed, clean git tree")


def build_manifest(
    config: dict[str, Any], *, device: torch.device | str | None = None,
    argv: list[str] | None = None, **extra: Any,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "device": str(device) if device is not None else None,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_num_threads": torch.get_num_threads(),
        "argv": list(sys.argv if argv is None else argv),
        "config_sha256": config_hash(config),
        **extra,
    }


def write_run_metadata(
    output_dir: Path, config: dict[str, Any], *, device: torch.device | str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Write `resolved_config.yaml` and a schema-v2 `manifest.json`; return the manifest."""
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = build_manifest(config, device=device, **extra)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest
