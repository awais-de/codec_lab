"""Run context: load an experiment's ``config.yaml`` and create a timestamped
result directory for one run.

Layout produced::

    results/<experiment>/results_<YYYYMMDDHHMMSS>/
        config.snapshot.yaml   exact config used for this run
        meta.json              git sha, seed, timestamps, host, duration
        <the experiment's own plots / csv>

The experiment's ``run.py`` calls :func:`load_config` then :func:`RunContext.start`,
writes its artifacts into ``ctx.dir``, and calls :func:`RunContext.finish`.
"""
from __future__ import annotations

import json
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config", "RunContext", "repo_root", "results_root"]


def repo_root() -> Path:
    """Repository root (the directory containing ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").is_file():
            return p
    raise RuntimeError("could not locate repo root (no pyproject.toml above codeclab)")


def results_root() -> Path:
    return repo_root() / "results"


def load_config(path: str | Path) -> dict[str, Any]:
    """Parse a YAML config file into a plain dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


@dataclass
class RunContext:
    """Handle to one timestamped run directory."""

    experiment: str
    config: dict[str, Any]
    dir: Path = field(init=False)
    _t0: float = field(init=False, default=0.0)

    def start(self) -> "RunContext":
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.dir = results_root() / self.experiment / f"results_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=False)
        with open(self.dir / "config.snapshot.yaml", "w") as fh:
            yaml.safe_dump(self.config, fh, sort_keys=False)
        self._t0 = time.time()
        return self

    def path(self, name: str) -> Path:
        """Path to an artifact inside this run directory."""
        return self.dir / name

    def finish(self, **extra: Any) -> Path:
        """Write ``meta.json`` and return the run directory."""
        meta = {
            "experiment": self.experiment,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.time() - self._t0, 3),
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
            "seed": self.config.get("source", {}).get("seed"),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            **extra,
        }
        with open(self.dir / "meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        return self.dir
