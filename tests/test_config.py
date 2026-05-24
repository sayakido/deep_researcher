"""Tests for runtime path configuration."""

from __future__ import annotations

from pathlib import Path

from config import PROJECT_ROOT, Configuration, resolve_project_path


def test_resolves_relative_paths_from_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(PROJECT_ROOT / "src")

    assert Path(resolve_project_path("runtime/logs/app.log")) == PROJECT_ROOT / "runtime/logs/app.log"


def test_notes_workspace_default_is_project_root_relative(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT / "src")

    config = Configuration()

    assert Path(config.notes_workspace) == PROJECT_ROOT / "runtime/notes/dev"
