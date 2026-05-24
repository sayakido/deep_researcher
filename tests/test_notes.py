"""Tests for NoteStore."""

from __future__ import annotations

from pathlib import Path

from services.notes import NoteStore


class TestNoteStore:
    def test_create_and_read(self, note_store: NoteStore):
        note_id = note_store.create(title="测试笔记", note_type="task_state", tags=["test"], content="内容")
        assert note_id is not None
        assert len(note_id) == 8

        content = note_store.read(note_id)
        assert content is not None
        assert "测试笔记" in content
        assert "内容" in content

    def test_read_nonexistent(self, note_store: NoteStore):
        assert note_store.read("nonexistent") is None

    def test_update_existing(self, note_store: NoteStore):
        note_id = note_store.create(title="原始", note_type="task_state", content="原始内容")
        assert note_store.update(note_id=note_id, content="更新后的内容") is True

        content = note_store.read(note_id)
        assert content is not None
        assert "更新后的内容" in content

    def test_update_nonexistent(self, note_store: NoteStore):
        assert note_store.update(note_id="ghost", content="任何内容") is False

    def test_get_path(self, note_store: NoteStore):
        note_id = note_store.create(title="路径测试", note_type="task_state", content="内容")
        path = note_store.get_path(note_id)
        assert path.endswith(f"{note_id}.md")
        assert Path(path).parent.exists()

    def test_workspace_created(self, tmp_path: Path):
        tmp = tmp_path / "test_notes_workspace"
        store = NoteStore(workspace=str(tmp))
        assert tmp.exists()
        assert store.workspace == tmp
