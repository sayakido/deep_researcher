"""Helpers for coordinating note tool usage instructions and a simple NoteStore replacement."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from langchain_core.tools import tool


class NoteStore:
    """Simple file-based note store."""

    def __init__(self, workspace: str = "./notes") -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        title: str,
        note_type: str = "task_state",
        tags: list[str] | None = None,
        content: str = "",
    ) -> str:
        note_id = uuid.uuid4().hex[:8]
        path = self.workspace / f"{note_id}.md"
        meta = {
            "id": note_id,
            "title": title,
            "type": note_type,
            "tags": tags or [],
        }
        full_content = f"---\n{json.dumps(meta, ensure_ascii=False, indent=2)}\n---\n\n{content}"
        path.write_text(full_content, encoding="utf-8")
        return note_id

    def read(self, note_id: str) -> str | None:
        path = self.workspace / f"{note_id}.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def update(
        self,
        note_id: str,
        title: str | None = None,
        note_type: str | None = None,
        tags: list[str] | None = None,
        content: str = "",
    ) -> bool:
        path = self.workspace / f"{note_id}.md"
        if not path.exists():
            return False

        existing = path.read_text(encoding="utf-8")
        meta_end = existing.find("---", 3)
        if meta_end != -1:
            existing_meta = json.loads(existing[4:meta_end].strip())
        else:
            existing_meta = {}

        if title:
            existing_meta["title"] = title
        if note_type:
            existing_meta["type"] = note_type
        if tags is not None:
            existing_meta["tags"] = tags
        existing_meta["id"] = note_id

        full_content = (
            f"---\n{json.dumps(existing_meta, ensure_ascii=False, indent=2)}\n---\n\n{content}"
        )
        path.write_text(full_content, encoding="utf-8")
        return True

    def get_path(self, note_id: str) -> str:
        return str(self.workspace / f"{note_id}.md")


def create_note_tools(note_store: NoteStore) -> list:
    """Create LangChain tools wrapping NoteStore for LLM tool calling."""

    @tool
    def create_note(title: str, note_type: str, tags: list[str], content: str) -> str:
        """创建一个新笔记来存储任务信息。调用此工具记录调研任务的进度、发现和总结。
        
        Args:
            title: 笔记标题，应包含任务编号和名称
            note_type: 笔记类型，通常是 "task_state" 或 "conclusion"
            tags: 标签列表，必须包含 "deep_research" 和 "task_{task_id}"
            content: 笔记正文内容
        """
        note_id = note_store.create(title=title, note_type=note_type, tags=tags, content=content)
        return f"笔记创建成功！ID: {note_id}"

    @tool
    def update_note(note_id: str, content: str) -> str:
        """更新已有笔记的内容。在获得新的调研结果后，调用此工具追加信息到已有笔记中。
        
        Args:
            note_id: 要更新的笔记 ID
            content: 追加或替换的笔记内容
        """
        if note_store.update(note_id=note_id, content=content):
            return f"笔记 {note_id} 已更新"
        return f"❌ 笔记 {note_id} 不存在"

    @tool
    def read_note(note_id: str) -> str:
        """读取已有笔记的全部内容。在开始任务总结前，先调用此工具获取先前记录的上下文。
        
        Args:
            note_id: 要读取的笔记 ID
        """
        content = note_store.read(note_id)
        if content is None:
            return f"笔记 {note_id} 不存在"
        return content

    return [create_note, update_note, read_note]




