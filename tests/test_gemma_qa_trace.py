from __future__ import annotations

from pathlib import Path

from scripts.gemma_qa.schemas import UPOS, CefrReviewBatch, CefrReviewRow, ReviewAction
from scripts.gemma_qa.trace import (
    configure,
    event,
    extract_thoughts,
    recent_events,
    summarize_parsed,
)


def test_extract_thoughts_from_thought_parts() -> None:
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "I am reasoning about POS.", "thought": True},
                        {"text": '{"rows":[]}'},
                    ]
                }
            }
        ]
    }
    assert extract_thoughts(response) == ["I am reasoning about POS."]


def test_summarize_parsed_counts_actions() -> None:
    batch = CefrReviewBatch(
        rows=[
            CefrReviewRow(
                id="german:A1:1",
                lemma="Haus",
                english_lemma="house",
                chinese_lemma="房子",
                upos=UPOS.NOUN,
                action=ReviewAction.KEEP,
            ),
            CefrReviewRow(
                id="german:A1:2",
                lemma="gehen",
                english_lemma="go",
                chinese_lemma="去",
                upos=UPOS.VERB,
                action=ReviewAction.FIX,
            ),
        ]
    )
    summary = summarize_parsed(batch)
    assert summary["row_count"] == 2
    assert summary["actions"] == {"keep": 1, "fix": 1}
    assert len(summary["sample"]) == 2


def test_event_writes_jsonl_and_recent(tmp_path: Path, monkeypatch) -> None:
    # Snapshot trace globals so this test cannot leak _CONFIGURED/_JSONL_PATH
    # state into other test modules (e.g. workflow/client) whose event() calls
    # would otherwise write to this tmp_path after cleanup.
    import scripts.gemma_qa.trace as _trace

    snapshot = {
        "_JSONL_PATH": _trace._JSONL_PATH,
        "_LEVEL": _trace._LEVEL,
        "_LOG_BODIES": _trace._LOG_BODIES,
        "_CONFIGURED": _trace._CONFIGURED,
    }

    def _restore() -> None:
        _trace._JSONL_PATH = snapshot["_JSONL_PATH"]
        _trace._LEVEL = snapshot["_LEVEL"]
        _trace._LOG_BODIES = snapshot["_LOG_BODIES"]
        _trace._CONFIGURED = snapshot["_CONFIGURED"]

    monkeypatch.setattr(_trace, "_JSONL_PATH", snapshot["_JSONL_PATH"])
    monkeypatch.setattr(_trace, "_LEVEL", snapshot["_LEVEL"])
    monkeypatch.setattr(_trace, "_LOG_BODIES", snapshot["_LOG_BODIES"])
    monkeypatch.setattr(_trace, "_CONFIGURED", snapshot["_CONFIGURED"])

    path = tmp_path / "events.jsonl"
    configure(root=tmp_path, level="INFO", log_bodies=False, jsonl_path=path)
    try:
        event("test.kind", model="gemma-x", wait_s=1.5, error="boom")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        recent = recent_events(path, limit=5)
        assert len(recent) == 1
        assert recent[0]["error"] == "boom"
    finally:
        _restore()


def test_summarize_parsed_and_recent_events_edge_cases(tmp_path: Path) -> None:
    assert summarize_parsed(None) == {"type": "none"}
    assert summarize_parsed(42) == {"type": "int", "repr": "42"}
    assert summarize_parsed({"other_key": 123})["keys"] == ["other_key"]
    assert recent_events(tmp_path / "nonexistent.jsonl") == []
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("not json\n\n", encoding="utf-8")
    assert recent_events(bad_file) == []
