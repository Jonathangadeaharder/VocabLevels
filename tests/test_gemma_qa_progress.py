from scripts.gemma_qa.progress import (
    batch_progress_line,
    eta_from_samples,
    format_duration,
    scale_progress_line,
)


def test_format_duration() -> None:
    assert format_duration(0) == "0s"
    assert format_duration(65) == "1m05s"
    assert format_duration(3661) == "1h01m"
    assert format_duration(None) == "?"


def test_eta_from_samples() -> None:
    assert eta_from_samples([], 3) is None
    assert eta_from_samples([10.0, 10.0], 2) == 20.0
    assert eta_from_samples([10.0], 0) == 0.0


def test_progress_lines_include_counts() -> None:
    scale = scale_progress_line(
        done=3,
        total=40,
        current="cefr:en:B2",
        status="running",
        durations=[100.0, 120.0],
        started_at=0.0,
    )
    assert "3/40" in scale
    assert "cefr:en:B2" in scale
    assert "eta=" in scale
    batch = batch_progress_line(
        lang="english",
        level="B2",
        batch_index=8,
        batch_count=55,
        rows_in_batch=36,
        rows_done=288,
        rows_total=1965,
        durations=[30.0, 35.0],
        started_at=0.0,
        status="ok",
    )
    assert "8/55" in batch
    assert "rows=288/1965" in batch
    waiting = batch_progress_line(
        lang="english",
        level="B2",
        batch_index=15,
        batch_count=55,
        rows_in_batch=36,
        rows_done=504,
        rows_total=1965,
        durations=[30.0, 35.0],
        started_at=0.0,
        status="waiting",
        wait_s=22.0,
    )
    assert "status=waiting" in waiting
    assert "wait=22s" in waiting
    assert "current_batch=15/55" in waiting


def test_batch_progress_concurrency_uses_completed_and_wall_eta() -> None:
    # Out-of-order batch 40 finished but only 10 completed overall.
    line = batch_progress_line(
        lang="french",
        level="C1",
        batch_index=40,
        batch_count=112,
        rows_in_batch=36,
        rows_done=360,
        rows_total=4000,
        durations=[60.0, 60.0],
        started_at=0.0,
        status="ok",
        completed=10,
        concurrency=4,
    )
    assert "10/112" in line
    assert "concurrency=4" in line
    assert "current_batch=40/112" in line
    # remaining 102 batches / 4 ≈ 26 wall slots * 60s
    assert "eta=26m" in line or "eta=25m" in line or "eta=27m" in line
