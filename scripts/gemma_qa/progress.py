"""Human progress / ETA lines for scale and CEFR batch loops."""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Sequence


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN
        return "?"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def eta_from_samples(
    durations: Sequence[float],
    remaining: int,
) -> float | None:
    if remaining <= 0:
        return 0.0
    samples = [d for d in durations if d > 0]
    if not samples:
        return None
    # Prefer last few samples so estimate tracks recent batch speed.
    window = samples[-8:]
    avg = sum(window) / len(window)
    return avg * remaining


def print_progress(line: str) -> None:
    """Always flush so ``tail -f`` shows progress immediately."""
    print(line, file=sys.stderr, flush=True)


def scale_progress_line(
    *,
    done: int,
    total: int,
    current: str,
    status: str,
    durations: Sequence[float],
    started_at: float,
) -> str:
    remaining = max(0, total - done)
    pct = (100.0 * done / total) if total else 100.0
    eta_s = eta_from_samples(list(durations), remaining)
    elapsed = time.time() - started_at
    last = durations[-1] if durations else None
    return (
        f"PROGRESS scale {done}/{total} ({pct:5.1f}%) "
        f"status={status} current={current} "
        f"remaining={remaining} last={format_duration(last)} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta_s)}"
    )


def batch_progress_line(
    *,
    lang: str,
    level: str,
    batch_index: int,
    batch_count: int,
    rows_in_batch: int,
    rows_done: int,
    rows_total: int,
    durations: Sequence[float],
    started_at: float,
    status: str = "ok",
    wait_s: float | None = None,
    completed: int | None = None,
    concurrency: int = 1,
) -> str:
    # For status=running, batch_index is the batch about to start (1-based);
    # remaining includes the current batch. Pass completed= when batches run
    # out of order under concurrency.
    if completed is not None:
        done = max(0, min(completed, batch_count))
        remaining_batches = max(0, batch_count - done)
        pct = (100.0 * done / batch_count) if batch_count else 100.0
    elif status == "running":
        done = max(0, batch_index - 1)
        remaining_batches = max(0, batch_count - done)
        pct = (100.0 * done / batch_count) if batch_count else 100.0
    else:
        done = batch_index
        remaining_batches = max(0, batch_count - batch_index)
        pct = (100.0 * batch_index / batch_count) if batch_count else 100.0
    wall_remaining = (
        math.ceil(remaining_batches / concurrency)
        if concurrency > 1
        else remaining_batches
    )
    eta_s = eta_from_samples(list(durations), wall_remaining)
    elapsed = time.time() - started_at
    last = durations[-1] if durations else None
    wait_part = (
        f" wait={format_duration(wait_s)}" if wait_s is not None else ""
    )
    concurrency_part = f" concurrency={concurrency}" if concurrency > 1 else ""
    return (
        f"PROGRESS batch {lang}/{level} "
        f"{done}/{batch_count} ({pct:5.1f}%) "
        f"status={status} current_batch={batch_index}/{batch_count} "
        f"rows={rows_done}/{rows_total} batch_rows={rows_in_batch} "
        f"remaining_batches={remaining_batches} last={format_duration(last)} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta_s)}"
        f"{wait_part}{concurrency_part}"
    )
