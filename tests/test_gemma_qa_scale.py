from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

from scripts.gemma_qa.cli import build_parser
from scripts.gemma_qa.languages import (
    LANGUAGE_CODES,
    LANGUAGE_DIRECTORIES,
    LEVELS,
    get_language,
)
from scripts.gemma_qa.scale import (
    ScaleConfig,
    ScaleState,
    ScaleTask,
    build_scale_tasks,
    run_scale,
)


def test_profiles_cover_all_languages_and_task_keys() -> None:
    assert LANGUAGE_CODES == ("en", "de", "es", "ar", "fr", "sv", "zh", "nl")
    assert LANGUAGE_DIRECTORIES == (
        "english",
        "german",
        "spanish",
        "arabic",
        "french",
        "swedish",
        "chinese",
        "dutch",
    )
    assert LEVELS == ("A1", "A2", "B1", "B2", "C1")
    assert get_language("german") is get_language("de")
    cefr = build_scale_tasks(phases=("cefr",))
    handcraft = build_scale_tasks(phases=("handcraft",))
    assert len(cefr) == 40
    assert len(handcraft) == 40
    assert cefr[0].key == "cefr:en:A1"
    assert cefr[-1].key == "cefr:nl:C1"
    assert handcraft[0].key == "handcraft:en:A1"
    assert handcraft[-1].key == "handcraft:nl:C1"


def test_scale_continues_resumes_and_retries_failures(tmp_path: Path) -> None:
    calls: list[str] = []

    def execute(task: ScaleTask, config: ScaleConfig) -> Path:
        calls.append(task.key)
        if task.key == "cefr:de:A1" and calls.count(task.key) == 1:
            raise RuntimeError("broken")
        return tmp_path / f"{task.key.replace(':', '-')}.out"

    config = ScaleConfig(
        root=tmp_path,
        lemmatizer_root=tmp_path / "lemmatizer",
        languages=("en", "de"),
        levels=("A1",),
        phases=("cefr",),
    )
    first = run_scale(config, execute=execute)
    assert first.failed == 1
    assert first.succeeded == 1
    assert calls == ["cefr:en:A1", "cefr:de:A1"]

    second = run_scale(config, execute=execute)
    assert second.skipped == 1
    assert second.failed == 1
    assert calls == ["cefr:en:A1", "cefr:de:A1"]

    retried = run_scale(
        dataclasses.replace(config, retry_failed=True),
        execute=execute,
    )
    assert retried.succeeded == 1
    assert retried.skipped == 1
    assert calls[-1] == "cefr:de:A1"


def test_scale_state_redacts_api_key(tmp_path: Path, monkeypatch) -> None:
    secret = "test-secret-key"
    monkeypatch.setenv("API_KEY", secret)
    config = ScaleConfig(
        root=tmp_path,
        lemmatizer_root=tmp_path / "lemmatizer",
        languages=("en",),
        levels=("A1",),
        phases=("cefr",),
    )

    def fail(task: ScaleTask, current: ScaleConfig) -> Path:
        raise RuntimeError(f"request failed with {secret}")

    result = run_scale(config, execute=fail)
    assert result.failed == 1
    state = ScaleState(tmp_path / ".gemma_qa" / "scale.sqlite3")
    record = state.get("cefr:en:A1")
    assert record is not None
    assert secret not in (record.error or "")
    assert "[REDACTED]" in (record.error or "")


def test_scale_defaults_to_proposals_without_refill_or_apply(tmp_path: Path) -> None:
    observed: list[ScaleConfig] = []

    def execute(task: ScaleTask, config: ScaleConfig) -> Path:
        observed.append(config)
        return tmp_path / "proposal"

    config = ScaleConfig(root=tmp_path, lemmatizer_root=tmp_path / "lemmatizer")
    run_scale(
        dataclasses.replace(
            config,
            languages=("en",),
            levels=("A1",),
            phases=("cefr",),
        ),
        execute=execute,
    )
    assert observed[0].apply is False
    assert observed[0].refill_to_target is False
    assert observed[0].handcraft_count == 20


def test_scale_cli_supports_explicit_apply_refill_and_selectors() -> None:
    args = build_parser().parse_args(
        [
            "scale",
            "--root",
            "/vocab",
            "--lemmatizer-root",
            "/lemmatizer",
            "--languages",
            "german",
            "es",
            "--levels",
            "A1",
            "C1",
            "--phase",
            "both",
            "--apply",
            "--refill-to-target",
            "--retry-failed",
        ]
    )
    assert args.languages == ["german", "es"]
    assert args.levels == ["A1", "C1"]
    assert args.apply is True
    assert args.refill_to_target is True
    assert args.retry_failed is True


def test_changed_apply_mode_is_not_skipped_as_prior_success(tmp_path: Path) -> None:
    calls: list[bool] = []

    def execute(task: ScaleTask, config: ScaleConfig) -> Path:
        calls.append(config.apply)
        return tmp_path / "output"

    base = ScaleConfig(
        root=tmp_path,
        lemmatizer_root=tmp_path / "lemmatizer",
        languages=("en",),
        levels=("A1",),
        phases=("cefr",),
    )
    run_scale(base, execute=execute)
    applied = dataclasses.replace(base, apply=True)
    result = run_scale(applied, execute=execute)
    assert result.succeeded == 1
    assert result.skipped == 0
    assert calls == [False, True]


def test_phase_workers_do_not_share_owned_resources(tmp_path: Path) -> None:
    """Each phase gets its own execute callable from the factory (no shared client)."""
    owners: list[object] = []

    def executor_factory() -> Callable[[ScaleTask, ScaleConfig], Path]:
        owner = object()
        owners.append(owner)

        def execute(task: ScaleTask, config: ScaleConfig) -> Path:
            _ = owner
            return tmp_path / task.key

        return execute

    config = ScaleConfig(
        root=tmp_path,
        lemmatizer_root=tmp_path / "lemmatizer",
        languages=("en",),
        levels=("A1",),
        phases=("cefr", "handcraft"),
    )
    result = run_scale(config, executor_factory=executor_factory)
    assert result.succeeded == 2
    assert len(owners) == 2
    assert owners[0] is not owners[1]


def test_both_phases_run_cefr_before_handcraft(tmp_path: Path) -> None:
    order: list[str] = []

    def execute(task: ScaleTask, config: ScaleConfig) -> Path:
        order.append(task.key)
        return tmp_path / task.key

    run_scale(
        ScaleConfig(
            root=tmp_path,
            lemmatizer_root=tmp_path / "lemmatizer",
            languages=("en",),
            levels=("A1", "A2"),
            phases=("cefr", "handcraft"),
        ),
        execute=execute,
    )
    assert order[:2] == ["cefr:en:A1", "cefr:en:A2"]
    assert order[2:] == ["handcraft:en:A1", "handcraft:en:A2"]


def test_handcraft_blocked_when_cefr_sibling_failed(tmp_path: Path) -> None:
    def cefr_only_fail(task: ScaleTask, config: ScaleConfig) -> Path:
        if task.phase == "cefr":
            raise RuntimeError("cefr boom")
        return tmp_path / task.key

    result = run_scale(
        ScaleConfig(
            root=tmp_path,
            lemmatizer_root=tmp_path / "lemmatizer",
            languages=("en",),
            levels=("A1",),
            phases=("cefr", "handcraft"),
        ),
        execute=cefr_only_fail,
    )
    assert result.failed == 2
    assert result.succeeded == 0
    state = ScaleState(tmp_path / ".gemma_qa" / "scale.sqlite3")
    handcraft = state.get("handcraft:en:A1")
    assert handcraft is not None
    assert handcraft.status == "failed"
    assert handcraft.error is not None
    assert "handcraft blocked" in handcraft.error
