from __future__ import annotations

import os
import json
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .cefr import run_cefr
from .client import GemmaClient
from .handcraft import run_handcraft
from .languages import LANGUAGE_CODES, LEVELS, get_language
from .ledger import Ledger
from .quota import QuotaGate

ScalePhase = Literal["cefr", "handcraft"]
ScaleStatus = Literal["pending", "running", "succeeded", "failed"]


@dataclass(frozen=True)
class ScaleTask:
    phase: ScalePhase
    language: str
    level: str

    @property
    def key(self) -> str:
        return f"{self.phase}:{self.language}:{self.level}"


@dataclass(frozen=True)
class ScaleConfig:
    root: Path
    lemmatizer_root: Path
    languages: tuple[str, ...] = LANGUAGE_CODES
    levels: tuple[str, ...] = LEVELS
    phases: tuple[ScalePhase, ...] = ("cefr", "handcraft")
    handcraft_count: int = 20
    apply: bool = False
    refill_to_target: bool = False
    retry_failed: bool = False
    single_model: str | None = None


@dataclass(frozen=True)
class ScaleTaskRecord:
    key: str
    phase: ScalePhase
    language: str
    level: str
    status: ScaleStatus
    output: str | None
    error: str | None
    attempts: int


@dataclass(frozen=True)
class ScaleRunResult:
    succeeded: int
    failed: int
    skipped: int

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


ScaleExecutor = Callable[[ScaleTask, ScaleConfig], Path]
ScaleExecutorFactory = Callable[[], ScaleExecutor]


class ScaleState:
    def __init__(self, database: Path) -> None:
        self.database = database
        database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scale_tasks (
                    task_key TEXT PRIMARY KEY,
                    phase TEXT NOT NULL,
                    language TEXT NOT NULL,
                    level TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    configuration_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(scale_tasks)")
            }
            if "configuration_json" not in columns:
                connection.execute(
                    "ALTER TABLE scale_tasks "
                    "ADD COLUMN configuration_json TEXT NOT NULL DEFAULT '{}'"
                )

    def prepare(
        self,
        tasks: Sequence[ScaleTask],
        config: ScaleConfig | None = None,
    ) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO scale_tasks (
                    task_key, phase, language, level, status,
                    configuration_json, updated_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(task_key) DO UPDATE SET
                    status = 'pending',
                    output = NULL,
                    error = NULL,
                    attempts = 0,
                    configuration_json = excluded.configuration_json,
                    updated_at = excluded.updated_at
                WHERE scale_tasks.configuration_json != excluded.configuration_json
                """,
                [
                    (
                        task.key,
                        task.phase,
                        task.language,
                        task.level,
                        _task_configuration(task, config),
                        now,
                    )
                    for task in tasks
                ],
            )

    def start(self, task: ScaleTask) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE scale_tasks
                SET status = 'running', output = NULL, error = NULL,
                    attempts = attempts + 1, updated_at = ?
                WHERE task_key = ?
                """,
                (time.time(), task.key),
            )

    def succeed(self, task: ScaleTask, output: Path) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE scale_tasks
                SET status = 'succeeded', output = ?, error = NULL, updated_at = ?
                WHERE task_key = ?
                """,
                (str(output), time.time(), task.key),
            )

    def fail(self, task: ScaleTask, error: Exception) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE scale_tasks
                SET status = 'failed', output = NULL, error = ?, updated_at = ?
                WHERE task_key = ?
                """,
                (_safe_error(error), time.time(), task.key),
            )

    def get(self, key: str) -> ScaleTaskRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT task_key, phase, language, level, status, output, error, attempts
                FROM scale_tasks
                WHERE task_key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return ScaleTaskRecord(
            key=str(row[0]),
            phase=row[1],
            language=str(row[2]),
            level=str(row[3]),
            status=row[4],
            output=None if row[5] is None else str(row[5]),
            error=None if row[6] is None else str(row[6]),
            attempts=int(row[7]),
        )

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) FROM scale_tasks GROUP BY status"
            ).fetchall()
        counts = {status: 0 for status in ("pending", "running", "succeeded", "failed")}
        counts.update({str(status): int(count) for status, count in rows})
        return counts

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection


def build_scale_tasks(
    *,
    languages: Sequence[str] = LANGUAGE_CODES,
    levels: Sequence[str] = LEVELS,
    phases: Sequence[ScalePhase] = ("cefr", "handcraft"),
) -> list[ScaleTask]:
    normalized_languages = tuple(get_language(language).code for language in languages)
    if len(normalized_languages) != len(set(normalized_languages)):
        raise ValueError("scale languages must be unique")
    unknown_levels = [level for level in levels if level not in LEVELS]
    if unknown_levels:
        raise ValueError(f"unsupported CEFR levels: {', '.join(unknown_levels)}")
    if len(levels) != len(set(levels)):
        raise ValueError("scale levels must be unique")
    if any(phase not in {"cefr", "handcraft"} for phase in phases):
        raise ValueError("scale phases must be cefr or handcraft")
    if len(phases) != len(set(phases)):
        raise ValueError("scale phases must be unique")
    return [
        ScaleTask(phase=phase, language=language, level=level)
        for phase in phases
        for language in normalized_languages
        for level in levels
    ]


def run_scale(
    config: ScaleConfig,
    *,
    execute: ScaleExecutor | None = None,
    executor_factory: ScaleExecutorFactory | None = None,
) -> ScaleRunResult:
    if config.handcraft_count <= 0:
        raise ValueError("handcraft count must be positive")
    tasks = build_scale_tasks(
        languages=config.languages,
        levels=config.levels,
        phases=config.phases,
    )
    state = ScaleState(config.root / ".gemma_qa" / "scale.sqlite3")
    state.prepare(tasks, config)
    factory = executor_factory or (
        (lambda: execute) if execute is not None else _default_executor_factory(config)
    )
    phase_tasks = {
        phase: [task for task in tasks if task.phase == phase]
        for phase in config.phases
    }
    with ThreadPoolExecutor(max_workers=len(phase_tasks)) as pool:
        results = [
            pool.submit(
                _run_phase,
                tasks_for_phase,
                config=config,
                state=state,
                execute=factory(),
            )
            for tasks_for_phase in phase_tasks.values()
        ]
        counters = [future.result() for future in results]
    totals = Counter[str]()
    for counter in counters:
        totals.update(counter)
    return ScaleRunResult(
        succeeded=totals["succeeded"],
        failed=totals["failed"],
        skipped=totals["skipped"],
    )


def _run_phase(
    tasks: Sequence[ScaleTask],
    *,
    config: ScaleConfig,
    state: ScaleState,
    execute: ScaleExecutor,
) -> Counter[str]:
    counts = Counter[str]()
    for task in tasks:
        record = state.get(task.key)
        if record is None:
            raise RuntimeError(f"missing scale state for {task.key}")
        if record.status == "succeeded":
            counts["skipped"] += 1
            continue
        if record.status == "failed" and not config.retry_failed:
            counts["failed"] += 1
            continue
        state.start(task)
        try:
            output = execute(task, config)
        except Exception as error:
            state.fail(task, error)
            counts["failed"] += 1
        else:
            state.succeed(task, output)
            counts["succeeded"] += 1
    return counts


def _default_executor_factory(config: ScaleConfig) -> ScaleExecutorFactory:
    state_directory = config.root / ".gemma_qa"

    def factory() -> ScaleExecutor:
        def execute(task: ScaleTask, current: ScaleConfig) -> Path:
            quota = QuotaGate(state_directory / "quota.sqlite3")
            ledger = Ledger(state_directory / "ledger.sqlite3")
            client = GemmaClient(quota=quota)
            try:
                profile = get_language(task.language)
                if task.phase == "cefr":
                    return run_cefr(
                        root=current.root,
                        lang=profile.directory,
                        level=task.level,
                        client=client,
                        ledger=ledger,
                        apply=current.apply,
                        single_model=current.single_model,
                        refill_to_target=current.refill_to_target,
                    )
                return run_handcraft(
                    vocab_root=current.root,
                    lemmatizer_root=current.lemmatizer_root,
                    lang=profile.code,
                    level=task.level,
                    count=current.handcraft_count,
                    client=client,
                    ledger=ledger,
                    apply=current.apply,
                    single_model=current.single_model,
                )
            finally:
                client.close()
                quota.close()
                ledger.close()

        return execute

    return factory


def _safe_error(error: Exception) -> str:
    message = f"{type(error).__name__}: {error}"
    secret = os.environ.get("GEMINI_API_KEY")
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return message[:4_000]


def _task_configuration(task: ScaleTask, config: ScaleConfig | None) -> str:
    if config is None:
        return "{}"
    values: dict[str, object] = {
        "apply": config.apply,
        "single_model": config.single_model,
    }
    if task.phase == "cefr":
        values["refill_to_target"] = config.refill_to_target
    else:
        values["handcraft_count"] = config.handcraft_count
        values["lemmatizer_root"] = str(config.lemmatizer_root.resolve())
    return json.dumps(values, sort_keys=True, separators=(",", ":"))
