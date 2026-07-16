from __future__ import annotations

import csv
import json
import os
import tempfile
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from vocab_schema import LEVELS

from .cefr_refill import load_other_level_collision_keys, normalized_key
from .language_repair import german_row_issues
from .schemas import CefrReviewRow, ReviewAction, UPOS
from .validated import ValidatedStore, validated_store_path

NonEmptyText = Annotated[str, StringConstraints(min_length=1)]


class ManualAction(StrEnum):
    FIX = "fix"
    DROP = "drop"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ReviewTuple(StrictModel):
    lemma: NonEmptyText
    english_lemma: NonEmptyText
    chinese_lemma: NonEmptyText
    upos: UPOS

    @field_validator("upos", mode="before")
    @classmethod
    def parse_upos(cls, value: object) -> object:
        return UPOS(value) if isinstance(value, str) else value

    def csv_tuple(self) -> tuple[str, str, str, str]:
        return (
            self.lemma,
            self.english_lemma,
            self.chinese_lemma,
            self.upos.value,
        )


class Expected(ReviewTuple):
    pass


class Replacement(ReviewTuple):
    pass


class Decision(StrictModel):
    line: int = Field(ge=2)
    expected: Expected
    action: ManualAction
    replacement: Replacement | None
    reason: NonEmptyText
    reviewer: NonEmptyText

    @field_validator("action", mode="before")
    @classmethod
    def parse_action(cls, value: object) -> object:
        return ManualAction(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_replacement(self) -> Self:
        if self.action is ManualAction.FIX and self.replacement is None:
            raise ValueError("fix decisions require a replacement")
        if self.action is ManualAction.DROP and self.replacement is not None:
            raise ValueError("drop decisions must not have a replacement")
        return self


@dataclass(frozen=True)
class ManualReviewResult:
    output: Path
    input_count: int
    fix_count: int
    drop_count: int
    output_count: int

    def __str__(self) -> str:
        return (
            f"input={self.input_count} fix={self.fix_count} "
            f"drop={self.drop_count} output={self.output_count} "
            f"path={self.output}"
        )


@dataclass(frozen=True)
class ReviewedRow:
    fields: tuple[str, str, str, str]
    original_line: int
    fixed: bool


def read_decisions(directory: Path) -> list[Decision]:
    paths = sorted(directory.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"no decision JSONL files found in {directory}")
    decisions: list[Decision] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for file_line, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    raise ValueError(f"{path}:{file_line}: blank decision line")
                try:
                    payload = json.loads(raw_line)
                    decisions.append(Decision.model_validate(payload))
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(
                        f"{path}:{file_line}: invalid decision: {error}"
                    ) from error
    return decisions


def run_manual_review(
    *,
    root: Path,
    lang: str,
    level: str,
    source: Path,
    decisions_directory: Path,
    apply: bool = False,
    append: bool = False,
    check_other_level_collisions: bool = False,
) -> ManualReviewResult:
    root = root.resolve()
    source = source if source.is_absolute() else root / source
    source = source.resolve()
    decisions_directory = (
        decisions_directory
        if decisions_directory.is_absolute()
        else root / decisions_directory
    )
    decisions_directory = decisions_directory.resolve()
    _validate_source_identity(source, root=root, lang=lang, level=level)
    decisions = read_decisions(decisions_directory)
    header, rows = _read_source(source, lang=lang)
    decisions_by_line = _validate_decisions(decisions, rows=rows, source=source)
    reviewed_rows = _apply_decisions(rows, decisions_by_line)
    _validate_final_rows(
        reviewed_rows,
        root=root,
        lang=lang,
        level=level,
        check_other_level_collisions=check_other_level_collisions or append,
    )
    if append and not apply:
        raise ValueError("append requires apply=True")
    if append:
        output = root / lang / f"{level}.csv"
        committed_header, committed_rows = _read_source(output, lang=lang)
        if committed_header != header:
            raise ValueError("gap proposal header does not match committed CSV")
        _atomic_write_csv(
            output,
            committed_header,
            [*committed_rows, *[row.fields for row in reviewed_rows]],
        )
    else:
        if apply:
            output = root / lang / f"{level}.csv"
        elif source.name.endswith(".gap.proposed.csv"):
            output = source.with_name(
                source.name.replace(".gap.proposed.csv", ".gap.reviewed.csv")
            )
        else:
            output = root / lang / f"{level}.reviewed.csv"
        _atomic_write_csv(output, header, [row.fields for row in reviewed_rows])
    fix_count = sum(decision.action is ManualAction.FIX for decision in decisions)
    drop_count = sum(decision.action is ManualAction.DROP for decision in decisions)
    if apply:
        store = ValidatedStore(validated_store_path(root))
        try:
            store.seed_from_csv(root, lang=lang, level=level)
        finally:
            store.close()
    return ManualReviewResult(
        output=output,
        input_count=len(rows),
        fix_count=fix_count,
        drop_count=drop_count,
        output_count=len(reviewed_rows),
    )


def _validate_source_identity(
    source: Path,
    *,
    root: Path,
    lang: str,
    level: str,
) -> None:
    if level not in LEVELS:
        raise ValueError(f"unsupported CEFR level: {level}")
    expected = {
        (root / lang / f"{level}.proposed.csv").resolve(),
        (root / lang / f"{level}.gap.proposed.csv").resolve(),
    }
    if source.resolve() not in expected:
        raise ValueError(
            "input proposal path must match language and level: "
            f"expected one of {sorted(str(path) for path in expected)}"
        )


def _read_source(
    source: Path,
    *,
    lang: str,
) -> tuple[tuple[str, str, str, str], list[tuple[str, str, str, str]]]:
    with source.open(encoding="utf-8", newline="") as handle:
        physical_rows = list(csv.reader(handle))
    if not physical_rows:
        raise ValueError(f"empty CSV: {source}")
    expected_header = [
        f"{lang.title()}_Lemma",
        "English_Lemma",
        "Chinese_Lemma",
        "POS",
    ]
    if physical_rows[0] != expected_header:
        raise ValueError(f"unexpected header in {source}: {physical_rows[0]!r}")
    rows: list[tuple[str, str, str, str]] = []
    for line, row in enumerate(physical_rows[1:], start=2):
        if len(row) != 4:
            raise ValueError(f"{source}:{line}: expected 4 fields")
        rows.append((row[0], row[1], row[2], row[3]))
    return (
        (
            expected_header[0],
            expected_header[1],
            expected_header[2],
            expected_header[3],
        ),
        rows,
    )


def _validate_decisions(
    decisions: list[Decision],
    *,
    rows: list[tuple[str, str, str, str]],
    source: Path,
) -> dict[int, Decision]:
    decisions_by_line: dict[int, Decision] = {}
    for decision in decisions:
        if decision.line in decisions_by_line:
            raise ValueError(f"duplicate decision for physical line {decision.line}")
        row_index = decision.line - 2
        if row_index >= len(rows):
            raise ValueError(
                f"decision physical line {decision.line} is outside {source}"
            )
        actual = rows[row_index]
        expected = decision.expected.csv_tuple()
        if actual != expected:
            raise ValueError(
                f"{source}:{decision.line}: expected {expected!r}, found {actual!r}"
            )
        decisions_by_line[decision.line] = decision
    return decisions_by_line


def _apply_decisions(
    rows: list[tuple[str, str, str, str]],
    decisions_by_line: dict[int, Decision],
) -> list[ReviewedRow]:
    reviewed: list[ReviewedRow] = []
    for physical_line, row in enumerate(rows, start=2):
        decision = decisions_by_line.get(physical_line)
        if decision is None:
            reviewed.append(
                ReviewedRow(fields=row, original_line=physical_line, fixed=False)
            )
        elif decision.action is ManualAction.FIX:
            if decision.replacement is None:
                raise AssertionError("validated fix decision lacks replacement")
            reviewed.append(
                ReviewedRow(
                    fields=decision.replacement.csv_tuple(),
                    original_line=physical_line,
                    fixed=True,
                )
            )
    return reviewed


def _validate_final_rows(
    rows: list[ReviewedRow],
    *,
    root: Path,
    lang: str,
    level: str,
    check_other_level_collisions: bool,
) -> None:
    collision_keys = (
        load_other_level_collision_keys(root, lang=lang, level=level)
        if check_other_level_collisions or any(row.fixed for row in rows)
        else set()
    )
    seen: set[tuple[str, UPOS]] = set()
    for output_line, reviewed_row in enumerate(rows, start=2):
        row = reviewed_row.fields
        location = (
            f"output line {output_line} "
            f"(original physical line {reviewed_row.original_line})"
        )
        if any(not field.strip() for field in row):
            raise ValueError(f"{location}: fields must be nonempty")
        for field in row:
            if unicodedata.normalize("NFC", field) != field:
                raise ValueError(f"{location}: fields must use NFC")
        try:
            upos = UPOS(row[3])
        except ValueError as error:
            raise ValueError(f"{location}: invalid UPOS {row[3]!r}") from error
        review_row = CefrReviewRow(
            id=f"{lang}:{level}:{output_line - 1}",
            lemma=row[0],
            english_lemma=row[1],
            chinese_lemma=row[2],
            upos=upos,
            action=ReviewAction.KEEP,
        )
        if lang == "german":
            issues = german_row_issues(review_row)
            if issues:
                codes = ", ".join(issue.code for issue in issues)
                raise ValueError(f"{location}: German language gates failed: {codes}")
        key = normalized_key(row[0], upos)
        if key in seen:
            raise ValueError(f"{location}: duplicate normalized lemma and UPOS")
        if key in collision_keys and (
            reviewed_row.fixed or check_other_level_collisions
        ):
            raise ValueError(f"{location}: lemma and UPOS collide with another level")
        seen.add(key)


def _atomic_write_csv(
    output: Path,
    header: tuple[str, str, str, str],
    rows: list[tuple[str, str, str, str]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
