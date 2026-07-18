from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.gemma_qa.cli import main
from scripts.gemma_qa.manual_review import run_manual_review


HEADER = ["German_Lemma", "English_Lemma", "Chinese_Lemma", "POS"]


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)


def write_decisions(path: Path, decisions: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "review.jsonl").write_text(
        "".join(
            json.dumps(decision, ensure_ascii=False) + "\n" for decision in decisions
        ),
        encoding="utf-8",
    )


def decision(
    line: int,
    expected: list[str],
    action: str,
    replacement: list[str] | None = None,
) -> dict[str, object]:
    fields = ("lemma", "english_lemma", "chinese_lemma", "upos")
    return {
        "line": line,
        "expected": dict(zip(fields, expected, strict=True)),
        "action": action,
        "replacement": (
            dict(zip(fields, replacement, strict=True))
            if replacement is not None
            else None
        ),
        "reason": "verified",
        "reviewer": "tester",
    }


def test_manual_review_applies_fixes_and_drops_by_original_physical_line(
    tmp_path: Path,
) -> None:
    source = tmp_path / "german" / "A1.proposed.csv"
    rows = [
        ["Abend", "evening", "晚上", "NOUN"],
        ["alt", "old", "老的", "ADJ"],
        ["gehen", "go", "去", "VERB"],
        ["Haus", "house", "房子", "NOUN"],
    ]
    write_csv(source, rows)
    reviews = tmp_path / "manual_reviews" / "german" / "A1"
    write_decisions(
        reviews,
        [
            decision(3, rows[1], "drop"),
            decision(
                4,
                rows[2],
                "fix",
                ["laufen", "run", "跑", "VERB"],
            ),
        ],
    )

    result = run_manual_review(
        root=tmp_path,
        lang="german",
        level="A1",
        source=source,
        decisions_directory=reviews,
    )

    assert result.input_count == 4
    assert result.fix_count == 1
    assert result.drop_count == 1
    assert result.output_count == 3
    assert result.output == tmp_path / "german" / "A1.reviewed.csv"
    assert source.read_text(encoding="utf-8") == (
        "German_Lemma,English_Lemma,Chinese_Lemma,POS\n"
        "Abend,evening,晚上,NOUN\n"
        "alt,old,老的,ADJ\n"
        "gehen,go,去,VERB\n"
        "Haus,house,房子,NOUN\n"
    )
    with result.output.open(encoding="utf-8", newline="") as handle:
        assert list(csv.reader(handle)) == [
            HEADER,
            rows[0],
            ["laufen", "run", "跑", "VERB"],
            rows[3],
        ]


def test_expected_mismatch_aborts_without_writing(tmp_path: Path) -> None:
    source = tmp_path / "german" / "A1.proposed.csv"
    rows = [["Abend", "evening", "晚上", "NOUN"]]
    write_csv(source, rows)
    reviews = tmp_path / "manual_reviews" / "german" / "A1"
    write_decisions(
        reviews,
        [
            decision(
                2,
                ["Morgen", "morning", "早晨", "NOUN"],
                "drop",
            )
        ],
    )

    with pytest.raises(ValueError, match="expected"):
        run_manual_review(
            root=tmp_path,
            lang="german",
            level="A1",
            source=source,
            decisions_directory=reviews,
        )

    assert not (tmp_path / "german" / "A1.reviewed.csv").exists()


def test_duplicate_decision_line_aborts_without_writing(tmp_path: Path) -> None:
    source = tmp_path / "german" / "A1.proposed.csv"
    rows = [["Abend", "evening", "晚上", "NOUN"]]
    write_csv(source, rows)
    reviews = tmp_path / "manual_reviews" / "german" / "A1"
    duplicate = decision(2, rows[0], "drop")
    write_decisions(reviews, [duplicate, duplicate])

    with pytest.raises(ValueError, match="duplicate decision"):
        run_manual_review(
            root=tmp_path,
            lang="german",
            level="A1",
            source=source,
            decisions_directory=reviews,
        )

    assert not (tmp_path / "german" / "A1.reviewed.csv").exists()


@pytest.mark.parametrize(
    "invalid_decision",
    [
        {
            **decision(
                2,
                ["Abend", "evening", "晚上", "NOUN"],
                "drop",
            ),
            "unexpected": True,
        },
        decision(
            2,
            ["Abend", "evening", "晚上", "NOUN"],
            "fix",
        ),
        decision(
            2,
            ["Abend", "evening", "晚上", "NOUN"],
            "drop",
            ["Morgen", "morning", "早晨", "NOUN"],
        ),
    ],
    ids=["extra-field", "fix-without-replacement", "drop-with-replacement"],
)
def test_malformed_or_inconsistent_decision_aborts(
    tmp_path: Path,
    invalid_decision: dict[str, object],
) -> None:
    source = tmp_path / "german" / "A1.proposed.csv"
    write_csv(source, [["Abend", "evening", "晚上", "NOUN"]])
    reviews = tmp_path / "manual_reviews" / "german" / "A1"
    write_decisions(reviews, [invalid_decision])

    with pytest.raises(ValueError, match="invalid decision"):
        run_manual_review(
            root=tmp_path,
            lang="german",
            level="A1",
            source=source,
            decisions_directory=reviews,
        )

    assert not (tmp_path / "german" / "A1.reviewed.csv").exists()


def test_fix_creating_duplicate_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "german" / "A1.proposed.csv"
    rows = [
        ["Abend", "evening", "晚上", "NOUN"],
        ["Morgen", "morning", "早晨", "NOUN"],
    ]
    write_csv(source, rows)
    reviews = tmp_path / "manual_reviews" / "german" / "A1"
    write_decisions(
        reviews,
        [
            decision(
                3,
                rows[1],
                "fix",
                ["Abend", "morning", "早晨", "NOUN"],
            )
        ],
    )

    with pytest.raises(ValueError, match="duplicate normalized"):
        run_manual_review(
            root=tmp_path,
            lang="german",
            level="A1",
            source=source,
            decisions_directory=reviews,
        )

    assert not (tmp_path / "german" / "A1.reviewed.csv").exists()


def test_fix_creating_cross_level_collision_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "german" / "A1.proposed.csv"
    rows = [["Abend", "evening", "晚上", "NOUN"]]
    write_csv(source, rows)
    write_csv(
        tmp_path / "german" / "A2.csv",
        [["Morgen", "morning", "早晨", "NOUN"]],
    )
    reviews = tmp_path / "manual_reviews" / "german" / "A1"
    write_decisions(
        reviews,
        [
            decision(
                2,
                rows[0],
                "fix",
                ["Morgen", "morning", "早晨", "NOUN"],
            )
        ],
    )

    with pytest.raises(ValueError, match="collide with another level"):
        run_manual_review(
            root=tmp_path,
            lang="german",
            level="A1",
            source=source,
            decisions_directory=reviews,
        )

    assert not (tmp_path / "german" / "A1.reviewed.csv").exists()


def test_full_cross_level_collision_check_is_optional(tmp_path: Path) -> None:
    source = tmp_path / "german" / "A1.proposed.csv"
    rows = [
        ["Abend", "evening", "晚上", "NOUN"],
        ["gehen", "go", "去", "VERB"],
    ]
    write_csv(source, rows)
    write_csv(
        tmp_path / "german" / "A2.csv",
        [["Abend", "evening", "晚上", "NOUN"]],
    )
    reviews = tmp_path / "manual_reviews" / "german" / "A1"
    write_decisions(reviews, [decision(3, rows[1], "drop")])

    run_manual_review(
        root=tmp_path,
        lang="german",
        level="A1",
        source=source,
        decisions_directory=reviews,
    )

    (tmp_path / "german" / "A1.reviewed.csv").unlink()
    with pytest.raises(ValueError, match="collide with another level"):
        run_manual_review(
            root=tmp_path,
            lang="german",
            level="A1",
            source=source,
            decisions_directory=reviews,
            check_other_level_collisions=True,
        )
    assert not (tmp_path / "german" / "A1.reviewed.csv").exists()


def test_fix_failing_german_language_gate_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "german" / "A1.proposed.csv"
    rows = [["gehen", "go", "去", "VERB"]]
    write_csv(source, rows)
    reviews = tmp_path / "manual_reviews" / "german" / "A1"
    write_decisions(
        reviews,
        [
            decision(
                2,
                rows[0],
                "fix",
                ["geht", "goes", "去", "VERB"],
            )
        ],
    )

    with pytest.raises(ValueError, match="language gates failed"):
        run_manual_review(
            root=tmp_path,
            lang="german",
            level="A1",
            source=source,
            decisions_directory=reviews,
        )

    assert not (tmp_path / "german" / "A1.reviewed.csv").exists()


def test_apply_validates_before_atomically_replacing_canonical_csv(
    tmp_path: Path,
) -> None:
    source = tmp_path / "german" / "A1.proposed.csv"
    rows = [["gehen", "go", "去", "VERB"]]
    write_csv(source, rows)
    original_source = source.read_bytes()
    canonical = tmp_path / "german" / "A1.csv"
    canonical.write_text("existing canonical\n", encoding="utf-8")
    reviews = tmp_path / "manual_reviews" / "german" / "A1"
    write_decisions(
        reviews,
        [
            decision(
                2,
                rows[0],
                "fix",
                ["geht", "goes", "去", "VERB"],
            )
        ],
    )

    with pytest.raises(ValueError, match="language gates failed"):
        run_manual_review(
            root=tmp_path,
            lang="german",
            level="A1",
            source=source,
            decisions_directory=reviews,
            apply=True,
        )
    assert canonical.read_text(encoding="utf-8") == "existing canonical\n"

    write_decisions(
        reviews,
        [
            decision(
                2,
                rows[0],
                "fix",
                ["laufen", "run", "跑", "VERB"],
            )
        ],
    )
    result = run_manual_review(
        root=tmp_path,
        lang="german",
        level="A1",
        source=source,
        decisions_directory=reviews,
        apply=True,
    )

    assert result.output == canonical
    assert source.read_bytes() == original_source
    assert b"\r\n" not in canonical.read_bytes()
    with canonical.open(encoding="utf-8", newline="") as handle:
        assert list(csv.reader(handle)) == [
            HEADER,
            ["laufen", "run", "跑", "VERB"],
        ]
    assert list((tmp_path / "german").glob("*.tmp")) == []


def test_manual_review_append_merges_gap_rows_into_committed_csv(
    tmp_path: Path,
) -> None:
    committed = tmp_path / "german" / "A1.csv"
    gap = tmp_path / "german" / "A1.gap.proposed.csv"
    write_csv(
        committed,
        [
            ["Abend", "evening", "晚上", "NOUN"],
            ["Haus", "house", "房子", "NOUN"],
        ],
    )
    write_csv(
        gap,
        [
            ["Morgen", "morning", "早晨", "NOUN"],
            ["schlecht", "bad", "坏", "ADJ"],
        ],
    )
    reviews = tmp_path / "manual_reviews" / "german" / "A1"
    write_decisions(
        reviews,
        [
            decision(3, ["schlecht", "bad", "坏", "ADJ"], "drop"),
        ],
    )

    result = run_manual_review(
        root=tmp_path,
        lang="german",
        level="A1",
        source=gap,
        decisions_directory=reviews,
        apply=True,
        append=True,
        check_other_level_collisions=True,
    )

    assert result.output_count == 1
    assert result.output == committed
    with committed.open(encoding="utf-8", newline="") as handle:
        assert list(csv.reader(handle)) == [
            HEADER,
            ["Abend", "evening", "晚上", "NOUN"],
            ["Haus", "house", "房子", "NOUN"],
            ["Morgen", "morning", "早晨", "NOUN"],
        ]


def test_manual_review_cli_runs_without_api_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "german" / "A1.proposed.csv"
    rows = [["Abend", "evening", "晚上", "NOUN"]]
    write_csv(source, rows)
    reviews = tmp_path / "manual_reviews" / "german" / "A1"
    write_decisions(reviews, [decision(2, rows[0], "drop")])
    monkeypatch.delenv("API_KEY", raising=False)

    assert (
        main(
            [
                "manual-review",
                "--root",
                str(tmp_path),
                "--lang",
                "german",
                "--level",
                "A1",
                "--input",
                "german/A1.proposed.csv",
                "--decisions",
                "manual_reviews/german/A1",
            ]
        )
        == 0
    )

    assert "input=1 fix=0 drop=1 output=0" in capsys.readouterr().out


def test_manual_review_spanish_applies_and_gates_english_echo(tmp_path: Path) -> None:
    header = ["Spanish_Lemma", "English_Lemma", "Chinese_Lemma", "POS"]

    def write_es(path: Path, rows: list[list[str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    source = tmp_path / "spanish" / "A1.proposed.csv"
    write_es(
        source,
        [
            ["casa", "house", "房子", "NOUN"],
            ["house", "house", "房子", "NOUN"],
        ],
    )
    reviews = tmp_path / "manual_reviews" / "spanish" / "A1"
    write_decisions(reviews, [decision(3, ["house", "house", "房子", "NOUN"], "drop")])

    result = run_manual_review(
        root=tmp_path,
        lang="spanish",
        level="A1",
        source=source,
        decisions_directory=reviews,
    )
    assert result.output_count == 1
    with result.output.open(encoding="utf-8", newline="") as handle:
        assert list(csv.reader(handle)) == [
            header,
            ["casa", "house", "房子", "NOUN"],
        ]

    write_es(source, [["house", "house", "房子", "NOUN"]])
    write_decisions(reviews, [])
    # empty decisions dir has no jsonl -> error; use keep-all with no decisions file
    # instead test gate on fix that creates English echo
    write_es(source, [["casa", "house", "房子", "NOUN"]])
    write_decisions(
        reviews,
        [
            decision(
                2,
                ["casa", "house", "房子", "NOUN"],
                "fix",
                ["house", "house", "房子", "NOUN"],
            )
        ],
    )
    with pytest.raises(ValueError, match="language gates failed"):
        run_manual_review(
            root=tmp_path,
            lang="spanish",
            level="A1",
            source=source,
            decisions_directory=reviews,
        )
