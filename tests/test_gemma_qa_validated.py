from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import cast

import pytest

from scripts.gemma_qa.cefr import CefrClient, run_cefr, run_cefr_gap_refill
from scripts.gemma_qa.cli import main
from scripts.gemma_qa.language_repair import german_row_issues, repair_german_rows
from scripts.gemma_qa.ledger import Ledger
from scripts.gemma_qa.manual_review import run_manual_review
from scripts.gemma_qa.schemas import CefrReviewBatch, CefrReviewRow, ReviewAction, UPOS
from scripts.gemma_qa.validated import ValidatedStore, fingerprint, validated_store_path
from vocab_schema import TARGETS


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


class RaisingClient:
    def generate(self, **_kwargs: object) -> object:
        raise AssertionError("generate must not be called for frozen rows")

    @staticmethod
    def parse_response(
        response_json: dict[str, object],
        response_model: type[CefrReviewBatch],
    ) -> tuple[CefrReviewBatch, object]:
        from scripts.gemma_qa.client import GemmaClient

        return GemmaClient.parse_response(response_json, response_model)


class TrackingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[CefrReviewBatch],
        max_output_tokens: int,
    ):
        from scripts.gemma_qa.client import GenerationResult, Usage

        self.calls.append((model, prompt))
        payload = json.loads(prompt.splitlines()[-1])
        inputs = payload["inputs"] if isinstance(payload, dict) else payload
        rows = [
            {
                **item,
                "chinese_lemma": item["chinese_lemma"] or "词",
                "action": "keep",
            }
            for item in inputs
        ]
        parsed = response_model.model_validate({"rows": rows})
        response_json = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    parsed.model_dump(mode="json"),
                                    ensure_ascii=False,
                                )
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        return GenerationResult(
            parsed=parsed,
            usage=Usage(10, 5, 15),
            request_json={"prompt": prompt},
            response_json=response_json,
        )

    @staticmethod
    def parse_response(
        response_json: dict[str, object],
        response_model: type[CefrReviewBatch],
    ) -> tuple[CefrReviewBatch, object]:
        from scripts.gemma_qa.client import GemmaClient

        return GemmaClient.parse_response(response_json, response_model)


def test_fingerprint_uses_nfc_and_preserves_lemma_case() -> None:
    key = fingerprint(
        "german",
        "A1",
        "Ha\u0308us",
        "house",
        "房子",
        "NOUN",
    )
    expected = fingerprint("german", "A1", "Häus", "house", "房子", "NOUN")
    assert key == expected


def test_seed_from_csv_marks_all_committed_rows(tmp_path: Path) -> None:
    rows = [
        ["Abend", "evening", "晚上", "NOUN"],
        ["alt", "old", "老的", "ADJ"],
    ]
    write_csv(tmp_path / "german" / "A1.csv", rows)
    store = ValidatedStore(validated_store_path(tmp_path))
    store.seed_from_csv(tmp_path, lang="german", level="A1")
    assert store.count("german", "A1") == 2
    for row in rows:
        assert store.contains(
            "german",
            "A1",
            row[0],
            row[1],
            row[2],
            row[3],
        )
    store.close()


def test_run_cefr_skips_generate_when_all_rows_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        ["Abend", "evening", "晚上", "NOUN"],
        ["alt", "old", "老的", "ADJ"],
    ]
    write_csv(tmp_path / "german" / "A1.csv", rows)
    monkeypatch.setitem(TARGETS, "A1", len(rows))
    store = ValidatedStore(validated_store_path(tmp_path))
    store.seed_from_csv(tmp_path, lang="german", level="A1")
    store.close()
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RaisingClient()
    output = run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=cast(CefrClient, client),
        ledger=ledger,
    )
    assert output == tmp_path / "german" / "A1.proposed.csv"
    assert output.read_text(encoding="utf-8").splitlines()[1:] == [
        "Abend,evening,晚上,NOUN",
        "alt,old,老的,ADJ",
    ]
    ledger.close()


def test_run_cefr_reviews_only_unvalidated_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        ["Abend", "evening", "晚上", "NOUN"],
        ["alt", "old", "老的", "ADJ"],
        ["Haus", "house", "房子", "NOUN"],
    ]
    write_csv(tmp_path / "german" / "A1.csv", rows)
    monkeypatch.setitem(TARGETS, "A1", len(rows))
    store = ValidatedStore(validated_store_path(tmp_path))
    store.mark_rows("german", "A1", [tuple(row) for row in rows[:2]])
    store.close()
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = TrackingClient()
    run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=cast(CefrClient, client),
        ledger=ledger,
    )
    reviewed_ids = _cefr_review_input_ids(client.calls)
    assert reviewed_ids == {"german:A1:3"}
    ledger.close()


def _cefr_review_input_ids(
    calls: list[tuple[str, str]],
) -> set[str]:
    reviewed_ids: set[str] = set()
    for _, prompt in calls:
        if "adjudication" in prompt or "language-repair" in prompt:
            continue
        if not prompt.startswith("prompt_version=cefr-"):
            continue
        payload = json.loads(prompt.splitlines()[-1])
        inputs = payload["inputs"] if isinstance(payload, dict) else payload
        reviewed_ids.update(item["id"] for item in inputs)
    return reviewed_ids


def test_manual_review_apply_resyncs_validated_store(tmp_path: Path) -> None:
    rows = [
        ["Abend", "evening", "晚上", "NOUN"],
        ["alt", "old", "老的", "ADJ"],
        ["gehen", "go", "去", "VERB"],
    ]
    source = tmp_path / "german" / "A1.proposed.csv"
    write_csv(source, rows)
    reviews = tmp_path / "manual_reviews" / "german" / "A1"
    write_decisions(
        reviews,
        [
            decision(3, rows[1], "drop"),
            decision(4, rows[2], "fix", ["laufen", "run", "跑", "VERB"]),
        ],
    )
    run_manual_review(
        root=tmp_path,
        lang="german",
        level="A1",
        source=source,
        decisions_directory=reviews,
        apply=True,
    )
    store = ValidatedStore(validated_store_path(tmp_path))
    assert store.count("german", "A1") == 2
    assert store.contains("german", "A1", "Abend", "evening", "晚上", "NOUN")
    assert store.contains("german", "A1", "laufen", "run", "跑", "VERB")
    assert not store.contains("german", "A1", "alt", "old", "老的", "ADJ")
    assert not store.contains("german", "A1", "gehen", "go", "去", "VERB")
    store.close()


def test_validated_german_row_skips_language_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = ["bringen", "bring", "带来", "NOUN"]
    write_csv(tmp_path / "german" / "A1.csv", [row])
    monkeypatch.setitem(TARGETS, "A1", 1)
    review_row = CefrReviewRow(
        id="german:A1:1",
        lemma=row[0],
        english_lemma=row[1],
        chinese_lemma=row[2],
        upos=UPOS.NOUN,
        action=ReviewAction.KEEP,
    )
    assert german_row_issues(review_row)
    store = ValidatedStore(validated_store_path(tmp_path))
    store.seed_from_csv(tmp_path, lang="german", level="A1")
    store.close()
    ledger = Ledger(tmp_path / "ledger.sqlite3")

    class RepairSpy:
        def __init__(self) -> None:
            self.calls: list[object] = []

    spy = RepairSpy()
    original_repair = repair_german_rows

    def tracked_repair(rows, **kwargs):
        spy.calls.append(list(rows))
        return original_repair(rows, **kwargs)

    monkeypatch.setattr(
        "scripts.gemma_qa.cefr.repair_german_rows",
        tracked_repair,
    )
    client = RaisingClient()
    run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=cast(CefrClient, client),
        ledger=ledger,
    )
    assert spy.calls == []
    ledger.close()


def test_gap_refill_seeds_committed_rows_as_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TARGETS, "A1", 3)
    german = tmp_path / "german"
    english = tmp_path / "english"
    german.mkdir()
    english.mkdir()
    committed = german / "A1.csv"
    committed.write_text(
        "German_Lemma,English_Lemma,Chinese_Lemma,POS\n"
        "eins,one,一,NUM\n"
        "zwei,two,二,NUM\n",
        encoding="utf-8",
    )
    (english / "A1.csv").write_text(
        "English_Lemma,English_Lemma,Chinese_Lemma,POS\n"
        "one,one,一,NUM\n"
        "two,two,二,NUM\n"
        "three,three,三,NUM\n",
        encoding="utf-8",
    )
    from tests.test_gemma_qa_refill import RefillClient

    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient(lemma_by_id={"english:A1:3": "drei"})
    run_cefr_gap_refill(
        root=tmp_path,
        lang="german",
        level="A1",
        client=cast(CefrClient, client),
        ledger=ledger,
    )
    store = ValidatedStore(validated_store_path(tmp_path))
    assert store.count("german", "A1") == 2
    assert store.contains("german", "A1", "eins", "one", "一", "NUM")
    assert store.contains("german", "A1", "zwei", "two", "二", "NUM")
    store.close()
    ledger.close()


def test_seed_validated_cli(tmp_path: Path) -> None:
    rows = [["Abend", "evening", "晚上", "NOUN"]]
    write_csv(tmp_path / "german" / "A1.csv", rows)
    assert (
        main(
            [
                "seed-validated",
                "--root",
                str(tmp_path),
                "--lang",
                "german",
                "--level",
                "A1",
            ]
        )
        == 0
    )
    store = ValidatedStore(validated_store_path(tmp_path))
    assert store.count("german", "A1") == 1
    store.close()
