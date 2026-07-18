from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from scripts.gemma_qa.cefr import CefrClient, run_cefr
from scripts.gemma_qa.client import GemmaClient, GenerationResult, Usage
from scripts.gemma_qa.config import MODEL_26B, MODEL_31B, MODEL_ADJUDICATION, INPUT_BATCH_TOKEN_CAP
from scripts.gemma_qa.language_repair import (
    canonicalize_repaired_german_noun,
    cefr_row_issues,
    german_row_issues,
    repair_german_rows,
)
from scripts.gemma_qa.ledger import Ledger
from scripts.gemma_qa.packing import TiktokenEstimator
from scripts.gemma_qa.prompts import (
    LANGUAGE_REPAIR_PROMPT_VERSION,
    build_language_repair_adjudication_prompt,
    build_language_repair_generation_prompt,
    build_language_repair_review_prompt,
)
from scripts.gemma_qa.schemas import (
    CefrLanguageRepairItem,
    CefrReviewBatch,
    CefrReviewRow,
    ReviewAction,
    UPOS,
)
from vocab_schema import TARGETS


class RepairClient:
    def __init__(
        self,
        corrections: dict[str, dict[str, object]],
        *,
        review_corrections: dict[str, dict[str, object]] | None = None,
        refill_lemma: str = "Neu",
    ) -> None:
        self.corrections = corrections
        self.review_corrections = review_corrections or corrections
        self.refill_lemma = refill_lemma
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[CefrReviewBatch],
        max_output_tokens: int,
    ) -> GenerationResult[CefrReviewBatch]:
        self.calls.append((model, prompt))
        payload = json.loads(prompt.splitlines()[-1])
        if isinstance(payload, list):
            rows = [
                {
                    **source,
                    "chinese_lemma": source["chinese_lemma"] or "含义",
                    "action": "keep",
                }
                for source in payload
            ]
        elif "prompt_version=cefr-language-repair-de-v1" not in prompt:
            if "slot_ids" in payload:
                if "candidate" in payload:
                    rows = payload["candidate"]["rows"]
                elif "generated" in payload:
                    rows = payload["reviewed"]["rows"]
                else:
                    rows = [
                        {
                            "id": slot_id,
                            "lemma": "Ersatz",
                            "english_lemma": "replacement",
                            "chinese_lemma": "替代",
                            "upos": "NOUN",
                            "action": "keep",
                        }
                        for slot_id in payload["slot_ids"]
                    ]
            elif "concepts" in payload:
                rows = [
                    {
                        "id": concept["id"],
                        "lemma": self.refill_lemma,
                        "chinese_lemma": concept["chinese_lemma"] or "新",
                        "action": "keep",
                    }
                    for concept in payload["concepts"]
                ]
            elif "candidate" in payload:
                rows = payload["candidate"]["rows"]
            else:
                rows = payload["reviewed"]["rows"]
        elif "candidate" in payload:
            source_rows = payload["candidate"]["rows"]
            corrections = self.review_corrections
            rows = [
                {
                    **source,
                    **corrections.get(source["id"], {}),
                }
                for source in source_rows
            ]
        elif "repaired" in payload:
            source_rows = payload["reviewed"]["rows"]
            corrections = {}
            rows = [
                {
                    **source,
                    **corrections.get(source["id"], {}),
                }
                for source in source_rows
            ]
        else:
            source_rows = [item["row"] for item in payload["items"]]
            corrections = self.corrections
            rows = [
                {
                    **source,
                    **corrections.get(source["id"], {}),
                }
                for source in source_rows
            ]
        parsed = response_model.model_validate({"rows": rows})
        response_json: dict[str, object] = {
            "candidates": [{"content": {"parts": [{"text": parsed.model_dump_json()}]}}]
        }
        return GenerationResult(
            parsed=parsed,
            usage=Usage(10, 5, 15),
            request_json={"prompt": prompt, "max_outputTokens": max_output_tokens},
            response_json=response_json,
        )

    @staticmethod
    def parse_response(
        response_json: dict[str, object],
        response_model: type[CefrReviewBatch],
    ) -> tuple[CefrReviewBatch, Usage]:
        return GemmaClient.parse_response(response_json, response_model)


def row(
    lemma: str,
    upos: UPOS,
    *,
    english_lemma: str = "meaning",
) -> CefrReviewRow:
    return CefrReviewRow(
        id=f"german:A1:{lemma}",
        lemma=lemma,
        english_lemma=english_lemma,
        chinese_lemma="含义",
        upos=upos,
        action=ReviewAction.KEEP,
    )


@pytest.mark.parametrize(
    ("lemma", "upos", "expected_code"),
    [
        ("buchstabe", UPOS.NOUN, "german.noun_requires_uppercase"),
        ("heute", UPOS.VERB, "german.verb_requires_infinitive"),
        ("Berlin", UPOS.PROPN, "cefr.forbidden_upos"),
    ],
)
def test_german_gate_returns_structured_issues(
    lemma: str,
    upos: UPOS,
    expected_code: str,
) -> None:
    issues = german_row_issues(row(lemma, upos))
    assert [issue.code for issue in issues] == [expected_code]
    assert all(issue.message for issue in issues)


@pytest.mark.parametrize(
    ("lemma", "upos"),
    [
        ("Buchstabe", UPOS.NOUN),
        ("bringen", UPOS.VERB),
        ("tun", UPOS.VERB),
        ("heute", UPOS.ADV),
    ],
)
def test_german_gate_accepts_valid_citation_forms(lemma: str, upos: UPOS) -> None:
    assert german_row_issues(row(lemma, upos)) == []


@pytest.mark.parametrize(
    ("lemma", "english_lemma", "upos"),
    [
        ("Get", "get", UPOS.VERB),
        ("best", "best", UPOS.ADJ),
        ("Work", "work", UPOS.NOUN),
        ("Pen", "pen", UPOS.NOUN),
        ("again", "again", UPOS.ADV),
    ],
)
def test_non_english_gate_rejects_exact_english_echo(
    lemma: str,
    english_lemma: str,
    upos: UPOS,
) -> None:
    candidate = row(lemma, upos, english_lemma=english_lemma)
    codes = [issue.code for issue in cefr_row_issues(candidate, lang="german")]
    assert "cefr.english_echo" in codes


def test_english_echo_gate_does_not_apply_to_english_rows() -> None:
    candidate = row("work", UPOS.NOUN, english_lemma="work")
    assert "cefr.english_echo" not in [
        issue.code for issue in cefr_row_issues(candidate, lang="english")
    ]


@pytest.mark.parametrize(
    ("lemma", "upos"),
    [
        ("Schlecht", UPOS.ADJ),
        ("Heute", UPOS.ADV),
        ("Hör", UPOS.INTJ),
        ("Bringen", UPOS.VERB),
    ],
)
def test_german_gate_rejects_non_noun_ascii_capitalized_lemmas(
    lemma: str,
    upos: UPOS,
) -> None:
    codes = [issue.code for issue in german_row_issues(row(lemma, upos))]
    assert codes == ["german.non_noun_capitalized"]


@pytest.mark.parametrize(
    ("lemma", "expected"),
    [
        ("äpfel", "Äpfel"),
        ("ßache", "SSache"),
    ],
)
def test_canonicalizes_sole_repaired_noun_case_issue(
    lemma: str,
    expected: str,
) -> None:
    repaired = row(lemma, UPOS.NOUN)
    canonical = canonicalize_repaired_german_noun(repaired)
    assert canonical.lemma == expected
    assert canonical.model_dump(exclude={"lemma"}) == repaired.model_dump(
        exclude={"lemma"}
    )


def test_canonicalization_leaves_wrong_pos_and_multiple_issues_unchanged() -> None:
    wrong_pos = row("äpfel", UPOS.ADJ)
    multiple_issues = row("zwei wörter", UPOS.NOUN)
    assert canonicalize_repaired_german_noun(wrong_pos) == wrong_pos
    assert canonicalize_repaired_german_noun(multiple_issues) == multiple_issues


def test_language_repair_prompts_stay_under_input_cap() -> None:
    rows = [row(f"wort{index}", UPOS.NOUN) for index in range(8)]
    items = [
        CefrLanguageRepairItem(
            row=current,
            issues=german_row_issues(current),
        )
        for current in rows
    ]
    repaired = CefrReviewBatch(
        rows=[
            current.model_copy(update={"lemma": current.lemma.title()})
            for current in rows
        ]
    )
    reviewed = repaired.model_copy(deep=True)
    reviewed.rows[0].lemma = "Wort"
    prompts = [
        build_language_repair_generation_prompt(
            items,
            lang="german",
            level="A1",
            pass_number=1,
        ),
        build_language_repair_review_prompt(
            items,
            repaired,
            lang="german",
            level="A1",
            pass_number=1,
        ),
        build_language_repair_adjudication_prompt(
            items,
            repaired,
            reviewed,
            lang="german",
            level="A1",
            pass_number=1,
        ),
    ]
    estimator = TiktokenEstimator()
    assert LANGUAGE_REPAIR_PROMPT_VERSION == "cefr-language-repair-de-v1"
    assert all(estimator.count(prompt) <= INPUT_BATCH_TOKEN_CAP for prompt in prompts)
    assert all("blind" in prompt.casefold() for prompt in prompts)


def test_dual_language_repair_fixes_noun_case_and_wrong_pos(tmp_path: Path) -> None:
    rows = [
        row("buchstabe", UPOS.NOUN),
        row("bringen", UPOS.NOUN),
    ]
    client = RepairClient(
        {
            rows[0].id: {"lemma": "Buchstabe", "action": "fix"},
            rows[1].id: {"upos": "VERB", "action": "fix"},
        }
    )
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    repaired = repair_german_rows(
        rows,
        client=client,
        ledger=ledger,
        lang="german",
        level="A1",
        pass_number=1,
    )
    assert [(current.lemma, current.upos) for current in repaired] == [
        ("Buchstabe", UPOS.NOUN),
        ("bringen", UPOS.VERB),
    ]
    assert [model for model, _ in client.calls] == [MODEL_31B, MODEL_26B]
    assert all(german_row_issues(current) == [] for current in repaired)
    ledger.close()


def write_workflow_data(
    root: Path,
    german_rows: list[str],
    english_rows: list[str],
) -> Path:
    german = root / "german"
    english = root / "english"
    german.mkdir()
    english.mkdir()
    source = german / "A1.csv"
    source.write_text(
        "German_Lemma,English_Lemma,Chinese_Lemma,POS\n"
        + "\n".join(german_rows)
        + "\n",
        encoding="utf-8",
    )
    (english / "A1.csv").write_text(
        "English_Lemma,English_Lemma,Chinese_Lemma,POS\n"
        + "\n".join(english_rows)
        + "\n",
        encoding="utf-8",
    )
    return source


def test_collision_created_by_post_repair_canonicalization_is_refilled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TARGETS, "A1", 2)
    source = write_workflow_data(
        tmp_path,
        ["buch,book,书,NOUN", "Buchstabe,letter,字母,NOUN"],
        ["book,book,书,NOUN", "new,new,新,NOUN"],
    )
    client = RepairClient(
        {"german:A1:1": {"lemma": "buchstabe", "action": "fix"}},
        refill_lemma="Neu",
    )
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    output = run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=cast(CefrClient, client),
        ledger=ledger,
    )
    assert output != source
    assert output.read_text(encoding="utf-8").splitlines()[1:] == [
        "Buchstabe,book,书,NOUN",
        "Neu,new,新,NOUN",
    ]
    assert source.read_text(encoding="utf-8").splitlines()[1] == "buch,book,书,NOUN"
    ledger.close()


def test_blind_capitalization_is_rejected_and_row_is_refilled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TARGETS, "A1", 1)
    source = write_workflow_data(
        tmp_path,
        ["bringen,bring,带来,NOUN"],
        ["new,new,新,NOUN"],
    )
    client = RepairClient(
        {"german:A1:1": {"lemma": "Bringen", "action": "fix"}},
        review_corrections={"german:A1:1": {"action": "drop"}},
        refill_lemma="Neu",
    )
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    output = run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=cast(CefrClient, client),
        ledger=ledger,
    )
    assert output.read_text(encoding="utf-8").splitlines()[1] == "Neu,new,新,NOUN"
    repair_models = [
        model
        for model, prompt in client.calls
        if "prompt_version=cefr-language-repair-de-v1" in prompt
    ]
    assert repair_models[0] == MODEL_31B
    assert repair_models[1] == MODEL_26B
    assert repair_models[2]  # multi-model adjudication pool
    assert source.read_text(encoding="utf-8").splitlines()[1] == (
        "bringen,bring,带来,NOUN"
    )
    ledger.close()


def test_invalid_trusted_refill_cannot_reintroduce_post_repair_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TARGETS, "A1", 1)
    source = write_workflow_data(
        tmp_path,
        ["geh,go,去,VERB"],
        ["go,go,去,VERB"],
    )
    original = source.read_bytes()
    client = RepairClient({}, refill_lemma="geh")
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    output = run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=cast(CefrClient, client),
        ledger=ledger,
    )
    assert source.read_bytes() == original
    written = output.read_text(encoding="utf-8").splitlines()
    assert written[1] == "Ersatz,replacement,替代,NOUN"
    assert "geh,go,去,VERB" not in written
    ledger.close()


def test_english_citation_mismatch_is_flagged() -> None:
    row = CefrReviewRow(
        id="english:A2:1",
        lemma="dreams",
        english_lemma="dream",
        chinese_lemma="梦",
        upos=UPOS.NOUN,
        action=ReviewAction.KEEP,
    )
    codes = [issue.code for issue in cefr_row_issues(row, lang="english")]
    assert "english.citation_mismatch" in codes


def test_english_matching_citation_is_clean() -> None:
    row = CefrReviewRow(
        id="english:A2:1",
        lemma="dream",
        english_lemma="dream",
        chinese_lemma="梦",
        upos=UPOS.NOUN,
        action=ReviewAction.KEEP,
    )
    assert cefr_row_issues(row, lang="english") == []


def test_canonicalize_english_citation_rewrites_lemma() -> None:
    from scripts.gemma_qa.language_repair import canonicalize_english_citation

    row = CefrReviewRow(
        id="english:B1:1",
        lemma="answered",
        english_lemma="answer",
        chinese_lemma="回答",
        upos=UPOS.VERB,
        action=ReviewAction.KEEP,
    )
    fixed = canonicalize_english_citation(row)
    assert fixed.lemma == "answer"
    assert fixed.english_lemma == "answer"
    assert fixed.action is ReviewAction.FIX
    assert cefr_row_issues(fixed, lang="english") == []


def test_non_english_still_flags_english_echo() -> None:
    row = CefrReviewRow(
        id="german:A1:1",
        lemma="Hotel",
        english_lemma="hotel",
        chinese_lemma="酒店",
        upos=UPOS.NOUN,
        action=ReviewAction.KEEP,
    )
    codes = [issue.code for issue in cefr_row_issues(row, lang="german")]
    assert "cefr.english_echo" in codes
