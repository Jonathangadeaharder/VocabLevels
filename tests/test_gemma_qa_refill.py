from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from scripts.gemma_qa.cefr import (
    CefrClient,
    CefrDocument,
    read_cefr_csv,
    run_cefr,
    run_cefr_gap_refill,
    write_reviewed_csv,
)
from scripts.gemma_qa.cefr_refill import (
    ReviewRequiredError,
    _bounded_novel_prompt,
    complete_cefr_rows,
    load_english_refill_concepts,
    load_gap_reject_keys,
    load_gap_rejected_english_keys,
    load_other_level_collision_keys,
)
from scripts.gemma_qa.client import GemmaClient, GenerationResult, Usage
from scripts.gemma_qa.config import MODEL_26B, MODEL_31B, MODEL_ADJUDICATION, INPUT_BATCH_TOKEN_CAP
from scripts.gemma_qa.ledger import Checkpoint, Ledger, prompt_hash
from scripts.gemma_qa.language_repair import german_row_issues
from scripts.gemma_qa.packing import TiktokenEstimator
from scripts.gemma_qa.prompts import (
    NOVEL_PROMPT_VERSION,
    REFILL_PROMPT_VERSION,
    build_novel_adjudication_prompt,
    build_novel_generation_prompt,
    build_novel_review_prompt,
    build_refill_adjudication_prompt,
    build_refill_generation_prompt,
    build_refill_review_prompt,
)
from scripts.gemma_qa.schemas import (
    CefrNovelBatch,
    CefrRefillBatch,
    CefrRefillConcept,
    CefrRefillRow,
    CefrNovelRow,
    CefrReviewRow,
    CefrReviewBatch,
    ReviewAction,
    UPOS,
)
from vocab_schema import TARGETS


def test_refill_output_schema_excludes_trusted_concept_fields() -> None:
    schema = CefrRefillRow.model_json_schema()
    assert set(schema["properties"]) == {
        "id",
        "lemma",
        "chinese_lemma",
        "action",
    }
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        CefrRefillRow.model_validate(
            {
                "id": "english:A1:1",
                "lemma": "Haus",
                "english_lemma": "house",
                "chinese_lemma": "房子",
                "upos": "NOUN",
                "action": "keep",
            }
        )


def test_novel_output_schema_owns_all_generated_lexical_fields() -> None:
    schema = CefrNovelRow.model_json_schema()
    assert set(schema["properties"]) == {
        "id",
        "lemma",
        "english_lemma",
        "chinese_lemma",
        "upos",
        "action",
    }
    with pytest.raises(ValueError, match="Field required"):
        CefrNovelRow.model_validate(
            {
                "id": "novel:german:A1:slot:1:round:1",
                "lemma": "Haus",
                "chinese_lemma": "房子",
                "upos": "NOUN",
                "action": "keep",
            }
        )


class RefillClient:
    def __init__(
        self,
        *,
        lemma_by_id: dict[str, str] | None = None,
        english_by_id: dict[str, str] | None = None,
        upos_by_id: dict[str, UPOS] | None = None,
        action_by_id: dict[str, ReviewAction] | None = None,
        disagree_novel: bool = False,
        repeat_until_excluded: bool = False,
    ) -> None:
        self.lemma_by_id = lemma_by_id or {}
        self.english_by_id = english_by_id or {}
        self.upos_by_id = upos_by_id or {}
        self.action_by_id = action_by_id or {}
        self.disagree_novel = disagree_novel
        self.repeat_until_excluded = repeat_until_excluded
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[CefrRefillBatch],
        max_output_tokens: int,
    ) -> GenerationResult[CefrRefillBatch]:
        self.calls.append((model, prompt))
        payload = json.loads(prompt.splitlines()[-1])
        if isinstance(payload, list):
            rows = [
                {
                    **row,
                    "chinese_lemma": row["chinese_lemma"] or "中",
                    "action": ReviewAction.KEEP.value,
                }
                for row in payload
            ]
        elif "slot_ids" in payload:
            if "candidate" in payload:
                rows = [dict(row) for row in payload["candidate"]["rows"]]
                if self.disagree_novel:
                    rows[0]["lemma"] = "Korrektur"
            elif "generated" in payload:
                rows = payload["reviewed"]["rows"]
            else:
                rows = [
                    self._novel_row(
                        slot_id,
                        exclusions=payload["accepted_exclusions"],
                    )
                    for slot_id in payload["slot_ids"]
                ]
        elif "concepts" in payload:
            rows = [
                {
                    "id": concept["id"],
                    "lemma": self.lemma_by_id.get(
                        concept["id"],
                        f"Neu{concept['id'].rsplit(':', 1)[-1]}",
                    ),
                    "chinese_lemma": concept["chinese_lemma"] or "新",
                    "action": self.action_by_id.get(
                        concept["id"],
                        ReviewAction.KEEP,
                    ).value,
                }
                for concept in payload["concepts"]
            ]
        elif "candidate" in payload:
            rows = payload["candidate"]["rows"]
        else:
            rows = payload["generated"]["rows"]
        parsed = response_model.model_validate({"rows": rows})
        response_json: dict[str, object] = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": parsed.model_dump_json()}],
                    }
                }
            ]
        }
        return GenerationResult(
            parsed=parsed,
            usage=Usage(10, 5, 15),
            request_json={"prompt": prompt, "max_output_tokens": max_output_tokens},
            response_json=response_json,
        )

    def _novel_row(
        self,
        slot_id: str,
        *,
        exclusions: Sequence[str],
    ) -> dict[str, object]:
        parts = slot_id.split(":")
        slot = int(parts[parts.index("slot") + 1])
        round_number = int(parts[parts.index("round") + 1])
        suffix = self._letters((slot - 1) * 5 + round_number)
        if self.repeat_until_excluded and "brot|NOUN" not in exclusions:
            return {
                "id": slot_id,
                "lemma": "Brot",
                "english_lemma": "bread",
                "chinese_lemma": "面包",
                "upos": "NOUN",
                "action": "keep",
            }
        return {
            "id": slot_id,
            "lemma": self.lemma_by_id.get(slot_id, f"Neu{suffix}"),
            "english_lemma": self.english_by_id.get(
                slot_id,
                f"novel concept {suffix}",
            ),
            "chinese_lemma": f"新概念{suffix}",
            "upos": self.upos_by_id.get(slot_id, UPOS.NOUN).value,
            "action": self.action_by_id.get(
                slot_id,
                ReviewAction.KEEP,
            ).value,
        }

    @staticmethod
    def _letters(value: int) -> str:
        letters = ""
        while value:
            value, remainder = divmod(value - 1, 26)
            letters = chr(ord("a") + remainder) + letters
        return letters

    @staticmethod
    def parse_response(
        response_json: dict[str, object],
        response_model: type[CefrRefillBatch],
    ) -> tuple[CefrRefillBatch, Usage]:
        return GemmaClient.parse_response(response_json, response_model)


class IdRepairClient(RefillClient):
    def __init__(self, expected_id: str) -> None:
        super().__init__()
        self.expected_id = expected_id

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[CefrRefillBatch],
        max_output_tokens: int,
    ) -> GenerationResult[CefrRefillBatch]:
        self.calls.append((model, prompt))
        row_id = "english:A1:altered" if len(self.calls) == 1 else self.expected_id
        parsed = response_model.model_validate(
            {
                "rows": [
                    {
                        "id": row_id,
                        "lemma": "Haus",
                        "chinese_lemma": "房子",
                        "action": "keep",
                    }
                ]
            }
        )
        response_json: dict[str, object] = {
            "candidates": [{"content": {"parts": [{"text": parsed.model_dump_json()}]}}]
        }
        return GenerationResult(
            parsed=parsed,
            usage=Usage(10, 5, 15),
            request_json={"prompt": prompt, "max_output_tokens": max_output_tokens},
            response_json=response_json,
        )


def review_row(index: int, *, duplicate_of: int | None = None) -> CefrReviewRow:
    identity = duplicate_of if duplicate_of is not None else index
    return CefrReviewRow(
        id=f"german:A1:{index + 1}",
        lemma=f"Alt{identity}",
        english_lemma=f"accepted{identity}",
        chinese_lemma=f"旧{identity}",
        upos=UPOS.NOUN,
        action=ReviewAction.KEEP,
    )


def concepts(count: int) -> list[CefrRefillConcept]:
    return [
        CefrRefillConcept(
            id=f"english:A1:{index + 1}",
            english_lemma=f"pivot{index}",
            chinese_lemma=f"枢{index}",
            upos=UPOS.NOUN,
        )
        for index in range(count)
    ]


def complete(
    accepted: Sequence[CefrReviewRow],
    pivot: Sequence[CefrRefillConcept],
    *,
    target: int,
    client: RefillClient,
    ledger: Ledger,
    collision_keys: set[tuple[str, UPOS]] | None = None,
) -> list[CefrReviewRow]:
    return complete_cefr_rows(
        accepted,
        concepts=pivot,
        collision_keys=collision_keys or set(),
        target=target,
        lang="german",
        level="A1",
        client=client,
        ledger=ledger,
    )


def test_refill_reattaches_trusted_english_and_upos_locally(tmp_path: Path) -> None:
    pivot = [
        CefrRefillConcept(
            id="english:A1:1",
            english_lemma="trusted concept",
            chinese_lemma="可信概念",
            upos=UPOS.VERB,
        )
    ]
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient(lemma_by_id={pivot[0].id: "vertrauen"})
    completed = complete(
        [],
        pivot,
        target=1,
        client=client,
        ledger=ledger,
    )
    assert completed[0].model_dump(mode="json") == {
        "id": "english:A1:1",
        "lemma": "vertrauen",
        "english_lemma": "trusted concept",
        "chinese_lemma": "可信概念",
        "upos": "VERB",
        "action": "keep",
    }
    ledger.close()


@pytest.mark.parametrize(
    ("lemma", "upos"),
    [
        ("wort", UPOS.NOUN),
        ("geh", UPOS.VERB),
    ],
)
def test_invalid_trusted_german_refill_is_rejected(
    tmp_path: Path,
    lemma: str,
    upos: UPOS,
) -> None:
    pivot = [
        CefrRefillConcept(
            id="english:A1:1",
            english_lemma="unclean pivot",
            chinese_lemma="不干净",
            upos=upos,
        )
    ]
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient(lemma_by_id={pivot[0].id: lemma})
    completed = complete(
        [],
        pivot,
        target=1,
        client=client,
        ledger=ledger,
    )
    assert completed[0].id.startswith("novel:german:A1:")
    assert all(not german_row_issues(row) for row in completed)
    ledger.close()


def test_dual_reviewed_novel_noun_is_canonicalized_before_acceptance(
    tmp_path: Path,
) -> None:
    slot_id = "novel:german:A1:slot:1:round:1"
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient(lemma_by_id={slot_id: "äpfel"})
    completed = complete(
        [],
        [],
        target=1,
        client=client,
        ledger=ledger,
    )
    assert completed[0].lemma == "Äpfel"
    assert german_row_issues(completed[0]) == []
    assert [model for model, _ in client.calls] == [MODEL_31B, MODEL_26B]
    ledger.close()


def test_german_completion_rejects_invalid_final_rows(tmp_path: Path) -> None:
    invalid = review_row(0).model_copy(update={"lemma": "wort"})
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    with pytest.raises(ReviewRequiredError, match="language issues"):
        complete(
            [invalid],
            [],
            target=1,
            client=RefillClient(),
            ledger=ledger,
        )
    ledger.close()


def test_altered_refill_id_is_repaired_before_checkpointing(tmp_path: Path) -> None:
    pivot = concepts(1)
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = IdRepairClient(pivot[0].id)
    completed = complete_cefr_rows(
        [],
        concepts=pivot,
        collision_keys=set(),
        target=1,
        lang="german",
        level="A1",
        client=client,
        ledger=ledger,
        single_model=MODEL_31B,
    )
    assert completed[0].id == pivot[0].id
    assert len(client.calls) == 2
    assert "refill IDs/order/cardinality differ" in client.calls[1][1]
    assert ledger.status().checkpoints == 1
    ledger.close()


def test_refill_batch_prompts_stay_under_input_cap() -> None:
    pivot = concepts(24)
    generated = CefrRefillBatch(
        rows=[
            CefrRefillRow(
                id=concept.id,
                lemma=f"DeutschesLemma{index}",
                chinese_lemma=f"精确中文翻译{index}",
                action=ReviewAction.KEEP,
            )
            for index, concept in enumerate(pivot)
        ]
    )
    reviewed = generated.model_copy(deep=True)
    reviewed.rows[0].lemma = "KorrigiertesLemma"
    prompts = [
        build_refill_generation_prompt(pivot, lang="german", level="A1"),
        build_refill_review_prompt(
            pivot,
            generated,
            lang="german",
            level="A1",
        ),
        build_refill_adjudication_prompt(
            pivot,
            generated,
            reviewed,
            lang="german",
            level="A1",
        ),
    ]
    estimator = TiktokenEstimator()
    prompt_token_counts = [estimator.count(prompt) for prompt in prompts]
    assert REFILL_PROMPT_VERSION == "cefr-refill-de-v2"
    assert prompt_token_counts == sorted(prompt_token_counts)
    assert all(count <= INPUT_BATCH_TOKEN_CAP for count in prompt_token_counts)


def test_novel_batch_prompts_stay_under_input_cap() -> None:
    slot_ids = [f"novel:german:A1:slot:{index}:round:1" for index in range(1, 11)]
    generated = CefrNovelBatch(
        rows=[
            CefrNovelRow(
                id=slot_id,
                lemma=f"Neuwort{index}",
                english_lemma=f"new concept {index}",
                chinese_lemma=f"新概念{index}",
                upos=UPOS.NOUN,
                action=ReviewAction.KEEP,
            )
            for index, slot_id in enumerate(slot_ids)
        ]
    )
    reviewed = generated.model_copy(deep=True)
    reviewed.rows[0].lemma = "Korrektur"
    exclusions = [
        "brot|NOUN",
        "neu|ADJ",
        *[
            f"ausführlicher-akzeptierter-ausschluss-{index}|NOUN"
            for index in range(2_000)
        ],
    ]
    prompts = [
        _bounded_novel_prompt(
            lambda included: build_novel_generation_prompt(
                slot_ids,
                lang="german",
                level="A1",
                exclusions=included,
            ),
            exclusions,
        ),
        _bounded_novel_prompt(
            lambda included: build_novel_review_prompt(
                slot_ids,
                generated,
                lang="german",
                level="A1",
                exclusions=included,
            ),
            exclusions,
        ),
        _bounded_novel_prompt(
            lambda included: build_novel_adjudication_prompt(
                slot_ids,
                generated,
                reviewed,
                lang="german",
                level="A1",
                exclusions=included,
            ),
            exclusions,
        ),
    ]
    estimator = TiktokenEstimator()
    assert NOVEL_PROMPT_VERSION == "cefr-novel-de-v2"
    assert all(estimator.count(prompt) <= INPUT_BATCH_TOKEN_CAP for prompt in prompts)
    included_by_prompt = [
        json.loads(prompt.splitlines()[-1])["accepted_exclusions"] for prompt in prompts
    ]
    assert all(included[:2] == exclusions[:2] for included in included_by_prompt)
    assert all(len(included) < len(exclusions) for included in included_by_prompt)


def test_round_ten_prompt_serialization_remains_checkpoint_stable() -> None:
    slot_ids = ["novel:german:A1:slot:7:round:10"]
    generated = CefrNovelBatch(
        rows=[
            CefrNovelRow(
                id=slot_ids[0],
                lemma="Haus",
                english_lemma="house",
                chinese_lemma="房子",
                upos=UPOS.NOUN,
                action=ReviewAction.KEEP,
            )
        ]
    )
    reviewed = generated.model_copy(deep=True)
    reviewed.rows[0].lemma = "Heim"
    prompts = [
        build_novel_generation_prompt(
            slot_ids,
            lang="german",
            level="A1",
            exclusions=["brot|NOUN"],
        ),
        build_novel_review_prompt(
            slot_ids,
            generated,
            lang="german",
            level="A1",
            exclusions=["brot|NOUN"],
        ),
        build_novel_adjudication_prompt(
            slot_ids,
            generated,
            reviewed,
            lang="german",
            level="A1",
            exclusions=["brot|NOUN"],
        ),
    ]
    assert [prompt_hash(prompt) for prompt in prompts] == [
        "0f422ca835f806a1a5d7913ced64b540f0da71fe7bab119ed76474ce20e5c809",
        "dba871624f57b3cd1bf54537764f5e27a6da7ff182b146929da13cbf080ccbe3",
        "20b06ddb350cb705bd70dcf74c77d6b6cf86b2b9880d3c16154a459bd6e15c69",
    ]


def test_round_eleven_prompt_adds_deterministic_initial_constraint() -> None:
    slot_id = "novel:german:A1:slot:1:round:11"
    prompt = build_novel_generation_prompt(
        [slot_id],
        lang="german",
        level="A1",
        exclusions=[],
    )
    payload = json.loads(prompt.splitlines()[-1])
    assert payload["domain_hints"] == [
        {
            "id": slot_id,
            "domain": "family",
            "initial": "a",
        }
    ]
    assert "Anfangsbuchstaben" in prompt


def test_bounded_prompt_keeps_maximal_rejected_first_prefix() -> None:
    rejected = ["brot|NOUN", "neu|ADJ", "kaufen|VERB"]
    accepted = [
        f"ausführlicher-akzeptierter-ausschluss-{index}|NOUN" for index in range(2_000)
    ]
    exclusions = [*rejected, *accepted]

    def build(included: Sequence[str]) -> str:
        return json.dumps(
            {"accepted_exclusions": list(included)},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    prompt = _bounded_novel_prompt(build, exclusions)
    included = json.loads(prompt)["accepted_exclusions"]
    estimator = TiktokenEstimator()
    assert included[:3] == rejected
    assert 3 < len(included) < len(exclusions)
    assert estimator.count(prompt) <= INPUT_BATCH_TOKEN_CAP
    assert estimator.count(build(exclusions[: len(included) + 1])) > INPUT_BATCH_TOKEN_CAP


def test_novel_v1_checkpoint_never_resumes_under_v2(tmp_path: Path) -> None:
    slot_id = "novel:german:A1:slot:1:round:1"
    current_prompt = build_novel_generation_prompt(
        [slot_id],
        lang="german",
        level="A1",
        exclusions=[],
    )
    old_prompt = current_prompt.replace("cefr-novel-de-v2", "cefr-novel-de-v1", 1)
    old_batch = CefrNovelBatch(
        rows=[
            CefrNovelRow(
                id=slot_id,
                lemma="Altcheckpoint",
                english_lemma="old checkpoint",
                chinese_lemma="旧检查点",
                upos=UPOS.NOUN,
                action=ReviewAction.KEEP,
            )
        ]
    )
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.store(
        Checkpoint(
            prompt_hash=prompt_hash(old_prompt),
            model=MODEL_31B,
            batch_id=(
                f"novel:cefr-novel-de-v1:german:A1:{slot_id}..{slot_id}:generation"
            ),
            request_json={"old": True},
            response_json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": old_batch.model_dump_json()}],
                        }
                    }
                ]
            },
            usage=Usage(1, 1, 2),
        )
    )
    client = RefillClient()
    completed = complete(
        [],
        [],
        target=1,
        client=client,
        ledger=ledger,
    )
    assert completed[0].lemma != "Altcheckpoint"
    assert len(client.calls) == 2
    ledger.close()


def test_rejected_blacklist_breaks_repetition_with_large_accepted_set(
    tmp_path: Path,
) -> None:
    accepted = [
        CefrReviewRow(
            id=f"german:A1:{index + 1}",
            lemma=f"Ausführlichesakzeptierteswort{index}",
            english_lemma=f"accepted concept {index}",
            chinese_lemma=f"已接受{index}",
            upos=UPOS.NOUN,
            action=ReviewAction.KEEP,
        )
        for index in range(300)
    ]
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient(repeat_until_excluded=True)
    completed = complete(
        accepted,
        [],
        target=302,
        client=client,
        ledger=ledger,
        collision_keys={("brot", UPOS.NOUN)},
    )
    assert len(completed) == 302
    novel_payloads = [
        json.loads(prompt.splitlines()[-1])
        for _, prompt in client.calls
        if "prompt_version=cefr-novel-de-v2" in prompt
    ]
    second_round = [
        payload
        for payload in novel_payloads
        if all(":round:2" in slot_id for slot_id in payload["slot_ids"])
    ]
    assert second_round
    assert all(
        payload["accepted_exclusions"][0] == "brot|NOUN" for payload in second_round
    )
    assert all(TiktokenEstimator().count(prompt) <= INPUT_BATCH_TOKEN_CAP for _, prompt in client.calls)
    ledger.close()


def test_pivot_exhaustion_triggers_novel_same_level_stage(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient()
    completed = complete(
        [review_row(0), review_row(1)],
        concepts(1),
        target=5,
        client=client,
        ledger=ledger,
    )
    assert len(completed) == 5
    assert [row.id for row in completed[-2:]] == [
        "novel:german:A1:slot:1:round:1",
        "novel:german:A1:slot:2:round:1",
    ]
    novel_prompts = [
        prompt
        for _, prompt in client.calls
        if "prompt_version=cefr-novel-de-v2" in prompt
    ]
    assert len(novel_prompts) == 2
    assert all('"level":"A1"' in prompt for prompt in novel_prompts)
    ledger.close()


def test_novel_collisions_and_drops_retry_stable_slots(tmp_path: Path) -> None:
    first_slot = "novel:german:A1:slot:1:round:1"
    second_slot = "novel:german:A1:slot:2:round:1"
    third_slot = "novel:german:A1:slot:3:round:1"
    fourth_slot = "novel:german:A1:slot:4:round:1"
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient(
        lemma_by_id={
            first_slot: "Alt0",
            fourth_slot: "Cross",
        },
        english_by_id={third_slot: "accepted0"},
        action_by_id={second_slot: ReviewAction.DROP},
    )
    completed = complete(
        [review_row(0), review_row(1)],
        [],
        target=6,
        client=client,
        ledger=ledger,
        collision_keys={("cross", UPOS.NOUN)},
    )
    assert [row.id for row in completed[-4:]] == [
        "novel:german:A1:slot:1:round:2",
        "novel:german:A1:slot:2:round:2",
        "novel:german:A1:slot:3:round:2",
        "novel:german:A1:slot:4:round:2",
    ]
    assert len(client.calls) == 4
    round_two_generation = next(
        json.loads(prompt.splitlines()[-1])
        for _, prompt in client.calls
        if "prompt_version=cefr-novel-de-v2\n" in prompt
        and '"candidate":' not in prompt
        and ":round:2" in prompt
    )
    assert round_two_generation["accepted_exclusions"][:4] == [
        "cross|NOUN",
        "neuk|NOUN",
        "neuf|NOUN",
        "alt0|NOUN",
    ]
    assert round_two_generation["accepted_exclusions"].count("alt0|NOUN") == 1
    ledger.close()


def test_novel_disagreement_uses_31b_adjudication(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient(disagree_novel=True)
    completed = complete(
        [],
        [],
        target=1,
        client=client,
        ledger=ledger,
    )
    assert completed[0].lemma == "Korrektur"
    models = [model for model, _ in client.calls]
    assert models[0] == MODEL_31B
    assert models[1] == MODEL_26B
    assert len(models) == 3
    assert models[2] not in {MODEL_31B, MODEL_26B}
    ledger.close()


def test_novel_lexical_hygiene_rejects_invalid_first_round(tmp_path: Path) -> None:
    slot_ids = [f"novel:german:A1:slot:{slot}:round:1" for slot in range(1, 5)]
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient(
        lemma_by_id={
            slot_ids[0]: "Name",
            slot_ids[1]: "Wort1",
            slot_ids[2]: "Wort?",
            slot_ids[3]: "zwei Wörter",
        },
        upos_by_id={slot_ids[0]: UPOS.PROPN},
    )
    completed = complete(
        [],
        [],
        target=4,
        client=client,
        ledger=ledger,
    )
    assert all(row.id.endswith("round:2") for row in completed)
    assert len(client.calls) == 4
    ledger.close()


def test_novel_rounds_twenty_one_through_thirty_remain_resumable(
    tmp_path: Path,
) -> None:
    rejected_ids = [
        f"novel:german:A1:slot:1:round:{round_number}" for round_number in range(1, 30)
    ]
    round_thirty = "novel:german:A1:slot:1:round:30"
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient(
        lemma_by_id={
            **{row_id: "Brot" for row_id in rejected_ids},
            round_thirty: "Vogel",
        },
        action_by_id={row_id: ReviewAction.DROP for row_id in rejected_ids},
    )
    completed = complete(
        [],
        [],
        target=1,
        client=client,
        ledger=ledger,
    )
    assert completed[0].id == round_thirty
    assert len(client.calls) == 60
    generation_payloads = [
        json.loads(prompt.splitlines()[-1])
        for _, prompt in client.calls
        if "prompt_version=cefr-novel-de-v2\n" in prompt
        and '"candidate":' not in prompt
    ]
    assert [payload["slot_ids"][0] for payload in generation_payloads] == [
        f"novel:german:A1:slot:1:round:{round_number}" for round_number in range(1, 31)
    ]
    assert all(
        payload["accepted_exclusions"][0] == "brot|NOUN"
        for payload in generation_payloads[1:]
    )

    resumed_client = RefillClient()
    resumed = complete(
        [],
        [],
        target=1,
        client=resumed_client,
        ledger=ledger,
    )
    assert resumed == completed
    assert resumed_client.calls == []
    ledger.close()


def test_round_eleven_initial_is_locally_enforced(tmp_path: Path) -> None:
    rejected_ids = [
        f"novel:german:A1:slot:1:round:{round_number}" for round_number in range(1, 11)
    ]
    round_twelve = "novel:german:A1:slot:1:round:12"
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient(
        lemma_by_id={round_twelve: "Bahn"},
        action_by_id={row_id: ReviewAction.DROP for row_id in rejected_ids},
    )
    completed = complete(
        [],
        [],
        target=1,
        client=client,
        ledger=ledger,
    )
    assert completed[0].id == round_twelve
    assert completed[0].lemma == "Bahn"
    assert len(client.calls) == 24
    ledger.close()


def test_collapsed_600_rows_refill_to_600_unique_and_resume(tmp_path: Path) -> None:
    accepted = [
        review_row(index, duplicate_of=index - 435 if index >= 435 else None)
        for index in range(600)
    ]
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient()
    completed = complete(
        accepted,
        concepts(600),
        target=600,
        client=client,
        ledger=ledger,
    )
    keys = {(row.lemma.casefold(), row.upos) for row in completed}
    assert len(completed) == 600
    assert len(keys) == 600
    assert len(client.calls) == 14
    assert {model for model, _ in client.calls} == {MODEL_31B, MODEL_26B}
    estimator = TiktokenEstimator()
    assert max(estimator.count(prompt) for _, prompt in client.calls) <= INPUT_BATCH_TOKEN_CAP
    assert (
        max(
            len(json.loads(prompt.splitlines()[-1])["concepts"])
            for _, prompt in client.calls
        )
        <= 24
    )

    resumed_client = RefillClient()
    resumed = complete(
        accepted,
        concepts(600),
        target=600,
        client=resumed_client,
        ledger=ledger,
    )
    assert resumed == completed
    assert resumed_client.calls == []
    ledger.close()


def test_pivot_exhaustion_refills_collapsed_source_to_exact_600_and_resumes(
    tmp_path: Path,
) -> None:
    accepted = [
        review_row(index, duplicate_of=index - 435 if index >= 435 else None)
        for index in range(600)
    ]
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient()
    completed = complete(
        accepted,
        concepts(92),
        target=600,
        client=client,
        ledger=ledger,
    )
    assert len(completed) == 600
    assert len({(row.lemma.casefold(), row.upos) for row in completed}) == 600
    novel_prompts = [
        prompt
        for _, prompt in client.calls
        if "prompt_version=cefr-novel-de-v2" in prompt
    ]
    assert len(client.calls) == 24
    assert len(novel_prompts) == 16
    assert (
        max(
            len(json.loads(prompt.splitlines()[-1])["slot_ids"])
            for prompt in novel_prompts
        )
        <= 10
    )
    estimator = TiktokenEstimator()
    assert all(estimator.count(prompt) <= INPUT_BATCH_TOKEN_CAP for prompt in novel_prompts)

    resumed_client = RefillClient()
    resumed = complete(
        accepted,
        concepts(92),
        target=600,
        client=resumed_client,
        ledger=ledger,
    )
    assert resumed == completed
    assert resumed_client.calls == []
    ledger.close()


def test_drop_and_cross_level_collision_consume_additional_concepts(
    tmp_path: Path,
) -> None:
    dropped_review = review_row(2).model_copy(
        update={"action": ReviewAction.DROP},
    )
    accepted = [review_row(0), review_row(1), dropped_review]
    pivot = concepts(4)
    client = RefillClient(
        lemma_by_id={
            pivot[0].id: "Blocked",
            pivot[1].id: "Dropped",
        },
        action_by_id={pivot[1].id: ReviewAction.DROP},
    )
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    completed = complete(
        accepted,
        pivot,
        target=4,
        client=client,
        ledger=ledger,
        collision_keys={("blocked", UPOS.NOUN)},
    )
    assert [row.lemma for row in completed] == ["Alt0", "Alt1", "Neu3", "Neu4"]
    ledger.close()


def test_bounded_novel_exhaustion_fails_without_writing(tmp_path: Path) -> None:
    accepted = [review_row(0)]
    pivot = concepts(1)
    rejected_ids = {
        f"novel:german:A1:slot:{slot}:round:{round_number}": ReviewAction.DROP
        for slot in (1, 2)
        for round_number in range(1, 31)
    }
    client = RefillClient(
        action_by_id={
            pivot[0].id: ReviewAction.DROP,
            **rejected_ids,
        }
    )
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    output = tmp_path / "A1.proposed.csv"
    with pytest.raises(ReviewRequiredError, match="novel refill exhausted 30 attempts"):
        complete(
            accepted,
            pivot,
            target=3,
            client=client,
            ledger=ledger,
        )
    assert not output.exists()
    assert len(client.calls) == 62
    ledger.close()


def test_bounded_novel_workflow_failure_leaves_source_and_proposal_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TARGETS, "A1", 2)
    german = tmp_path / "german"
    english = tmp_path / "english"
    german.mkdir()
    english.mkdir()
    source = german / "A1.csv"
    source.write_text(
        "German_Lemma,English_Lemma,Chinese_Lemma,POS\neins,one,一,NUM\n",
        encoding="utf-8",
    )
    (english / "A1.csv").write_text(
        "English_Lemma,English_Lemma,Chinese_Lemma,POS\none,one,一,NUM\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    rejected = {
        "english:A1:1": ReviewAction.DROP,
        **{
            f"novel:german:A1:slot:1:round:{round_number}": ReviewAction.DROP
            for round_number in range(1, 31)
        },
    }
    client = RefillClient(action_by_id=rejected)
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    with pytest.raises(ReviewRequiredError, match="novel refill exhausted 30 attempts"):
        run_cefr(
            root=tmp_path,
            lang="german",
            level="A1",
            client=cast(CefrClient, client),
            ledger=ledger,
        )
    assert source.read_bytes() == original
    assert not source.with_suffix(".proposed.csv").exists()
    ledger.close()


def test_overfilled_unique_review_requires_manual_review(tmp_path: Path) -> None:
    client = RefillClient()
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    with pytest.raises(ReviewRequiredError, match="exceed target"):
        complete(
            [review_row(0), review_row(1), review_row(2)],
            concepts(3),
            target=2,
            client=client,
            ledger=ledger,
        )
    assert client.calls == []
    ledger.close()


def test_loads_cross_level_keys_and_stable_english_concepts(tmp_path: Path) -> None:
    german = tmp_path / "german"
    english = tmp_path / "english"
    german.mkdir()
    english.mkdir()
    (german / "A2.csv").write_text(
        "German_Lemma,English_Lemma,Chinese_Lemma,POS\nHa\u0308us,house,房子,NOUN\n",
        encoding="utf-8",
    )
    (english / "A1.csv").write_text(
        "English_Lemma,English_Lemma,Chinese_Lemma,POS\n"
        "house,house,房子,NOUN\n"
        "go,go,,VERB\n",
        encoding="utf-8",
    )
    assert load_other_level_collision_keys(
        tmp_path,
        lang="german",
        level="A1",
    ) == {("häus", UPOS.NOUN)}
    loaded = load_english_refill_concepts(tmp_path, level="A1")
    assert [concept.id for concept in loaded] == ["english:A1:1", "english:A1:2"]
    assert loaded[0].chinese_lemma == "房子"
    assert loaded[1].chinese_lemma is None


def test_dry_run_writes_only_proposal_and_preserves_source(tmp_path: Path) -> None:
    german = tmp_path / "german"
    german.mkdir()
    source = german / "A1.csv"
    source.write_text(
        "German_Lemma,English_Lemma,Chinese_Lemma,POS\nalt,old,旧,ADJ\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    document: CefrDocument = read_cefr_csv(source, lang="german", level="A1")
    rows = [
        CefrReviewRow(
            id="english:A1:1",
            lemma="neu",
            english_lemma="new",
            chinese_lemma="新",
            upos=UPOS.ADJ,
            action=ReviewAction.KEEP,
        )
    ]
    output = write_reviewed_csv(
        document,
        CefrReviewBatch(rows=rows),
        apply=False,
    )
    assert source.read_bytes() == original
    assert output == german / "A1.proposed.csv"
    assert output.read_text(encoding="utf-8").splitlines() == [
        "German_Lemma,English_Lemma,Chinese_Lemma,POS",
        "neu,new,新,ADJ",
    ]


def test_underfilled_committed_level_refills_before_dry_run_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TARGETS, "A1", 4)
    german = tmp_path / "german"
    english = tmp_path / "english"
    german.mkdir()
    english.mkdir()
    source = german / "A1.csv"
    source.write_text(
        "German_Lemma,English_Lemma,Chinese_Lemma,POS\n"
        "eins,one,一,NUM\n"
        "zwei,two,二,NUM\n"
        "drei,three,三,NUM\n",
        encoding="utf-8",
    )
    (english / "A1.csv").write_text(
        "English_Lemma,English_Lemma,Chinese_Lemma,POS\n"
        "one,one,一,NUM\n"
        "two,two,二,NUM\n"
        "three,three,三,NUM\n"
        "four,four,四,NUM\n",
        encoding="utf-8",
    )
    original = source.read_bytes()
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient()
    output = run_cefr(
        root=tmp_path,
        lang="german",
        level="A1",
        client=cast(CefrClient, client),
        ledger=ledger,
    )
    assert source.read_bytes() == original
    assert len(output.read_text(encoding="utf-8").splitlines()) == 5
    ledger.close()


def test_gap_refill_writes_only_missing_rows_without_touching_committed_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TARGETS, "A1", 4)
    german = tmp_path / "german"
    english = tmp_path / "english"
    german.mkdir()
    english.mkdir()
    committed = german / "A1.csv"
    committed.write_text(
        "German_Lemma,English_Lemma,Chinese_Lemma,POS\n"
        "eins,one,一,NUM\n"
        "zwei,two,二,NUM\n"
        "drei,three,三,NUM\n",
        encoding="utf-8",
    )
    (english / "A1.csv").write_text(
        "English_Lemma,English_Lemma,Chinese_Lemma,POS\n"
        "one,one,一,NUM\n"
        "two,two,二,NUM\n"
        "three,three,三,NUM\n"
        "four,four,四,NUM\n",
        encoding="utf-8",
    )
    original = committed.read_bytes()
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient(lemma_by_id={"english:A1:4": "vier"})
    output = run_cefr_gap_refill(
        root=tmp_path,
        lang="german",
        level="A1",
        client=cast(CefrClient, client),
        ledger=ledger,
    )
    assert committed.read_bytes() == original
    assert output == german / "A1.gap.proposed.csv"
    assert output.read_text(encoding="utf-8").splitlines() == [
        "German_Lemma,English_Lemma,Chinese_Lemma,POS",
        "vier,four,四,NUM",
    ]
    ledger.close()


def test_gap_refill_writes_partial_rows_when_novel_exhausts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TARGETS, "A1", 5)
    german = tmp_path / "german"
    english = tmp_path / "english"
    german.mkdir()
    english.mkdir()
    committed = german / "A1.csv"
    committed.write_text(
        "German_Lemma,English_Lemma,Chinese_Lemma,POS\n"
        "eins,one,一,NUM\n"
        "zwei,two,二,NUM\n"
        "drei,three,三,NUM\n",
        encoding="utf-8",
    )
    (english / "A1.csv").write_text(
        "English_Lemma,English_Lemma,Chinese_Lemma,POS\n"
        "one,one,一,NUM\n"
        "two,two,二,NUM\n"
        "three,three,三,NUM\n",
        encoding="utf-8",
    )
    original = committed.read_bytes()
    client = RefillClient()
    original_novel = client._novel_row

    def selective_novel(slot_id: str, *, exclusions: list[str]) -> dict[str, object]:
        row = original_novel(slot_id, exclusions=exclusions)
        if ":slot:1:" in slot_id and ":round:1" in slot_id:
            row["action"] = ReviewAction.KEEP.value
            row["lemma"] = "Neuheit"
            row["english_lemma"] = "novelty"
            row["chinese_lemma"] = "新奇"
            row["upos"] = UPOS.NOUN.value
        else:
            row["action"] = ReviewAction.DROP.value
        return row

    client._novel_row = selective_novel  # type: ignore[method-assign]
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    output = run_cefr_gap_refill(
        root=tmp_path,
        lang="german",
        level="A1",
        client=cast(CefrClient, client),
        ledger=ledger,
    )
    assert committed.read_bytes() == original
    assert output == german / "A1.gap.proposed.csv"
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "German_Lemma,English_Lemma,Chinese_Lemma,POS"
    assert len(lines) == 2
    assert lines[1].startswith("Neuheit,")
    ledger.close()


def write_gap_decisions(
    directory: Path,
    decisions: list[dict[str, object]],
    *,
    filename: str = "review.jsonl",
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(
        "".join(
            json.dumps(decision, ensure_ascii=False) + "\n" for decision in decisions
        ),
        encoding="utf-8",
    )


def gap_decision(
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


def test_load_gap_reject_keys_collects_drop_and_fix_expected_lemmas(
    tmp_path: Path,
) -> None:
    write_gap_decisions(
        tmp_path,
        [
            gap_decision(2, ["Get", "get", "得到", "VERB"], "drop"),
            gap_decision(
                3,
                ["best", "best", "最好的", "ADJ"],
                "fix",
                ["gut", "best", "最好的", "ADJ"],
            ),
        ],
    )
    keys = load_gap_reject_keys(tmp_path)
    assert keys == {
        ("get", UPOS.VERB),
        ("best", UPOS.ADJ),
    }


def test_load_gap_rejected_english_keys_collects_expected_english_concepts(
    tmp_path: Path,
) -> None:
    write_gap_decisions(
        tmp_path,
        [
            gap_decision(2, ["Get", "get", "得到", "VERB"], "drop"),
            gap_decision(
                3,
                ["best", "best", "最好的", "ADJ"],
                "fix",
                ["gut", "best", "最好的", "ADJ"],
            ),
        ],
    )
    keys = load_gap_rejected_english_keys(tmp_path)
    assert keys == {
        ("get", UPOS.VERB),
        ("best", UPOS.ADJ),
    }


def test_zero_accept_pivot_pass_skips_represented_retry_and_uses_novel(
    tmp_path: Path,
) -> None:
    accepted = [review_row(0)]
    pivot = [
        CefrRefillConcept(
            id="english:A1:2",
            english_lemma="fresh concept",
            chinese_lemma="新",
            upos=UPOS.NOUN,
        ),
        CefrRefillConcept(
            id="english:A1:1",
            english_lemma="accepted0",
            chinese_lemma="旧0",
            upos=UPOS.NOUN,
        ),
    ]
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient(lemma_by_id={"english:A1:2": "wort"})
    completed = complete(
        accepted,
        pivot,
        target=2,
        client=client,
        ledger=ledger,
    )
    assert len(completed) == 2
    assert completed[1].id.startswith("novel:german:A1:")
    refill_payloads = [
        json.loads(prompt.splitlines()[-1])
        for _, prompt in client.calls
        if "concepts" in json.loads(prompt.splitlines()[-1])
    ]
    refill_concept_ids = {
        concept["id"] for payload in refill_payloads for concept in payload["concepts"]
    }
    assert refill_concept_ids == {"english:A1:2"}
    ledger.close()


def test_gap_refill_applies_reject_decisions_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TARGETS, "A1", 4)
    german = tmp_path / "german"
    english = tmp_path / "english"
    german.mkdir()
    english.mkdir()
    committed = german / "A1.csv"
    committed.write_text(
        "German_Lemma,English_Lemma,Chinese_Lemma,POS\n"
        "eins,one,一,NUM\n"
        "zwei,two,二,NUM\n"
        "drei,three,三,NUM\n",
        encoding="utf-8",
    )
    (english / "A1.csv").write_text(
        "English_Lemma,English_Lemma,Chinese_Lemma,POS\n"
        "one,one,一,NUM\n"
        "two,two,二,NUM\n"
        "three,three,三,NUM\n"
        "four,four,四,NUM\n"
        "get,get,得到,VERB\n",
        encoding="utf-8",
    )
    decisions = tmp_path / "manual_reviews" / "german" / "A1" / "gap"
    write_gap_decisions(
        decisions,
        [gap_decision(2, ["Get", "get", "得到", "VERB"], "drop")],
    )
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    client = RefillClient(lemma_by_id={"english:A1:4": "vier"})
    output = run_cefr_gap_refill(
        root=tmp_path,
        lang="german",
        level="A1",
        client=cast(CefrClient, client),
        ledger=ledger,
        reject_decisions_dir=decisions,
    )
    refill_payloads = [
        json.loads(prompt.splitlines()[-1])
        for _, prompt in client.calls
        if "concepts" in json.loads(prompt.splitlines()[-1])
    ]
    refill_english_lemmas = {
        concept["english_lemma"]
        for payload in refill_payloads
        for concept in payload["concepts"]
    }
    assert refill_english_lemmas == {"four"}
    assert "get" not in refill_english_lemmas
    assert output.read_text(encoding="utf-8").splitlines() == [
        "German_Lemma,English_Lemma,Chinese_Lemma,POS",
        "vier,four,四,NUM",
    ]
    ledger.close()


def test_completion_protects_committed_loanwords_from_language_assert(
    tmp_path: Path,
) -> None:
    """Gap refill must not re-fail frozen cognates (Hotel, Hand, Name, ...)."""
    committed = review_row(0).model_copy(
        update={"lemma": "Hotel", "english_lemma": "hotel", "upos": UPOS.NOUN}
    )
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    completed = complete_cefr_rows(
        [committed],
        concepts=[],
        collision_keys=set(),
        target=1,
        lang="german",
        level="A1",
        client=RefillClient(),
        ledger=ledger,
        single_model=MODEL_31B,
        protect_accepted=True,
    )
    assert len(completed) == 1
    assert completed[0].lemma == "Hotel"
    ledger.close()
