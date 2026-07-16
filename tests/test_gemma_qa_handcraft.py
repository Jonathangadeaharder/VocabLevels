from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts.gemma_qa.client import GemmaClient, GenerationResult, Usage
from scripts.gemma_qa.cli import build_parser
from scripts.gemma_qa.config import MODEL_26B, MODEL_31B, INPUT_BATCH_TOKEN_CAP
from scripts.gemma_qa.handcraft import (
    HANDCRAFT_MAX_OUTPUT_TOKENS,
    MAX_HANDCRAFT_SENTENCES_PER_BATCH,
    SentenceTargets,
    run_handcraft,
    select_handcraft_targets,
    validate_handcraft_batch,
    write_handcraft,
)
from scripts.gemma_qa.ledger import Checkpoint, Ledger, prompt_hash
from scripts.gemma_qa.packing import TiktokenEstimator
from scripts.gemma_qa.prompts import (
    HANDCRAFT_PROMPT_VERSION,
    build_handcraft_generation_prompt,
)
from scripts.gemma_qa.schemas import (
    HandcraftBatch,
    HandcraftSentence,
    HandcraftToken,
    UPOS,
)


class QueueClient:
    def __init__(
        self,
        responses: Sequence[HandcraftBatch],
        *,
        ledger: Ledger | None = None,
    ) -> None:
        self.responses = list(responses)
        self.ledger = ledger
        self.calls: list[str] = []
        self.prompts: list[str] = []
        self.output_limits: list[int] = []
        self.checkpoint_counts: list[int] = []

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[HandcraftBatch],
        max_output_tokens: int,
    ) -> GenerationResult[HandcraftBatch]:
        self.calls.append(model)
        self.prompts.append(prompt)
        self.output_limits.append(max_output_tokens)
        if self.ledger is not None:
            self.checkpoint_counts.append(self.ledger.status().checkpoints)
        parsed = self.responses.pop(0)
        response_json: dict[str, object] = {
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
            ]
        }
        return GenerationResult(
            parsed=parsed,
            usage=Usage(10, 10, 20),
            request_json={
                "prompt": prompt,
                "maxOutputTokens": max_output_tokens,
                "schema": response_model.__name__,
            },
            response_json=response_json,
        )

    @staticmethod
    def parse_response(
        response_json: dict[str, object],
        response_model: type[HandcraftBatch],
    ) -> tuple[HandcraftBatch, Usage]:
        return GemmaClient.parse_response(response_json, response_model)


def make_vocab(root: Path, rows: int = 60) -> None:
    directory = root / "german"
    directory.mkdir(parents=True)
    with (directory / "A1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["German_Lemma", "English_Lemma", "Chinese_Lemma", "POS"])
        for index in range(1, rows + 1):
            writer.writerow([f"Wort{index}", f"word{index}", "", "NOUN"])


def make_cross_batch_duplicate_vocab(root: Path) -> None:
    directory = root / "german"
    directory.mkdir(parents=True)
    with (directory / "A1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["German_Lemma", "English_Lemma", "Chinese_Lemma", "POS"])
        for index in range(1, 13):
            lemma_index = index - 9 if index >= 10 else index
            writer.writerow([f"Wort{lemma_index}", f"word{lemma_index}", "", "NOUN"])


def batch_for(assignments: Sequence[SentenceTargets]) -> HandcraftBatch:
    sentences = []
    for assignment in assignments:
        forms = [target.lemma for target in assignment.targets]
        text = " ".join(forms) + "."
        tokens = [
            HandcraftToken(
                id=str(index),
                form=target.lemma,
                lemma=target.lemma,
                upos=target.upos,
            )
            for index, target in enumerate(assignment.targets, start=1)
        ]
        tokens.append(
            HandcraftToken(
                id=str(len(tokens) + 1),
                form=".",
                lemma=".",
                upos=UPOS.PUNCT,
            )
        )
        sentences.append(
            HandcraftSentence(
                sent_id=assignment.sent_id,
                text=text,
                target_ids=[target.id for target in assignment.targets],
                tokens=tokens,
            )
        )
    return HandcraftBatch(sentences=sentences)


def response_json_for(batch: HandcraftBatch) -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                batch.model_dump(mode="json"),
                                ensure_ascii=False,
                            )
                        }
                    ]
                }
            }
        ]
    }


def generation_prompt_for(
    assignments: Sequence[SentenceTargets],
    *,
    lang: str = "de",
    level: str = "A1",
) -> str:
    payload = [
        {
            "sent_id": assignment.sent_id,
            "targets": [
                {
                    "id": target.id,
                    "lemma": target.lemma,
                    "upos": target.upos.value,
                }
                for target in assignment.targets
            ],
        }
        for assignment in assignments
    ]
    return build_handcraft_generation_prompt(
        payload,
        lang=lang,
        level=level,
    )


def generation_batch_id(assignments: Sequence[SentenceTargets]) -> str:
    return (
        f"handcraft:{HANDCRAFT_PROMPT_VERSION}:de:A1:"
        f"{assignments[0].sent_id}..{assignments[-1].sent_id}:generation"
    )


def one_assignment() -> SentenceTargets:
    root = Path()
    return SentenceTargets.from_values(
        sent_id="handcraft-de-a1-001",
        targets=[("german:A1:1", "Abend", UPOS.NOUN)],
        source=root / "german" / "A1.csv",
    )


def one_batch() -> HandcraftBatch:
    return HandcraftBatch(
        sentences=[
            HandcraftSentence(
                sent_id="handcraft-de-a1-001",
                text="Guten Abend.",
                target_ids=["german:A1:1"],
                tokens=[
                    HandcraftToken(
                        id="1",
                        form="Guten",
                        lemma="gut",
                        upos=UPOS.ADJ,
                    ),
                    HandcraftToken(
                        id="2",
                        form="Abend",
                        lemma="Abend",
                        upos=UPOS.NOUN,
                    ),
                    HandcraftToken(
                        id="3",
                        form=".",
                        lemma=".",
                        upos=UPOS.PUNCT,
                    ),
                ],
            )
        ]
    )


def test_mocked_dual_model_pipeline_writes_twenty_sentences(tmp_path: Path) -> None:
    vocab_root = tmp_path / "vocab"
    lemmatizer_root = tmp_path / "lemmatizer"
    make_vocab(vocab_root)
    assignments = select_handcraft_targets(
        vocab_root=vocab_root,
        lang="de",
        level="A1",
        count=20,
    )
    assignment_batches = [
        assignments[index : index + MAX_HANDCRAFT_SENTENCES_PER_BATCH]
        for index in range(0, len(assignments), MAX_HANDCRAFT_SENTENCES_PER_BATCH)
    ]
    responses: list[HandcraftBatch] = []
    for assignment_batch in assignment_batches:
        generated = batch_for(assignment_batch)
        reviewed = generated.model_copy(deep=True)
        reviewed.sentences[0].text = reviewed.sentences[0].text[:-1] + "!"
        reviewed.sentences[0].tokens[-1].form = "!"
        reviewed.sentences[0].tokens[-1].lemma = "!"
        responses.extend([generated, reviewed, generated])
    client = QueueClient(responses)
    ledger = Ledger(vocab_root / ".gemma_qa" / "ledger.sqlite3")

    output = run_handcraft(
        vocab_root=vocab_root,
        lemmatizer_root=lemmatizer_root,
        lang="de",
        level="A1",
        count=20,
        client=client,
        ledger=ledger,
    )

    assert output == lemmatizer_root / "data/handcraft/de/train/a1.proposed.conllu"
    rendered = output.read_text(encoding="utf-8")
    assert rendered.count("# sent_id = ") == 20
    assert [
        line.removeprefix("# sent_id = ")
        for line in rendered.splitlines()
        if line.startswith("# sent_id = ")
    ] == [assignment.sent_id for assignment in assignments]
    assert client.calls == [
        model for _ in assignment_batches for model in (MODEL_31B, MODEL_26B, MODEL_31B)
    ]
    estimator = TiktokenEstimator()
    assert all(estimator.count(prompt) <= INPUT_BATCH_TOKEN_CAP for prompt in client.prompts)
    assert client.output_limits == [HANDCRAFT_MAX_OUTPUT_TOKENS] * len(client.calls)
    assert all(
        len(json.loads(prompt.splitlines()[-1]).get("sentences", []))
        <= MAX_HANDCRAFT_SENTENCES_PER_BATCH
        for prompt in client.prompts
        if f"prompt_version={HANDCRAFT_PROMPT_VERSION}\n" in prompt
    )
    assert len({assignment.sent_id for assignment in assignments}) == 20
    first_prompt = client.prompts[0]
    first_span_end = assignment_batches[0][-1].sent_id
    assert (
        ledger.get(
            prompt_hash(first_prompt),
            MODEL_31B,
            (
                f"handcraft:{HANDCRAFT_PROMPT_VERSION}:de:A1:"
                f"handcraft-de-a1-001..{first_span_end}:generation"
            ),
        )
        is not None
    )

    resumed_client = QueueClient([])
    resumed_output = run_handcraft(
        vocab_root=vocab_root,
        lemmatizer_root=lemmatizer_root,
        lang="de",
        level="A1",
        count=20,
        client=resumed_client,
        ledger=ledger,
    )
    assert resumed_output == output
    assert resumed_client.calls == []
    ledger.close()


def test_duplicate_text_across_batches_fails_without_partial_file(
    tmp_path: Path,
) -> None:
    vocab_root = tmp_path / "vocab"
    lemmatizer_root = tmp_path / "lemmatizer"
    make_cross_batch_duplicate_vocab(vocab_root)
    assignments = select_handcraft_targets(
        vocab_root=vocab_root,
        lang="de",
        level="A1",
        count=4,
    )
    from scripts.gemma_qa.handcraft import _pack_handcraft_assignments

    assignment_batches = _pack_handcraft_assignments(
        assignments,
        lang="de",
        level="A1",
    )
    # Structured retries may re-call generate; keep a deep queue of duplicate batches.
    responses = [
        batch_for(assignment_batch)
        for assignment_batch in assignment_batches
        for _ in range(12)
    ]
    client = QueueClient(responses)
    ledger = Ledger(vocab_root / ".gemma_qa" / "ledger.sqlite3")
    output = lemmatizer_root / "data/handcraft/de/train/a1.proposed.conllu"
    with pytest.raises(ValueError, match="duplicate text"):
        run_handcraft(
            vocab_root=vocab_root,
            lemmatizer_root=lemmatizer_root,
            lang="de",
            level="A1",
            count=4,
            client=client,
            ledger=ledger,
        )
    assert not output.exists()
    assert not output.with_name(f".{output.name}.tmp").exists()
    ledger.close()


def test_oversized_review_candidate_reduces_batch_deterministically(
    tmp_path: Path,
) -> None:
    vocab_root = tmp_path / "vocab"
    make_vocab(vocab_root)
    assignments = select_handcraft_targets(
        vocab_root=vocab_root,
        lang="de",
        level="A1",
        count=3,
    )
    oversized = batch_for(assignments)
    for sentence in oversized.sentences:
        punctuation = sentence.tokens.pop()
        for _ in range(2_500):
            sentence.tokens.append(
                HandcraftToken(
                    id="1",
                    form="sehr",
                    lemma="sehr",
                    upos=UPOS.ADV,
                )
            )
        sentence.tokens.append(punctuation)
        for index, token in enumerate(sentence.tokens, start=1):
            token.id = str(index)
        sentence.text = " ".join(token.form for token in sentence.tokens[:-1]) + "."
    first = batch_for(assignments[:1])
    remaining = batch_for(assignments[1:])
    client = QueueClient([oversized, first, first, remaining, remaining])
    ledger = Ledger(vocab_root / ".gemma_qa" / "ledger.sqlite3")
    output = run_handcraft(
        vocab_root=vocab_root,
        lemmatizer_root=tmp_path / "lemmatizer",
        lang="de",
        level="A1",
        count=3,
        client=client,
        ledger=ledger,
    )
    assert output.read_text(encoding="utf-8").count("# sent_id = ") == 3
    assert client.calls == [
        MODEL_31B,
        MODEL_31B,
        MODEL_26B,
        MODEL_31B,
        MODEL_26B,
    ]
    assert all(TiktokenEstimator().count(prompt) <= INPUT_BATCH_TOKEN_CAP for prompt in client.prompts)
    ledger.close()


def test_material_review_difference_uses_31b_adjudication(tmp_path: Path) -> None:
    vocab_root = tmp_path / "vocab"
    make_vocab(vocab_root)
    assignments = select_handcraft_targets(
        vocab_root=vocab_root,
        lang="de",
        level="A1",
        count=1,
    )
    generated = batch_for(assignments)
    reviewed = generated.model_copy(deep=True)
    reviewed.sentences[0].text = reviewed.sentences[0].text[:-1] + "!"
    reviewed.sentences[0].tokens[-1].form = "!"
    reviewed.sentences[0].tokens[-1].lemma = "!"
    client = QueueClient([generated, reviewed, generated])
    ledger = Ledger(vocab_root / ".gemma_qa" / "ledger.sqlite3")

    run_handcraft(
        vocab_root=vocab_root,
        lemmatizer_root=tmp_path / "lemmatizer",
        lang="de",
        level="A1",
        count=1,
        client=client,
        ledger=ledger,
    )

    assert client.calls == [MODEL_31B, MODEL_26B, MODEL_31B]
    ledger.close()


def test_single_model_runs_explicit_smoke_without_review(tmp_path: Path) -> None:
    vocab_root = tmp_path / "vocab"
    make_vocab(vocab_root)
    assignments = select_handcraft_targets(
        vocab_root=vocab_root,
        lang="de",
        level="A1",
        count=1,
    )
    client = QueueClient([batch_for(assignments)])
    ledger = Ledger(vocab_root / ".gemma_qa" / "ledger.sqlite3")

    run_handcraft(
        vocab_root=vocab_root,
        lemmatizer_root=tmp_path / "lemmatizer",
        lang="de",
        level="A1",
        count=1,
        client=client,
        ledger=ledger,
        single_model=MODEL_26B,
    )

    assert client.calls == [MODEL_26B]
    ledger.close()


def test_handcraft_full_semantic_repair_precedes_checkpoint_storage(
    tmp_path: Path,
) -> None:
    vocab_root = tmp_path / "vocab"
    make_vocab(vocab_root)
    assignments = select_handcraft_targets(
        vocab_root=vocab_root,
        lang="de",
        level="A1",
        count=1,
    )
    valid = batch_for(assignments)
    invalid = valid.model_copy(deep=True)
    invalid.sentences[0].tokens[-1].lemma = "!"
    ledger = Ledger(vocab_root / ".gemma_qa" / "ledger.sqlite3")
    client = QueueClient([invalid, valid], ledger=ledger)

    run_handcraft(
        vocab_root=vocab_root,
        lemmatizer_root=tmp_path / "lemmatizer",
        lang="de",
        level="A1",
        count=1,
        client=client,
        ledger=ledger,
        single_model=MODEL_31B,
    )

    assert client.calls == [MODEL_31B, MODEL_31B]
    assert client.checkpoint_counts == [0, 0]
    assert client.prompts[0] in client.prompts[1]
    assert "punctuation lemma must equal form" in client.prompts[1]
    for target in assignments[0].targets:
        assert target.id in client.prompts[1]
    checkpoint = ledger.get(
        prompt_hash(client.prompts[0]),
        MODEL_31B,
        (
            f"handcraft:{HANDCRAFT_PROMPT_VERSION}:de:A1:"
            "handcraft-de-a1-001..handcraft-de-a1-001:generation"
        ),
    )
    assert checkpoint is not None
    attempts = checkpoint.request_json["semantic_attempts"]
    assert isinstance(attempts, list)
    assert len(attempts) == 2
    parsed, _ = client.parse_response(checkpoint.response_json, HandcraftBatch)
    assert validate_handcraft_batch(parsed, assignments) == valid
    ledger.close()


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("punctuation", "punctuation lemma"),
        ("text", "text mismatch"),
        ("target", "target lemma"),
    ],
)
def test_full_validation_failure_never_reaches_checkpoint(
    tmp_path: Path,
    defect: str,
    message: str,
) -> None:
    vocab_root = tmp_path / "vocab"
    make_vocab(vocab_root)
    assignments = select_handcraft_targets(
        vocab_root=vocab_root,
        lang="de",
        level="A1",
        count=1,
    )
    invalid = batch_for(assignments)
    if defect == "punctuation":
        invalid.sentences[0].tokens[-1].lemma = "!"
    elif defect == "text":
        invalid.sentences[0].text = "Falscher Text."
    else:
        invalid.sentences[0].tokens[0].lemma = "fehlen"
    ledger = Ledger(vocab_root / ".gemma_qa" / "ledger.sqlite3")
    client = QueueClient(
        [invalid, invalid.model_copy(deep=True), invalid.model_copy(deep=True)],
        ledger=ledger,
    )
    output = tmp_path / "lemmatizer/data/handcraft/de/train/a1.proposed.conllu"
    with pytest.raises(ValueError, match=message):
        run_handcraft(
            vocab_root=vocab_root,
            lemmatizer_root=tmp_path / "lemmatizer",
            lang="de",
            level="A1",
            count=1,
            client=client,
            ledger=ledger,
            single_model=MODEL_31B,
        )
    assert len(client.calls) == 3
    assert client.checkpoint_counts == [0, 0, 0]
    assert ledger.status().checkpoints == 0
    assert not output.exists()
    ledger.close()


def test_invalid_existing_full_checkpoint_is_deleted_and_regenerated(
    tmp_path: Path,
) -> None:
    vocab_root = tmp_path / "vocab"
    make_vocab(vocab_root)
    assignments = select_handcraft_targets(
        vocab_root=vocab_root,
        lang="de",
        level="A1",
        count=1,
    )
    prompt = generation_prompt_for(assignments)
    batch_id = generation_batch_id(assignments)
    invalid = batch_for(assignments)
    invalid.sentences[0].tokens[-1].lemma = "!"
    valid = batch_for(assignments)
    ledger = Ledger(vocab_root / ".gemma_qa" / "ledger.sqlite3")
    ledger.store(
        Checkpoint(
            prompt_hash=prompt_hash(prompt),
            model=MODEL_31B,
            batch_id=batch_id,
            request_json={"stale": True},
            response_json=response_json_for(invalid),
            usage=Usage(1, 1, 2),
        )
    )
    client = QueueClient([valid], ledger=ledger)
    output = run_handcraft(
        vocab_root=vocab_root,
        lemmatizer_root=tmp_path / "lemmatizer",
        lang="de",
        level="A1",
        count=1,
        client=client,
        ledger=ledger,
        single_model=MODEL_31B,
    )
    assert output.exists()
    assert client.calls == [MODEL_31B]
    checkpoint = ledger.get(prompt_hash(prompt), MODEL_31B, batch_id)
    assert checkpoint is not None
    parsed, _ = client.parse_response(checkpoint.response_json, HandcraftBatch)
    assert validate_handcraft_batch(parsed, assignments) == valid
    ledger.close()


def test_validation_rejects_upos_x() -> None:
    batch = one_batch()
    batch.sentences[0].tokens[0].upos = UPOS.X
    with pytest.raises(ValueError, match="UPOS X"):
        validate_handcraft_batch(batch, [one_assignment()])


def test_validation_rejects_missing_target() -> None:
    batch = one_batch()
    batch.sentences[0].tokens[1].lemma = "Morgen"
    with pytest.raises(ValueError, match="target lemma"):
        validate_handcraft_batch(batch, [one_assignment()])


def test_validation_rejects_text_mismatch() -> None:
    batch = one_batch()
    batch.sentences[0].text = "Guten Morgen."
    with pytest.raises(ValueError, match="text mismatch"):
        validate_handcraft_batch(batch, [one_assignment()])


def test_validation_rejects_duplicate_text() -> None:
    batch = one_batch()
    duplicate = batch.sentences[0].model_copy(update={"sent_id": "handcraft-de-a1-002"})
    batch = HandcraftBatch(sentences=[batch.sentences[0], duplicate])
    second = SentenceTargets.from_values(
        sent_id="handcraft-de-a1-002",
        targets=[("german:A1:1", "Abend", UPOS.NOUN)],
        source=Path("german/A1.csv"),
    )
    with pytest.raises(ValueError, match="duplicate text"):
        validate_handcraft_batch(batch, [one_assignment(), second])


def test_dry_run_and_apply_paths_never_touch_test_data(tmp_path: Path) -> None:
    test_file = tmp_path / "data/handcraft/de_test.conllu"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("reference\n", encoding="utf-8")
    batch = one_batch()

    proposed = write_handcraft(
        batch,
        lemmatizer_root=tmp_path,
        lang="de",
        level="A1",
        apply=False,
    )
    applied = write_handcraft(
        batch,
        lemmatizer_root=tmp_path,
        lang="de",
        level="A1",
        apply=True,
    )

    assert proposed == tmp_path / "data/handcraft/de/train/a1.proposed.conllu"
    assert applied == tmp_path / "data/handcraft/de/train/a1.conllu"
    assert test_file.read_text(encoding="utf-8") == "reference\n"


def test_supported_external_lemma_checker_blocks_output(tmp_path: Path) -> None:
    checker = tmp_path / "src/lemmatizer/data/lemma_checker.py"
    checker.parent.mkdir(parents=True)
    checker.write_text(
        "class Result:\n"
        "    errors = ['bad lemma']\n"
        "\n"
        "def check_text(text, lang):\n"
        "    return Result()\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bad lemma"):
        write_handcraft(
            one_batch(),
            lemmatizer_root=tmp_path,
            lang="de",
            level="A1",
            apply=False,
        )
    assert not (tmp_path / "data/handcraft/de/train/a1.proposed.conllu").exists()


def test_handcraft_cli_accepts_required_roots_and_smoke_model() -> None:
    args = build_parser().parse_args(
        [
            "handcraft",
            "--vocab-root",
            "/vocab",
            "--lemmatizer-root",
            "/lemmatizer",
            "--lang",
            "de",
            "--level",
            "A1",
            "--count",
            "20",
            "--apply",
            "--single-model",
            MODEL_31B,
        ]
    )
    assert args.vocab_root == Path("/vocab")
    assert args.lemmatizer_root == Path("/lemmatizer")
    assert args.count == 20
    assert args.apply is True
