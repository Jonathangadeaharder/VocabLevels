from __future__ import annotations

import unicodedata
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from runpy import run_path
from typing import Protocol, cast

from .cefr import read_cefr_csv
from .client import GenerationResult
from .config import (
    INPUT_BATCH_TOKEN_CAP,
    MODEL_26B,
    MODEL_31B,
    MODEL_ADJUDICATION,
    MODEL_IDS,
)
from .language_repair import cefr_row_issues
from .ledger import Ledger
from .languages import get_language, has_arabic_script
from .packing import TiktokenEstimator
from .prompts import (
    HANDCRAFT_PROMPT_VERSION,
    build_handcraft_adjudication_prompt,
    build_handcraft_generation_prompt,
    build_handcraft_review_prompt,
)
from .schemas import (
    CefrReviewRow,
    HandcraftBatch,
    HandcraftSentence,
    ReviewAction,
    UPOS,
)
from .semantic_generation import checkpointed_semantic_generate

HANDCRAFT_MAX_OUTPUT_TOKENS = 4_096
MAX_HANDCRAFT_SENTENCES_PER_BATCH = 5
TARGETS_PER_SENTENCE = 3
# Cap issue reporting so gate output stays readable on large CSVs.
_MAX_READY_ISSUES = 20
# Loanword echo (Team/Baby) is allowed on committed CEFR lists used as handcraft
# targets; structural citation rules still apply.
_HANDCRAFT_READY_SKIP_CODES = frozenset(
    {
        "cefr.english_echo",
        "cefr.drop",
        # Formal Sie / similar CEFR-accepted PRON capitalisation.
        "german.non_noun_capitalized",
    }
)


class HandcraftClient(Protocol):
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_model: type[HandcraftBatch],
        max_output_tokens: int,
    ) -> GenerationResult[HandcraftBatch]: ...

    def parse_response(
        self,
        response_json: dict[str, object],
        response_model: type[HandcraftBatch],
    ) -> tuple[HandcraftBatch, object]: ...


class LemmaCheckResult(Protocol):
    errors: list[str]


@dataclass(frozen=True)
class TargetLemma:
    id: str
    lemma: str
    upos: UPOS


@dataclass(frozen=True)
class SentenceTargets:
    sent_id: str
    targets: tuple[TargetLemma, ...]
    source: Path

    @classmethod
    def from_values(
        cls,
        *,
        sent_id: str,
        targets: Sequence[tuple[str, str, UPOS]],
        source: Path,
    ) -> SentenceTargets:
        return cls(
            sent_id=sent_id,
            targets=tuple(
                TargetLemma(id=target_id, lemma=lemma, upos=upos)
                for target_id, lemma, upos in targets
            ),
            source=source,
        )


@dataclass(frozen=True)
class HandcraftReadyReport:
    """Whether a CEFR cell is safe to turn into handcraft CoNLL-U."""

    ready: bool
    lang: str
    level: str
    csv_path: Path
    row_count: int
    required_rows: int
    issues: tuple[str, ...]

    def summary(self) -> str:
        status = "ready" if self.ready else "not_ready"
        head = (
            f"{status} lang={self.lang} level={self.level} "
            f"rows={self.row_count} need>={self.required_rows} "
            f"path={self.csv_path}"
        )
        if not self.issues:
            return head
        return head + "\n" + "\n".join(f"  - {item}" for item in self.issues)


def assess_handcraft_ready(
    *,
    vocab_root: Path,
    lang: str,
    level: str,
    count: int = 20,
    max_issue_rows: int = _MAX_READY_ISSUES,
) -> HandcraftReadyReport:
    """Gate: committed CEFR CSV must be loadable, large enough, citation-clean."""
    if count <= 0:
        raise ValueError("count must be positive")
    profile = get_language(lang)
    source = vocab_root / profile.directory / f"{level}.csv"
    required = count  # select_handcraft_targets needs at least one row per sentence
    issues: list[str] = []
    row_count = 0
    if not source.is_file():
        issues.append(f"missing CEFR csv: {source}")
        return HandcraftReadyReport(
            ready=False,
            lang=profile.code,
            level=level,
            csv_path=source,
            row_count=0,
            required_rows=required,
            issues=tuple(issues),
        )
    try:
        document = read_cefr_csv(source, lang=profile.directory, level=level)
    except Exception as error:  # noqa: BLE001 — surface parse errors as gate fails
        issues.append(f"csv unreadable: {error}")
        return HandcraftReadyReport(
            ready=False,
            lang=profile.code,
            level=level,
            csv_path=source,
            row_count=0,
            required_rows=required,
            issues=tuple(issues),
        )
    row_count = len(document.rows)
    if row_count < required:
        issues.append(
            f"not enough rows for handcraft count={count} "
            f"(have {row_count}, need >={required})"
        )
    bad = 0
    for row in document.rows:
        review = CefrReviewRow(
            id=row.id,
            lemma=row.lemma,
            english_lemma=row.english_lemma,
            chinese_lemma=row.chinese_lemma or row.lemma,
            upos=row.upos,
            action=ReviewAction.KEEP,
        )
        row_issues = [
            issue
            for issue in cefr_row_issues(review, lang=profile.directory)
            if issue.code not in _HANDCRAFT_READY_SKIP_CODES
        ]
        if not row_issues:
            continue
        bad += 1
        if len(issues) < max_issue_rows:
            codes = ", ".join(issue.code for issue in row_issues)
            issues.append(
                f"row {row.id} lemma={row.lemma!r} upos={row.upos.value}: {codes}"
            )
    if bad > max_issue_rows:
        issues.append(f"... and {bad - max_issue_rows} more rows with citation issues")
    elif bad:
        # Keep a total for scripts even when all fit under the cap.
        if not any(item.startswith("... and ") for item in issues):
            issues.append(f"citation_issue_rows={bad}")
    return HandcraftReadyReport(
        ready=not issues,
        lang=profile.code,
        level=level,
        csv_path=source,
        row_count=row_count,
        required_rows=required,
        issues=tuple(issues),
    )


# Prefer open-class targets so models can realize lemma+UPOS in short sentences.
_HANDCRAFT_CONTENT_UPOS = frozenset(
    {UPOS.NOUN, UPOS.VERB, UPOS.ADJ, UPOS.ADV, UPOS.PROPN}
)


def select_handcraft_targets(
    *,
    vocab_root: Path,
    lang: str,
    level: str,
    count: int,
) -> list[SentenceTargets]:
    if count <= 0:
        raise ValueError("count must be positive")
    profile = get_language(lang)
    source = vocab_root / profile.directory / f"{level}.csv"
    document = read_cefr_csv(source, lang=profile.directory, level=level)
    if len(document.rows) < count:
        raise ValueError("not enough CEFR target lemmas for requested sentence count")
    need = count * TARGETS_PER_SENTENCE
    content = [row for row in document.rows if row.upos in _HANDCRAFT_CONTENT_UPOS]
    # Content first (CSV order within band), then fill from remaining rows.
    preferred = content + [row for row in document.rows if row not in content]
    selected = preferred[: min(len(preferred), need)]
    if len(selected) < count:
        raise ValueError("not enough CEFR target lemmas for requested sentence count")
    minimum, remainder = divmod(len(selected), count)
    assignments: list[SentenceTargets] = []
    offset = 0
    for index in range(count):
        size = minimum + (1 if index < remainder else 0)
        rows = selected[offset : offset + size]
        offset += size
        assignments.append(
            SentenceTargets(
                sent_id=f"handcraft-{lang}-{level.casefold()}-{index + 1:03d}",
                targets=tuple(
                    TargetLemma(id=row.id, lemma=row.lemma, upos=row.upos)
                    for row in rows
                ),
                source=source,
            )
        )
    return assignments


def validate_handcraft_batch(
    batch: HandcraftBatch,
    assignments: Sequence[SentenceTargets],
    *,
    lang: str = "de",
) -> HandcraftBatch:
    profile = get_language(lang)
    expected_ids = [assignment.sent_id for assignment in assignments]
    actual_ids = [sentence.sent_id for sentence in batch.sentences]
    if actual_ids != expected_ids:
        raise ValueError("sentence IDs/order/cardinality differ from assignments")
    texts: set[str] = set()
    for sentence, assignment in zip(batch.sentences, assignments, strict=True):
        _validate_sentence(sentence, assignment, lang=profile.code)
        if sentence.text in texts:
            raise ValueError(f"{sentence.sent_id}: duplicate text")
        texts.add(sentence.text)
    return batch


def _validate_sentence(
    sentence: HandcraftSentence,
    assignment: SentenceTargets,
    *,
    lang: str,
) -> None:
    expected_target_ids = [target.id for target in assignment.targets]
    if sentence.target_ids != expected_target_ids:
        raise ValueError(f"{sentence.sent_id}: target IDs differ from assignment")
    expected_token_ids = [str(index) for index in range(1, len(sentence.tokens) + 1)]
    if [token.id for token in sentence.tokens] != expected_token_ids:
        raise ValueError(f"{sentence.sent_id}: token IDs must be consecutive integers")
    values = [
        sentence.sent_id,
        sentence.text,
        *sentence.target_ids,
        *(
            value
            for token in sentence.tokens
            for value in (token.id, token.form, token.lemma)
        ),
    ]
    if any(unicodedata.normalize("NFC", value) != value for value in values):
        raise ValueError(f"{sentence.sent_id}: all text must use NFC normalization")
    for token in sentence.tokens:
        if token.upos is UPOS.X:
            raise ValueError(f"{sentence.sent_id}: UPOS X is forbidden")
        if token.upos is UPOS.PUNCT and token.lemma != token.form:
            raise ValueError(f"{sentence.sent_id}: punctuation lemma must equal form")
        if lang == "zh" and token.upos is not UPOS.PUNCT and token.lemma != token.form:
            raise ValueError(f"{sentence.sent_id}: Chinese lemma must equal form")
        if (
            lang == "ar"
            and token.upos is not UPOS.PUNCT
            and not has_arabic_script(token.lemma)
        ):
            raise ValueError(f"{sentence.sent_id}: Arabic lemma must use Arabic script")
    joined_forms = "".join(token.form for token in sentence.tokens)
    squeezed_text = "".join(sentence.text.split())
    if joined_forms != squeezed_text:
        raise ValueError(f"{sentence.sent_id}: text mismatch")
    for target in assignment.targets:
        if not any(
            token.lemma.casefold() == target.lemma.casefold()
            and token.upos is target.upos
            and bool(token.form)
            for token in sentence.tokens
        ):
            raise ValueError(
                f"{sentence.sent_id}: target {target.lemma!r}/"
                f"{target.upos.value} is not represented "
                "(lemma and UPOS must both match a token)"
            )
    for token in sentence.tokens:
        if lang == "de" and token.upos is UPOS.NOUN and not token.lemma[0].isupper():
            raise ValueError(
                f"{sentence.sent_id}: German noun lemma must be capitalized"
            )
        if (
            lang == "de"
            and token.upos is UPOS.VERB
            and not token.lemma.casefold().endswith(("en", "n"))
        ):
            raise ValueError(
                f"{sentence.sent_id}: German verb lemma must be infinitive"
            )


def render_handcraft_conllu(batch: HandcraftBatch) -> str:
    lines: list[str] = []
    for sentence in batch.sentences:
        lines.extend(
            [
                f"# sent_id = {sentence.sent_id}",
                f"# text = {sentence.text}",
            ]
        )
        lines.extend(
            "\t".join(
                [
                    token.id,
                    token.form,
                    token.lemma,
                    token.upos.value,
                    "_",
                    "_",
                    "_",
                    "_",
                    "_",
                    "_",
                ]
            )
            for token in sentence.tokens
        )
        lines.append("")
    return "\n".join(lines)


def write_handcraft(
    batch: HandcraftBatch,
    *,
    lemmatizer_root: Path,
    lang: str,
    level: str,
    apply: bool,
) -> Path:
    rendered = render_handcraft_conllu(batch)
    _run_supported_lemma_checker(
        rendered,
        lemmatizer_root=lemmatizer_root,
        lang=lang,
    )
    directory = lemmatizer_root / "data" / "handcraft" / lang / "train"
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".conllu" if apply else ".proposed.conllu"
    output = directory / f"{level.casefold()}{suffix}"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    return output


def _run_supported_lemma_checker(
    text: str,
    *,
    lemmatizer_root: Path,
    lang: str,
) -> None:
    if lang not in {"de", "ar", "zh"}:
        return
    checker_path = lemmatizer_root / "src" / "lemmatizer" / "data" / "lemma_checker.py"
    if not checker_path.exists():
        return
    namespace = run_path(str(checker_path))
    check_text = cast(
        Callable[..., LemmaCheckResult],
        namespace["check_text"],
    )
    result = check_text(text, lang=lang)
    if result.errors:
        raise ValueError(
            "lemma checker rejected handcraft output: " + "; ".join(result.errors)
        )


def run_handcraft(
    *,
    vocab_root: Path,
    lemmatizer_root: Path,
    lang: str,
    level: str,
    count: int,
    client: HandcraftClient,
    ledger: Ledger,
    apply: bool = False,
    single_model: str | None = None,
) -> Path:
    if single_model is not None and single_model not in MODEL_IDS:
        raise ValueError(f"unsupported model: {single_model}")
    assignments = select_handcraft_targets(
        vocab_root=vocab_root,
        lang=lang,
        level=level,
        count=count,
    )
    sentences: list[HandcraftSentence] = []
    for assignment_batch in _pack_handcraft_assignments(
        assignments,
        lang=lang,
        level=level,
    ):
        chosen = _run_handcraft_assignment_batch(
            assignment_batch,
            lang=lang,
            level=level,
            client=client,
            ledger=ledger,
            single_model=single_model,
        )
        sentences.extend(chosen.sentences)
    merged = HandcraftBatch(sentences=sentences)
    validate_handcraft_batch(merged, assignments, lang=lang)
    return write_handcraft(
        merged,
        lemmatizer_root=lemmatizer_root,
        lang=lang,
        level=level,
        apply=apply,
    )


def _run_handcraft_assignment_batch(
    assignments: Sequence[SentenceTargets],
    *,
    lang: str,
    level: str,
    client: HandcraftClient,
    ledger: Ledger,
    single_model: str | None,
) -> HandcraftBatch:
    assignment_payload = _assignment_payload(assignments)
    span = f"{assignments[0].sent_id}..{assignments[-1].sent_id}"
    prompt_version = (
        HANDCRAFT_PROMPT_VERSION if lang == "de" else f"handcraft-{lang}-v2"
    )
    namespace = f"handcraft:{prompt_version}:{lang}:{level}:{span}"
    generation_prompt = build_handcraft_generation_prompt(
        assignment_payload,
        lang=lang,
        level=level,
    )
    generator_model = single_model or MODEL_31B
    generated = _checkpointed_generate(
        client=client,
        ledger=ledger,
        model=generator_model,
        batch_id=f"{namespace}:generation",
        prompt=generation_prompt,
        assignments=assignments,
        lang=lang,
    )
    chosen = generated
    if single_model is None:
        review_prompt = build_handcraft_review_prompt(
            assignment_payload,
            generated,
            lang=lang,
            level=level,
        )
        if _handcraft_prompt_exceeds_cap(review_prompt):
            return _rerun_split_handcraft_batch(
                assignments,
                lang=lang,
                level=level,
                client=client,
                ledger=ledger,
                single_model=single_model,
            )
        reviewed = _checkpointed_generate(
            client=client,
            ledger=ledger,
            model=MODEL_26B,
            batch_id=f"{namespace}:review",
            prompt=review_prompt,
            assignments=assignments,
            lang=lang,
        )
        if _materially_different(generated, reviewed):
            adjudication_prompt = build_handcraft_adjudication_prompt(
                assignment_payload,
                generated,
                reviewed,
                lang=lang,
                level=level,
            )
            if _handcraft_prompt_exceeds_cap(adjudication_prompt):
                return _rerun_split_handcraft_batch(
                    assignments,
                    lang=lang,
                    level=level,
                    client=client,
                    ledger=ledger,
                    single_model=single_model,
                )
            chosen = _checkpointed_generate(
                client=client,
                ledger=ledger,
                # Fixed adjudicator: rotating pool breaks ledger re-entry in tests
                # and CEFR already spreads load via dual+adj rotation.
                model=MODEL_ADJUDICATION,
                batch_id=f"{namespace}:adjudication",
                prompt=adjudication_prompt,
                assignments=assignments,
                lang=lang,
            )
    return chosen


def _rerun_split_handcraft_batch(
    assignments: Sequence[SentenceTargets],
    *,
    lang: str,
    level: str,
    client: HandcraftClient,
    ledger: Ledger,
    single_model: str | None,
) -> HandcraftBatch:
    if len(assignments) <= 1:
        raise ValueError("single handcraft candidate exceeds input token cap")
    split_at = len(assignments) // 2
    sentences: list[HandcraftSentence] = []
    for subset in (assignments[:split_at], assignments[split_at:]):
        completed = _run_handcraft_assignment_batch(
            subset,
            lang=lang,
            level=level,
            client=client,
            ledger=ledger,
            single_model=single_model,
        )
        sentences.extend(completed.sentences)
    return HandcraftBatch(sentences=sentences)


def _handcraft_prompt_exceeds_cap(prompt: str) -> bool:
    return TiktokenEstimator().count(prompt) > INPUT_BATCH_TOKEN_CAP


def _pack_handcraft_assignments(
    assignments: Sequence[SentenceTargets],
    *,
    lang: str,
    level: str,
) -> list[list[SentenceTargets]]:
    estimator = TiktokenEstimator()
    batches: list[list[SentenceTargets]] = []
    current: list[SentenceTargets] = []
    for assignment in assignments:
        candidate = [*current, assignment]
        prompt = build_handcraft_generation_prompt(
            _assignment_payload(candidate),
            lang=lang,
            level=level,
        )
        if (
            len(candidate) <= MAX_HANDCRAFT_SENTENCES_PER_BATCH
            and estimator.count(prompt) <= INPUT_BATCH_TOKEN_CAP
        ):
            current = candidate
            continue
        if not current:
            raise ValueError("single handcraft assignment exceeds input token cap")
        batches.append(current)
        current = [assignment]
        prompt = build_handcraft_generation_prompt(
            _assignment_payload(current),
            lang=lang,
            level=level,
        )
        if estimator.count(prompt) > INPUT_BATCH_TOKEN_CAP:
            raise ValueError("single handcraft assignment exceeds input token cap")
    if current:
        batches.append(current)
    return batches


def _assignment_payload(
    assignments: Sequence[SentenceTargets],
) -> list[dict[str, object]]:
    return [
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


def _materially_different(
    generated: HandcraftBatch,
    reviewed: HandcraftBatch,
) -> bool:
    return generated.model_dump(mode="json") != reviewed.model_dump(mode="json")


def _checkpointed_generate(
    *,
    client: HandcraftClient,
    ledger: Ledger,
    model: str,
    batch_id: str,
    prompt: str,
    assignments: Sequence[SentenceTargets],
    lang: str,
) -> HandcraftBatch:
    prompt_tokens = TiktokenEstimator().count(prompt)
    if prompt_tokens > INPUT_BATCH_TOKEN_CAP:
        raise ValueError("handcraft prompt exceeds input token cap")
    return checkpointed_semantic_generate(
        client=client,
        ledger=ledger,
        model=model,
        batch_id=batch_id,
        prompt=prompt,
        response_model=HandcraftBatch,
        max_output_tokens=HANDCRAFT_MAX_OUTPUT_TOKENS,
        validate=lambda batch: validate_handcraft_batch(batch, assignments, lang=lang),
        expected_identity={
            "sentences": [
                {
                    "sent_id": assignment.sent_id,
                    "target_ids": [target.id for target in assignment.targets],
                }
                for assignment in assignments
            ]
        },
    )
