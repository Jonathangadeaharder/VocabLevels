from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.gemma_qa.cefr import read_cefr_csv
from scripts.gemma_qa.handcraft import select_handcraft_targets
from scripts.gemma_qa.language_repair import cefr_row_issues
from scripts.gemma_qa.languages import LANGUAGE_CODES, get_language
from scripts.gemma_qa.prompts import (
    build_cefr_prompt,
    build_handcraft_generation_prompt,
    build_novel_generation_prompt,
    build_refill_generation_prompt,
)
from scripts.gemma_qa.schemas import (
    CefrInputRow,
    CefrRefillConcept,
    CefrReviewRow,
    ReviewAction,
    UPOS,
)


@pytest.mark.parametrize("code", LANGUAGE_CODES)
def test_profile_drives_all_prompt_families(code: str) -> None:
    profile = get_language(code)
    row = CefrInputRow(
        id=f"{profile.directory}:A1:1",
        lemma="词" if code == "zh" else "lemma",
        english_lemma="word",
        chinese_lemma="词",
        upos=UPOS.NOUN,
    )
    concept = CefrRefillConcept(
        id="english:A1:1",
        english_lemma="word",
        chinese_lemma="词",
        upos=UPOS.NOUN,
    )
    prompts = (
        build_cefr_prompt([row], lang=profile.directory),
        build_refill_generation_prompt(
            [concept],
            lang=profile.directory,
            level="A1",
        ),
        build_novel_generation_prompt(
            [f"novel:{profile.directory}:A1:slot:1:round:1"],
            lang=profile.directory,
            level="A1",
            exclusions=[],
        ),
        build_handcraft_generation_prompt(
            [{"sent_id": f"handcraft-{code}-a1-001", "targets": []}],
            lang=code,
            level="A1",
        ),
    )
    assert all(profile.display_name in prompt or code == "de" for prompt in prompts)
    assert all(profile.citation_rules in prompt or code == "de" for prompt in prompts)


def test_german_cefr_prompt_remains_byte_stable() -> None:
    row = CefrInputRow(
        id="german:A1:1",
        lemma="Haus",
        english_lemma="house",
        chinese_lemma="房子",
        upos=UPOS.NOUN,
    )
    assert build_cefr_prompt([row]) == build_cefr_prompt([row], lang="german")


@pytest.mark.parametrize(
    ("language", "lemma", "english_lemma", "expected"),
    [
        ("arabic", "house", "house", "arabic.script_required"),
        ("arabic", "بيت", "house", None),
        ("chinese", "house", "house", "chinese.script_required"),
        ("chinese", "房子", "house", None),
        ("english", "running", "running", None),
        ("spanish", "fui", "went", None),
    ],
)
def test_generic_cefr_gates_only_enforce_sound_constraints(
    language: str,
    lemma: str,
    english_lemma: str,
    expected: str | None,
) -> None:
    row = CefrReviewRow(
        id=f"{language}:A1:1",
        lemma=lemma,
        english_lemma=english_lemma,
        chinese_lemma="房子",
        upos=UPOS.VERB,
        action=ReviewAction.KEEP,
    )
    codes = [issue.code for issue in cefr_row_issues(row, lang=language)]
    if expected is None:
        assert codes == []
    else:
        assert expected in codes


@pytest.mark.parametrize("code", LANGUAGE_CODES)
def test_handcraft_target_mapping_reads_profile_directory(
    tmp_path: Path,
    code: str,
) -> None:
    profile = get_language(code)
    source_directory = tmp_path / profile.directory
    source_directory.mkdir()
    with (source_directory / "A1.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                profile.csv_header,
                "English_Lemma",
                "Chinese_Lemma",
                "POS",
            ]
        )
        writer.writerow(["词" if code == "zh" else "lemma", "word", "词", "NOUN"])
    selected = select_handcraft_targets(
        vocab_root=tmp_path,
        lang=code,
        level="A1",
        count=1,
    )
    assert selected[0].source == source_directory / "A1.csv"
    assert selected[0].sent_id == f"handcraft-{code}-a1-001"


def test_read_csv_rejects_directory_code_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "A1.csv"
    source.write_text(
        "German_Lemma,English_Lemma,Chinese_Lemma,POS\nHaus,house,房子,NOUN\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected header"):
        read_cefr_csv(source, lang="english", level="A1")
