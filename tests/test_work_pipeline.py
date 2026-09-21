"""Tests for the Gemini expansion pipeline in work/: prompt generation from
per-pair deficits, merging Gemini CSVs into expansion.csv, and cleaning
criterion-6 violations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from work import clean_expansion, merge_gemini
from work import gen_prompts_r2 as gen_prompts

EXPANSION_HEADER = ["Lemma", "English_Lemma", "Chinese_Lemma", "POS", "CEFR"]
GEMINI_HEADER = ["Lang", "Lemma", "English_Lemma", "Chinese_Lemma", "POS", "CEFR"]
ALL_LANGS = {"ar", "de", "en", "es", "fr", "nl", "sv", "zh"}


def write_expansion(root: Path, lang_dir: str, rows: list[list[str]]) -> None:
    path = root / lang_dir / "expansion.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPANSION_HEADER)
        writer.writerows(rows)


def write_gemini_csv(out_dir: Path, name: str, rows: list[list[str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(GEMINI_HEADER)
        writer.writerows(rows)


def read_data_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))[1:]


# ---------------------------------------------------------------- merge_gemini


def test_merge_appends_uncovered_and_skips_covered_or_invalid(
    tmp_path: Path,
) -> None:
    write_expansion(tmp_path, "english", [["book", "book", "书", "NOUN", "A1"]])
    out = tmp_path / "gemini_out"
    write_gemini_csv(
        out,
        "en_80.csv",
        [
            ["en", "read", "read", "读", "VERB", "A1"],  # new -> merged
            ["en", "book", "book", "书", "NOUN", "A1"],  # covered -> skipped
            ["en", "run", "run", "跑", "VERB", "B1"],  # new -> merged
            ["en", "walk", "walk", "走", "VERB", "C9"],  # bad CEFR -> skipped
            ["en", "", "", "", "NOUN", "A1"],  # empty -> skipped
            ["en", "swim", "swim", "游", "VERB", "C1"],  # C1 -> Advanced
        ],
    )
    merge_gemini.merge(tmp_path, out)
    rows = read_data_rows(tmp_path / "english" / "expansion.csv")
    assert ["read", "read", "读", "VERB", "A1"] in rows
    assert ["run", "run", "跑", "VERB", "B1"] in rows
    assert ["swim", "swim", "游", "VERB", "Advanced"] in rows
    assert not any(r[0] == "walk" for r in rows)
    assert len(rows) == 4  # book + read + run + swim


def test_merge_maps_c1_to_advanced(tmp_path: Path) -> None:
    write_expansion(tmp_path, "english", [])
    out = tmp_path / "gemini_out"
    write_gemini_csv(out, "en_80.csv", [["en", "swim", "swim", "游", "VERB", "C1"]])
    merge_gemini.merge(tmp_path, out)
    rows = read_data_rows(tmp_path / "english" / "expansion.csv")
    assert rows == [["swim", "swim", "游", "VERB", "Advanced"]]


def test_merge_no_files_reports(tmp_path: Path, capsys) -> None:
    write_expansion(tmp_path, "english", [["book", "book", "书", "NOUN", "A1"]])
    merge_gemini.merge(tmp_path, tmp_path / "empty")
    assert "no gemini files" in capsys.readouterr().out


# ------------------------------------------------------------ clean_expansion


def test_classify_single_copy_goes_to_allowlist() -> None:
    singles, multis, prefixes, delete_keys = clean_expansion.classify(
        [("harira", "harira", "NOUN")], "es"
    )
    assert singles == ["harira"]
    assert not multis and not prefixes and not delete_keys


def test_classify_multiword_copy_deleted_unless_loan() -> None:
    singles, multis, _, delete_keys = clean_expansion.classify(
        [("do one's best", "do one's best", "VERB"), ("de facto", "de facto", "ADJ")],
        "de",
    )
    assert not singles
    assert "do one's best" in multis
    assert "de facto" not in multis
    assert ("do one's best", "do one's best", "VERB") in delete_keys
    assert not delete_keys or ("de facto", "de facto", "ADJ") not in delete_keys


def test_classify_kept_loan_skips_prefix_rule() -> None:
    # 'a priori' is a genuine loan AND starts with English function prefix
    # 'a ': the copy branch must keep it without consulting the prefix rule.
    singles, multis, prefixes, delete_keys = clean_expansion.classify(
        [("a priori", "a priori", "ADJ"), ("a posteriori", "a posteriori", "ADJ")],
        "fr",
    )
    assert not singles and not multis and not prefixes and not delete_keys


def test_classify_prefix_copy_deleted_native_prefix_kept() -> None:
    singles, multis, prefixes, delete_keys = clean_expansion.classify(
        [("to book", "reserve", "VERB"), ("in Ordnung", "in order", "ADV")], "de"
    )
    assert not singles and not multis
    assert "to book" in prefixes
    assert ("to book", "reserve", "VERB") in delete_keys


def test_classify_zh_latin_lemma_deleted() -> None:
    _, _, _, delete_keys = clean_expansion.classify(
        [("hello", "hello", "INTJ"), ("你好", "hello", "INTJ")], "zh"
    )
    assert ("hello", "hello", "INTJ") in delete_keys
    assert ("你好", "hello", "INTJ") not in delete_keys


def test_classify_ar_latin_lemma_deleted() -> None:
    _, _, _, delete_keys = clean_expansion.classify([("cafe", "cafe", "NOUN")], "ar")
    assert ("cafe", "cafe", "NOUN") in delete_keys


def test_classify_extended_allowlist_copy_is_clean() -> None:
    # Gate parity: the copy check allowlists COGNATE_ALLOWLIST union
    # EXTENDED_COGNATE_ALLOWLIST; 'aberrant' is only in the extended set.
    singles, multis, prefixes, delete_keys = clean_expansion.classify(
        [("aberrant", "aberrant", "ADJ")], "de"
    )
    assert not singles and not multis and not prefixes and not delete_keys


def test_classify_loan_phrase_skips_prefix_rule_without_copy() -> None:
    # Gate parity: MULTIWORD_LOAN_PHRASES also guard the prefix rule, even
    # when the lemma is not a gloss copy.
    singles, multis, prefixes, delete_keys = clean_expansion.classify(
        [("a priori", "deduced from earlier knowledge", "ADV")], "fr"
    )
    assert not singles and not multis and not prefixes and not delete_keys


def test_classify_uses_gate_native_prefix_table() -> None:
    # Gate parity: the gate's NATIVE_FUNCTION_PREFIXES table guards the
    # prefix rule; the tool must not carry its own diverging table.
    singles, multis, prefixes, delete_keys = clean_expansion.classify(
        [("for example", "par exemple", "ADV")], "fr"
    )
    assert not singles and not multis and not prefixes and not delete_keys


# ------------------------------------------------------------ gen_prompts_r2


def make_delivery(
    base: list[SimpleNamespace], en: list[SimpleNamespace]
) -> dict[str, list[SimpleNamespace]]:
    """Every language carries `base` rows; only en may differ. Pairs between
    non-en languages are then covered, so tests see only intended deficits."""
    return {code: (en if code == "en" else base) for code in sorted(ALL_LANGS)}


def patch_pipeline(monkeypatch: pytest.MonkeyPatch, delivery: dict[str, list]) -> None:
    monkeypatch.setattr(gen_prompts, "load_csv_records", lambda root: [])
    monkeypatch.setattr(gen_prompts, "load_expansion_records", lambda root: [])
    monkeypatch.setattr(gen_prompts, "build_delivery_rows", lambda records: delivery)


def test_gen_prompts_picks_deficit_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    delivery = make_delivery(
        [
            SimpleNamespace(lemma="买", gloss="buy", pos="VERB"),
            SimpleNamespace(lemma="书", gloss="book", pos="NOUN"),
        ],
        [SimpleNamespace(lemma="buy", gloss="buy", pos="VERB")],
    )
    patch_pipeline(monkeypatch, delivery)
    gen_prompts.main(root=tmp_path, round_no="t")
    files = sorted((tmp_path / "work" / "gemini_prompts_rt").glob("*.txt"))
    assert [f.name for f in files] == ["en_ra.txt"]
    text = files[0].read_text(encoding="utf-8")
    assert "buy (target: en, VERB)" not in text  # covered cell not re-asked
    assert "book (target: en, NOUN)" in text  # deficit cell asked
    assert "en: 1 cells" in capsys.readouterr().out


def test_gen_prompts_skips_en_homonym_black_hole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # en already has 'book' as ADJ: a NOUN row collides on (lang, lemma,
    # gloss_norm) and harmonization drops it, so the generator must skip it.
    delivery = make_delivery(
        [SimpleNamespace(lemma="书", gloss="book", pos="NOUN")],
        [SimpleNamespace(lemma="book", gloss="book", pos="ADJ")],
    )
    patch_pipeline(monkeypatch, delivery)
    gen_prompts.main(root=tmp_path, round_no="t")
    assert not list((tmp_path / "work" / "gemini_prompts_rt").glob("en_*.txt"))
    assert "en: 0 cells" in capsys.readouterr().out


def test_gen_prompts_skips_key_already_in_raw_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    delivery = make_delivery(
        [SimpleNamespace(lemma="书", gloss="book", pos="NOUN")], []
    )
    patch_pipeline(monkeypatch, delivery)
    write_expansion(tmp_path, "english", [["book2", "book", "书", "NOUN", "B1"]])
    gen_prompts.main(root=tmp_path, round_no="t")
    assert not list((tmp_path / "work" / "gemini_prompts_rt").glob("en_*.txt"))


def test_gen_prompts_prefers_clean_over_fallback_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # en carries 'near' as ADJ in delivery: the ADP cell is an en homonym
    # black hole (skipped), while the fully-missing 'book' cell is asked.
    delivery = make_delivery(
        [
            SimpleNamespace(lemma="附近", gloss="near", pos="ADP"),
            SimpleNamespace(lemma="书", gloss="book", pos="NOUN"),
        ],
        [SimpleNamespace(lemma="nabij", gloss="near", pos="ADJ")],
    )
    delivery["en"][0] = SimpleNamespace(lemma="near", gloss="near", pos="ADJ")
    patch_pipeline(monkeypatch, delivery)
    gen_prompts.main(root=tmp_path, round_no="t")
    text = (tmp_path / "work" / "gemini_prompts_rt" / "en_ra.txt").read_text(
        encoding="utf-8"
    )
    assert "book (target: en, NOUN)" in text
    assert "near (target: en, ADP)" not in text


def test_gen_prompts_fallback_cells_for_non_en_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # For nl (lemma != gloss allowed) a gloss that exists under another POS is
    # merely fallback padding, not a black hole, so the cell must be asked.
    base = [SimpleNamespace(lemma="书", gloss="book", pos="NOUN")]
    delivery = dict.fromkeys(sorted(ALL_LANGS), base)
    delivery["nl"] = [SimpleNamespace(lemma="nabij", gloss="near", pos="ADJ")]
    delivery["zh"] = [SimpleNamespace(lemma="附近", gloss="near", pos="ADP")]
    patch_pipeline(monkeypatch, delivery)
    gen_prompts.main(root=tmp_path, round_no="t")
    text = (tmp_path / "work" / "gemini_prompts_rt" / "nl_ra.txt").read_text(
        encoding="utf-8"
    )
    assert "near (target: nl, ADP)" in text


# ------------------------------------------------- clean_expansion --apply IO


def test_clean_expansion_apply_deletes_and_writes_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    write_expansion(
        tmp_path,
        "spanish",
        [
            ["harira", "harira", "汤", "NOUN", "B2"],  # single copy -> allowlist
            ["do one's best", "do one's best", "尽力", "VERB", "B2"],  # multi -> delete
            ["comer", "eat", "吃", "VERB", "A1"],  # clean -> kept
        ],
    )
    write_expansion(tmp_path, "arabic", [["cafe", "cafe", "咖啡", "NOUN", "A1"]])
    monkeypatch.setattr(clean_expansion, "ROOT", tmp_path)
    (tmp_path / "work").mkdir()
    monkeypatch.setattr("sys.argv", ["clean_expansion.py", "--apply"])
    clean_expansion.main()
    rows = read_data_rows(tmp_path / "spanish" / "expansion.csv")
    assert rows == [
        ["harira", "harira", "汤", "NOUN", "B2"],  # single copy kept + allowlisted
        ["comer", "eat", "吃", "VERB", "A1"],
    ]
    out = capsys.readouterr().out
    assert "DELETED 2 rows" in out  # multi copy + ar script junk
    allow = json.loads(
        (tmp_path / "work" / "allowlist_extra.json").read_text(encoding="utf-8")
    )
    assert allow["es"] == ["harira"]


def test_clean_expansion_dry_run_deletes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    write_expansion(
        tmp_path, "spanish", [["do one's best", "do one's best", "尽力", "VERB", "B2"]]
    )
    monkeypatch.setattr(clean_expansion, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["clean_expansion.py"])
    clean_expansion.main()
    assert len(read_data_rows(tmp_path / "spanish" / "expansion.csv")) == 1
    assert "dry run" in capsys.readouterr().out


# --------------------------------------------------------- flush_checkpoints


def test_flush_checkpoints_appends_uncovered_approved_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from scripts.expand_concepts import LANG_DIRS
    from work import flush_checkpoints

    state_root = tmp_path / "state"
    for name, _code in LANG_DIRS.items():
        write_expansion(
            tmp_path, name, [["covered", "covered gloss", "盖", "NOUN", "B2"]]
        )
    for code in ("de", "ar", "en", "es", "fr", "nl", "sv", "zh"):
        state_dir = state_root / code
        state_dir.mkdir(parents=True)
        (state_dir / ".checkpoint.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "event": "approved",
                            "lang": code,
                            "gloss": " covered gloss ",
                            "pos": "NOUN",
                            "lemma": "skipme",
                            "zh": "",
                            "level": "B2",
                        }
                    ),
                    "{not json",
                    json.dumps(
                        {
                            "event": "generated",
                            "lang": code,
                            "gloss": "g2",
                            "pos": "VERB",
                            "lemma": "x",
                            "zh": "",
                        }
                    ),
                    json.dumps(
                        {
                            "event": "approved",
                            "lang": "other",
                            "gloss": "wrong lang",
                            "pos": "NOUN",
                            "lemma": "y",
                            "zh": "",
                        }
                    ),
                    json.dumps(
                        {
                            "event": "approved",
                            "lang": code,
                            "gloss": "new gloss",
                            "pos": "VERB",
                            "lemma": "neu",
                            "zh": "新",
                            "level": "B1",
                        }
                    ),
                    json.dumps(
                        {
                            "event": "approved",
                            "lang": code,
                            "gloss": "new gloss",
                            "pos": "VERB",
                            "lemma": "dupe",
                            "zh": "新2",
                        }
                    ),
                    json.dumps(
                        {
                            "event": "approved",
                            "lang": code,
                            "gloss": "",
                            "pos": "VERB",
                            "lemma": "emptygloss",
                            "zh": "",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(flush_checkpoints, "ROOT", tmp_path)
    monkeypatch.setattr(
        flush_checkpoints,
        "STATE_DIRS",
        {
            code: state_root / code
            for code in ("de", "ar", "en", "es", "fr", "nl", "sv", "zh")
        },
    )
    flush_checkpoints.main()
    for name, _code in LANG_DIRS.items():
        rows = read_data_rows(tmp_path / name / "expansion.csv")
        assert rows == [
            ["covered", "covered gloss", "盖", "NOUN", "B2"],
            ["neu", "new gloss", "新", "VERB", "B1"],
        ]
    out = capsys.readouterr().out
    assert "de: +1 rows from checkpoint" in out
    assert "en: +1 rows from checkpoint" in out


def test_flush_checkpoints_empty_state_appends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.expand_concepts import LANG_DIRS
    from work import flush_checkpoints

    state_root = tmp_path / "state"
    for name, _code in LANG_DIRS.items():
        write_expansion(tmp_path, name, [])
    for code in ("de", "ar", "en", "es", "fr", "nl", "sv", "zh"):
        (state_root / code).mkdir(parents=True)
        (state_root / code / ".checkpoint.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(flush_checkpoints, "ROOT", tmp_path)
    monkeypatch.setattr(
        flush_checkpoints,
        "STATE_DIRS",
        {
            code: state_root / code
            for code in ("de", "ar", "en", "es", "fr", "nl", "sv", "zh")
        },
    )
    flush_checkpoints.main()
    for name, _code in LANG_DIRS.items():
        assert read_data_rows(tmp_path / name / "expansion.csv") == []
