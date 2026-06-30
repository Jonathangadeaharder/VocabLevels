"""Tests for vocab_manager.py — CLI manager for multilingual CEFR vocab CSVs."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import pytest

import vocab_manager as vm


# ---------------------------------------------------------------------------
# Fixtures: temp directory with minimal CSVs
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a minimal repo structure with sample CSVs; patch ROOT."""
    monkeypatch.setattr(vm, "ROOT", tmp_path)
    for lang, cfg in vm.LANGS.items():
        (tmp_path / lang).mkdir(exist_ok=True)
        for level in vm.LEVELS:
            fields = [cfg["lemma_col"], *cfg["trans_cols"]]
            path = tmp_path / lang / f"{level}.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                # Add a couple of seed rows per level
                writer.writerow(
                    {
                        cfg["lemma_col"]: f"{lang[:2]}_{level}_1",
                        cfg["trans_cols"][0]: "t1",
                        cfg["trans_cols"][1]: "t2",
                    }
                )
                writer.writerow(
                    {
                        cfg["lemma_col"]: f"{lang[:2]}_{level}_2",
                        cfg["trans_cols"][0]: "t1",
                        cfg["trans_cols"][1]: "t2",
                    }
                )
    return tmp_path


# ---------------------------------------------------------------------------
# read_level / write_level / file_path
# ---------------------------------------------------------------------------


class TestFilePath:
    def test_constructs_path(self, tmp_repo: Path) -> None:
        result = vm.file_path("english", "A1")
        assert result == tmp_repo / "english" / "A1.csv"

    def test_missing_level_returns_empty(self, tmp_repo: Path) -> None:
        (tmp_repo / "english" / "X1.csv").unlink(missing_ok=True)
        assert vm.read_level("english", "X1") == []


class TestReadWriteLevel:
    def test_round_trip(self, tmp_repo: Path) -> None:
        rows = vm.read_level("english", "A1")
        assert len(rows) == 2
        assert rows[0]["English_Lemma"] == "en_A1_1"

    def test_write_then_read(self, tmp_repo: Path) -> None:
        rows = vm.read_level("english", "A1")
        rows.append(
            {
                "English_Lemma": "apple",
                "German_Translation": "Apfel",
                "Spanish_Translation": "manzana",
            }
        )
        vm.write_level("english", "A1", rows)
        result = vm.read_level("english", "A1")
        lemmas = [r["English_Lemma"] for r in result]
        assert "apple" in lemmas
        assert not (tmp_repo / "english" / ".A1.csv.tmp").exists()

    def test_write_sorts_by_lemma(self, tmp_repo: Path) -> None:
        rows = [
            {
                "English_Lemma": "zebra",
                "German_Translation": "Zebra",
                "Spanish_Translation": "cebra",
            },
            {
                "English_Lemma": "apple",
                "German_Translation": "Apfel",
                "Spanish_Translation": "manzana",
            },
        ]
        vm.write_level("english", "A1", rows)
        result = vm.read_level("english", "A1")
        lemmas = [r["English_Lemma"] for r in result]
        assert lemmas[0] == "apple"
        assert lemmas[1] == "zebra"

    def test_read_missing_file(self, tmp_repo: Path) -> None:
        (tmp_repo / "english" / "C1.csv").unlink()
        assert vm.read_level("english", "C1") == []


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------


class TestFind:
    def test_find_existing(self, tmp_repo: Path) -> None:
        levels = vm.find("english", "en_A1_1")
        assert "A1" in levels

    def test_find_case_insensitive(self, tmp_repo: Path) -> None:
        levels = vm.find("english", "EN_A1_1")
        assert "A1" in levels

    def test_find_missing(self, tmp_repo: Path) -> None:
        levels = vm.find("english", "nonexistent")
        assert levels == []

    def test_find_across_levels(self, tmp_repo: Path) -> None:
        # Add same lemma to A2 as well
        rows = vm.read_level("english", "A2")
        rows.append(
            {
                "English_Lemma": "en_A1_1",
                "German_Translation": "x",
                "Spanish_Translation": "x",
            }
        )
        vm.write_level("english", "A2", rows)
        levels = vm.find("english", "en_A1_1")
        assert "A1" in levels
        assert "A2" in levels


# ---------------------------------------------------------------------------
# cmd_find
# ---------------------------------------------------------------------------


class TestCmdFind:
    def test_found(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        args = argparse.Namespace(lang="english", lemma="en_A1_1")
        ret = vm.cmd_find(args)
        assert ret == 0
        assert "en_A1_1" in capsys.readouterr().out

    def test_not_found(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(lang="english", lemma="nope")
        ret = vm.cmd_find(args)
        assert ret == 1
        assert "not found" in capsys.readouterr().out

    def test_strips_whitespace(self, tmp_repo: Path) -> None:
        args = argparse.Namespace(lang="english", lemma="  en_A1_1  ")
        ret = vm.cmd_find(args)
        assert ret == 0


# ---------------------------------------------------------------------------
# cmd_add
# ---------------------------------------------------------------------------


class TestCmdAdd:
    def test_add_new(self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        args = argparse.Namespace(
            lang="english", level="A1", lemma="hello", t1="Hallo", t2="hola"
        )
        ret = vm.cmd_add(args)
        assert ret == 0
        assert "hello" in capsys.readouterr().out
        assert "hello" in [r["English_Lemma"] for r in vm.read_level("english", "A1")]

    def test_add_duplicate(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(
            lang="english", level="A1", lemma="en_A1_1", t1="x", t2="x"
        )
        ret = vm.cmd_add(args)
        assert ret == 1
        assert "already in" in capsys.readouterr().out

    def test_add_empty_lemma(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(
            lang="english", level="A1", lemma="  ", t1="x", t2="x"
        )
        ret = vm.cmd_add(args)
        assert ret == 1
        assert "empty" in capsys.readouterr().out.lower()

    def test_add_multiword_lemma(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(
            lang="english", level="A1", lemma="big apple", t1="x", t2="x"
        )
        ret = vm.cmd_add(args)
        assert ret == 1
        assert "single-word" in capsys.readouterr().out.lower()

    def test_add_empty_translation(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(
            lang="english", level="A1", lemma="word", t1="", t2="hola"
        )
        ret = vm.cmd_add(args)
        assert ret == 1
        assert "cannot be empty" in capsys.readouterr().out

    def test_add_german(self, tmp_repo: Path) -> None:
        args = argparse.Namespace(
            lang="german", level="B1", lemma="Haus", t1="house", t2="casa"
        )
        ret = vm.cmd_add(args)
        assert ret == 0
        assert "Haus" in [r["German_Lemma"] for r in vm.read_level("german", "B1")]


# ---------------------------------------------------------------------------
# cmd_remove
# ---------------------------------------------------------------------------


class TestCmdRemove:
    def test_remove_existing(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(lang="english", lemma="en_A1_1")
        ret = vm.cmd_remove(args)
        assert ret == 0
        assert vm.find("english", "en_A1_1") == []

    def test_remove_missing(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(lang="english", lemma="nope")
        ret = vm.cmd_remove(args)
        assert ret == 1
        assert "not found" in capsys.readouterr().out

    def test_remove_case_insensitive(self, tmp_repo: Path) -> None:
        args = argparse.Namespace(lang="english", lemma="EN_A1_1")
        ret = vm.cmd_remove(args)
        assert ret == 0
        assert vm.find("english", "en_A1_1") == []


# ---------------------------------------------------------------------------
# cmd_move
# ---------------------------------------------------------------------------


class TestCmdMove:
    def test_move_to_different_level(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(lang="english", target_level="A2", lemma="en_A1_1")
        ret = vm.cmd_move(args)
        assert ret == 0
        assert vm.find("english", "en_A1_1") == ["A2"]
        out = capsys.readouterr().out
        assert "A1" in out and "A2" in out

    def test_move_to_same_level(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(lang="english", target_level="A1", lemma="en_A1_1")
        ret = vm.cmd_move(args)
        assert ret == 0
        assert "already in" in capsys.readouterr().out

    def test_move_missing_lemma(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(lang="english", target_level="A2", lemma="nope")
        ret = vm.cmd_move(args)
        assert ret == 1
        assert "not found" in capsys.readouterr().out

    def test_move_invalid_level(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # cmd_move checks LEVELS — argparse would normally prevent this
        # but we test the internal check
        args = argparse.Namespace(lang="english", target_level="X9", lemma="en_A1_1")
        ret = vm.cmd_move(args)
        assert ret == 1
        assert "Invalid level" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# cmd_update
# ---------------------------------------------------------------------------


class TestCmdUpdate:
    def test_update_translation(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(
            lang="english", lemma="en_A1_1", t1="neu1", t2="nuevo1", rename=None
        )
        ret = vm.cmd_update(args)
        assert ret == 0
        rows = vm.read_level("english", "A1")
        row = next(r for r in rows if r["English_Lemma"] == "en_A1_1")
        assert row["German_Translation"] == "neu1"
        assert row["Spanish_Translation"] == "nuevo1"

    def test_update_partial_t1_only(self, tmp_repo: Path) -> None:
        args = argparse.Namespace(
            lang="english", lemma="en_A1_1", t1="neu1", t2=None, rename=None
        )
        ret = vm.cmd_update(args)
        assert ret == 0
        rows = vm.read_level("english", "A1")
        row = next(r for r in rows if r["English_Lemma"] == "en_A1_1")
        assert row["German_Translation"] == "neu1"
        # t2 unchanged
        assert row["Spanish_Translation"] == "t2"

    def test_update_rename(self, tmp_repo: Path) -> None:
        args = argparse.Namespace(
            lang="english", lemma="en_A1_1", t1=None, t2=None, rename="renamed"
        )
        ret = vm.cmd_update(args)
        assert ret == 0
        assert vm.find("english", "renamed") == ["A1"]
        assert vm.find("english", "en_A1_1") == []

    def test_update_rename_to_existing(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(
            lang="english", lemma="en_A1_1", t1=None, t2=None, rename="en_A1_2"
        )
        ret = vm.cmd_update(args)
        assert ret == 1
        assert "already exists" in capsys.readouterr().out

    def test_update_empty_rename(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(
            lang="english", lemma="en_A1_1", t1=None, t2=None, rename=""
        )
        ret = vm.cmd_update(args)
        assert ret == 1
        assert "cannot be empty" in capsys.readouterr().out.lower()

    def test_update_multiword_rename(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(
            lang="english", lemma="en_A1_1", t1=None, t2=None, rename="big word"
        )
        ret = vm.cmd_update(args)
        assert ret == 1
        assert "single-word" in capsys.readouterr().out.lower()

    def test_update_empty_translation(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(
            lang="english", lemma="en_A1_1", t1="", t2=None, rename=None
        )
        ret = vm.cmd_update(args)
        assert ret == 1
        assert "cannot be empty" in capsys.readouterr().out

    def test_update_not_found(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(
            lang="english", lemma="nope", t1="x", t2="x", rename=None
        )
        ret = vm.cmd_update(args)
        assert ret == 1
        assert "not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# cmd_lookup
# ---------------------------------------------------------------------------


class TestCmdLookup:
    def test_lookup_by_lemma(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(term="en_A1_1")
        ret = vm.cmd_lookup(args)
        assert ret == 0
        assert "english" in capsys.readouterr().out

    def test_lookup_by_translation(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(term="t1")
        ret = vm.cmd_lookup(args)
        assert ret == 0
        out = capsys.readouterr().out.splitlines()
        expected = [
            f"  {lang}/{level}: {lang[:2]}_{level}_{row_number} | t1 | t2"
            for lang in vm.LANGS
            for level in vm.LEVELS
            for row_number in (1, 2)
        ]
        assert out == expected

    def test_lookup_not_found(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = argparse.Namespace(term="zzznotexist")
        ret = vm.cmd_lookup(args)
        assert ret == 1
        assert "not found" in capsys.readouterr().out

    def test_lookup_case_insensitive(self, tmp_repo: Path) -> None:
        args = argparse.Namespace(term="EN_A1_1")
        ret = vm.cmd_lookup(args)
        assert ret == 0


# ---------------------------------------------------------------------------
# cmd_lint
# ---------------------------------------------------------------------------


class TestCmdLint:
    def test_lint_runs_against_cli_root(
        self, tmp_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import check_quality

        monkeypatch.setattr(check_quality, "ROOT", Path("/tmp/unused-vocab-root"))
        args = argparse.Namespace()
        ret = vm.cmd_lint(args)
        assert check_quality.ROOT == tmp_repo
        assert isinstance(ret, int)


# ---------------------------------------------------------------------------
# main (integration)
# ---------------------------------------------------------------------------


class TestMain:
    def test_find_command(
        self, tmp_repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ret = vm.main(["vocab_manager.py", "find", "english", "en_A1_1"])
        assert ret == 0
        assert "en_A1_1" in capsys.readouterr().out

    def test_add_command(self, tmp_repo: Path) -> None:
        ret = vm.main(
            ["vocab_manager.py", "add", "english", "A1", "hello", "Hallo", "hola"]
        )
        assert ret == 0

    def test_remove_command(self, tmp_repo: Path) -> None:
        ret = vm.main(["vocab_manager.py", "remove", "english", "en_A1_1"])
        assert ret == 0

    def test_lookup_command(self, tmp_repo: Path) -> None:
        ret = vm.main(["vocab_manager.py", "lookup", "en_A1_1"])
        assert ret == 0

    def test_move_command(self, tmp_repo: Path) -> None:
        ret = vm.main(["vocab_manager.py", "move", "english", "A2", "en_A1_1"])
        assert ret == 0

    def test_update_command(self, tmp_repo: Path) -> None:
        ret = vm.main(
            ["vocab_manager.py", "update", "english", "en_A1_1", "--t1", "newt1"]
        )
        assert ret == 0

    def test_no_command_errors(self, tmp_repo: Path) -> None:
        with pytest.raises(SystemExit):
            vm.main(["vocab_manager.py"])
