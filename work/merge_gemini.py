"""Merge agent-studio Gemini CSVs into per-language expansion.csv files.

Reads work/gemini_out/{lang}_*.csv (Lang,Lemma,English_Lemma,Chinese_Lemma,POS,CEFR),
normalizes CEFR (C1 -> Advanced per vocab_schema.LEVELS), drops invalid POS/CEFR
rows and empty fields, dedupes on (normalize_gloss(english), POS) against the
existing expansion.csv, then appends.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.expand_concepts import LANG_DIRS, normalize_gloss

LEVELS = ("A1", "A2", "B1", "B2", "Advanced")
POS_OK = {
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "PRON",
    "ADP",
    "NUM",
    "SCONJ",
    "INTJ",
    "DET",
    "PART",
    "AUX",
    "CONJ",
    "CCONJ",
}


def merge(root: Path, out_dir: Path) -> None:
    for name, code in LANG_DIRS.items():
        path = root / name / "expansion.csv"
        covered: set[tuple[str, str]] = set()
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for cols in reader:
                    if len(cols) >= 4 and cols[0].strip() and cols[1].strip():
                        covered.add(
                            (normalize_gloss(cols[1].strip()), cols[3].strip().upper())
                        )

        files = sorted(out_dir.glob(f"{code}_80*.csv")) + sorted(
            out_dir.glob(f"{code}_r*.csv")
        )
        if not files:
            print(f"{code}: no gemini files")
            continue
        seen: set[tuple[str, str]] = set()
        new_rows: list[list[str]] = []
        dropped = 0
        for f in files:
            with f.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for cols in reader:
                    if len(cols) < 6:
                        dropped += 1
                        continue
                    _, lemma, gloss, zh, pos, cefr = (c.strip() for c in cols[:6])
                    cefr = {"C1": "Advanced"}.get(cefr, cefr)
                    if (
                        not lemma
                        or not gloss
                        or pos not in POS_OK
                        or cefr not in LEVELS
                    ):
                        dropped += 1
                        continue
                    key = (normalize_gloss(gloss), pos)
                    if key in covered or key in seen:
                        dropped += 1
                        continue
                    seen.add(key)
                    new_rows.append([lemma, gloss, zh, pos, cefr])
        if new_rows:
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerows(new_rows)
        print(f"{code}: +{len(new_rows)} rows, {dropped} skipped")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "work" / "gemini_out"
    merge(root, out_dir)


if __name__ == "__main__":
    main()
