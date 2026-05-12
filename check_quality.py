"""Quality checker for trilingual CEFR vocab CSVs.

Each language directory holds A1-C1 CSVs. The lemma column is named after the
language; the two translation columns cover the other two languages.

Run from repo root:
    python check_quality.py
    python check_quality.py english   # single language
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
LEVELS = ["A1", "A2", "B1", "B2", "C1"]
TARGETS = {"A1": 600, "A2": 600, "B1": 1000, "B2": 2000, "C1": 4000}

LANGS = {
    "english": {
        "lemma_col": "English_Lemma",
        "trans_cols": ("German_Translation", "Spanish_Translation"),
    },
    "german": {
        "lemma_col": "German_Lemma",
        "trans_cols": ("English_Translation", "Spanish_Translation"),
    },
    "spanish": {
        "lemma_col": "Spanish_Lemma",
        "trans_cols": ("English_Translation", "German_Translation"),
    },
}

SPECIAL_CHARS = re.compile(r"[?!@#$%^&*()_=+\[\]{};:\"\\|<>~`]")


def check_language(lang: str) -> int:
    cfg = LANGS[lang]
    lang_dir = ROOT / lang
    print(f"\n=== {lang.upper()} ===")

    seen_lemmas: dict[str, str] = {}  # lemma -> first level it appeared in
    issues = 0

    for level in LEVELS:
        path = lang_dir / f"{level}.csv"
        if not path.exists():
            print(f"  [WARN] {path} missing")
            continue

        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != [cfg["lemma_col"], *cfg["trans_cols"]]:
                print(f"  [ERROR] {level}: bad header {reader.fieldnames}")
                issues += 1
                continue

            rows = list(reader)

        count = len(rows)
        target = TARGETS[level]
        delta = count - target
        status = "OK" if abs(delta) <= target * 0.05 else f"OFF ({delta:+d})"
        print(f"  {level}: {count} rows (target {target}) — {status}")

        intra_lemmas: set[str] = set()
        for idx, row in enumerate(rows, start=2):
            lemma = (row.get(cfg["lemma_col"]) or "").strip()
            t1 = (row.get(cfg["trans_cols"][0]) or "").strip()
            t2 = (row.get(cfg["trans_cols"][1]) or "").strip()

            if not lemma:
                print(f"    L{idx}: empty lemma")
                issues += 1
                continue
            if " " in lemma:
                print(f"    L{idx}: multi-word lemma '{lemma}'")
                issues += 1
            if re.search(r"[0-9]", lemma):
                print(f"    L{idx}: digits in lemma '{lemma}'")
                issues += 1
            if SPECIAL_CHARS.search(lemma):
                print(f"    L{idx}: special chars in lemma '{lemma}'")
                issues += 1
            if not t1:
                print(f"    L{idx}: '{lemma}' missing {cfg['trans_cols'][0]}")
                issues += 1
            if not t2:
                print(f"    L{idx}: '{lemma}' missing {cfg['trans_cols'][1]}")
                issues += 1

            key = lemma.lower()
            if key in intra_lemmas:
                print(f"    L{idx}: intra-level duplicate '{lemma}'")
                issues += 1
            intra_lemmas.add(key)

            if key in seen_lemmas and seen_lemmas[key] != level:
                print(f"    L{idx}: '{lemma}' already in {seen_lemmas[key]}")
                issues += 1
            else:
                seen_lemmas.setdefault(key, level)

    return issues


def main(argv: list[str]) -> int:
    langs = argv[1:] if len(argv) > 1 else list(LANGS)
    total = 0
    for lang in langs:
        if lang not in LANGS:
            print(f"Unknown language: {lang}")
            return 2
        total += check_language(lang)
    print(f"\nTotal issues: {total}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
