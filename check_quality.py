"""Quality checker for multilingual CEFR vocab CSVs.

Each language directory holds A1-C1 CSVs. The lemma column is named after the
language; translation columns are configured in vocab_schema.py.

Run from repo root:
    python check_quality.py
    python check_quality.py english   # single language
    python check_quality.py --show-shared-translations
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from vocab_schema import LANGS, LEVELS, TARGETS, HSK_LEVELS, HSK_TARGETS

ROOT = Path(__file__).parent

SPECIAL_CHARS = re.compile(r"[?!@#$%^&*()_=+\[\]{};:\"\\|<>~`]")


def check_language(lang: str, *, show_shared_translations: bool = False) -> int:
    cfg = LANGS[lang]
    lang_dir = ROOT / lang
    print(f"\n=== {lang.upper()} ===")

    # All languages now use harmonized CEFR A1-C1 levels.
    levels = LEVELS
    targets = TARGETS

    seen_lemmas: dict[str, str] = {}  # lemma -> first level it appeared in
    issues = 0

    for level in levels:
        path = lang_dir / f"{level}.csv"
        if not path.exists():
            print(f"  [WARN] {path} missing")
            continue

        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != [cfg["lemma_col"], *cfg["trans_cols"], "POS"]:
                print(f"  [ERROR] {level}: bad header {reader.fieldnames}")
                issues += 1
                continue

            rows = list(reader)

        count = len(rows)
        target = targets[level]
        delta = count - target
        status = "OK" if abs(delta) <= target * 0.05 else f"OFF ({delta:+d})"
        print(f"  {level}: {count} rows (target {target}) — {status}")

        intra_lemmas: set[str] = set()
        intra_trans: dict[str, set[str]] = {}  # translation -> lemmas in this level
        for idx, row in enumerate(rows, start=2):
            lemma_raw = row.get(cfg["lemma_col"]) or ""
            t1_raw = row.get(cfg["trans_cols"][0]) or ""
            t2_raw = row.get(cfg["trans_cols"][1]) or ""

            lemma = lemma_raw.strip()
            t1 = t1_raw.strip()
            t2 = t2_raw.strip()

            if not lemma:
                print(f"    L{idx}: empty lemma")
                issues += 1
                continue
            if lemma != lemma_raw:
                print(f"    L{idx}: lemma '{lemma}' has leading/trailing whitespace")
                issues += 1
            if " " in lemma:
                print(f"    L{idx}: multi-word lemma '{lemma}'")
                issues += 1
            if re.search(r"[0-9]", lemma):
                print(f"    L{idx}: digits in lemma '{lemma}'")
                issues += 1
            if SPECIAL_CHARS.search(lemma):
                print(f"    L{idx}: special chars in lemma '{lemma}'")
                issues += 1
            # Capitalized lemmas allowed (German nouns, proper nouns in any language)

            if not t1:
                print(f"    L{idx}: '{lemma}' missing {cfg['trans_cols'][0]}")
                issues += 1
            elif t1 != t1_raw:
                print(f"    L{idx}: '{lemma}' {cfg['trans_cols'][0]} has whitespace")
                issues += 1
            if not t2:
                print(f"    L{idx}: '{lemma}' missing {cfg['trans_cols'][1]}")
                issues += 1
            elif t2 != t2_raw:
                print(f"    L{idx}: '{lemma}' {cfg['trans_cols'][1]} has whitespace")
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

            for trans in {t1.lower(), t2.lower()}:
                if trans:
                    intra_trans.setdefault(trans, set()).add(lemma)

        shared_translations = [
            (trans, lemmas) for trans, lemmas in intra_trans.items() if len(lemmas) > 1
        ]
        if show_shared_translations:
            for trans, lemmas in shared_translations:
                print(
                    f"    {level}: '{trans}' shared by {len(lemmas)} lemmas: {', '.join(sorted(lemmas))}"
                )
        elif shared_translations:
            print(
                f"    {level}: {len(shared_translations)} shared translation groups "
                "(use --show-shared-translations for details)"
            )

    return issues


def main(argv: list[str]) -> int:
    args = argv[1:]
    show_shared_translations = "--show-shared-translations" in args
    langs = [arg for arg in args if arg != "--show-shared-translations"] or list(LANGS)
    total = 0
    for lang in langs:
        if lang not in LANGS:
            print(f"Unknown language: {lang}")
            return 2
        total += check_language(lang, show_shared_translations=show_shared_translations)
    print(f"\nTotal issues: {total}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
