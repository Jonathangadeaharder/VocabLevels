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

from vocab_schema import LANGS, LEVELS, TARGETS

ROOT = Path(__file__).parent

SPECIAL_CHARS = re.compile(r"[?!@#$%^&*()_=+\[\]{};:\"\\|<>~`]")

# Harmonized dual-pivot column layout on disk (commit c122a99), read
# positionally rather than through a name-keyed dict reader. English and
# Chinese are the two pivot languages: their lemma column name legitimately
# collides with one of these two fixed names (English_Lemma / Chinese_Lemma
# respectively). A dict-keyed reader collapses same-named columns into one
# key and silently keeps only the last column's value — for Chinese that
# means the real lemma (column 0) would never be read at all, which is how
# the C1 "confronting" row went undetected. Reading columns by index instead
# of by name keeps both physical columns visible regardless of name clashes.
LEMMA_IDX, T1_IDX, T2_IDX, POS_IDX = 0, 1, 2, 3
T1_NAME = "English_Lemma"
T2_NAME = "Chinese_Lemma"


def check_language(lang: str, *, show_shared_translations: bool = False) -> int:
    cfg = LANGS[lang]
    lang_dir = ROOT / lang
    print(f"\n=== {lang.upper()} ===")

    # All languages now use harmonized CEFR A1-C1 levels.
    levels = LEVELS
    targets = TARGETS

    seen_lemmas: dict[str, str] = {}  # lemma -> first level it appeared in
    issues = 0
    warnings = 0

    for level in levels:
        path = lang_dir / f"{level}.csv"
        if not path.exists():
            print(f"  [WARN] {path} missing")
            continue

        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            # Real on-disk header is always [<Lang>_Lemma, English_Lemma,
            # Chinese_Lemma, POS]. cfg["lemma_col"] already equals
            # "English_Lemma"/"Chinese_Lemma" for the two pivot languages,
            # so no per-language special-casing is needed here.
            expected = [cfg["lemma_col"], T1_NAME, T2_NAME, "POS"]
            if header != expected:
                print(f"  [ERROR] {level}: bad header {header}")
                issues += 1
                continue

            rows = [row for row in reader if row]

        count = len(rows)
        target = targets[level]
        delta = count - target
        status = "OK" if abs(delta) <= target * 0.05 else f"OFF ({delta:+d})"
        print(f"  {level}: {count} rows (target {target}) — {status}")

        intra_lemmas: set[str] = set()
        intra_trans: dict[str, set[str]] = {}  # translation -> lemmas in this level
        for idx, row in enumerate(rows, start=2):
            lemma_raw = row[LEMMA_IDX] if len(row) > LEMMA_IDX else ""
            t1_raw = row[T1_IDX] if len(row) > T1_IDX else ""
            t2_raw = row[T2_IDX] if len(row) > T2_IDX else ""
            pos_raw = row[POS_IDX] if len(row) > POS_IDX else ""

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
                print(f"    L{idx}: '{lemma}' missing {T1_NAME}")
                issues += 1
            elif t1 != t1_raw:
                print(f"    L{idx}: '{lemma}' {T1_NAME} has whitespace")
                issues += 1
            if not t2:
                # Chinese_Lemma translations are 100% unfilled for 7/8
                # languages (tracked as a known gap in the audit doc) — a
                # translation-completeness gap, not a lemma-cleanliness
                # defect, so it must not drown out real issues in the count.
                print(f"    L{idx}: [WARN] '{lemma}' missing {T2_NAME}")
                warnings += 1
            elif t2 != t2_raw:
                print(f"    L{idx}: '{lemma}' {T2_NAME} has whitespace")
                issues += 1

            # Duplicate key includes POS: same lemma with different POS
            # (e.g. "run" as VERB and NOUN) is NOT a duplicate.
            pos = (pos_raw or "X").strip()
            key = f"{lemma.lower()}|{pos}"

            if key in intra_lemmas:
                print(f"    L{idx}: intra-level duplicate '{lemma}' ({pos})")
                issues += 1
            intra_lemmas.add(key)

            # Cross-level check also keyed on lemma+POS
            if key in seen_lemmas and seen_lemmas[key] != level:
                print(f"    L{idx}: '{lemma}' ({pos}) already in {seen_lemmas[key]}")
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

    if warnings:
        print(
            f"  [WARN] {warnings} rows missing {T2_NAME} (see 'Deferred' in audit doc)"
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
