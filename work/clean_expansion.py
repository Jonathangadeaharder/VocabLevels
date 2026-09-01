"""Classify + fix criterion-6 violations in the expansion CSVs.

Reads each language dir's expansion.csv, replays check_script_and_substance,
and:
  - zh non_chinese_script + junk_lemma rows -> DELETE from expansion.csv
  - ungrounded_gloss_copy single-word lemmas -> emit as extra cognate
    allowlist proposal (work/allowlist_extra.json), merge into
    extended_cognates.json by hand
  - ungrounded_gloss_copy multi-word: keep true loans, DELETE the rest
  - english_function_prefix_copy: emit list; native-prefix false positives get
    a native-prefix skip rule in the checker, true copies DELETE

Usage: .venv/bin/python work/clean_expansion.py [--apply]
Default is dry-run: prints counts only.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from check_data_contract import (  # noqa: E402
    CHINESE_SCRIPT,
    COGNATE_ALLOWLIST,
    ENGLISH_FUNCTION_PREFIXES,
    FORBIDDEN_JUNK_LEMMAS,
    LANG_DIRS,
    normalize_gloss,
)

ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

# Genuine foreign-loan phrases that legitimately equal their English gloss.
MULTIWORD_KEEP = {"de facto", "a priori", "a posteriori", "dim sum"}

# Native function-word prefixes per language. A lemma starting with one of
# these is a native multiword expression, not an English copy, so the
# english_function_prefix_copy rule must not fire on it.
NATIVE_PREFIXES: dict[str, tuple[str, ...]] = {
    "de": (
        "in ",
        "an ",
        "auf ",
        "aus ",
        "bei ",
        "mit ",
        "nach ",
        "vor ",
        "zu ",
        "zum ",
        "zur ",
        "der ",
        "die ",
        "das ",
        "den ",
        "dem ",
        "des ",
        "ein ",
        "eine ",
        "einen ",
    ),
    "es": (
        "a ",
        "por ",
        "de ",
        "del ",
        "en ",
        "con ",
        "sin ",
        "para ",
        "el ",
        "la ",
        "los ",
        "las ",
        "un ",
        "una ",
        "al ",
    ),
    "fr": (
        "à ",
        "de ",
        "du ",
        "des ",
        "en ",
        "par ",
        "pour ",
        "avec ",
        "sans ",
        "le ",
        "la ",
        "les ",
        "un ",
        "une ",
        "au ",
        "aux ",
        "d'un ",
        "d'une ",
        "dans ",
        "sur ",
        "chez ",
    ),
    "nl": (
        "in ",
        "voor ",
        "te ",
        "tot ",
        "met ",
        "van ",
        "op ",
        "bij ",
        "de ",
        "het ",
        "een ",
        "der ",
        "den ",
        "ter ",
        "ten ",
    ),
    "sv": (
        "i ",
        "på ",
        "för ",
        "till ",
        "med ",
        "av ",
        "från ",
        "vid ",
        "den ",
        "det ",
        "de ",
        "en ",
        "ett ",
        "om ",
        "ur ",
    ),
}


def load_expansion(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return [header] + [row for row in reader if row and row[0].strip()]


def classify(rows: list[tuple[str, str, str]], lang: str):
    """Replay check_script_and_substance on (lemma, gloss, pos) triples."""
    single_copies: list[str] = []
    multi_copies: list[str] = []
    prefix_copies: list[str] = []
    delete_keys: set[tuple[str, str, str]] = set()
    for lemma, gloss, pos in rows:
        lemma_s = lemma.strip()
        if not lemma_s:
            continue
        if lemma_s in FORBIDDEN_JUNK_LEMMAS:
            delete_keys.add((lemma, gloss, pos))
            continue
        if lang == "ar":
            if any(c.isalpha() for c in lemma_s) and not ARABIC_SCRIPT.search(lemma_s):
                delete_keys.add((lemma, gloss, pos))
            continue
        if lang == "zh":
            if any(c.isalpha() for c in lemma_s) and not CHINESE_SCRIPT.search(lemma_s):
                delete_keys.add((lemma, gloss, pos))
            continue
        if lang == "en":
            continue
        gloss_norm = normalize_gloss(gloss)
        lemma_norm = normalize_gloss(lemma_s)
        lemma_lower = lemma_s.lower()
        gloss_lower = gloss.lower()
        allowlist = COGNATE_ALLOWLIST.get(lang, set())
        is_copy = lemma_norm == gloss_norm or lemma_lower == gloss_lower
        if is_copy and lemma_lower not in allowlist and lemma_norm not in allowlist:
            if " " in lemma_lower:
                if lemma_lower not in MULTIWORD_KEEP:
                    multi_copies.append(lemma_s)
                    delete_keys.add((lemma, gloss, pos))
            else:
                single_copies.append(lemma_s)
            continue
        if lemma_lower.startswith(ENGLISH_FUNCTION_PREFIXES):
            if lemma_lower.startswith(NATIVE_PREFIXES.get(lang, ())):
                continue  # false positive: native preposition phrase
            prefix_copies.append(lemma_s)
            delete_keys.add((lemma, gloss, pos))
    return single_copies, multi_copies, prefix_copies, delete_keys


def main() -> None:
    apply = "--apply" in sys.argv
    all_singles: dict[str, set[str]] = {}
    total_deleted = 0
    for name, code in LANG_DIRS.items():
        path = ROOT / name / "expansion.csv"
        if not path.exists():
            continue
        table = load_expansion(path)
        triples = [(r[0], r[1], r[3]) for r in table[1:]]
        singles, multis, prefixes, delete_keys = classify(triples, code)
        if singles:
            all_singles[code] = set(singles)
        if delete_keys or multis or prefixes:
            print(
                f"{code}: singles={len(singles)} multi_delete={len(multis)} "
                f"prefix_delete={len(prefixes)} "
                f"zh_script_junk={len(delete_keys) - len(multis) - len(prefixes)}"
            )
        if apply and delete_keys:
            kept = [table[0]] + [
                r for r in table[1:] if (r[0], r[1], r[3]) not in delete_keys
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(kept)
            total_deleted += len(table) - len(kept)
    if total_deleted:
        print(f"DELETED {total_deleted} rows")
    else:
        print("dry run: nothing deleted (use --apply)")
    if all_singles:
        out = ROOT / "work" / "allowlist_extra.json"
        proposal = {
            code: sorted({w.lower() for w in all_singles[code]})
            for code in sorted(all_singles)
        }
        out.write_text(
            json.dumps(proposal, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(f"wrote {out} ({sum(len(v) for v in proposal.values())} words)")


if __name__ == "__main__":
    main()
