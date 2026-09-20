"""Audit lemmatization: find plurals, verb forms, and other inflections."""

import csv
from pathlib import Path

from vocab_schema import LANGS, LEVELS

ROOT = Path(__file__).parent


def _collect_lemmas(lang: str, lemma_col: str) -> dict[str, list]:
    """Index lemma_lower -> list of (level, row) across all levels."""
    all_lemmas: dict[str, list] = {}
    for level in LEVELS:
        path = ROOT / lang / f"{level}.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                lemma = row[lemma_col].strip().lower()
                if lemma not in all_lemmas:
                    all_lemmas[lemma] = []
                all_lemmas[lemma].append((level, row))
    return all_lemmas


def _suffix_candidate(
    lemma: str, ftype: str, cut: int, all_lemmas: dict
) -> tuple | None:
    base = lemma[:-cut]
    if base in all_lemmas and base != lemma:
        return (lemma, ftype, base)
    return None


def _match_candidate(lemma: str, all_lemmas: dict) -> tuple | None:
    if lemma.endswith("s") and not lemma.endswith("ss"):
        return _suffix_candidate(lemma, "plural", 1, all_lemmas)
    if lemma.endswith("ing"):
        return _suffix_candidate(lemma, "gerund", 3, all_lemmas)
    if lemma.endswith("ed") and len(lemma) > 3:
        return _suffix_candidate(lemma, "past", 2, all_lemmas)
    return None


def _find_candidates(all_lemmas: dict) -> list[tuple]:
    candidates = []
    for lemma in sorted(all_lemmas.keys()):
        candidate = _match_candidate(lemma, all_lemmas)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _print_candidates(candidates: list[tuple], all_lemmas: dict) -> None:
    if not candidates:
        print("No obvious inflected forms found")
        return
    print(f"Found {len(candidates)} potential non-lemmatized forms:")
    for form, ftype, base in candidates[:20]:
        levels = [lv for lv, _ in all_lemmas[form]]
        base_levels = [lv for lv, _ in all_lemmas[base]]
        print(
            f"  {form:20} ({ftype:10}) → {base:15} | "
            f"{form}: {levels}, {base}: {base_levels}"
        )


def find_inflected_forms():
    """Scan for likely inflected forms (plurals, verb forms)."""
    for lang, cfg in LANGS.items():
        print(f"\n=== {lang.upper()} ===")
        all_lemmas = _collect_lemmas(lang, cfg["lemma_col"])
        _print_candidates(_find_candidates(all_lemmas), all_lemmas)


if __name__ == "__main__":
    find_inflected_forms()
