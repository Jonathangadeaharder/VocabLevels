"""Remove obvious inflected forms and exact duplicates, keeping lemmas.

Two independent, language-agnostic cleanups:

1. ``remove_exact_duplicates`` — for every language, drop verbatim
   (lemma, POS) duplicate rows within the same level, keeping the first
   occurrence. Pure CSV comparison, no NLP required.
2. ``remove_english_plural_duplicates`` — English-only suffix heuristic
   (kept for backward compatibility with the original tool).

Real inflected-form-vs-citation-form cleanup (e.g. Swedish "används" vs
"använda") requires morphological analysis and lives in
``remove_inflected_duplicates`` (see that function's docstring) — it is
driven by each language's own Stanza lemmatizer rather than a hardcoded
word list, and is opt-in via ``--stanza-langs`` since it needs the (large,
network-downloaded) Stanza models.

Run from repo root:
    uv run python cleanup_inflections.py                       # exact dups only, all langs
    uv run python cleanup_inflections.py --stanza-langs german spanish french swedish dutch
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import fix_pos_and_overflow as fpo
from vocab_schema import LANGS, LEVELS

ROOT = Path(__file__).parent


def _read_level(lang: str, level: str) -> list[dict[str, str]]:
    """Read a level CSV via fix_pos_and_overflow's schema-correct I/O.

    vocab_manager.read_level/write_level assume the wrong (3-column,
    no-POS) schema from vocab_schema.trans_cols — real files are the
    4-column dual-pivot shape [<Lang>_Lemma, English_Lemma, Chinese_Lemma,
    POS]. Using vocab_manager.write_level here would silently drop the POS
    column from every row it touches. fix_pos_and_overflow.load_csv/
    save_csv already match the real on-disk shape, so cleanup reuses them.
    """
    fpo.ROOT = ROOT
    return fpo.load_csv(lang, level)


def _write_level(lang: str, level: str, rows: list[dict[str, str]]) -> None:
    fpo.ROOT = ROOT
    fpo.save_csv(lang, level, rows)


def _file_path(lang: str, level: str) -> Path:
    return ROOT / lang / f"{level}.csv"


# Languages whose morphology is well served by Stanza's neural lemmatizer
# and where the audit found genuine inflected-form-as-lemma duplicates.
# Arabic/Chinese are excluded: Arabic's root-and-pattern morphology makes
# naive lemma-diff unreliable (per audit), and Chinese has no inflection.
STANZA_INFLECTION_LANGS = ("english", "german", "spanish", "french", "swedish", "dutch")

STANZA_LANG_CODE = {
    "english": "en",
    "german": "de",
    "spanish": "es",
    "french": "fr",
    "swedish": "sv",
    "dutch": "nl",
}

_COARSE_POS = {
    "VERB": "V",
    "AUX": "V",
    "NOUN": "N",
    "PROPN": "N",
    "ADJ": "ADJ",
    "ADV": "ADV",
}


def _coarse_pos(upos: str) -> str:
    return _COARSE_POS.get(upos, upos)


def remove_exact_duplicates(lang: str) -> int:
    """Remove verbatim (lemma, POS) duplicate rows within the same level.

    Keeps the first occurrence. Applies to every language — this is the
    class of defect found in German (12 rows: Antwort/kaufen/Jagd/Angst
    etc. each appearing twice verbatim in the same level file).
    """
    cfg = LANGS[lang]
    lemma_col = cfg["lemma_col"]

    total_removed = 0
    for level in LEVELS:
        if not _file_path(lang, level).exists():
            continue
        rows = _read_level(lang, level)

        seen: set[tuple[str, str]] = set()
        kept = []
        removed_here = 0
        for row in rows:
            lemma = (row.get(lemma_col) or "").strip()
            pos = (row.get("POS") or "").strip()
            key = (lemma.lower(), pos)
            if key in seen:
                removed_here += 1
                continue
            seen.add(key)
            kept.append(row)

        if removed_here > 0:
            _write_level(lang, level, kept)
            print(
                f"{lang}/{level}: removed {removed_here} exact duplicate rows "
                f"({len(kept)} rows remain)"
            )
            total_removed += removed_here

    return total_removed


def remove_english_plural_duplicates(lang: str) -> int:
    """Remove same-level English plurals when the singular already exists."""
    cfg = LANGS[lang]
    if lang != "english":
        return 0

    lemma_col = cfg["lemma_col"]

    total_removed = 0
    for level in LEVELS:
        if not _file_path(lang, level).exists():
            continue
        rows = _read_level(lang, level)

        lemmas_in_level = set(row[lemma_col].strip().lower() for row in rows)

        kept = []
        removed_here = 0
        for row in rows:
            lemma = row[lemma_col].strip()
            lemma_lower = lemma.lower()

            skip = False
            if lemma_lower.endswith("s") and not lemma_lower.endswith("ss"):
                singular = lemma_lower[:-1]
                if (
                    singular in lemmas_in_level
                    and len(singular) > 2
                    and singular != lemma_lower
                ):
                    skip = True
                    removed_here += 1

            if not skip:
                kept.append(row)

        if removed_here > 0:
            _write_level(lang, level, kept)
            print(
                f"{lang}/{level}: removed {removed_here} plural forms ({len(kept)} rows remain)"
            )
            total_removed += removed_here

    return total_removed


def cleanup_language(lang: str) -> int:
    """Run the fast, stanza-free cleanups for a language.

    Always removes exact duplicates; additionally applies the English
    plural-suffix heuristic for English. Real cross-language inflected-form
    cleanup (Swedish, Dutch, German, Spanish, French) requires
    ``remove_inflected_duplicates`` and Stanza models — see that function.
    """
    total = remove_exact_duplicates(lang)
    total += remove_english_plural_duplicates(lang)
    return total


# --- Stanza-driven inflected-form dedup (opt-in, needs models) -----------


def _stanza_tag_fn(lang: str) -> Callable[[str], tuple[str, str]]:
    """Build a (word) -> (base_lemma, upos) function using Stanza.

    Lazy import so importing this module never requires stanza to be
    installed or its models to be downloaded (mirrors the pattern used by
    fix_pos_and_overflow.get_pipeline).
    """
    import stanza

    nlp: Any = stanza.Pipeline(
        lang=STANZA_LANG_CODE[lang],
        processors="tokenize,pos,lemma",
        verbose=False,
        use_gpu=False,
    )

    def tag(word: str) -> tuple[str, str]:
        doc = nlp(word)
        if not doc.sentences or not doc.sentences[0].words:
            return word, ""
        w = doc.sentences[0].words[0]
        return (w.lemma or word), w.upos

    return tag


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


class _Record:
    __slots__ = ("level", "row", "text", "base", "coarse", "is_self")

    def __init__(self, level: str, row: dict, text: str, base: str, coarse: str):
        self.level = level
        self.row = row
        self.text = text
        self.base = base
        self.coarse = coarse
        self.is_self = base.lower() == text.lower()


def _find_inflected_removals(records: list[_Record]) -> list[_Record]:
    """Pure clustering logic: decide which records are inflected duplicates.

    Groups records by (coarse POS, base lemma text). Within a group that
    has *exactly one* self-citation row (a row whose own text already is
    its own base lemma), every other row in the group is an inflected
    duplicate of that citation form and is staged for removal. Groups with
    zero or multiple self-citation rows are left untouched (ambiguous).

    A second pass catches near-miss lemmatizer typos (e.g. Swedish
    "fortsätt" -> stanza lemma "fortsäta", one edit away from the real
    citation form "fortsätta") by fuzzy-matching *true singleton* non-self
    records (no group-mates at all) against the pool of confirmed citation
    forms, using a small edit distance rather than a hardcoded word list.
    """
    groups: dict[tuple[str, str], list[_Record]] = {}
    for r in records:
        if not r.text:
            continue
        groups.setdefault((r.coarse, r.base.lower()), []).append(r)

    to_remove: list[_Record] = []
    # Only true singletons (a lone non-self record with no group-mates at
    # all) go through the fuzzy fallback below. Records in an *ambiguous*
    # multi-self group already found an exact match but it was ambiguous —
    # leave them untouched entirely rather than re-attempt a fuzzy match
    # that could inappropriately fold them into an unrelated citation form.
    singletons: list[_Record] = []
    for _key, group in groups.items():
        if len(group) == 1:
            if not group[0].is_self:
                singletons.append(group[0])
            continue
        self_rows = [r for r in group if r.is_self]
        if len(self_rows) == 1:
            to_remove.extend(r for r in group if r is not self_rows[0])
        # else: 0 or >=2 self rows in this group — ambiguous, skip entirely.

    if singletons:
        citation_pool: dict[str, list[str]] = {}
        for r in records:
            if r.is_self and r.text:
                citation_pool.setdefault(r.coarse, []).append(r.text.lower())
        for r in singletons:
            candidates = citation_pool.get(r.coarse, [])
            base_lower = r.base.lower()
            best = min(
                (c for c in candidates if abs(len(c) - len(base_lower)) <= 2),
                key=lambda c: _edit_distance(base_lower, c),
                default=None,
            )
            if best is not None and _edit_distance(base_lower, best) <= 1:
                to_remove.append(r)

    return to_remove


def remove_inflected_duplicates(
    lang: str, tag_fn: Callable[[str], tuple[str, str]] | None = None
) -> int:
    """Remove inflected-form rows whose citation form exists elsewhere.

    Uses the language's own morphological lemmatizer (Stanza by default,
    injectable via ``tag_fn`` for testing) rather than a hardcoded
    word/suffix list — a row is only removed when the lemmatizer itself
    judges it to be a non-citation form *and* the corpus already contains
    an unambiguous citation-form row of the same coarse POS. Operates
    across all levels combined (an inflected form in one level and its
    infinitive in another, e.g. Swedish "beskrev" (C1) vs "beskriva" (A1),
    is still a duplicate).
    """
    if lang not in STANZA_INFLECTION_LANGS:
        return 0

    cfg = LANGS[lang]
    lemma_col = cfg["lemma_col"]
    if tag_fn is None:
        tag_fn = _stanza_tag_fn(lang)

    per_level_rows: dict[str, list[dict]] = {}
    records: list[_Record] = []
    for level in LEVELS:
        if not _file_path(lang, level).exists():
            continue
        rows = _read_level(lang, level)
        per_level_rows[level] = rows
        for row in rows:
            text = (row.get(lemma_col) or "").strip()
            if not text:
                continue
            base, upos = tag_fn(text)
            records.append(_Record(level, row, text, base or text, _coarse_pos(upos)))

    removals = _find_inflected_removals(records)
    removed_row_ids = {id(r.row) for r in removals}
    if not removed_row_ids:
        return 0

    total_removed = 0
    for level, rows in per_level_rows.items():
        kept = [row for row in rows if id(row) not in removed_row_ids]
        removed_here = len(rows) - len(kept)
        if removed_here > 0:
            _write_level(lang, level, kept)
            print(
                f"{lang}/{level}: removed {removed_here} inflected-duplicate rows "
                f"({len(kept)} rows remain)"
            )
            total_removed += removed_here

    return total_removed


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Remove exact duplicates and (opt-in) inflected-form duplicates."
    )
    parser.add_argument(
        "--stanza-langs",
        nargs="*",
        default=None,
        metavar="LANG",
        help=(
            "Also run Stanza-based inflected-duplicate removal for these "
            f"languages (subset of {STANZA_INFLECTION_LANGS}). Downloads "
            "Stanza models on first use."
        ),
    )
    args = parser.parse_args(argv[1:])

    print("Removing exact intra-level duplicates from all language CSVs...")
    total = 0
    for lang in LANGS:
        total += cleanup_language(lang)
    print(f"Total exact/plural duplicates removed: {total}")

    if args.stanza_langs is not None:
        stanza_total = 0
        for lang in args.stanza_langs:
            print(f"\nRunning Stanza-based inflected-duplicate removal: {lang}...")
            stanza_total += remove_inflected_duplicates(lang)
        print(f"Total inflected duplicates removed: {stanza_total}")
        total += stanza_total

    print("\nRunning quality check...")
    import check_quality

    return check_quality.main(["check_quality.py"])


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
