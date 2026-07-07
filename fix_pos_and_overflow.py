"""Fix POS mislabels via Stanza, trim overflow, redistribute.

Three operations on the harmonized CEFR CSVs:
1. POS audit — use Stanza NLP to correct misclassified POS in german and
   arabic CSVs. Stanza provides trained UD POS taggers (de, ar, fr, es, en,
   sv, zh). Dialectal/colloquial words (e.g. Moroccan darija) return X
   (unknown); these are left untouched unless a minimal gloss-based fallback
   applies (e.g. gloss "to X" → VERB).
2. Trim redundant overflow — drop rows past TARGET where the lemma (any POS)
   already exists in the target zone of the same file.
3. Redistribute unique overflow — relocate C1-appropriate arabic B2 overflow
   to arabic/C1 (underfilled), then trim remaining unique overflow.

Usage:
    uv run python fix_pos_and_overflow.py            # dry-run, report only
    uv run python fix_pos_and_overflow.py --apply     # write changes

Models are auto-downloaded on first run (~1-2 GB total for all 7 languages).
Subsequent runs use the cache at ~/stanza_resources.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent

LEVELS = ("A1", "A2", "B1", "B2", "C1")
TARGETS = {"A1": 600, "A2": 600, "B1": 1000, "B2": 2000, "C1": 4000}

LANG_LEMMA_COL = {
    "arabic": "Arabic_Lemma",
    "chinese": "Chinese_Lemma",
    "english": "English_Lemma",
    "french": "French_Lemma",
    "german": "German_Lemma",
    "spanish": "Spanish_Lemma",
    "swedish": "Swedish_Lemma",
}

# Stanza language codes per repo language.
STANZA_LANG = {
    "arabic": "ar",
    "chinese": "zh",
    "english": "en",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "swedish": "sv",
}

# Minimal gloss-based fallback for words Stanza tags as X.
# Only applied when Stanza returns X (uncertain). Conservative.
TO_GLOSS_PREFIX = "to "


@dataclass
class PosChange:
    file: str
    line: int
    lemma: str
    old_pos: str
    new_pos: str
    reason: str


@dataclass
class TrimChange:
    file: str
    line: int
    lemma: str
    pos: str
    reason: str


@dataclass
class RelocateChange:
    src_file: str
    src_line: int
    lemma: str
    dst_file: str
    reason: str


@dataclass
class ChangeSet:
    pos_changes: list[PosChange] = field(default_factory=list)
    trims: list[TrimChange] = field(default_factory=list)
    relocations: list[RelocateChange] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


# --- CSV I/O --------------------------------------------------------------


def load_csv(lang: str, level: str) -> list[dict[str, str]]:
    path = ROOT / lang / f"{level}.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(lang: str, level: str, rows: list[dict[str, str]]) -> None:
    path = ROOT / lang / f"{level}.csv"
    lemma_col = LANG_LEMMA_COL[lang]
    fieldnames = [lemma_col, "English_Lemma", "Chinese_Lemma", "POS"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\r\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    lemma_col: r.get(lemma_col, ""),
                    "English_Lemma": r.get("English_Lemma", ""),
                    "Chinese_Lemma": r.get("Chinese_Lemma", ""),
                    "POS": r.get("POS", ""),
                }
            )


def norm_pos(row: dict[str, str]) -> str:
    return (row.get("POS") or "").strip()


# --- Stanza POS tagging ---------------------------------------------------


# Cache loaded pipelines by language to avoid re-initialization.
# Cache loaded pipelines by language to avoid re-initialization.
_pipelines: dict[str, Any] = {}


def get_pipeline(lang: str) -> Any:
    """Load (and cache) a Stanza pipeline for the given repo language."""
    stanza_lang = STANZA_LANG[lang]
    if lang not in _pipelines:
        import stanza

        _pipelines[lang] = stanza.Pipeline(
            lang=stanza_lang,
            processors="tokenize,pos",
            verbose=False,
            use_gpu=False,
        )
    return _pipelines[lang]


def tag_lemma(lang: str, lemma: str) -> str:
    """Return Stanza UPOS tag for a lemma, or empty string if uncertain.

    Returns the first word's UPOS. If Stanza returns X (unknown), returns
    empty string so the caller knows to try the gloss fallback.
    """
    if not lemma.strip():
        return ""
    nlp = get_pipeline(lang)
    doc = nlp(lemma)
    if not doc.sentences or not doc.sentences[0].words:
        return ""
    upos = doc.sentences[0].words[0].upos
    if upos == "X":
        return ""
    return upos


def gloss_fallback(gloss: str) -> str:
    """Minimal gloss-based POS fallback for words Stanza tagged as X.

    Only applies the most obvious patterns:
    - gloss starts with "to " → VERB
    - gloss ends in -ed/-ing/-ous/-ish (adjective participle pattern) → ADJ
    Otherwise returns empty (uncertain, leave unchanged).
    """
    gl = gloss.strip().lower()
    if not gl:
        return ""
    if gl.startswith(TO_GLOSS_PREFIX):
        return "VERB"
    if gl.endswith(("-ed", "-ing", "-ous", "-ish")):
        return "ADJ"
    return ""


# --- POS audit ------------------------------------------------------------


def audit_pos(lang: str, cs: ChangeSet) -> None:
    """Audit POS for a language using Stanza, stage corrections."""
    lemma_col = LANG_LEMMA_COL[lang]
    for level in LEVELS:
        rows = load_csv(lang, level)
        for idx, row in enumerate(rows, start=2):
            lemma = (row.get(lemma_col) or "").strip()
            gloss = (row.get("English_Lemma") or "").strip()
            old_pos = norm_pos(row)
            if not lemma or not old_pos:
                continue
            new_pos = tag_lemma(lang, lemma)
            reason = "stanza UPOS"
            if not new_pos:
                # Stanza uncertain (X). Try minimal gloss fallback.
                fb = gloss_fallback(gloss)
                if not fb or fb == old_pos:
                    continue
                new_pos = fb
                reason = "gloss fallback (stanza=X)"
            if new_pos != old_pos:
                cs.pos_changes.append(
                    PosChange(
                        file=f"{lang}/{level}.csv",
                        line=idx,
                        lemma=lemma,
                        old_pos=old_pos,
                        new_pos=new_pos,
                        reason=reason,
                    )
                )


# --- Overflow analysis ----------------------------------------------------


def analyze_redundant_overflow(lang: str, cs: ChangeSet) -> None:
    """Stage trims for redundant overflow (lemma already in target zone)."""
    lemma_col = LANG_LEMMA_COL[lang]
    for level in LEVELS:
        rows = load_csv(lang, level)
        target = TARGETS[level]
        if len(rows) <= target:
            continue
        overflow = rows[target:]
        target_lemmas = {r.get(lemma_col, "").strip().lower() for r in rows[:target]}
        for i, row in enumerate(overflow, start=target + 2):
            lemma = (row.get(lemma_col) or "").strip()
            pos = norm_pos(row)
            if lemma.lower() in target_lemmas:
                cs.trims.append(
                    TrimChange(
                        file=f"{lang}/{level}.csv",
                        line=i,
                        lemma=lemma,
                        pos=pos,
                        reason="redundant (lemma in target zone)",
                    )
                )


def dedup_after_pos_fix(langs: list[str], cs: ChangeSet) -> None:
    """Stage trims for lemma+POS duplicates revealed by POS fixes.

    POS corrections can make two rows (previously distinct by lemma+POS)
    collide. Detect this by simulating the POS changes on each file, then
    trimming the second occurrence of any (lemma, pos) pair. Skips rows
    already staged for trimming by analyze_redundant_overflow.

    Only trims when the file is over target (the dupes sit in the overflow
    zone). Files at or under target keep their dupes — trimming would drop
    below target, and those are pre-existing data-quality issues to fix by
    adding replacement words, not by shrinking the corpus.
    """
    pos_fixes: dict[tuple[str, int], str] = {
        (c.file, c.line): c.new_pos for c in cs.pos_changes
    }
    already_trimmed = {(t.file, t.line) for t in cs.trims}
    for lang in langs:
        lemma_col = LANG_LEMMA_COL[lang]
        for level in LEVELS:
            rows = load_csv(lang, level)
            fname = f"{lang}/{level}.csv"
            target = TARGETS[level]
            if len(rows) <= target:
                continue  # at/under target: keep dupes
            seen: set[tuple[str, str]] = set()
            for i, row in enumerate(rows, start=2):
                if (fname, i) in already_trimmed:
                    continue
                lemma = (row.get(lemma_col) or "").strip().lower()
                pos = pos_fixes.get((fname, i), norm_pos(row))
                key = (lemma, pos)
                if key in seen:
                    cs.trims.append(
                        TrimChange(
                            file=fname,
                            line=i,
                            lemma=(row.get(lemma_col) or "").strip(),
                            pos=pos,
                            reason="lemma+POS duplicate after POS fix",
                        )
                    )
                else:
                    seen.add(key)


# --- Relocation: arabic B2 → C1 -------------------------------------------

# 14 C1-appropriate arabic B2 overflow words to relocate.
ARABIC_B2_RELOCATE_TO_C1 = {
    "اعتكاف",  # spiritual retreat
    "عشور",  # tithe
    "كرامات",  # saintly miracles
    "مشعوذ",  # charlatan
    "شعوذة",  # sorcery
    "تبرك",  # seeking blessing
    "تقاسيم",  # instrumental improvisation
    "حنبل",  # woven rug
    "جبصية",  # carved plasterwork
    "تذهيب",  # gilding
    "شوار",  # bridal trousseau
    "كياس",  # bath scrubber
    "طيابة",  # bath attendant
    "شيخة",  # folk singer/sheikha
}


def redistribute_arabic_b2(cs: ChangeSet) -> None:
    """Relocate 14 C1-appropriate words from arabic/B2 to arabic/C1."""
    c1_rows = load_csv("arabic", "C1")
    room = TARGETS["C1"] - len(c1_rows)
    if room < len(ARABIC_B2_RELOCATE_TO_C1):
        cs.blockers.append(
            f"arabic/C1 only has room for {room}, "
            f"but {len(ARABIC_B2_RELOCATE_TO_C1)} to relocate"
        )
        return
    b2_rows = load_csv("arabic", "B2")
    lemma_col = LANG_LEMMA_COL["arabic"]
    for i, row in enumerate(b2_rows, start=2):
        lemma = (row.get(lemma_col) or "").strip()
        if lemma in ARABIC_B2_RELOCATE_TO_C1:
            cs.relocations.append(
                RelocateChange(
                    src_file="arabic/B2.csv",
                    src_line=i,
                    lemma=lemma,
                    dst_file="arabic/C1.csv",
                    reason="C1-appropriate cultural term",
                )
            )


def trim_unique_overflow(cs: ChangeSet) -> None:
    """Stage trims for unique overflow after relocation.

    - arabic/B1: +5 food nouns (regional, not fundamental)
    - arabic/B2: remaining overflow after 14 relocated → trim to 2000
    - chinese/C1: +9 near-synonym verbs (not fundamental)
    """
    lemma_col = LANG_LEMMA_COL["arabic"]
    b1_rows = load_csv("arabic", "B1")
    b1_target = TARGETS["B1"]
    for i, row in enumerate(b1_rows[b1_target:], start=b1_target + 2):
        lemma = (row.get(lemma_col) or "").strip()
        pos = norm_pos(row)
        cs.trims.append(
            TrimChange(
                file="arabic/B1.csv",
                line=i,
                lemma=lemma,
                pos=pos,
                reason="overflow (regional food noun)",
            )
        )

    b2_rows = load_csv("arabic", "B2")
    b2_target = TARGETS["B2"]
    for i, row in enumerate(b2_rows[b2_target:], start=b2_target + 2):
        lemma = (row.get(lemma_col) or "").strip()
        pos = norm_pos(row)
        if lemma in ARABIC_B2_RELOCATE_TO_C1:
            continue
        cs.trims.append(
            TrimChange(
                file="arabic/B2.csv",
                line=i,
                lemma=lemma,
                pos=pos,
                reason="overflow (non-fundamental descriptor)",
            )
        )

    lemma_col_c = LANG_LEMMA_COL["chinese"]
    c1_rows = load_csv("chinese", "C1")
    c1_target = TARGETS["C1"]
    for i, row in enumerate(c1_rows[c1_target:], start=c1_target + 2):
        lemma = (row.get(lemma_col_c) or "").strip()
        pos = norm_pos(row)
        cs.trims.append(
            TrimChange(
                file="chinese/C1.csv",
                line=i,
                lemma=lemma,
                pos=pos,
                reason="overflow (near-synonym verb)",
            )
        )


# --- Apply changes --------------------------------------------------------


def apply_changes(cs: ChangeSet) -> None:
    """Apply POS changes, relocations, then trims to all files."""
    pos_by_file: dict[str, list[PosChange]] = {}
    trims_by_file: dict[str, list[TrimChange]] = {}
    reloc_by_src: dict[str, list[RelocateChange]] = {}
    for c in cs.pos_changes:
        pos_by_file.setdefault(c.file, []).append(c)
    for c in cs.trims:
        trims_by_file.setdefault(c.file, []).append(c)
    for c in cs.relocations:
        reloc_by_src.setdefault(c.src_file, []).append(c)

    all_files: set[str] = set()
    all_files.update(pos_by_file)
    all_files.update(trims_by_file)
    all_files.update(reloc_by_src)
    for r in cs.relocations:
        all_files.add(r.dst_file)

    cache: dict[str, list[dict[str, str]]] = {}
    for f in all_files:
        lang, lvl_csv = f.split("/")
        cache[f] = load_csv(lang, lvl_csv.replace(".csv", ""))

    # 1. Apply POS changes (by line number).
    for f, changes in pos_by_file.items():
        rows = cache[f]
        for c in changes:
            idx = c.line - 2
            if 0 <= idx < len(rows):
                rows[idx]["POS"] = c.new_pos

    # 2. Apply relocations: move rows from src to dst.
    for src, relocs in reloc_by_src.items():
        lang = src.split("/")[0]
        lemma_col = LANG_LEMMA_COL[lang]
        reloc_lemmas = {r.lemma for r in relocs}
        moved = [
            r for r in cache[src] if (r.get(lemma_col) or "").strip() in reloc_lemmas
        ]
        cache[src] = [
            r
            for r in cache[src]
            if (r.get(lemma_col) or "").strip() not in reloc_lemmas
        ]
        dst = relocs[0].dst_file
        cache[dst].extend(moved)

    # 3. Apply trims: remove overflow-zone rows whose lemma is in trim set.
    for f, trims in trims_by_file.items():
        lang = f.split("/")[0]
        lemma_col = LANG_LEMMA_COL[lang]
        rows = cache[f]
        trim_lemmas = {t.lemma for t in trims}
        target = TARGETS[f.split("/")[1].replace(".csv", "")]
        cache[f] = [
            row
            for i, row in enumerate(rows)
            if not (i >= target and (row.get(lemma_col) or "").strip() in trim_lemmas)
        ]

    for f, rows in cache.items():
        lang, lvl_csv = f.split("/")
        save_csv(lang, lvl_csv.replace(".csv", ""), rows)


# --- Reporting -----------------------------------------------------------


def report(cs: ChangeSet) -> int:
    """Print dry-run report. Returns exit code (0 ok, 1 blockers)."""
    print("=" * 72)
    print(f"POS CHANGES ({len(cs.pos_changes)} proposed)")
    print("=" * 72)
    by_lang: dict[str, list[PosChange]] = {}
    for c in cs.pos_changes:
        by_lang.setdefault(c.file.split("/")[0], []).append(c)
    for lang in sorted(by_lang):
        changes = by_lang[lang]
        print(f"\n--- {lang} ({len(changes)} POS changes) ---")
        for c in changes[:60]:
            print(
                f"  {c.file}:L{c.line}  {c.lemma!r:20} "
                f"{c.old_pos:6} → {c.new_pos:6}  ({c.reason})"
            )
        if len(changes) > 60:
            print(f"  ... +{len(changes) - 60} more")

    print("\n" + "=" * 72)
    print(f"RELOCATIONS ({len(cs.relocations)} proposed)")
    print("=" * 72)
    for r in cs.relocations:
        print(f"  {r.src_file}:L{r.src_line}  {r.lemma!r:18} → {r.dst_file}")

    print("\n" + "=" * 72)
    print(f"TRIMS ({len(cs.trims)} proposed)")
    print("=" * 72)
    trims_by_file: dict[str, list[TrimChange]] = {}
    for t in cs.trims:
        trims_by_file.setdefault(t.file, []).append(t)
    for f in sorted(trims_by_file):
        trims = trims_by_file[f]
        print(f"\n--- {f} ({len(trims)} trims) ---")
        for t in trims[:25]:
            print(f"  L{t.line}  {t.lemma!r:20} ({t.pos:6})  {t.reason}")
        if len(trims) > 25:
            print(f"  ... +{len(trims) - 25} more")

    if cs.blockers:
        print("\n" + "=" * 72)
        print(f"BLOCKERS ({len(cs.blockers)})")
        print("=" * 72)
        for b in cs.blockers:
            print(f"  {b}")
        return 1

    print("\n" + "=" * 72)
    print("PROJECTED ROW COUNTS (after apply)")
    print("=" * 72)
    affected = {c.file.split("/")[0] for c in cs.pos_changes}
    affected.update({t.file.split("/")[0] for t in cs.trims})
    affected.update({r.src_file.split("/")[0] for r in cs.relocations})
    affected.update({r.dst_file.split("/")[0] for r in cs.relocations})
    affected.update({"arabic", "chinese", "french", "german", "spanish"})
    for lang in sorted(affected):
        print(f"\n--- {lang} ---")
        for level in LEVELS:
            rows = load_csv(lang, level)
            n = len(rows)
            target = TARGETS[level]
            trims_n = sum(1 for t in cs.trims if t.file == f"{lang}/{level}.csv")
            reloc_out = sum(
                1 for r in cs.relocations if r.src_file == f"{lang}/{level}.csv"
            )
            reloc_in = sum(
                1 for r in cs.relocations if r.dst_file == f"{lang}/{level}.csv"
            )
            projected = n - trims_n - reloc_out + reloc_in
            delta = projected - target
            status = "OK" if delta == 0 else f"OFF ({delta:+d})"
            print(f"  {level}: {n} → {projected}  (target {target})  {status}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Fix POS via Stanza, trim overflow, redistribute."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes (default: dry-run only)",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["german", "arabic"],
        help="Languages to POS-audit (default: german arabic)",
    )
    args = parser.parse_args(argv[1:])

    cs = ChangeSet()

    for lang in args.langs:
        audit_pos(lang, cs)

    for lang in ("french", "german", "spanish"):
        analyze_redundant_overflow(lang, cs)

    # After POS fixes, detect lemma+POS duplicates revealed by the changes.
    dedup_after_pos_fix(args.langs, cs)

    redistribute_arabic_b2(cs)
    trim_unique_overflow(cs)

    rc = report(cs)
    if rc != 0:
        return rc
    if args.apply:
        apply_changes(cs)
        print("\nApplied. Re-run check_quality.py to verify.")
    else:
        print("\nDry-run. Re-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
