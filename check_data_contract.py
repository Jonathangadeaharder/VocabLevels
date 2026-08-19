"""Acceptance checks for the vocab data contract of 2026-08-19.

The contract (Vidiom docs/specs/2026-08-19-vocab-data-contract.md) makes the
normalized English gloss the join key between languages and defines five
acceptance criteria for every delivery. This script runs them mechanically.

Run from repo root:
    python check_data_contract.py                # measure the CSV baseline
    python check_data_contract.py delivery/      # gate a TSV delivery

The delivery directory holds one TSV per language, UTF-8, with the header:
    lemma	pos	english_gloss	english_pos	cefr	rank	concept_key
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from vocab_schema import LEVELS

ROOT = Path(__file__).parent

LANG_DIRS = {
    "english": "en",
    "german": "de",
    "spanish": "es",
    "french": "fr",
    "swedish": "sv",
    "arabic": "ar",
    "dutch": "nl",
    "chinese": "zh",
}
CODE_TO_DIR = {code: name for name, code in LANG_DIRS.items()}

GLOSS_ASCII = re.compile(r"^[A-Za-z '-]+$")
MIN_COVERAGE = 0.60

TSV_HEADER = [
    "lemma",
    "pos",
    "english_gloss",
    "english_pos",
    "cefr",
    "rank",
    "concept_key",
]


@dataclass(frozen=True)
class Row:
    lang: str
    lemma: str
    pos: str
    english_gloss: str
    rank: int | None


def normalize_gloss(gloss: str) -> str:
    """Match the app's key: lower, drop a leading 'to ', strip spaces only."""
    return re.sub("^to ", "", gloss.lower()).strip(" ")


def _unquote(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1].replace('""', '"')
    return value


def load_csv_rows(root: Path) -> list[Row]:
    """Read the authored CEFR CSVs (levels A1-C1) of every language."""
    rows: list[Row] = []
    for name, code in LANG_DIRS.items():
        for level in LEVELS:
            path = root / name / f"{level}.csv"
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for cols in reader:
                    lemma = _unquote(cols[0]) if len(cols) > 0 else ""
                    if not lemma:
                        continue
                    gloss = _unquote(cols[1]) if len(cols) > 1 else ""
                    pos = cols[3].strip() if len(cols) > 3 else ""
                    rows.append(Row(code, lemma, pos, gloss, None))
    return rows


def load_delivery_rows(delivery: Path) -> list[Row]:
    """Read contract TSVs; the filename is the language (dir name or code)."""
    rows: list[Row] = []
    for path in sorted(delivery.glob("*.tsv")):
        code = LANG_DIRS.get(path.stem, path.stem)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
            if header != TSV_HEADER:
                raise ValueError(f"{path.name}: unexpected header {header}")
            for cols in reader:
                if not cols or not cols[0].strip():
                    continue
                if len(cols) < 4:
                    raise ValueError(
                        f"{path.name}: row has {len(cols)} columns: {cols}"
                    )
                rank_text = cols[5].strip() if len(cols) > 5 else ""
                rows.append(
                    Row(
                        lang=code,
                        lemma=cols[0].strip(),
                        pos=cols[3].strip(),
                        english_gloss=cols[2].strip(),
                        rank=int(rank_text) if rank_text else None,
                    )
                )
    return rows


def check_shrunken_glosses(delivery: list[Row], source: list[Row]) -> list[str]:
    """Criterion 1: a multi-word source gloss must not shrink to one word."""
    source_gloss = {(r.lang, r.lemma): r.english_gloss for r in source}
    violations = []
    for row in delivery:
        original = source_gloss.get((row.lang, row.lemma))
        if original is None:
            continue
        if len(original.split()) > 1 and len(row.english_gloss.split()) == 1:
            violations.append(
                f"{row.lang}:{row.lemma}: '{original}' -> '{row.english_gloss}'"
            )
    return violations


def check_duplicates(rows: list[Row]) -> list[str]:
    """Criterion 2: (lang, lemma, gloss_norm) is unique."""
    counts = Counter((r.lang, r.lemma, normalize_gloss(r.english_gloss)) for r in rows)
    return [f"{k[0]}:{k[1]}:{k[2]}" for k, n in counts.items() if n > 1]


def check_rank_gaps(rows: list[Row]) -> list[str]:
    """Criterion 3: exactly one rank-1 row per (lang, gloss_norm, pos)."""
    groups: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        if row.rank == 1:
            groups[(row.lang, normalize_gloss(row.english_gloss), row.pos)] += 1
    seen = {(r.lang, normalize_gloss(r.english_gloss), r.pos) for r in rows}
    violations = []
    for key in sorted(seen):
        if groups.get(key, 0) != 1:
            violations.append(f"{key[0]}:{key[1]}:{key[2]}")
    return violations


def pair_coverage(rows: list[Row], source: str, target: str) -> tuple[int, int, int]:
    """Criterion 4 per pair: (eindeutig, mehrdeutig, ohne)."""
    index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row.lang == target:
            index[(normalize_gloss(row.english_gloss), row.pos)].add(row.lemma)
    eindeutig = mehrdeutig = ohne = 0
    for row in rows:
        if row.lang != source:
            continue
        gloss = normalize_gloss(row.english_gloss)
        if not gloss:
            ohne += 1
            continue
        found = len(index.get((gloss, row.pos), set()))
        if found == 1:
            eindeutig += 1
        elif found > 1:
            mehrdeutig += 1
        else:
            ohne += 1
    return eindeutig, mehrdeutig, ohne


def check_ascii(rows: list[Row]) -> list[str]:
    """Criterion 5: the pivot gloss is English ASCII (or blank)."""
    violations = []
    for row in rows:
        gloss = row.english_gloss
        if not gloss or not GLOSS_ASCII.match(gloss):
            violations.append(f"{row.lang}:{row.lemma}: '{gloss}'")
    return violations


def run_baseline(root: Path) -> int:
    rows = load_csv_rows(root)
    print(f"CSV baseline: {len(rows)} rows")
    print("criterion 1: n/a (needs a delivery)")
    dups = check_duplicates(rows)
    print(f"criterion 2: {len(dups)} duplicate keys")
    print("criterion 3: n/a (CSVs carry no rank)")
    print("criterion 4: eindeutige Abdeckung je Sprachpaar")
    codes = sorted(set(LANG_DIRS.values()))
    for source in codes:
        for target in codes:
            if source == target:
                continue
            ein, mehr, ohne = pair_coverage(rows, source, target)
            total = ein + mehr + ohne
            ratio = ein / total if total else 0.0
            print(
                f"  {source}->{target}: {ein}/{total} = {ratio:.0%} "
                f"(mehrdeutig {mehr}, ohne {ohne})"
            )
    ascii_bad = check_ascii(rows)
    print(f"criterion 5: {len(ascii_bad)} non-ASCII or blank glosses")
    return 0


def run_delivery(delivery: Path, root: Path) -> int:
    rows = load_delivery_rows(delivery)
    if not rows:
        print(f"no TSV rows found in {delivery}")
        return 1
    source = load_csv_rows(root)
    failed = False

    shrunk = check_shrunken_glosses(rows, source)
    print(f"criterion 1: {len(shrunk)} shrunken multi-word glosses")
    for item in shrunk[:20]:
        print(f"  {item}")
    failed |= bool(shrunk)

    dups = check_duplicates(rows)
    print(f"criterion 2: {len(dups)} duplicate keys")
    for item in dups[:20]:
        print(f"  {item}")
    failed |= bool(dups)

    gaps = check_rank_gaps(rows)
    print(f"criterion 3: {len(gaps)} group(s) without exactly one rank 1")
    for item in gaps[:20]:
        print(f"  {item}")
    failed |= bool(gaps)

    print("criterion 4: eindeutige Abdeckung je Sprachpaar (>= 60 %)")
    codes = sorted({r.lang for r in rows})
    for source_lang in codes:
        for target in codes:
            if source_lang == target:
                continue
            ein, mehr, ohne = pair_coverage(rows, source_lang, target)
            total = ein + mehr + ohne
            ratio = ein / total if total else 0.0
            status = "OK " if ratio >= MIN_COVERAGE else "LOW"
            print(
                f"  [{status}] {source_lang}->{target}: {ein}/{total} "
                f"= {ratio:.0%} (mehrdeutig {mehr}, ohne {ohne})"
            )
            failed |= ratio < MIN_COVERAGE

    ascii_bad = check_ascii(rows)
    print(f"criterion 5: {len(ascii_bad)} non-ASCII or blank glosses")
    for item in ascii_bad[:20]:
        print(f"  {item}")
    failed |= bool(ascii_bad)

    print("RESULT: FAIL" if failed else "RESULT: PASS")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "delivery",
        nargs="?",
        help="directory with one contract TSV per language",
    )
    args = parser.parse_args(argv)
    if args.delivery:
        return run_delivery(Path(args.delivery), ROOT)
    return run_baseline(ROOT)


if __name__ == "__main__":
    sys.exit(main())
