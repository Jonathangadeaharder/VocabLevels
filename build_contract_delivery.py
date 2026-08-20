"""Harmonization pipeline: CEFR CSVs -> contract TSV deliveries.

Reads the authored per-language CSVs, aligns rows into concepts across
languages using the shared English and Chinese gloss columns, derives a
canonical English gloss per concept, dedups on (lang, lemma, gloss_norm),
assigns one rank-1 per (lang, gloss, pos), and emits a stable concept_key
across languages. Output follows the 2026-08-19 data contract:

    lemma  pos  english_gloss  english_pos  cefr  rank  concept_key
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from check_data_contract import LANG_DIRS, LEVELS, normalize_gloss

ROOT = Path(__file__).parent

CSV_HEADER = "Language_Lemma,English_Lemma,Chinese_Lemma,POS"
TSV_HEADER = [
    "lemma",
    "pos",
    "english_gloss",
    "english_pos",
    "cefr",
    "rank",
    "concept_key",
]

# English gloss must be ASCII letters/spaces/hyphen/apostrophe (contract rule 5).
GLOSS_ASCII = re.compile(r"^[A-Za-z '.-]+$")


@dataclass(frozen=True)
class DeliveryRow:
    lemma: str
    pos: str
    gloss: str
    english_pos: str
    cefr: str
    rank: int
    concept_key: str


def _unquote(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1].replace('""', '"')
    return value


def _clean_lemma(value: str) -> str:
    """Strip stray quote characters and control whitespace from dirty sources."""
    return value.replace('"', "").replace("\t", " ").replace("\r", " ").strip()


def normalize_cn(gloss: str) -> str:
    """Normalize a Chinese gloss for concept bridging: strip spaces only.

    Slashes separate alternative senses and stay put so only identical
    authored strings bridge across languages.
    """
    return gloss.replace(" ", "").strip()


def clean_gloss(gloss: str) -> str:
    """Clean a source English gloss to a contract-safe form.

    Keeps every word (criterion 1 forbids shrinking a multi-word gloss):
    parentheticals and slashes become spaces, accents fold to ASCII
    (criterion 5). 'orange (color)' -> 'orange color', 'fiancé' -> 'fiance'.
    """
    if not gloss:
        return ""
    folded = unicodedata.normalize("NFKD", gloss)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    no_paren = re.sub(r"[()]", " ", folded)
    no_slash = re.sub(r"[/]", " ", no_paren)
    return re.sub(r"\s+", " ", no_slash).strip()


def load_csv_records(root: Path) -> list[tuple[str, str, str, str, str, str]]:
    """Return (lang, level, lemma, english_gloss, chinese_gloss, pos) rows."""
    records: list[tuple[str, str, str, str, str, str]] = []
    for name, code in LANG_DIRS.items():
        for level in LEVELS:
            path = root / name / f"{level}.csv"
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for cols in reader:
                    lemma = _clean_lemma(_unquote(cols[0])) if len(cols) > 0 else ""
                    if not lemma:
                        continue
                    english = clean_gloss(_unquote(cols[1])) if len(cols) > 1 else ""
                    chinese = _unquote(cols[2]) if len(cols) > 2 else ""
                    pos = cols[3].strip() if len(cols) > 3 else ""
                    records.append((code, level, lemma, english, chinese, pos))
    return records


def load_expansion_records(
    root: Path,
) -> list[tuple[str, str, str, str, str, str]]:
    """Return (lang, level, lemma, english_gloss, chinese_gloss, pos) rows
    from the generated per-language expansion CSVs (if present)."""
    records: list[tuple[str, str, str, str, str, str]] = []
    for name, code in LANG_DIRS.items():
        path = root / name / "expansion.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for cols in reader:
                lemma = _clean_lemma(_unquote(cols[0])) if len(cols) > 0 else ""
                if not lemma:
                    continue
                english = clean_gloss(_unquote(cols[1])) if len(cols) > 1 else ""
                chinese = _unquote(cols[2]) if len(cols) > 2 else ""
                pos = cols[3].strip() if len(cols) > 3 else ""
                level = (cols[4].strip() if len(cols) > 4 else "") or "B1"
                records.append((code, level, lemma, english, chinese, pos))
    return records


def _make_english_key(
    record: tuple[str, str, str, str, str, str],
) -> tuple[str, str] | None:
    gloss = normalize_gloss(record[3])
    return (gloss, record[5]) if gloss else None


def _make_chinese_key(
    record: tuple[str, str, str, str, str, str],
) -> tuple[str, str] | None:
    gloss = normalize_cn(record[4])
    return (gloss, record[5]) if gloss else None


def align_concepts(
    records: list[tuple[str, str, str, str, str, str]],
) -> dict[int, list[tuple[str, str, str, str, str, str]]]:
    """Union rows into concepts across languages.

    Primary identity is the English (gloss_norm, pos): rows that share it are
    the same concept in every language (contract rule 1). A row with a blank
    English gloss joins a concept through its Chinese (gloss, pos) when another
    language row with that Chinese gloss carries a real English gloss. Chinese
    never merges two rows that both have distinct English glosses.
    """
    parent = list(range(len(records)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    english_index: dict[tuple[str, str], int] = {}
    chinese_anchor: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        english_key = _make_english_key(record)
        if english_key is not None:
            if english_key in english_index:
                union(index, english_index[english_key])
            else:
                english_index[english_key] = index
        chinese_key = _make_chinese_key(record)
        if chinese_key is not None and english_key is not None:
            chinese_anchor.setdefault(chinese_key, index)
    for index, record in enumerate(records):
        english_key = _make_english_key(record)
        if english_key is not None:
            continue
        chinese_key = _make_chinese_key(record)
        if chinese_key is None:
            continue
        anchor = chinese_anchor.get(chinese_key)
        if anchor is not None:
            union(index, anchor)

    concepts: dict[int, list[tuple[str, str, str, str, str, str]]] = defaultdict(list)
    for index, record in enumerate(records):
        concepts[find(index)].append(record)
    return dict(concepts)


def canonical_gloss(glosses: list[str]) -> str:
    """Pick the canonical English gloss for a concept.

    Prefers non-blank ASCII values, then multi-word forms, then majority.
    """
    candidates = [g for g in glosses if g and GLOSS_ASCII.match(g)]
    if not candidates:
        return ""
    multiword = [g for g in candidates if " " in g]
    pool = multiword if multiword else candidates
    return Counter(pool).most_common(1)[0][0]


def canonical_pos(poss: list[str]) -> str:
    return Counter(p for p in poss if p).most_common(1)[0][0] if any(poss) else ""


def build_delivery_rows(
    records: list[tuple[str, str, str, str, str, str]],
) -> dict[str, list[DeliveryRow]]:
    """Harmonize records into per-language delivery rows.

    For each concept: derive the canonical English gloss and pos, then emit
    one row per (lang, lemma, gloss). Rows whose normalized gloss collides
    (e.g. 'answer' noun and 'to answer' verb both normalize to 'answer')
    merge into one row with the lowest CEFR, so (lang, lemma, gloss_norm) is
    unique. A rank orders alternatives per (lang, gloss, pos).
    """
    concepts = align_concepts(records)

    emitted: list[tuple[str, DeliveryRow]] = []
    for members in concepts.values():
        gloss = canonical_gloss([m[3] for m in members])
        pos = canonical_pos([m[5] for m in members])
        concept_key = f"{normalize_gloss(gloss) or 'blank'}-{pos or 'X'}"
        per_lemma: dict[tuple[str, str], list[tuple[str, str, str, str, str, str]]] = (
            defaultdict(list)
        )
        for member in members:
            per_lemma[(member[0], member[2])].append(member)
        for (lang, lemma), member_rows in per_lemma.items():
            level = min((m[1] for m in member_rows), key=lambda x: LEVELS.index(x))
            emitted.append(
                (
                    lang,
                    DeliveryRow(
                        lemma=lemma,
                        pos=pos,
                        gloss=gloss,
                        english_pos=pos,
                        cefr=level,
                        rank=1,
                        concept_key=concept_key,
                    ),
                )
            )

    deduped: dict[tuple[str, str, str], tuple[str, DeliveryRow]] = {}
    for lang, row in emitted:
        key = (lang, row.lemma, normalize_gloss(row.gloss))
        prior = deduped.get(key)
        if prior is None or LEVELS.index(row.cefr) < LEVELS.index(prior[1].cefr):
            deduped[key] = (lang, row)

    ranked: dict[str, list[DeliveryRow]] = defaultdict(list)
    for lang, row in deduped.values():
        ranked[lang].append(row)
    for lang, rows in ranked.items():
        groups: dict[tuple[str, str], list[DeliveryRow]] = defaultdict(list)
        for row in rows:
            groups[(row.gloss, row.english_pos)].append(row)
        ranked[lang] = []
        for group in groups.values():
            for position, item in enumerate(sorted(group, key=lambda r: r.lemma)):
                ranked[lang].append(
                    DeliveryRow(
                        lemma=item.lemma,
                        pos=item.pos,
                        gloss=item.gloss,
                        english_pos=item.english_pos,
                        cefr=item.cefr,
                        rank=position + 1,
                        concept_key=item.concept_key,
                    )
                )
    return dict(ranked)


def write_tsv(out_dir: Path, lang: str, rows: list[DeliveryRow]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{lang}.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(TSV_HEADER) + "\n")
        for row in sorted(rows, key=lambda r: (r.lemma, r.rank)):
            handle.write(
                "\t".join(
                    [
                        row.lemma,
                        row.pos,
                        row.gloss,
                        row.english_pos,
                        row.cefr,
                        str(row.rank),
                        row.concept_key,
                    ]
                )
                + "\n"
            )


def build_all(out_dir: Path, root: Path = ROOT) -> dict[str, int]:
    records = load_csv_records(root) + load_expansion_records(root)
    per_lang = build_delivery_rows(records)
    counts: dict[str, int] = {}
    for lang, rows in per_lang.items():
        write_tsv(out_dir, lang, rows)
        counts[lang] = len(rows)
    return counts


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", help="directory for the per-language TSVs")
    args = parser.parse_args(argv)
    counts = build_all(Path(args.out_dir))
    for lang in sorted(counts):
        print(f"{lang}: {counts[lang]} rows")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
