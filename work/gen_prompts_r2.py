"""Generate round-2 Gemini prompts from actual per-pair gate deficits.

Universe/coverage mirror check_data_contract criterion 4: concepts(lang) =
{(norm gloss, pos)} from harmonized delivery rows (core + expansion). For each
target lang T, collect cells from source languages whose pair S->T is below
80% until each pair's deficit is covered, then chunk at 2000 concepts/file.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_contract_delivery import (  # noqa: E402
    build_delivery_rows,
    load_csv_records,
    load_expansion_records,
)
from scripts.expand_concepts import LANG_DIRS, normalize_gloss  # noqa: E402

GOAL = 0.80
CHUNK = 2000

LANGS = {
    "ar": {
        "lemma": "the canonical ar lemma in Arabic script. Use diacritics only where essential for disambiguation.",
        "extra": "- Lemma must be authentic Arabic script, never the English gloss.\n",
        "examples": "ar,مكتبة,library,图书馆,NOUN,A2\nar,كتب,to write,写,VERB,A1\nar,جميل,beautiful,美丽的,ADJ,A1",
    },
    "de": {
        "lemma": "the canonical de lemma. infinitive verb, capitalized noun, lowercase adjective/adverb. Use diacritics where correct (ä ö ü ß).",
        "extra": "- IMPORTANT: the Lemma must be an authentic native word in your language. NEVER use the English gloss itself as the Lemma (no verbatim English copies). Use the genuine loanword spelling only when that IS the native word.\n",
        "examples": "de,Bibliothek,library,图书馆,NOUN,A1\nde,schreiben,to write,写,VERB,A1\nde,schön,beautiful,美丽的,ADJ,A1",
    },
    "en": {
        "lemma": "the canonical en lemma. lowercase citation form.",
        "extra": "- For en the Lemma and English_Lemma are the same text: repeat the concept verbatim as the Lemma.\n",
        "examples": "en,library,library,图书馆,NOUN,A1\nen,write,write,写,VERB,A1\nen,beautiful,beautiful,美丽的,ADJ,A1",
    },
    "es": {
        "lemma": "the canonical es lemma. infinitive verb, lowercase adjective/adverb, citation noun. Use diacritics where correct.",
        "extra": "- IMPORTANT: the Lemma must be an authentic native word in your language. NEVER use the English gloss itself as the Lemma (no verbatim English copies). Use the genuine loanword spelling only when that IS the native word.\n",
        "examples": "es,biblioteca,library,图书馆,NOUN,A1\nes,escribir,to write,写,VERB,A1\nes,bonito,beautiful,美丽的,ADJ,A1",
    },
    "fr": {
        "lemma": "the canonical fr lemma. infinitive verb, lowercase adjective/adverb, citation noun. Use diacritics where correct.",
        "extra": "- IMPORTANT: the Lemma must be an authentic native word in your language. NEVER use the English gloss itself as the Lemma (no verbatim English copies). Use the genuine loanword spelling only when that IS the native word.\n",
        "examples": "fr,bibliothèque,library,图书馆,NOUN,A1\nfr,écrire,to write,写,VERB,A1\nfr,beau,beautiful,美丽的,ADJ,A1",
    },
    "nl": {
        "lemma": "the canonical nl lemma. infinitive verb, lowercase adjective/adverb, citation noun. Use diacritics where correct.",
        "extra": "- IMPORTANT: the Lemma must be an authentic native word in your language. NEVER use the English gloss itself as the Lemma (no verbatim English copies). Use the genuine loanword spelling only when that IS the native word.\n",
        "examples": "nl,bibliotheek,library,图书馆,NOUN,A1\nnl,schrijven,to write,写,VERB,A1\nnl,mooi,beautiful,美丽的,ADJ,A1",
    },
    "sv": {
        "lemma": "the canonical sv lemma. infinitive verb, lowercase adjective/adverb, citation noun. Use diacritics where correct (å ä ö).",
        "extra": "- IMPORTANT: the Lemma must be an authentic native word in your language. NEVER use the English gloss itself as the Lemma (no verbatim English copies). Use the genuine loanword spelling only when that IS the native word.\n",
        "examples": "sv,bibliotek,library,图书馆,NOUN,A1\nsv,skriva,to write,写,VERB,A1\nsv,vacker,beautiful,美丽的,ADJ,A1",
    },
    "zh": {
        "lemma": "the canonical zh lemma in Simplified Chinese (1-5 Chinese characters, no punctuation).",
        "extra": "- Lemma must be a natural Simplified Chinese word, not a transliteration. Do NOT copy the English gloss verbatim.\n",
        "examples": "zh,图书馆,library,图书馆,NOUN,A1\nzh,写,to write,写,VERB,A1\nzh,美丽,beautiful,美丽的,ADJ,A1",
    },
}

TEMPLATE = """You are a native {code} lexicographer. Below is a list of {n} English concepts with EXACT POS tags, one per line. Each line ends with a target-language code in parentheses.

Produce ONE raw CSV table. No markdown fences, no commentary, no row numbers, nothing before the header or after the last row.

Line 1 must be exactly:
Lang,Lemma,English_Lemma,Chinese_Lemma,POS,CEFR

Then one data row per concept, in the exact same order as the list.

Column rules:
- Lang: {code} for every row.
- Lemma: {lemma} Multi-word lemmas allowed. If a concept has no single canonical lemma, skip that row entirely (skip at most 15 rows per 200).
- English_Lemma: copy the concept VERBATIM, letter for letter, same case. Do NOT include the "(target: ...)" marker or the POS tag.
- Chinese_Lemma: Simplified Chinese translation of the concept (1 to 5 Chinese words, no punctuation).
- POS: use EXACTLY the POS tag shown in parentheses after the concept. Do not change it.
- CEFR: exactly one of: A1, A2, B1, B2, C1 (the level at which a learner of {code} meets this word)
- Never repeat the same (Lang, Lemma, English_Lemma, POS) combination in two rows.
- Do not quote fields unless a field itself contains a comma; fields with a comma must be quoted with double quotes.

{extra}
Calibration examples (format reference only, not part of your output):
{examples}

Concepts:
{concepts}"""
ORDER = ("ar", "de", "en", "es", "fr", "nl", "sv", "zh")


def main(root: Path | None = None, round_no: str | None = None) -> None:
    root = root or Path(__file__).resolve().parent.parent
    round_no = round_no or (sys.argv[1] if len(sys.argv) > 1 else "2")
    out_dir = root / f"work/gemini_prompts_r{round_no}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.txt"):
        old.unlink()

    records = load_csv_records(root) + load_expansion_records(root)
    delivery = build_delivery_rows(records)

    # Gate-exact criterion 4 bookkeeping: per target lang, (norm gloss, pos)
    # -> set of lemmas; per src lang, row list for pair counting.
    index: dict[str, dict[tuple[str, str], set[str]]] = {}
    for lang, rows in delivery.items():
        idx: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            if not row.gloss:
                continue
            idx.setdefault((normalize_gloss(row.gloss), row.pos), set()).add(row.lemma)
        index[lang] = idx
    canonical: dict[tuple[str, str], str] = {}
    for rows in delivery.values():
        for row in rows:
            if not row.gloss:
                continue
            key = (normalize_gloss(row.gloss), row.pos)
            cur = canonical.get(key)
            if cur is None or len(row.gloss) > len(cur):
                canonical[key] = row.gloss

    # Keys a fresh target row cannot cover: either already present in the raw
    # expansion CSV (harmonization dropped it once, it will drop again), or
    # (en only) the gloss already exists in the delivery under another POS so
    # the new row collides on (lang, lemma, gloss_norm) and is dropped.
    import csv as _csv

    raw_keys: dict[str, set[tuple[str, str]]] = {}
    for lang, code in LANG_DIRS.items():
        ks: set[tuple[str, str]] = set()
        p = root / lang / "expansion.csv"
        if p.exists():
            with p.open(newline="", encoding="utf-8") as handle:
                reader = _csv.reader(handle)
                next(reader, None)
                for cols in reader:
                    if len(cols) >= 4 and cols[0].strip():
                        ks.add(
                            (normalize_gloss(cols[1].strip()), cols[3].strip().upper())
                        )
        raw_keys[code] = ks
    bygloss: dict[str, dict[str, set[str]]] = {}
    for lang, rows in delivery.items():
        bg: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row.gloss:
                bg[normalize_gloss(row.gloss)].add(row.pos)
        bygloss[lang] = bg

    total = 0
    for tgt in ORDER:
        covered = index[tgt]
        deficits: list[tuple[int, set[tuple[str, str]]]] = []
        for src in ORDER:
            if src == tgt:
                continue
            src_rows = delivery[src]
            n = len(src_rows)
            eindeutig = 0
            missing: set[tuple[str, str]] = set()
            for row in src_rows:
                if not row.gloss:
                    continue
                key = (normalize_gloss(row.gloss), row.pos)
                found = len(covered.get(key, ()))
                if found == 1:
                    eindeutig += 1
                elif found == 0:
                    missing.add(key)
            need = max(0, math.ceil(0.8 * n) - eindeutig)
            if need:
                deficits.append((need, missing))
        if not deficits:
            print(f"{tgt}: all pairs >= 80%")
            continue
        chosen: set[tuple[str, str]] = set()
        for need, cells in sorted(deficits, key=lambda d: -d[0]):
            clean: list[tuple[str, str]] = []
            fallback: list[tuple[str, str]] = []
            for k in sorted(cells - chosen):
                if k in raw_keys[tgt]:
                    continue  # dropped before; regeneration is a black hole
                if (
                    tgt == "en"
                    and k[0] in bygloss["en"]
                    and k[1] not in bygloss["en"][k[0]]
                ):
                    continue  # en lemma == gloss: guaranteed (lang,lemma,gloss) collision
                if k[0] in bygloss[tgt]:
                    fallback.append(k)  # gloss exists under another POS
                else:
                    clean.append(k)
            # take what the pair needs, then pad with spare clean cells so
            # Gemini POS drift or invalid rows do not stall the round
            take = clean[:need] + fallback[:need] + clean[need : 2 * need]
            chosen.update(take[: 2 * need])
        cell_list = sorted(chosen, key=lambda k: (canonical.get(k, k[0]), k[1]))
        print(
            f"{tgt}: {len(cell_list)} cells for "
            f"{[f'{src}->{tgt}: {need}' for need, _ in deficits]}"
        )
        for i in range(0, len(cell_list), CHUNK):
            part = cell_list[i : i + CHUNK]
            letter = chr(ord("a") + i // CHUNK)
            lines = [
                f"{canonical[k]} (target: {tgt}, {k[1]})"
                for k in part
                if k in canonical
            ]
            text = TEMPLATE.format(
                code=tgt,
                n=len(lines),
                lemma=LANGS[tgt]["lemma"],
                extra=LANGS[tgt]["extra"],
                examples=LANGS[tgt]["examples"],
                concepts="\n".join(lines),
            )
            (out_dir / f"{tgt}_r{letter}.txt").write_text(text, encoding="utf-8")
        total += len(cell_list)
    print(f"total: {total} concepts -> {out_dir}")


if __name__ == "__main__":
    main()
