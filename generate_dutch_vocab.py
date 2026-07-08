"""Generate Dutch CEFR vocabulary CSVs (A1-C1) with Stanza POS tags.

Uses wordfreq for frequency-ranked Dutch word lists and Stanza nl_alpino
for UD POS tags. English translations are populated via deep-translator
(Google Translate) with batching for speed.

CEFR level mapping (by wordfreq rank):
- A1: ranks 1-600
- A2: ranks 601-1200
- B1: ranks 1201-2200
- B2: ranks 2201-4200
- C1: ranks 4201-8200

Produces dutch/A1.csv through dutch/C1.csv with the harmonized header:
    Dutch_Lemma, English_Lemma, Chinese_Lemma, POS

Usage:
    uv run python generate_dutch_vocab.py            # generate all levels
    uv run python generate_dutch_vocab.py --check     # validate only
    uv run python generate_dutch_vocab.py --translate-only  # fill English_Lemma only
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent
LEVELS = ("A1", "A2", "B1", "B2", "C1")
TARGETS = {"A1": 600, "A2": 600, "B1": 1000, "B2": 2000, "C1": 4000}

RANK_BOUNDS = {
    "A1": (1, 600),
    "A2": (601, 1200),
    "B1": (1201, 2200),
    "B2": (2201, 4200),
    "C1": (4201, 8200),
}


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["Dutch_Lemma", "English_Lemma", "Chinese_Lemma", "POS"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\r\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def get_dutch_words() -> list[str]:
    """Get frequency-ranked Dutch words from wordfreq."""
    from wordfreq import top_n_list

    return top_n_list("nl", 8200)


def translate_words(words: list[str]) -> dict[str, str]:
    """Translate Dutch words to English via Google Translate (batched)."""
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="nl", target="en")
    translations: dict[str, str] = {}
    total = len(words)
    batch_size = 100
    for start in range(0, total, batch_size):
        batch = words[start : start + batch_size]
        try:
            results = translator.translate_batch(batch)
            for word, en in zip(batch, results, strict=False):
                translations[word] = en if en else word
        except Exception as e:
            print(f"  error at batch {start}: {e}", flush=True)
            for w in batch:
                translations[w] = w
        done = min(start + batch_size, total)
        if done % 500 == 0 or done == total:
            print(f"  translated {done}/{total}", flush=True)
    return translations


def tag_with_stanza(lemmas: list[str]) -> dict[str, str]:
    """Tag lemmas with Stanza nl model."""
    import stanza

    nlp = stanza.Pipeline(
        lang="nl",
        processors="tokenize,pos",
        verbose=False,
        use_gpu=False,
    )
    tags: dict[str, str] = {}
    total = len(lemmas)
    for i, lemma in enumerate(lemmas, start=1):
        if not lemma.strip():
            continue
        doc = nlp(lemma)
        sentences = getattr(doc, "sentences", [])
        if sentences and sentences[0].words:
            upos = sentences[0].words[0].upos
            tags[lemma] = upos
        else:
            tags[lemma] = "X"
        if i % 500 == 0:
            print(f"  tagged {i}/{total}", flush=True)
    return tags


def generate() -> int:
    """Generate Dutch CSVs for all levels."""
    dutch_dir = ROOT / "dutch"
    dutch_dir.mkdir(exist_ok=True)

    print("Loading Dutch frequency word list...", flush=True)
    all_words = get_dutch_words()
    print(f"  {len(all_words)} words loaded", flush=True)

    print("Translating Dutch -> English...", flush=True)
    translations = translate_words(all_words)
    print(f"  {len(translations)} translations", flush=True)

    print("POS tagging with Stanza nl...", flush=True)
    tags = tag_with_stanza(all_words)
    print(f"  {len(tags)} tags", flush=True)

    issues = 0
    for level in LEVELS:
        start, end = RANK_BOUNDS[level]
        target = TARGETS[level]
        level_words = all_words[start - 1 : end]

        rows = []
        for word in level_words:
            en = translations.get(word, word)
            pos = tags.get(word, "X")
            rows.append(
                {
                    "Dutch_Lemma": word.strip(),
                    "English_Lemma": en.strip(),
                    "Chinese_Lemma": "",
                    "POS": pos,
                }
            )

        n = len(rows)
        delta = n - target
        status = "OK" if delta == 0 else f"OFF ({delta:+d})"
        print(f"  {level}: {n} rows (target {target}) — {status}", flush=True)
        if delta != 0:
            issues += 1

        save_csv(dutch_dir / f"{level}.csv", rows)

    return issues


def translate_only() -> int:
    """Fill English_Lemma for existing CSVs that have empty translations."""
    dutch_dir = ROOT / "dutch"
    all_words: list[str] = []
    for level in LEVELS:
        rows = load_csv(dutch_dir / f"{level}.csv")
        for r in rows:
            en = r.get("English_Lemma", "").strip()
            if not en or en == r.get("Dutch_Lemma", "").strip():
                all_words.append(r["Dutch_Lemma"])

    if not all_words:
        print("All translations already present.")
        return 0

    print(f"Translating {len(all_words)} words...", flush=True)
    translations = translate_words(all_words)

    for level in LEVELS:
        path = dutch_dir / f"{level}.csv"
        rows = load_csv(path)
        changed = False
        for r in rows:
            en = r.get("English_Lemma", "").strip()
            nl = r.get("Dutch_Lemma", "").strip()
            if not en or en == nl:
                r["English_Lemma"] = translations.get(nl, nl)
                changed = True
        if changed:
            save_csv(path, rows)
            print(f"  {level}: updated translations", flush=True)

    return 0


def check() -> int:
    """Validate generated CSVs."""
    issues = 0
    dutch_dir = ROOT / "dutch"
    for level in LEVELS:
        path = dutch_dir / f"{level}.csv"
        if not path.exists():
            print(f"  {level}: MISSING")
            issues += 1
            continue
        rows = load_csv(path)
        n = len(rows)
        target = TARGETS[level]
        delta = n - target
        status = "OK" if delta == 0 else f"OFF ({delta:+d})"
        print(f"  {level}: {n} rows (target {target}) — {status}")
        if delta != 0:
            issues += 1
        seen: set[tuple[str, str]] = set()
        dupes = 0
        for r in rows:
            key = (
                r.get("Dutch_Lemma", "").strip().lower(),
                r.get("POS", "").strip(),
            )
            if key in seen:
                dupes += 1
            seen.add(key)
        if dupes:
            print(f"    {dupes} duplicate lemma+POS")
            issues += 1
    return issues


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate Dutch CEFR CSVs.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--translate-only", action="store_true")
    args = parser.parse_args(argv[1:])

    if args.check:
        print("Checking Dutch CSVs...")
        issues = check()
    elif args.translate_only:
        issues = translate_only()
    else:
        print("Generating Dutch CSVs...", flush=True)
        issues = generate()

    if issues:
        print(f"\n{issues} issue(s) found.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
