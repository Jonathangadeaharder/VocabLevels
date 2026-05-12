"""Remove obvious inflected forms, keeping lemmas."""
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
LEVELS = ["A1", "A2", "B1", "B2", "C1"]

def get_all_lemmas():
    """Build index of all lemmas per language."""
    index = {}
    for lang in ["english", "german", "spanish"]:
        lemma_col = {"english": "English_Lemma", "german": "German_Lemma", "spanish": "Spanish_Lemma"}[lang]
        index[lang] = {}
        for level in LEVELS:
            path = ROOT / lang / f"{level}.csv"
            with path.open() as f:
                for row in csv.DictReader(f):
                    lemma = row[lemma_col].strip().lower()
                    if lemma not in index[lang]:
                        index[lang][lemma] = []
                    index[lang][lemma].append((level, row[lemma_col].strip()))
    return index

def is_inflected(lemma, all_lemmas):
    """Check if lemma is likely an inflected form."""
    lemma_lower = lemma.lower()

    # English: check for obvious plurals and verb forms
    if lemma_lower.endswith("s") and not lemma_lower.endswith("ss"):
        singular = lemma_lower[:-1]
        if singular in all_lemmas and len(singular) > 2:
            return singular
    if lemma_lower.endswith("ed") and len(lemma_lower) > 3:
        base = lemma_lower[:-2]
        if base in all_lemmas:
            return base
    if lemma_lower.endswith("ing") and len(lemma_lower) > 4:
        base = lemma_lower[:-3]
        if base in all_lemmas:
            return base

    return None

def cleanup_language(lang):
    """Remove inflected forms from a language."""
    lemma_col = {"english": "English_Lemma", "german": "German_Lemma", "spanish": "Spanish_Lemma"}[lang]

    total_removed = 0
    for level in LEVELS:
        path = ROOT / lang / f"{level}.csv"
        with path.open() as f:
            rows = list(csv.DictReader(f))

        # Build lemma set for this level only (to find same-level pairs)
        lemmas_in_level = set(row[lemma_col].strip().lower() for row in rows)

        # Only remove for English: obvious plurals where singular exists in same level
        if lang == "english":
            kept = []
            removed_here = 0
            for row in rows:
                lemma = row[lemma_col].strip()
                lemma_lower = lemma.lower()

                # Remove if: ends in 's', singular exists in level, base is >2 chars
                skip = False
                if lemma_lower.endswith("s") and not lemma_lower.endswith("ss"):
                    singular = lemma_lower[:-1]
                    if singular in lemmas_in_level and len(singular) > 2 and singular != lemma_lower:
                        skip = True
                        removed_here += 1

                if not skip:
                    kept.append(row)

            if removed_here > 0:
                # Rewrite file
                with path.open("w") as f:
                    writer = csv.DictWriter(f, fieldnames=[lemma_col, "German_Translation", "Spanish_Translation"])
                    writer.writeheader()
                    kept_sorted = sorted(kept, key=lambda r: r[lemma_col].lower())
                    writer.writerows(kept_sorted)
                print(f"{lang}/{level}: removed {removed_here} plural forms ({len(kept)} rows remain)")
                total_removed += removed_here

    return total_removed

if __name__ == "__main__":
    print("Cleaning up inflected forms from English CSVs...")
    total = cleanup_language("english")
    print(f"Total removed: {total}")
    print("\nRunning quality check...")
    import check_quality
    check_quality.main(["check_quality.py", "english"])
