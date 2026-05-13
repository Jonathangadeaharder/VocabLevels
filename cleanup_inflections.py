"""Remove obvious inflected forms, keeping lemmas."""
import csv
from pathlib import Path

ROOT = Path(__file__).parent
LEVELS = ["A1", "A2", "B1", "B2", "C1"]
TRANS_COLS = {
    "english": ("German_Translation", "Spanish_Translation"),
    "german": ("English_Translation", "Spanish_Translation"),
    "spanish": ("English_Translation", "German_Translation"),
}


def cleanup_language(lang):
    """Remove same-level plurals from a language (English only for now)."""
    lemma_col = {"english": "English_Lemma", "german": "German_Lemma", "spanish": "Spanish_Lemma"}[lang]

    total_removed = 0
    for level in LEVELS:
        path = ROOT / lang / f"{level}.csv"
        with path.open() as f:
            rows = list(csv.DictReader(f))

        lemmas_in_level = set(row[lemma_col].strip().lower() for row in rows)

        if lang == "english":
            kept = []
            removed_here = 0
            for row in rows:
                lemma = row[lemma_col].strip()
                lemma_lower = lemma.lower()

                skip = False
                if lemma_lower.endswith("s") and not lemma_lower.endswith("ss"):
                    singular = lemma_lower[:-1]
                    if singular in lemmas_in_level and len(singular) > 2 and singular != lemma_lower:
                        skip = True
                        removed_here += 1

                if not skip:
                    kept.append(row)

            if removed_here > 0:
                fieldnames = [lemma_col, *TRANS_COLS[lang]]
                with path.open("w") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
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
