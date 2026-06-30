"""Remove obvious inflected forms, keeping lemmas."""

from pathlib import Path

import vocab_manager
from vocab_schema import LANGS, LEVELS

ROOT = Path(__file__).parent


def cleanup_language(lang: str) -> int:
    """Remove same-level English plurals."""
    cfg = LANGS[lang]
    if lang != "english":
        return 0

    vocab_manager.ROOT = ROOT
    lemma_col = cfg["lemma_col"]

    total_removed = 0
    for level in LEVELS:
        path = vocab_manager.file_path(lang, level)
        if not path.exists():
            continue
        rows = vocab_manager.read_level(lang, level)

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
            vocab_manager.write_level(lang, level, kept)
            print(
                f"{lang}/{level}: removed {removed_here} plural forms ({len(kept)} rows remain)"
            )
            total_removed += removed_here

    return total_removed


if __name__ == "__main__":
    print("Cleaning up inflected forms from English CSVs...")
    total = cleanup_language("english")
    print(f"Total removed: {total}")
    print("\nRunning quality check...")
    import check_quality

    check_quality.main(["check_quality.py", "english"])
