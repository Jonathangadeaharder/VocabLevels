"""Audit lemmatization: find plurals, verb forms, and other inflections."""
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
LEVELS = ["A1", "A2", "B1", "B2", "C1"]

# Common inflection patterns to detect
PLURAL_SUFFIXES = {"s", "es", "ies", "n"}  # English, Spanish, German
VERB_SUFFIXES = {"ing", "ed", "er", "en", "te", "ado", "ido", "anza"}  # Common -ing, -ed, -er, German -en, Spanish -ado/-ido

def find_inflected_forms():
    """Scan for likely inflected forms (plurals, verb forms)."""
    for lang in ["english", "german", "spanish"]:
        print(f"\n=== {lang.upper()} ===")
        lemma_col = {"english": "English_Lemma", "german": "German_Lemma", "spanish": "Spanish_Lemma"}[lang]

        all_lemmas = {}  # lemma_lower -> list of (level, row)

        for level in LEVELS:
            path = ROOT / lang / f"{level}.csv"
            with path.open() as f:
                for row in csv.DictReader(f):
                    lemma = row[lemma_col].strip().lower()
                    if lemma not in all_lemmas:
                        all_lemmas[lemma] = []
                    all_lemmas[lemma].append((level, row))

        # Find candidates
        candidates = []
        for lemma in sorted(all_lemmas.keys()):
            # Check if a singular form exists
            if lemma.endswith("s") and not lemma.endswith("ss"):
                singular = lemma[:-1]
                if singular in all_lemmas and singular != lemma:
                    candidates.append((lemma, "plural", singular))
            # Check for -ing forms
            elif lemma.endswith("ing"):
                base = lemma[:-3]
                if base in all_lemmas and base != lemma:
                    candidates.append((lemma, "gerund", base))
            # Check for -ed forms
            elif lemma.endswith("ed") and len(lemma) > 3:
                base = lemma[:-2]
                if base in all_lemmas and base != lemma:
                    candidates.append((lemma, "past", base))

        if candidates:
            print(f"Found {len(candidates)} potential non-lemmatized forms:")
            for form, ftype, base in candidates[:20]:
                levels = [lv for lv, _ in all_lemmas[form]]
                base_levels = [lv for lv, _ in all_lemmas[base]]
                print(f"  {form:20} ({ftype:10}) → {base:15} | {form}: {levels}, {base}: {base_levels}")
        else:
            print("No obvious inflected forms found")

if __name__ == "__main__":
    find_inflected_forms()
