# VocabLevels

Multilingual CEFR vocabulary CSVs. Currently tracked languages: English, German, Spanish, Arabic, French, Swedish, Chinese, and Dutch.

## Structure

```
VocabLevels/
├── english/   A1-C1 CSV files + frequency reference TSV
├── german/    A1-C1 CSV files + frequency reference TSV
├── spanish/   A1-C1 CSV files + frequency reference TSV
├── arabic/    A1-C1 CSV files
├── chinese/   A1-C1 CSV files
├── dutch/     A1-C1 CSV files
├── french/    A1 CSV (in progress)
├── swedish/   A1 CSV (in progress)
├── vocab_schema.py     Shared language columns, CEFR levels, and targets
├── vocab_manager.py    CLI tool for managing entries
├── check_quality.py    Quality validation
├── audit_lemmatization.py
├── cleanup_inflections.py
```

CEFR-level CSV files use 600/600/1000/2000/4000 target entries per complete language.
The repository currently contains 18 CSV files and 26,818 entries.

## Usage

```bash
# Run the data quality checker
uv run python check_quality.py [lang]
uv run python check_quality.py --show-shared-translations [lang]

# Manage vocabulary
uv run python vocab_manager.py lint                              # run quality checks across all languages
uv run python vocab_manager.py find [lang] [lemma]                # find lemma in a specific language
uv run python vocab_manager.py add [lang] [level] [lemma] [t1] [t2]  # add entry (t1/t2 = translations)
uv run python vocab_manager.py remove [lang] [lemma]              # remove from all levels
uv run python vocab_manager.py move [lang] [target_level] [lemma] # move to target level
uv run python vocab_manager.py update [lang] [lemma] [--t1 X] [--t2 Y] [--rename X]
uv run python vocab_manager.py lookup [term]                      # search across all languages
```

## Quality Gates

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run python -m pytest
```
