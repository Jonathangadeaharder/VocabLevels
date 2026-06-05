# VocabLevels

Trilingual CEFR vocabulary database for English, German, and Spanish.

## Structure

```
VocabLevels/
├── english/   A1-C1 CSV files + frequency reference TSV
├── german/    A1-C1 CSV files + frequency reference TSV
├── spanish/   A1-C1 CSV files + frequency reference TSV
├── vocab_manager.py    CLI tool for managing entries
├── check_quality.py    Quality validation
├── audit_lemmatization.py
├── cleanup_inflections.py
```

15 files, ~24,000 vocabulary entries (600/600/1000/2000/4000 per CEFR level × 3 languages).

## Usage

```bash
# Run quality check (standalone scripts, no uv sync needed)
python check_quality.py [lang]

# Manage vocabulary
python vocab_manager.py lint                              # run quality checks across all languages
python vocab_manager.py find [lang] [lemma]              # find lemma in a specific language
python vocab_manager.py add [lang] [level] [lemma] [t1] [t2]  # add entry (t1/t2 = translations)
python vocab_manager.py remove [lang] [lemma]            # remove from all levels
python vocab_manager.py move [lang] [target_level] [lemma]    # move to target level (auto-detects source)
python vocab_manager.py update [lang] [lemma] [--t1 X] [--t2 Y] [--rename X]  # update fields
python vocab_manager.py lookup [term]                    # search across all languages
```

## Quality Gates

```bash
uvx ruff check
uvx ruff format
uvx pyright
```
