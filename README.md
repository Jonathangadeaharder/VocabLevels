# VocabLevels

Trilingual CEFR vocabulary database for English, German, and Spanish.

## Structure

```
VocabLevels/
├── english/   A1-C1 CSV files
├── german/    A1-C1 CSV files
├── spanish/   A1-C1 CSV files
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
python vocab_manager.py find [lang] [query]
python vocab_manager.py add [lang] [level] [lemma] [translation]
python vocab_manager.py remove [lang] [level] [lemma]
python vocab_manager.py move [lang] [from_level] [to_level] [lemma]
python vocab_manager.py update [lang] [level] [lemma] --rename [new] --translation [new]
```

## Quality Gates

```bash
uvx ruff check
uvx ruff format
uvx pyright
```
