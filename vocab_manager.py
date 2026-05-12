"""CLI manager for trilingual CEFR vocab CSVs.

Run from repo root. Operates on english/, german/, spanish/ directories.

Examples:
    python vocab_manager.py lint
    python vocab_manager.py find german Haus
    python vocab_manager.py add german A1 Haus house casa
    python vocab_manager.py move german A2 Haus  # move 'Haus' to level A2
    python vocab_manager.py remove german Haus
    python vocab_manager.py update german Haus --t1 house --t2 casa
    python vocab_manager.py lookup english cat   # search across all langs
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent
LEVELS = ["A1", "A2", "B1", "B2", "C1"]

LANGS = {
    "english": {
        "lemma_col": "English_Lemma",
        "trans_cols": ("German_Translation", "Spanish_Translation"),
    },
    "german": {
        "lemma_col": "German_Lemma",
        "trans_cols": ("English_Translation", "Spanish_Translation"),
    },
    "spanish": {
        "lemma_col": "Spanish_Lemma",
        "trans_cols": ("English_Translation", "German_Translation"),
    },
}


def file_path(lang: str, level: str) -> Path:
    return ROOT / lang / f"{level}.csv"


def read_level(lang: str, level: str) -> list[dict[str, str]]:
    path = file_path(lang, level)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_level(lang: str, level: str, rows: list[dict[str, str]]) -> None:
    cfg = LANGS[lang]
    fields = [cfg["lemma_col"], *cfg["trans_cols"]]
    rows_sorted = sorted(rows, key=lambda r: r[cfg["lemma_col"]].lower())
    with file_path(lang, level).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_sorted)


def find(lang: str, lemma: str) -> list[str]:
    cfg = LANGS[lang]
    needle = lemma.lower()
    return [
        level for level in LEVELS
        if any(r[cfg["lemma_col"]].lower() == needle for r in read_level(lang, level))
    ]


def cmd_lint(_: argparse.Namespace) -> int:
    import check_quality  # reuse logic
    return check_quality.main(["check_quality.py", *LANGS])


def cmd_find(args: argparse.Namespace) -> int:
    lemma = args.lemma.strip()
    levels = find(args.lang, lemma)
    if not levels:
        print(f"'{lemma}' not found in {args.lang}")
        return 1
    print(f"'{lemma}' found in {args.lang}: {', '.join(levels)}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    cfg = LANGS[args.lang]
    lemma = args.lemma.strip()
    t1 = args.t1.strip()
    t2 = args.t2.strip()
    if " " in lemma:
        print("Lemma must be single-word")
        return 1
    if not t1 or not t2:
        print("Translations cannot be empty")
        return 1
    existing = find(args.lang, lemma)
    if existing:
        print(f"'{lemma}' already in {args.lang}: {existing}")
        return 1
    rows = read_level(args.lang, args.level)
    rows.append({
        cfg["lemma_col"]: lemma,
        cfg["trans_cols"][0]: t1,
        cfg["trans_cols"][1]: t2,
    })
    write_level(args.lang, args.level, rows)
    print(f"Added '{lemma}' to {args.lang}/{args.level}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    cfg = LANGS[args.lang]
    lemma = args.lemma.strip()
    needle = lemma.lower()
    found = False
    for level in LEVELS:
        rows = read_level(args.lang, level)
        kept = [r for r in rows if r[cfg["lemma_col"]].lower() != needle]
        if len(kept) != len(rows):
            write_level(args.lang, level, kept)
            print(f"Removed '{lemma}' from {args.lang}/{level}")
            found = True
    if not found:
        print(f"'{lemma}' not found in {args.lang}")
        return 1
    return 0


def cmd_move(args: argparse.Namespace) -> int:
    cfg = LANGS[args.lang]
    lemma = args.lemma.strip()
    needle = lemma.lower()
    if args.target_level not in LEVELS:
        print(f"Invalid level '{args.target_level}'")
        return 1
    src_row = None
    src_level = None
    for level in LEVELS:
        for row in read_level(args.lang, level):
            if row[cfg["lemma_col"]].lower() == needle:
                src_row, src_level = row, level
                break
        if src_row:
            break
    if not src_row:
        print(f"'{lemma}' not found in {args.lang}")
        return 1
    if src_level == args.target_level:
        print(f"'{lemma}' already in {args.target_level}")
        return 0
    cmd_remove(argparse.Namespace(lang=args.lang, lemma=lemma))
    target_rows = read_level(args.lang, args.target_level)
    target_rows.append(src_row)
    write_level(args.lang, args.target_level, target_rows)
    print(f"Moved '{lemma}': {src_level} → {args.target_level}")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    cfg = LANGS[args.lang]
    needle = args.lemma.lower()
    rename = args.rename.strip() if args.rename else None
    t1 = args.t1.strip() if args.t1 else None
    t2 = args.t2.strip() if args.t2 else None

    if rename and " " in rename:
        print("New lemma must be single-word")
        return 1
    if rename and find(args.lang, rename):
        print(f"'{rename}' already exists in {args.lang}")
        return 1
    if t1 == "" or t2 == "":
        print("Translations cannot be empty")
        return 1

    for level in LEVELS:
        rows = read_level(args.lang, level)
        changed = False
        for row in rows:
            if row[cfg["lemma_col"]].lower() == needle:
                if t1 is not None:
                    row[cfg["trans_cols"][0]] = t1
                if t2 is not None:
                    row[cfg["trans_cols"][1]] = t2
                if rename is not None:
                    row[cfg["lemma_col"]] = rename
                changed = True
        if changed:
            write_level(args.lang, level, rows)
            print(f"Updated '{args.lemma}' in {args.lang}/{level}")
            return 0
    print(f"'{args.lemma}' not found in {args.lang}")
    return 1


def cmd_lookup(args: argparse.Namespace) -> int:
    """Search across all languages by lemma or any translation column (exact match)."""
    needle = args.term.lower()
    hits: list[tuple[str, str, dict[str, str]]] = []
    for lang, cfg in LANGS.items():
        for level in LEVELS:
            for row in read_level(lang, level):
                values = (row[cfg["lemma_col"]], *(row[c] for c in cfg["trans_cols"]))
                if any(needle == v.lower() for v in values):
                    hits.append((lang, level, row))
    if not hits:
        print(f"'{args.term}' not found")
        return 1
    for lang, level, row in hits:
        cols = LANGS[lang]
        print(f"  {lang}/{level}: {row[cols['lemma_col']]} | {row[cols['trans_cols'][0]]} | {row[cols['trans_cols'][1]]}")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Trilingual CEFR vocab manager")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("lint", help="Run check_quality across all languages")

    fp = sub.add_parser("find", help="Find lemma in a specific language")
    fp.add_argument("lang", choices=LANGS)
    fp.add_argument("lemma")

    ap = sub.add_parser("add", help="Add a lemma to a level")
    ap.add_argument("lang", choices=LANGS)
    ap.add_argument("level", choices=LEVELS)
    ap.add_argument("lemma")
    ap.add_argument("t1", help="First translation (see schema)")
    ap.add_argument("t2", help="Second translation")

    rp = sub.add_parser("remove", help="Remove a lemma from all levels")
    rp.add_argument("lang", choices=LANGS)
    rp.add_argument("lemma")

    mp = sub.add_parser("move", help="Move a lemma to a different level")
    mp.add_argument("lang", choices=LANGS)
    mp.add_argument("target_level", choices=LEVELS)
    mp.add_argument("lemma")

    up = sub.add_parser("update", help="Update translations or rename a lemma")
    up.add_argument("lang", choices=LANGS)
    up.add_argument("lemma")
    up.add_argument("--t1")
    up.add_argument("--t2")
    up.add_argument("--rename")

    lp = sub.add_parser("lookup", help="Search a term across all languages")
    lp.add_argument("term")

    args = p.parse_args(argv[1:])
    dispatch = {
        "lint": cmd_lint,
        "find": cmd_find,
        "add": cmd_add,
        "remove": cmd_remove,
        "move": cmd_move,
        "update": cmd_update,
        "lookup": cmd_lookup,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
