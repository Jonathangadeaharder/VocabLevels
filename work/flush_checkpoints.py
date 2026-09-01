"""Flush approved checkpoint cells into per-language expansion.csv files.

Reads each state dir's .checkpoint.jsonl "approved" events and appends cells
not already covered in expansion.csv (same covered-set semantics as
_write_expansion_csv). Safe to run repeatedly.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.expand_concepts import LANG_DIRS, normalize_gloss  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATE_DIRS = {
    "de": ROOT / "work" / "expand_test",
    **{
        code: ROOT / "work" / "expand_state" / code
        for code in ("ar", "en", "es", "fr", "nl", "sv", "zh")
    },
}


def main() -> None:
    for name, code in LANG_DIRS.items():
        path = ROOT / name / "expansion.csv"
        covered: set[tuple[str, str]] = set()
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for cols in reader:
                if len(cols) >= 4 and cols[0].strip() and cols[1].strip():
                    covered.add(
                        (normalize_gloss(cols[1].strip()), cols[3].strip().upper())
                    )
        state = STATE_DIRS[code]
        new_rows: list[list[str]] = []
        seen: set[tuple[str, str]] = set()
        for line in (
            (state / ".checkpoint.jsonl").read_text(encoding="utf-8").splitlines()
        ):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") != "approved" or ev.get("lang") != code:
                continue
            gloss, pos = ev["gloss"].strip(), ev["pos"].strip()
            key = (normalize_gloss(gloss), pos)
            if not gloss or key in covered or key in seen:
                continue
            seen.add(key)
            new_rows.append(
                [
                    ev.get("lemma", ""),
                    gloss,
                    ev.get("zh", ""),
                    pos,
                    ev.get("level", "B2"),
                ]
            )
        if new_rows:
            with path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerows(new_rows)
        print(f"{code}: +{len(new_rows)} rows from checkpoint")


if __name__ == "__main__":
    main()
