"""Build stratified p20 (95% ±20%) CEFR audit samples from proposed CSVs.

Compares each proposed row to the committed level CSV when present to label
change type, then draws a finite-population-corrected sample stratified by
change type. Writes manual_reviews/{lang}/tng-audit-sample/ packs.

Usage:
  uv run python -m scripts.gemma_qa.build_audit_sample --root .
  uv run python -m scripts.gemma_qa.build_audit_sample --root . --languages fr nl
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

LANG_DIRS = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "nl": "dutch",
    "sv": "swedish",
    "ar": "arabic",
    "zh": "chinese",
}

CHANGE_ORDER = (
    "edited",
    "pos_or_key_changed",
    "new_or_renamed",
    "unchanged",
)


@dataclass(frozen=True)
class Row:
    csv_line: int
    lemma: str
    english_lemma: str
    chinese_lemma: str
    upos: str
    change: str
    committed_before: str


def sample_size_fpc(population: int, *, margin: float = 0.20, z: float = 1.96, p: float = 0.5) -> int:
    if population <= 0:
        return 0
    n0 = (z * z * p * (1.0 - p)) / (margin * margin)
    n = n0 / (1.0 + (n0 - 1.0) / population)
    return max(1, min(population, int(math.ceil(n))))


def load_level_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            return []
        lemma_key = fieldnames[0]
        rows: list[dict[str, str]] = []
        for index, raw in enumerate(reader, start=2):
            lemma = (raw.get(lemma_key) or "").strip()
            english = (raw.get("English_Lemma") or "").strip()
            if lemma_key == "English_Lemma":
                english = lemma
            chinese = (raw.get("Chinese_Lemma") or "").strip()
            upos = (raw.get("POS") or "").strip()
            rows.append(
                {
                    "csv_line": str(index),
                    "lemma": lemma,
                    "english_lemma": english,
                    "chinese_lemma": chinese,
                    "upos": upos,
                    "fp": f"{lemma.casefold()}\0{upos.casefold()}",
                    "lemma_key": lemma.casefold(),
                }
            )
        return rows


def classify_rows(proposed: list[dict[str, str]], committed: list[dict[str, str]] | None) -> list[Row]:
    if not committed:
        return [
            Row(
                csv_line=int(item["csv_line"]),
                lemma=item["lemma"],
                english_lemma=item["english_lemma"],
                chinese_lemma=item["chinese_lemma"],
                upos=item["upos"],
                change="unchanged",
                committed_before="",
            )
            for item in proposed
        ]

    by_fp: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_lemma: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in committed:
        by_fp[item["fp"]].append(item)
        by_lemma[item["lemma_key"]].append(item)

    classified: list[Row] = []
    for item in proposed:
        matches = by_fp.get(item["fp"], [])
        if matches:
            prior = matches[0]
            same = (
                prior["english_lemma"] == item["english_lemma"]
                and prior["chinese_lemma"] == item["chinese_lemma"]
            )
            change = "unchanged" if same else "edited"
            before = (
                f"{prior['lemma']}|{prior['english_lemma']}|"
                f"{prior['chinese_lemma']}|{prior['upos']}"
            )
        elif item["lemma_key"] in by_lemma:
            prior = by_lemma[item["lemma_key"]][0]
            change = "pos_or_key_changed"
            before = (
                f"{prior['lemma']}|{prior['english_lemma']}|"
                f"{prior['chinese_lemma']}|{prior['upos']}"
            )
        else:
            change = "new_or_renamed"
            before = ""
        classified.append(
            Row(
                csv_line=int(item["csv_line"]),
                lemma=item["lemma"],
                english_lemma=item["english_lemma"],
                chinese_lemma=item["chinese_lemma"],
                upos=item["upos"],
                change=change,
                committed_before=before,
            )
        )
    return classified


def stratified_sample(rows: list[Row], n: int, rng: random.Random) -> list[Row]:
    if n >= len(rows):
        return list(rows)
    by_change: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_change[row.change].append(row)
    for bucket in by_change.values():
        rng.shuffle(bucket)

    # Proportional allocation; guarantee ≥1 from each non-empty stratum when n allows.
    strata = [key for key in CHANGE_ORDER if by_change.get(key)]
    strata += [key for key in by_change if key not in CHANGE_ORDER]
    total = len(rows)
    allocation: dict[str, int] = {}
    remaining = n
    for index, key in enumerate(strata):
        size = len(by_change[key])
        if index == len(strata) - 1:
            take = min(size, remaining)
        else:
            take = max(1, round(n * size / total)) if remaining > 0 else 0
            take = min(size, take, remaining - (len(strata) - index - 1))
            take = max(0, take)
        allocation[key] = take
        remaining -= take
    # Fix drift if remaining slots left.
    if remaining > 0:
        for key in strata:
            spare = len(by_change[key]) - allocation[key]
            add = min(spare, remaining)
            allocation[key] += add
            remaining -= add
            if remaining <= 0:
                break

    picked: list[Row] = []
    for key in strata:
        picked.extend(by_change[key][: allocation[key]])
    rng.shuffle(picked)
    return picked[:n]


def write_level_pack(
    out_dir: Path,
    *,
    lang_dir: str,
    level: str,
    population: int,
    sample: list[Row],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{level}.sample-p20.csv"
    md_path = out_dir / f"{level}.sample-p20.md"
    fieldnames = [
        "level",
        "sample_index",
        "csv_line",
        "change",
        "lemma",
        "english_lemma",
        "chinese_lemma",
        "upos",
        "committed_before",
        "verdict",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(sample, start=1):
            writer.writerow(
                {
                    "level": level,
                    "sample_index": index,
                    "csv_line": row.csv_line,
                    "change": row.change,
                    "lemma": row.lemma,
                    "english_lemma": row.english_lemma,
                    "chinese_lemma": row.chinese_lemma,
                    "upos": row.upos,
                    "committed_before": row.committed_before,
                    "verdict": "",
                    "notes": "",
                }
            )

    lines = [
        f"# Audit sample — {lang_dir} {level} (n={len(sample)}, 95%±20%) — **UNREVIEWED**",
        "",
        f"Population (proposed): **{population}**. Fill `verdict` = keep / drop / fix.",
        "",
        "| # | line | change | lemma | en | zh | POS | verdict | notes |",
        "|--:|-----:|--------|-------|----|----|-----|---------|-------|",
    ]
    for index, row in enumerate(sample, start=1):
        lines.append(
            f"| {index} | {row.csv_line} | {row.change} | {row.lemma} | "
            f"{row.english_lemma} | {row.chinese_lemma} | {row.upos} |  |  |"
        )
    lines.append("")
    lines.append(f"**Defects this level:** _/ {len(sample)} (fill after review)_")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(
    out_dir: Path,
    *,
    lang_code: str,
    lang_dir: str,
    inventory: list[tuple[str, int, int, dict[str, int]]],
    seed: int,
    skip_existing: bool,
) -> None:
    rows = [
        f"# TNG CEFR audit samples — {lang_dir} (p20)",
        "",
        f"Language code: `{lang_code}`. Source: `{{level}}.proposed.csv` vs committed `{{level}}.csv`.",
        "",
        "## Sample design",
        "",
        "- Stratified random sample **per level** by change type",
        "  (`edited` / `pos_or_key_changed` / `new_or_renamed` / `unchanged`)",
        "- **95% confidence, ±20% margin**, p=0.5, finite-population corrected",
        f"- Seed: `{seed}`",
        "",
        "## Inventory",
        "",
        "| Level | Proposed N | n (p20) | Change mix |",
        "|-------|----------:|--------:|------------|",
    ]
    total_n = 0
    for level, pop, n, mix in inventory:
        mix_s = ", ".join(f"{k}={v}" for k, v in mix.items() if v)
        rows.append(f"| {level} | {pop} | {n} | {mix_s} |")
        total_n += n
    rows.append(f"| **Σ** | | **{total_n}** | |")
    rows.extend(
        [
            "",
            "## Files",
            "",
            "| File | |",
            "|------|--|",
            "| `ALL.sample-p20.csv` | combined |",
            "| `{level}.sample-p20.csv` + `.md` | per-level checklist |",
            "",
            "## Score",
            "",
            "Verdict: `keep` / `drop` / `fix`. Defect rate = (drop+fix)/n.",
            "Packs are **UNREVIEWED** until a human/agent fills verdicts.",
            "",
        ]
    )
    if skip_existing:
        rows.append("Existing REVIEWED English pack was left untouched when present.")
        rows.append("")
    (out_dir / "README.md").write_text("\n".join(rows), encoding="utf-8")


def succeeded_tasks(root: Path, languages: set[str] | None) -> list[tuple[str, str, Path]]:
    db = root / ".gemma_qa" / "scale.sqlite3"
    if not db.exists():
        raise SystemExit(f"missing scale db: {db}")
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            """
            SELECT language, level, output
            FROM scale_tasks
            WHERE phase = 'cefr' AND status = 'succeeded'
            ORDER BY language, level
            """
        ).fetchall()
    finally:
        conn.close()
    tasks: list[tuple[str, str, Path]] = []
    for lang, level, output in rows:
        if languages is not None and lang not in languages:
            continue
        path = root / output if output else root / LANG_DIRS.get(lang, lang) / f"{level}.proposed.csv"
        tasks.append((lang, level, path))
    return tasks


def build_for_language(
    root: Path,
    lang_code: str,
    levels: list[tuple[str, Path]],
    *,
    seed: int,
    force: bool,
) -> Path:
    lang_dir = LANG_DIRS.get(lang_code, lang_code)
    out_dir = root / "manual_reviews" / lang_dir / "tng-audit-sample"
    if (
        not force
        and lang_code == "en"
        and (out_dir / "ALL.sample-p20.csv").exists()
        and "REVIEWED" in (out_dir / "A1.sample-p20.md").read_text(encoding="utf-8")
    ):
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    inventory: list[tuple[str, int, int, dict[str, int]]] = []

    for level, proposed_path in levels:
        if not proposed_path.exists():
            raise SystemExit(f"missing proposed: {proposed_path}")
        committed_path = proposed_path.with_name(f"{level}.csv")
        proposed = load_level_csv(proposed_path)
        committed = load_level_csv(committed_path) if committed_path.exists() else None
        classified = classify_rows(proposed, committed)
        n = sample_size_fpc(len(classified))
        rng = random.Random(f"{seed}:{lang_code}:{level}")
        sample = stratified_sample(classified, n, rng)
        write_level_pack(
            out_dir,
            lang_dir=lang_dir,
            level=level,
            population=len(classified),
            sample=sample,
        )
        mix: dict[str, int] = defaultdict(int)
        for index, row in enumerate(sample, start=1):
            mix[row.change] += 1
            all_rows.append(
                {
                    "level": level,
                    "sample_index": index,
                    "csv_line": row.csv_line,
                    "change": row.change,
                    "lemma": row.lemma,
                    "english_lemma": row.english_lemma,
                    "chinese_lemma": row.chinese_lemma,
                    "upos": row.upos,
                    "committed_before": row.committed_before,
                    "verdict": "",
                    "notes": "",
                }
            )
        inventory.append((level, len(classified), len(sample), dict(mix)))

    fieldnames = [
        "level",
        "sample_index",
        "csv_line",
        "change",
        "lemma",
        "english_lemma",
        "chinese_lemma",
        "upos",
        "committed_before",
        "verdict",
        "notes",
    ]
    with (out_dir / "ALL.sample-p20.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    write_readme(
        out_dir,
        lang_code=lang_code,
        lang_dir=lang_dir,
        inventory=inventory,
        seed=seed,
        skip_existing=False,
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Optional language codes (en fr de …). Default: all succeeded.",
    )
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if English REVIEWED pack exists.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    languages = set(args.languages) if args.languages else None
    tasks = succeeded_tasks(root, languages)
    if not tasks:
        raise SystemExit("no succeeded cefr tasks matched")

    by_lang: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for lang, level, path in tasks:
        by_lang[lang].append((level, path))

    for lang_code, levels in sorted(by_lang.items()):
        if (
            not args.force
            and lang_code == "en"
            and (
                root
                / "manual_reviews"
                / "english"
                / "tng-audit-sample"
                / "ALL.sample-p20.csv"
            ).exists()
        ):
            print("skip en (existing REVIEWED pack)")
            continue
        out = build_for_language(
            root,
            lang_code,
            levels,
            seed=args.seed,
            force=args.force,
        )
        print(f"wrote {out} levels={[lv for lv, _ in levels]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
