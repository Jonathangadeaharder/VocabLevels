from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .cefr import run_cefr, run_cefr_gap_refill
from .client import GemmaClient
from .config import MODEL_IDS, default_batch_concurrency
from .handcraft import assess_handcraft_ready, run_handcraft
from .language_repair import normalize_english_csv_file
from .languages import LANGUAGE_CODES, LANGUAGE_DIRECTORIES, LEVELS, get_language
from .ledger import Ledger
from .manual_review import run_manual_review
from .routing import UnifiedQaClient
from .scale import ScaleConfig, ScaleState, run_scale
from .trace import configure, event, recent_events, run_id
from .validated import ValidatedStore, validated_store_path


def _build_unified_client() -> UnifiedQaClient:
    return UnifiedQaClient(gemma=GemmaClient())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.gemma_qa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cefr = subparsers.add_parser("cefr")
    cefr.add_argument("--root", type=Path, required=True)
    cefr.add_argument("--lang", choices=LANGUAGE_DIRECTORIES, required=True)
    cefr.add_argument("--level", choices=LEVELS, required=True)
    cefr.add_argument("--limit", type=_positive_int)
    cefr.add_argument("--apply", action="store_true")
    cefr.add_argument("--refill-to-target", action="store_true")
    cefr.add_argument("--single-model", choices=MODEL_IDS)
    cefr.add_argument(
        "--batch-concurrency",
        type=_positive_int,
        default=None,
        help=(
            "parallel CEFR row-batches (default: env GEMMA_QA_BATCH_CONCURRENCY "
            f"or {default_batch_concurrency()}); ledger-safe on restart"
        ),
    )

    refill = subparsers.add_parser("refill")
    refill.add_argument("--root", type=Path, required=True)
    refill.add_argument("--lang", choices=LANGUAGE_DIRECTORIES, required=True)
    refill.add_argument("--level", choices=LEVELS, required=True)
    refill.add_argument("--single-model", choices=MODEL_IDS)
    refill.add_argument("--reject-decisions", type=Path)

    handcraft = subparsers.add_parser("handcraft")
    handcraft.add_argument("--vocab-root", type=Path, required=True)
    handcraft.add_argument("--lemmatizer-root", type=Path, required=True)
    handcraft.add_argument("--lang", choices=LANGUAGE_CODES, required=True)
    handcraft.add_argument(
        "--level",
        choices=LEVELS,
        required=True,
    )
    handcraft.add_argument("--count", type=_positive_int, required=True)
    handcraft.add_argument("--apply", action="store_true")
    handcraft.add_argument("--single-model", choices=MODEL_IDS)

    scale = subparsers.add_parser("scale")
    scale.add_argument("--root", type=Path, required=True)
    scale.add_argument("--lemmatizer-root", type=Path, required=True)
    scale.add_argument(
        "--languages",
        nargs="+",
        choices=(*LANGUAGE_CODES, *LANGUAGE_DIRECTORIES),
        default=list(LANGUAGE_CODES),
    )
    scale.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    scale.add_argument(
        "--phase",
        choices=["cefr", "handcraft", "both"],
        default="both",
    )
    scale.add_argument("--handcraft-count", type=_positive_int, default=20)
    scale.add_argument("--apply", action="store_true")
    scale.add_argument("--refill-to-target", action="store_true")
    scale.add_argument("--retry-failed", action="store_true")
    scale.add_argument("--single-model", choices=MODEL_IDS)
    scale.add_argument(
        "--batch-concurrency",
        type=_positive_int,
        default=None,
        help=(
            "parallel CEFR row-batches per language/level task "
            f"(default: env or {default_batch_concurrency()}); does not reset "
            "succeeded scale tasks"
        ),
    )

    manual_review = subparsers.add_parser("manual-review")
    manual_review.add_argument("--root", type=Path, required=True)
    manual_review.add_argument(
        "--lang",
        choices=LANGUAGE_DIRECTORIES,
        required=True,
    )
    manual_review.add_argument(
        "--level",
        choices=LEVELS,
        required=True,
    )
    manual_review.add_argument(
        "--input",
        "--source",
        dest="source",
        type=Path,
        required=True,
    )
    manual_review.add_argument("--decisions", type=Path, required=True)
    manual_review.add_argument(
        "--check-other-level-collisions",
        action="store_true",
    )
    manual_review.add_argument("--apply", action="store_true")
    manual_review.add_argument("--append", action="store_true")

    seed_validated = subparsers.add_parser("seed-validated")
    seed_validated.add_argument("--root", type=Path, required=True)
    seed_validated.add_argument("--lang", choices=LANGUAGE_DIRECTORIES, required=True)
    seed_validated.add_argument("--level", choices=LEVELS, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--root", type=Path, default=Path.cwd())
    status.add_argument(
        "--events",
        type=int,
        default=15,
        help="how many recent .gemma_qa/events.jsonl lines to print",
    )

    normalize_english = subparsers.add_parser(
        "normalize-english",
        help=(
            "Rewrite English CEFR CSVs so lemma matches english_lemma; "
            "dedupe lemma+POS"
        ),
    )
    normalize_english.add_argument("--root", type=Path, required=True)
    normalize_english.add_argument(
        "--levels",
        nargs="+",
        choices=LEVELS,
        default=list(LEVELS),
    )
    normalize_english.add_argument(
        "--proposed",
        action="store_true",
        help="normalize *.proposed.csv instead of committed *.csv",
    )

    handcraft_ready = subparsers.add_parser(
        "handcraft-ready",
        help=(
            "Exit 0 if committed CEFR CSV is large enough and citation-clean "
            "for handcraft generation"
        ),
    )
    handcraft_ready.add_argument("--vocab-root", type=Path, required=True)
    handcraft_ready.add_argument("--lang", choices=LANGUAGE_CODES, required=True)
    handcraft_ready.add_argument("--level", choices=LEVELS, required=True)
    handcraft_ready.add_argument(
        "--count",
        type=_positive_int,
        default=20,
        help="handcraft sentence count (default 20)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root_for_log = getattr(args, "root", None) or getattr(args, "vocab_root", None)
    if root_for_log is not None:
        configure(root=Path(root_for_log))
    else:
        configure()
    event(
        "cli.start",
        command=args.command,
        run=run_id(),
        argv=[str(item) for item in (argv or [])][:40] or None,
    )
    if args.command == "normalize-english":
        suffix = ".proposed.csv" if args.proposed else ".csv"
        total_rewritten = 0
        total_dupes = 0
        for level in args.levels:
            path = args.root / "english" / f"{level}{suffix}"
            if not path.exists() and args.proposed:
                path = args.root / "english" / f"{level}.proposed.csv"
            if not path.exists():
                print(f"skip missing {path}")
                continue
            stats = normalize_english_csv_file(path)
            total_rewritten += stats["rewritten"]
            total_dupes += stats["dropped_dupes"]
            print(
                f"{path}: in={stats['rows_in']} out={stats['rows_out']} "
                f"rewritten={stats['rewritten']} dupes={stats['dropped_dupes']}"
            )
            event(
                "normalize_english.ok",
                level_name=level,
                path=str(path),
                **stats,
            )
        print(f"total_rewritten={total_rewritten} total_dupes={total_dupes}")
        return 0
    if args.command == "handcraft-ready":
        report = assess_handcraft_ready(
            vocab_root=args.vocab_root,
            lang=args.lang,
            level=args.level,
            count=args.count,
        )
        print(report.summary())
        event(
            "handcraft_ready",
            lang=report.lang,
            level_name=report.level,
            ready=report.ready,
            row_count=report.row_count,
            issue_count=len(report.issues),
        )
        return 0 if report.ready else 1
    if args.command == "status":
        ledger = Ledger(args.root / ".gemma_qa" / "ledger.sqlite3")
        status = ledger.status()
        scale_state = ScaleState(args.root / ".gemma_qa" / "scale.sqlite3")
        scale_counts = scale_state.counts()
        total = sum(scale_counts.values())
        done = scale_counts["succeeded"] + scale_counts["failed"]
        remaining = scale_counts["pending"] + scale_counts["running"]
        pct = (100.0 * done / total) if total else 0.0
        print(
            f"checkpoints={status.checkpoints} "
            f"prompt_tokens={status.prompt_tokens} "
            f"candidate_tokens={status.candidate_tokens} "
            f"total_tokens={status.total_tokens}"
        )
        print(
            f"PROGRESS scale {done}/{total} ({pct:5.1f}%) "
            f"succeeded={scale_counts['succeeded']} "
            f"failed={scale_counts['failed']} "
            f"running={scale_counts['running']} "
            f"pending={scale_counts['pending']} "
            f"remaining={remaining}"
        )
        running = [record.key for record in scale_state.list_by_status("running")]
        if running:
            print(f"running_tasks={' '.join(running)}")
        events_path = args.root / ".gemma_qa" / "events.jsonl"
        recent = recent_events(events_path, limit=args.events if hasattr(args, "events") else 15)
        if recent:
            print(f"events_path={events_path} recent={len(recent)}")
            for item in recent:
                kind = item.get("kind", "?")
                iso = item.get("iso", "")
                level = item.get("level", "")
                model = item.get("model", "")
                batch = item.get("batch_id", "")
                err = item.get("error", "")
                wait = item.get("wait_s", "")
                progress = ""
                if item.get("done") is not None and item.get("total") is not None:
                    progress = f" done={item.get('done')}/{item.get('total')}"
                if item.get("batch_count") is not None:
                    progress += (
                        f" batch={item.get('attempt')}/{item.get('batch_count')}"
                    )
                print(
                    f"  {iso} {level} {kind}"
                    f"{' model=' + str(model) if model else ''}"
                    f"{' batch=' + str(batch) if batch else ''}"
                    f"{progress}"
                    f"{' wait_s=' + str(wait) if wait != '' else ''}"
                    f"{' err=' + str(err)[:120] if err else ''}"
                )
        else:
            print(f"events_path={events_path} recent=0")
        ledger.close()
        return 0
    if args.command == "manual-review":
        result = run_manual_review(
            root=args.root,
            lang=args.lang,
            level=args.level,
            source=args.source,
            decisions_directory=args.decisions,
            apply=args.apply,
            append=args.append,
            check_other_level_collisions=args.check_other_level_collisions,
        )
        print(result)
        return 0
    if args.command == "seed-validated":
        store = ValidatedStore(validated_store_path(args.root))
        try:
            store.seed_from_csv(args.root, lang=args.lang, level=args.level)
            print(store.count(args.lang, args.level))
        finally:
            store.close()
        return 0
    if args.command == "refill":
        state_directory = args.root / ".gemma_qa"
        ledger = Ledger(state_directory / "ledger.sqlite3")
        client = _build_unified_client()
        try:
            output = run_cefr_gap_refill(
                root=args.root,
                lang=args.lang,
                level=args.level,
                client=client,
                ledger=ledger,
                single_model=args.single_model,
                reject_decisions_dir=args.reject_decisions,
            )
            print(output)
            return 0
        finally:
            client.close()
            ledger.close()
    if args.command == "scale":
        phases = ("cefr", "handcraft") if args.phase == "both" else (args.phase,)
        result = run_scale(
            ScaleConfig(
                root=args.root,
                lemmatizer_root=args.lemmatizer_root,
                languages=tuple(
                    get_language(language).code for language in args.languages
                ),
                levels=tuple(args.levels),
                phases=phases,
                handcraft_count=args.handcraft_count,
                apply=args.apply,
                refill_to_target=args.refill_to_target,
                retry_failed=args.retry_failed,
                single_model=args.single_model,
                batch_concurrency=args.batch_concurrency,
            )
        )
        print(
            f"succeeded={result.succeeded} failed={result.failed} "
            f"skipped={result.skipped}"
        )
        return result.exit_code
    state_root = args.root if args.command == "cefr" else args.vocab_root
    state_directory = state_root / ".gemma_qa"
    ledger = Ledger(state_directory / "ledger.sqlite3")
    client = _build_unified_client()
    try:
        if args.command == "cefr":
            output = run_cefr(
                root=args.root,
                lang=args.lang,
                level=args.level,
                client=client,
                ledger=ledger,
                limit=args.limit,
                apply=args.apply,
                single_model=args.single_model,
                refill_to_target=args.refill_to_target,
                batch_concurrency=args.batch_concurrency,
            )
        else:
            output = run_handcraft(
                vocab_root=args.vocab_root,
                lemmatizer_root=args.lemmatizer_root,
                lang=args.lang,
                level=args.level,
                count=args.count,
                client=client,
                ledger=ledger,
                apply=args.apply,
                single_model=args.single_model,
            )
        print(output)
        return 0
    finally:
        client.close()
        ledger.close()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed
