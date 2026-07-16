from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .antigravity import AntigravityClient
from .cefr import run_cefr, run_cefr_gap_refill
from .client import GemmaClient
from .config import MODEL_IDS
from .handcraft import run_handcraft
from .languages import LANGUAGE_CODES, LANGUAGE_DIRECTORIES, LEVELS, get_language
from .ledger import Ledger
from .manual_review import run_manual_review
from .quota import QuotaGate
from .routing import UnifiedQaClient
from .scale import ScaleConfig, ScaleState, run_scale
from .validated import ValidatedStore, validated_store_path


def _build_unified_client(quota: QuotaGate) -> UnifiedQaClient:
    gemma = GemmaClient(quota=quota)
    antigravity = AntigravityClient(quota=quota)
    return UnifiedQaClient(gemma=gemma, antigravity=antigravity, quota=quota)


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

    manual_review = subparsers.add_parser("manual-review")
    manual_review.add_argument("--root", type=Path, required=True)
    manual_review.add_argument("--lang", choices=["german"], required=True)
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        ledger = Ledger(args.root / ".gemma_qa" / "ledger.sqlite3")
        status = ledger.status()
        scale_counts = ScaleState(args.root / ".gemma_qa" / "scale.sqlite3").counts()
        print(
            f"checkpoints={status.checkpoints} "
            f"prompt_tokens={status.prompt_tokens} "
            f"candidate_tokens={status.candidate_tokens} "
            f"total_tokens={status.total_tokens} "
            f"scale_pending={scale_counts['pending']} "
            f"scale_running={scale_counts['running']} "
            f"scale_succeeded={scale_counts['succeeded']} "
            f"scale_failed={scale_counts['failed']}"
        )
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
        quota = QuotaGate(state_directory / "quota.sqlite3")
        client = _build_unified_client(quota)
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
            quota.close()
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
    quota = QuotaGate(state_directory / "quota.sqlite3")
    client = _build_unified_client(quota)
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
        quota.close()
        ledger.close()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed
