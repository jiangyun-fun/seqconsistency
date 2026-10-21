"""Command-line interface: a thin wrapper around the library pipeline.

Exit codes (spec §9):
- 0: run completed; every group is COMPATIBLE.
- 1: run completed; one or more groups are not COMPATIBLE (scientific result).
- 2: input, configuration, dependency, or execution failure.

Scientific outcomes (REVIEW / INCOMPATIBLE / INSUFFICIENT_EVIDENCE) are results in
the output tables and exit code 1 — never exit code 2.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from seqconsistency import __version__
from seqconsistency.errors import SeqConsistencyError
from seqconsistency.pipeline import (
    EXIT_FAILURE,
    CheckConfig,
    run_check,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seqconsistency",
        description=(
            "Assess structural validity, technical consistency, and pooling "
            "suitability of sequencing datasets. Every persistent output requires "
            "its own explicit path; no files are written implicitly."
        ),
    )
    parser.add_argument("--version", action="version", version=f"seqconsistency {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check", help="validate, compare, and decide on one manifest's pooling groups"
    )
    check.add_argument(
        "--manifest", required=True, type=Path, help="input manifest (.parquet or .tsv)"
    )
    check.add_argument("--policy", required=True, type=Path, help="pooling policy YAML")
    check.add_argument(
        "--metrics-out", required=True, type=Path, help="metrics Parquet output path"
    )
    check.add_argument(
        "--distributions-out", required=True, type=Path, help="distributions Parquet output path"
    )
    check.add_argument(
        "--comparisons-out", required=True, type=Path, help="comparisons Parquet output path"
    )
    check.add_argument(
        "--findings-out", required=True, type=Path, help="findings Parquet output path"
    )
    check.add_argument(
        "--decisions-out", required=True, type=Path, help="decisions Parquet output path"
    )
    check.add_argument(
        "--provenance-out", required=True, type=Path, help="provenance JSON output path"
    )
    check.add_argument(
        "--report-out", type=Path, default=None, help="optional self-contained HTML report path"
    )
    check.add_argument(
        "--workers", type=int, default=4, help="maximum simultaneous dataset scans (default 4)"
    )
    check.add_argument(
        "--threads",
        type=int,
        default=4,
        help="total logical CPU budget shared by workers and the main process (default 4)",
    )
    check.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="maximum reservoir sample per dataset "
        "(sampling unit: read pair for PE FASTQ, eligible "
        "primary record for BAM; default 50000)",
    )
    check.add_argument(
        "--seed",
        type=int,
        default=42,
        help="master sampling seed; per-dataset seeds derive deterministically (default 42)",
    )
    check.add_argument(
        "--max-pairs",
        type=int,
        default=10000,
        help="abort before processing when all-pairs comparison "
        "would exceed this many dataset pairs (default 10000)",
    )
    check.add_argument(
        "--overwrite", action="store_true", help="allow replacing existing output files"
    )
    check.add_argument(
        "--hash-inputs",
        action="store_true",
        help="also SHA-256 the input files into provenance (expensive on large inputs)",
    )
    check.add_argument("--debug", action="store_true", help="print full tracebacks on failure")
    check.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default INFO)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "check":
        config = CheckConfig(
            manifest=args.manifest,
            policy=args.policy,
            metrics_out=args.metrics_out,
            distributions_out=args.distributions_out,
            comparisons_out=args.comparisons_out,
            findings_out=args.findings_out,
            decisions_out=args.decisions_out,
            provenance_out=args.provenance_out,
            report_out=args.report_out,
            workers=args.workers,
            threads=args.threads,
            sample_size=args.sample_size,
            seed=args.seed,
            max_pairs=args.max_pairs,
            overwrite=args.overwrite,
            hash_inputs=args.hash_inputs,
            debug=args.debug,
            cli_args={k: str(v) for k, v in vars(args).items()},
        )
        if config.sample_size < 0:
            print("seqconsistency: --sample-size must be >= 0", file=sys.stderr)
            return EXIT_FAILURE
        if config.seed < 0:
            print("seqconsistency: --seed must be >= 0", file=sys.stderr)
            return EXIT_FAILURE
        if config.workers < 1 or config.threads < 1:
            print("seqconsistency: --workers and --threads must be >= 1", file=sys.stderr)
            return EXIT_FAILURE
        if config.max_pairs < 0:
            print("seqconsistency: --max-pairs must be >= 0", file=sys.stderr)
            return EXIT_FAILURE
        try:
            return run_check(config)
        except SeqConsistencyError as exc:
            if config.debug:
                raise
            ctx = f" [{exc.context}]" if exc.context else ""
            print(f"seqconsistency: error: {exc.message}{ctx}", file=sys.stderr)
            return EXIT_FAILURE
        except KeyboardInterrupt:
            print("seqconsistency: interrupted", file=sys.stderr)
            return 130
    parser.error(f"unknown command {args.command!r}")
    return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
