"""`aced` — analyze a bouldering video from the command line.

The same pipeline the API worker runs, without the service around it: point it at a file and
it writes the report the service would have produced for that video and that bundle. That
equivalence is the point of the command, not a convenience — it is what makes the analysis
reproducible off a laptop, and what section 11's end-to-end test compares the API against.

The report goes to stdout by default so it can be piped straight into `jq`; everything else
the command has to say goes to stderr, so redirecting stdout to a file yields JSON and only
JSON.

Exit codes are distinct because the remedies are:

    0  the video was analyzed — including the valid outcome of no climber being found
    2  the command line itself was wrong (argparse)
    3  the video could not be read
    4  the bundle could not be loaded
    5  the analysis succeeded but the overlay could not be rendered
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from aced import __version__
from aced.analysis import analyze
from aced.inference.bundle import BUNDLE_ENV_VAR, BundleError, default_bundle_path, load_bundle
from aced.pose.extractor import UnreadableVideoError
from aced.report.types import AnalysisReport

EXIT_OK = 0
EXIT_UNREADABLE_VIDEO = 3
EXIT_BUNDLE_ERROR = 4
EXIT_RENDER_ERROR = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aced",
        description="Detect arm-technique faults in a bouldering video.",
    )
    parser.add_argument("--version", action="version", version=f"aced {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subcommands.add_parser(
        "analyze",
        help="analyze a video file and write a JSON report",
        description="Analyze a local video file and write the report as JSON.",
    )
    analyze_parser.add_argument("video", type=Path, help="path to the video file")
    analyze_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write the report here instead of to stdout",
    )
    analyze_parser.add_argument(
        "-b",
        "--bundle",
        type=Path,
        help=(
            "model bundle directory to analyze with; defaults to the newest shipped "
            f"bundle, or to ${BUNDLE_ENV_VAR} when that is set"
        ),
    )
    analyze_parser.add_argument(
        "--overlay",
        type=Path,
        metavar="PATH",
        help="also render a skeleton-overlay video here (off unless asked for)",
    )
    analyze_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress the human-readable summary on stderr",
    )
    return parser


def _write_report(report: AnalysisReport, output: Path | None) -> None:
    payload = report.to_json(indent=2)
    if output is None:
        sys.stdout.write(payload + "\n")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n")


def _summarize(report: AnalysisReport, output: Path | None, overlay: Path | None) -> None:
    print(report.summary.message, file=sys.stderr)
    for flag in report.quality_flags:
        print(f"  warning: {flag.message}", file=sys.stderr)
    if output is not None:
        print(f"  report:  {output}", file=sys.stderr)
    if overlay is not None:
        print(f"  overlay: {overlay}", file=sys.stderr)


def run_analyze(args: argparse.Namespace) -> int:
    bundle_path = args.bundle
    try:
        if bundle_path is None:
            bundle_path = default_bundle_path()
        bundle = load_bundle(bundle_path)
    except BundleError as exc:
        print(f"aced: {exc}", file=sys.stderr)
        return EXIT_BUNDLE_ERROR
    except Exception as exc:
        # Deserialising a model is the one step here that can fail in as many ways as there
        # are model formats: a truncated `.keras` archive, a joblib pickle written under an
        # incompatible scikit-learn, a permissions error. The user's remedy is the same for
        # all of them and a traceback is not it, so they are reported as what they are.
        print(f"aced: could not load bundle {bundle_path or '(default)'}: {exc}", file=sys.stderr)
        return EXIT_BUNDLE_ERROR

    try:
        analysis = analyze(args.video, bundle)
    except UnreadableVideoError as exc:
        print(f"aced: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE_VIDEO

    # The report is written before the overlay is rendered: rendering is the slow, optional,
    # codec-dependent half, and a failure there must not cost the analysis that succeeded.
    _write_report(analysis.report, args.output)

    if args.overlay is not None:
        from aced.render.overlay import OverlayRenderError, render_overlay

        try:
            render_overlay(
                args.video,
                analysis.landmarks,
                analysis.frame_classes,
                args.overlay,
            )
        except (OverlayRenderError, UnreadableVideoError) as exc:
            print(f"aced: the report was written, but the overlay failed: {exc}", file=sys.stderr)
            return EXIT_RENDER_ERROR

    if not args.quiet:
        _summarize(analysis.report, args.output, args.overlay)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        return run_analyze(args)
    raise AssertionError(f"unreachable: argparse accepted unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
