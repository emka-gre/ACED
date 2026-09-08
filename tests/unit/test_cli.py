"""The `aced` command line: argument handling, output routing, and exit codes.

The pipeline is stubbed out throughout. What the CLI is responsible for is choosing a bundle,
putting the report somewhere, and turning each kind of failure into an exit code a script can
branch on — none of which needs a real model or a real video, and all of which needs to be
pinned down because a shell script is the one caller that cannot ask what went wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aced.analysis import Analysis
from aced.cli import main as cli
from aced.inference.bundle import (
    BundleError,
    BundleManifest,
    Confidence,
    ModelBundle,
    ModelKind,
    Scaling,
)
from aced.pose.extractor import UnreadableVideoError
from aced.pose.landmarks import QualitySignals, VideoMeta
from aced.render.overlay import OverlayRenderError, OverlayResult
from aced.report.build import build_report
from aced.report.types import AnalysisReport

VIDEO = "climb.mov"


def manifest() -> BundleManifest:
    return BundleManifest(
        id="test-bundle",
        version="1.0.0",
        model_kind=ModelKind.FLAT,
        model_file="model.joblib",
        features=("left_elbow_angle",),
        window_length=60,
        stride=20,
        landmarks=(),
        threshold=0.5,
        scaling=Scaling.INTERNAL,
        confidence=Confidence.NONE,
    )


def report(*, frames_detected: int = 600) -> AnalysisReport:
    return build_report(
        video=VideoMeta(fps=30.0, frame_count=600, width=1080, height=1920),
        quality=QualitySignals(
            frames_read=600, frames_detected=frames_detected, mean_visibility=0.9
        ),
        windows=[],
        results=[],
        manifest=manifest(),
    )


def analysis(**kwargs: Any) -> Analysis:
    return Analysis(report=report(**kwargs), landmarks=(), frame_classes={}, windows=())


def loaded_bundle() -> ModelBundle:
    return ModelBundle(manifest=manifest(), model=object(), scaler=None, path=Path("bundle"))


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A CLI whose pipeline succeeds, with a record of what it was asked to do."""
    calls: dict[str, Any] = {}

    def fake_load(path: Path) -> ModelBundle:
        calls["bundle_path"] = path
        return loaded_bundle()

    def fake_default() -> Path:
        calls["default_consulted"] = True
        return Path("models/default/v1")

    def fake_analyze(source: Any, bundle: ModelBundle, **kwargs: Any) -> Analysis:
        calls["source"] = source
        return analysis()

    monkeypatch.setattr(cli, "load_bundle", fake_load)
    monkeypatch.setattr(cli, "default_bundle_path", fake_default)
    monkeypatch.setattr(cli, "analyze", fake_analyze)
    return calls


class TestReportOutput:
    def test_the_report_goes_to_stdout_by_default(
        self, stubbed: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["analyze", VIDEO]) == cli.EXIT_OK

        payload = json.loads(capsys.readouterr().out)
        assert payload["summary"]["message"]
        assert payload["model"]["bundle_id"] == "test-bundle"

    def test_stdout_carries_json_and_nothing_else(
        self, stubbed: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`aced analyze x.mov > report.json` has to yield a parseable file, so every
        human-readable word the command emits belongs on stderr."""
        cli.main(["analyze", VIDEO])
        captured = capsys.readouterr()

        json.loads(captured.out)
        assert captured.err.strip()

    def test_output_option_writes_a_file_and_leaves_stdout_empty(
        self, stubbed: dict[str, Any], capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        destination = tmp_path / "reports" / "climb.json"

        assert cli.main(["analyze", VIDEO, "-o", str(destination)]) == cli.EXIT_OK

        assert json.loads(destination.read_text())["summary"]
        assert capsys.readouterr().out == ""

    def test_the_summary_names_the_quality_flags(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "load_bundle", lambda path: loaded_bundle())
        monkeypatch.setattr(cli, "default_bundle_path", lambda: Path("bundle"))
        monkeypatch.setattr(
            cli, "analyze", lambda source, bundle, **kw: analysis(frames_detected=100)
        )

        cli.main(["analyze", VIDEO])

        assert "warning:" in capsys.readouterr().err

    def test_quiet_suppresses_the_summary_but_not_the_report(
        self, stubbed: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["analyze", VIDEO, "--quiet"])
        captured = capsys.readouterr()

        json.loads(captured.out)
        assert captured.err == ""


class TestBundleSelection:
    def test_an_explicit_bundle_is_used_verbatim(self, stubbed: dict[str, Any]) -> None:
        cli.main(["analyze", VIDEO, "--bundle", "/models/svm/v1"])

        assert stubbed["bundle_path"] == Path("/models/svm/v1")
        assert "default_consulted" not in stubbed

    def test_the_default_bundle_is_used_when_none_is_named(self, stubbed: dict[str, Any]) -> None:
        cli.main(["analyze", VIDEO])

        assert stubbed["default_consulted"]
        assert stubbed["bundle_path"] == Path("models/default/v1")


class TestFailureReporting:
    def test_an_unreadable_video_exits_non_zero_with_a_message(
        self,
        stubbed: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def unreadable(source: Any, bundle: Any, **kwargs: Any) -> Analysis:
            raise UnreadableVideoError("could not open climb.mov as video")

        monkeypatch.setattr(cli, "analyze", unreadable)

        assert cli.main(["analyze", VIDEO]) == cli.EXIT_UNREADABLE_VIDEO
        captured = capsys.readouterr()
        assert "could not open climb.mov" in captured.err
        assert captured.out == ""

    def test_a_bundle_error_exits_non_zero_with_a_message(
        self,
        stubbed: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def broken(path: Path) -> ModelBundle:
            raise BundleError("bundle lstm declares external scaling but ships no scaler")

        monkeypatch.setattr(cli, "load_bundle", broken)

        assert cli.main(["analyze", VIDEO]) == cli.EXIT_BUNDLE_ERROR
        assert "ships no scaler" in capsys.readouterr().err

    def test_an_unexpected_load_failure_is_reported_not_raised(
        self,
        stubbed: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A truncated `.keras` file raises from deep inside Keras. The user's remedy is the
        same as for any other bad bundle, and a traceback is not a message."""

        def exploding(path: Path) -> ModelBundle:
            raise ValueError("File not found: filepath=model.keras")

        monkeypatch.setattr(cli, "load_bundle", exploding)

        assert cli.main(["analyze", VIDEO]) == cli.EXIT_BUNDLE_ERROR
        assert "could not load bundle" in capsys.readouterr().err

    def test_the_two_failures_are_told_apart(
        self, stubbed: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A script retrying with another bundle must not retry on an unreadable video."""
        assert cli.EXIT_UNREADABLE_VIDEO != cli.EXIT_BUNDLE_ERROR

    def test_an_unknown_command_line_is_argparse_usage(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            cli.main([])
        assert exit_info.value.code == 2

    def test_no_pose_detected_is_a_success(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A readable video with no climber is an answer, not an error — the report says so
        and the command exits clean."""
        monkeypatch.setattr(cli, "load_bundle", lambda path: loaded_bundle())
        monkeypatch.setattr(cli, "default_bundle_path", lambda: Path("bundle"))
        monkeypatch.setattr(
            cli, "analyze", lambda source, bundle, **kw: analysis(frames_detected=0)
        )

        assert cli.main(["analyze", VIDEO]) == cli.EXIT_OK
        assert json.loads(capsys.readouterr().out)["quality_flags"]


class TestOverlay:
    def test_no_overlay_is_rendered_unless_asked_for(
        self, stubbed: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import aced.render.overlay as overlay_module

        def forbidden(*args: Any, **kwargs: Any) -> OverlayResult:
            raise AssertionError("rendering must not happen without --overlay")

        monkeypatch.setattr(overlay_module, "render_overlay", forbidden)

        assert cli.main(["analyze", VIDEO]) == cli.EXIT_OK

    def test_the_overlay_is_rendered_where_asked(
        self, stubbed: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import aced.render.overlay as overlay_module

        seen: dict[str, Any] = {}

        def record(source: Any, landmarks: Any, classes: Any, destination: Any) -> OverlayResult:
            seen["source"] = source
            seen["destination"] = destination
            return OverlayResult(Path(destination), 1, 1, 1, 0, 0)

        monkeypatch.setattr(overlay_module, "render_overlay", record)
        destination = tmp_path / "overlay.mp4"

        assert cli.main(["analyze", VIDEO, "--overlay", str(destination)]) == cli.EXIT_OK
        assert seen["destination"] == destination
        assert str(seen["source"]) == VIDEO

    def test_a_render_failure_does_not_cost_the_report(
        self,
        stubbed: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """Rendering is the slow, codec-dependent half. If it fails, the analysis that
        already succeeded is still on disk and the exit code says which half went wrong."""
        import aced.render.overlay as overlay_module

        def failing(*args: Any, **kwargs: Any) -> OverlayResult:
            raise OverlayRenderError("codec mp4v unavailable")

        monkeypatch.setattr(overlay_module, "render_overlay", failing)
        destination = tmp_path / "report.json"

        code = cli.main(
            ["analyze", VIDEO, "-o", str(destination), "--overlay", str(tmp_path / "o.mp4")]
        )

        assert code == cli.EXIT_RENDER_ERROR
        assert json.loads(destination.read_text())["summary"]
        assert "the report was written" in capsys.readouterr().err
