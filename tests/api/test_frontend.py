"""Serving the built web app from the API.

One container, one origin. The frontend routes on the hash, so every deep link is still a GET
of `/` — which is what makes this a static mount rather than a history fallback each
deployment would have to configure.

Two properties matter and neither is obvious from reading the mount: an API path can never be
shadowed by an asset that happens to share its name, and a service with no built frontend
still starts and still serves the API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from api.config import Settings
from tests.api.conftest import upload


@pytest.fixture
def built_frontend(tmp_path: Path) -> Path:
    """What `npm run build` leaves behind, in miniature."""
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>ACED</title><div id=root></div>")
    (root / "assets" / "index-abc123.js").write_text("console.log('aced');")
    return root


class TestServingTheApp:
    def test_the_app_is_served_at_the_root(self, make_client: Any, built_frontend: Path) -> None:
        client = make_client(web_root=built_frontend)

        response = client.get("/")

        assert response.status_code == 200
        assert "<div id=root>" in response.text

    def test_assets_are_served(self, make_client: Any, built_frontend: Path) -> None:
        client = make_client(web_root=built_frontend)

        response = client.get("/assets/index-abc123.js")

        assert response.status_code == 200
        assert "console.log" in response.text

    def test_an_unknown_path_is_not_found(self, make_client: Any, built_frontend: Path) -> None:
        """Not a fallback to index.html. Routing is hash-based, so a path that is not an
        asset is a mistake, and answering 200 with the app would hide it."""
        client = make_client(web_root=built_frontend)

        assert client.get("/not-an-asset.png").status_code == 404


class TestTheApiKeepsPrecedence:
    def test_api_routes_win_over_the_mount(self, make_client: Any, built_frontend: Path) -> None:
        client = make_client(web_root=built_frontend)

        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/analyses/does-not-exist").status_code == 404

    def test_an_upload_still_works_with_a_frontend_mounted(
        self, make_client: Any, built_frontend: Path, queue: Any, analyze_stub: Any
    ) -> None:
        client = make_client(web_root=built_frontend)

        created = upload(client)
        assert created.status_code == 202
        queue.run_all()

        analysis_id = created.json()["id"]
        assert client.get(f"/analyses/{analysis_id}").json()["state"] == "succeeded"
        assert client.get(f"/analyses/{analysis_id}/report").status_code == 200

    def test_an_asset_cannot_shadow_an_api_path(
        self, make_client: Any, built_frontend: Path
    ) -> None:
        """A file literally named `health` in the build output must not be able to take over
        the health endpoint. The mount is registered last precisely so it cannot."""
        (built_frontend / "health").write_text("not the health endpoint")
        client = make_client(web_root=built_frontend)

        assert client.get("/health").json()["status"] == "ok"


class TestWithoutABuiltFrontend:
    def test_the_service_starts_and_serves_the_api(self, make_client: Any, tmp_path: Path) -> None:
        """The API is useful on its own — the CLI, the parity suite and anyone driving it with
        curl never build the frontend. A missing build is not a failure to start."""
        client = make_client(web_root=tmp_path / "never-built")

        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 404

    def test_a_directory_without_an_index_is_not_mounted(
        self, make_client: Any, tmp_path: Path
    ) -> None:
        empty = tmp_path / "dist"
        empty.mkdir()

        client = make_client(web_root=empty)

        assert client.get("/").status_code == 404
        assert client.get("/health").status_code == 200


class TestConfiguration:
    def test_the_web_root_comes_from_the_environment(self) -> None:
        settings = Settings.from_env({"ACED_WEB_ROOT": "/srv/aced/dist"})
        assert settings.resolved_web_root == Path("/srv/aced/dist")

    def test_it_falls_back_to_the_repository_build_directory(self) -> None:
        settings = Settings.from_env({})
        assert settings.resolved_web_root.name == "dist"
        assert settings.resolved_web_root.parent.name == "web"
