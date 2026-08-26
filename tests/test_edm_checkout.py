"""EDM checkout installation without network access."""

from pathlib import Path

import pytest

from specdiff import edm_checkout


def make_checkout(path: Path) -> None:
    """Create the minimum directory structure accepted as an EDM checkout."""
    (path / "dnnlib").mkdir(parents=True)
    (path / "dnnlib/__init__.py").touch()
    (path / "torch_utils").mkdir()
    (path / "torch_utils/persistence.py").touch()


def test_default_checkout_is_inside_the_source_tree():
    assert edm_checkout.default_edm_checkout() == (
        Path(__file__).resolve().parents[1] / "edm"
    )


def test_existing_checkout_is_reused(tmp_path, monkeypatch):
    destination = tmp_path / "edm"
    make_checkout(destination)
    monkeypatch.setattr(
        edm_checkout.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("git must not run"),
    )

    assert edm_checkout.download_edm(destination) == destination


def test_download_clones_and_checks_out_the_requested_revision(tmp_path, monkeypatch):
    destination = tmp_path / "edm"
    calls = []

    def fake_run(command, *, check):
        assert check is True
        calls.append(command)
        if command[1] == "clone":
            make_checkout(Path(command[-1]))

    monkeypatch.setattr(edm_checkout.shutil, "which", lambda executable: "/usr/bin/git")
    monkeypatch.setattr(edm_checkout.subprocess, "run", fake_run)

    result = edm_checkout.download_edm(
        destination,
        repository="https://example.test/edm.git",
        revision="tested-revision",
    )

    assert result == destination
    assert edm_checkout.is_edm_checkout(destination)
    assert calls[0][0:4] == [
        "git", "clone", "--filter=blob:none", "https://example.test/edm.git"
    ]
    assert calls[1][0:2] == ["git", "-C"]
    assert Path(calls[1][2]).name == "edm"
    assert calls[1][3:] == ["checkout", "--detach", "tested-revision"]


def test_non_checkout_destination_is_refused(tmp_path):
    destination = tmp_path / "edm"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="not an EDM checkout"):
        edm_checkout.download_edm(destination)
