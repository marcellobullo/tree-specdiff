"""Download and locate the NVlabs EDM source checkout used by image experiments.

NVlabs/edm is a source repository rather than an installable Python package.
The :command:`specdiff-download-edm` command places a tested revision in the
project's ``edm/`` directory so its ``dnnlib`` and ``torch_utils`` modules are
available when loading official checkpoints.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence


EDM_REPOSITORY = "https://github.com/NVlabs/edm.git"
EDM_REVISION = "008a4e5316c8e3bfe61a62f874bddba254295afb"


def project_root() -> Path:
    """Return the source root for an editable specdiff installation."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(
            "cannot locate the specdiff source checkout; pass --destination "
            "when using a non-editable installation"
        )
    return root


def default_edm_checkout() -> Path:
    """Return the default ``edm/`` directory in the specdiff source checkout."""
    return project_root() / "edm"


def is_edm_checkout(path: Path) -> bool:
    """Return whether *path* contains the modules required by EDM pickles."""
    return all(
        (path / relative).is_file()
        for relative in ("dnnlib/__init__.py", "torch_utils/persistence.py")
    )


def download_edm(
    destination: Optional[Path] = None,
    *,
    repository: str = EDM_REPOSITORY,
    revision: str = EDM_REVISION,
) -> Path:
    """Clone a tested EDM revision into *destination* and return its path.

    Existing valid checkouts are reused without modification. Cloning occurs in
    a temporary directory and is moved into place only after validation, so a
    failed network operation does not leave a partial checkout at the target.
    """
    destination = (destination or default_edm_checkout()).expanduser().resolve()
    if destination.exists():
        if is_edm_checkout(destination):
            return destination
        raise FileExistsError(
            f"destination exists but is not an EDM checkout: {destination}"
        )
    if shutil.which("git") is None:
        raise RuntimeError("git is required to download NVlabs/edm")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".specdiff-edm-", dir=destination.parent
    ) as temporary:
        checkout = Path(temporary) / "edm"
        subprocess.run(
            ["git", "clone", "--filter=blob:none", repository, str(checkout)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", revision],
            check=True,
        )
        if not is_edm_checkout(checkout):
            raise RuntimeError(
                f"downloaded repository does not contain the EDM modules: {repository}"
            )
        checkout.replace(destination)
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the :command:`specdiff-download-edm` command."""
    parser = argparse.ArgumentParser(
        description="Download NVlabs/edm into the specdiff source checkout."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="checkout directory (default: <specdiff source>/edm)",
    )
    parser.add_argument("--repository", default=EDM_REPOSITORY)
    parser.add_argument("--revision", default=EDM_REVISION)
    args = parser.parse_args(argv)

    destination = args.destination or default_edm_checkout()
    existed = is_edm_checkout(destination.expanduser().resolve())
    path = download_edm(
        destination, repository=args.repository, revision=args.revision
    )
    status = "Using existing" if existed else "Installed"
    print(f"{status} EDM checkout: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
