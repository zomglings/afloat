"""Run records, source snapshots, and content hashes."""

import hashlib
import json
import platform
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path


def write_json(path: Path, data: object) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def append_json(path: Path, data: object) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_source(output: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source = output / "source"
    source.mkdir()
    for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE"):
        shutil.copyfile(root / name, source / name)
    shutil.copytree(
        root / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    revision: str | None = None
    dirty: bool | None = None
    if (root / ".git").exists():
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                text=True,
                capture_output=True,
            ).stdout
        )
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                name: version(name) for name in ("torch", "numpy", "matplotlib")
            },
            "git_revision": revision,
            "git_dirty": dirty,
            "source_hashes": {
                str(path.relative_to(source)): sha256(path)
                for path in sorted(source.rglob("*"))
                if path.is_file()
            },
        },
    )


def record_manifest(output: Path) -> None:
    write_json(
        output / "manifest.json",
        {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
