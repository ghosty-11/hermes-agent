"""Strict metadata reads distinguish an absent entry from an unavailable target."""
from __future__ import annotations

from pathlib import Path
import stat


def strict_file_present(path: Path) -> bool:
    """Allow absent first-run files; refuse dangling links, wrong types and I/O errors."""
    try:
        info = path.stat()
    except FileNotFoundError as error:
        try:
            path.lstat()
        except FileNotFoundError:
            return False
        raise ValueError(f"configuration entry has no readable target: {path}") from error
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"configuration entry is not a regular file: {path}")
    return True

def strict_directory_present(path: Path) -> bool:
    """Allow an absent directory, but never a dangling link or failed metadata."""
    try:
        info = path.stat()
    except FileNotFoundError as error:
        try:
            path.lstat()
        except FileNotFoundError:
            return False
        raise ValueError(f"directory has no readable target: {path}") from error
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"entry is not a directory: {path}")
    return True
