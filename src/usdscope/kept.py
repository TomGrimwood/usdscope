"""What was read from a stage, kept between runs: a scene opened again reads none of its
geometry if none of its layers changed. Kept by what it is, how it was read (its settings),
the stage's layers as they are now (each one's path, size and time changed) and the code that
read it: a layer edited, a setting changed or usdscope changed, it is read afresh.

Kept in the temporary folder (`KEPT`: gone when the machine restarts), the newest `KEEP` of
them and no more than `MOST` bytes; what cannot be kept or read back is read as if never kept."""

from __future__ import annotations

import getpass
import hashlib
import os
import pickle
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pxr import Usd


def _user() -> str:
    """The user's login name as a folder name (letters, digits, `-`, `_`, `.`), or "user"."""
    try:
        name = getpass.getuser()
    except Exception:  # no login name to be found (no USER, LOGNAME or password entry)
        return "user"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "user"


#: Where reads are kept (the temporary folder, one a user), how many (the newest) and how much.
KEPT = Path(tempfile.gettempdir()) / f"usdscope-{_user()}" / "reads"
KEEP, MOST = 32, 2_000_000_000


def kept[T](what: str, stage: Usd.Stage, settings: tuple[Any, ...], read: Callable[[], T]) -> T:
    """`read()`, or what it gave before for `what` of `stage` read with `settings`."""
    path = KEPT / f"{_stamp(what, stage, settings)}.pickle"
    try:
        with path.open("rb") as f:
            found = pickle.load(f)
        os.utime(path)  # newest: kept longest
        return found
    except Exception:  # not kept, or kept by a usdscope that reads differently: read afresh
        pass
    made = read()
    try:
        KEPT.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        with part.open("wb") as f:
            pickle.dump(made, f, protocol=pickle.HIGHEST_PROTOCOL)
        part.replace(path)  # whole, or not at all
        _prune()
    except Exception:
        pass  # read again next time, then
    return made


def _stamp(what: str, stage: Usd.Stage, settings: tuple[Any, ...]) -> str:
    """What was read, how, from what layers as they are now, by what code: a name for it."""
    layers = []
    for layer in stage.GetUsedLayers():
        real = layer.realPath
        if not real:  # in memory (a session layer): stamped by what it holds
            layers.append(layer.ExportToString())
            continue
        try:
            s = os.stat(real)
            layers.append(f"{real}|{s.st_size}|{s.st_mtime_ns}")
        except OSError:
            layers.append(real)
    code = [(p.name, p.stat().st_mtime_ns) for p in sorted(Path(__file__).parent.glob("*.py"))]
    said = repr((what, settings, sorted(layers), code)).encode()
    return hashlib.sha256(said).hexdigest()[:32]


def _prune() -> None:
    """The newest `KEEP` kept while they come to no more than `MOST` bytes, the rest gone."""
    found = sorted(KEPT.glob("*.pickle"), key=lambda p: p.stat().st_mtime, reverse=True)
    total = 0
    for k, path in enumerate(found):
        total += path.stat().st_size
        if k >= KEEP or (k > 0 and total > MOST):
            path.unlink(missing_ok=True)
