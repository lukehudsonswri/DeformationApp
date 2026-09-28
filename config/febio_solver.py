"""Locate a working FEBio4 console-solver executable.

Resolution order: a copy bundled directly in this repo first, then an
environment-variable override, then PATH, then common FEBioStudio install
locations. Mirrors (and stays in sync with) the resolution order used by
the Digital Twin project's own ``ik_check/build_hip_flexion_sim.py``
(``_find_febio_executable``).

Why a bundled copy is worth keeping directly in this repo
------------------------------------------------------------
This project's mesh-validity gate (``remesh/febio_smoke_test.py``) needs a
working ``febio4.exe`` to run at all -- it should not silently depend on a
sibling project's checkout existing on disk, or on FEBioStudio being
installed system-wide. So the standalone console-solver copy (originally
assembled for the Digital Twin project's ``ik_check/`` -- see
``febio_bin/README.md`` here for how it was built: recursive PE
import-table resolution via ``pefile``, not the whole FEBioStudio ``bin/``
folder, which also carries ~300 MB of GUI/Qt/FFmpeg/CAD libraries the
console solver never touches) is copied wholesale into ``febio_bin/`` at
this repo's root, so this project is self-contained and "packaged
together" rather than reaching across to another project's directory.

**Not committed to git** (~146 MB of binaries -- see ``.gitignore``; this
mirrors the exact same policy the Digital Twin project uses for its own
copy, for the same reason: these are large third-party redistributable
binaries, not project source). If ``febio_bin/`` is empty/missing here,
resolution falls through to ``$FEBIO_BIN_ROOT``, then PATH, then common
FEBioStudio install locations -- so the smoke test still degrades to a
clear "not found" skip rather than a hard crash when the folder hasn't been
populated on a given machine (see ``febio_bin/README.md``'s own
regeneration instructions).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

# This repo's own bundled copy -- see febio_bin/README.md.
_LOCAL_FEBIO_BIN_DIR = Path(__file__).resolve().parent.parent / "febio_bin"

# Fallback if the local copy isn't populated on this machine: the Digital
# Twin project's own copy, which this repo's febio_bin/ was copied from.
_FALLBACK_FEBIO_BIN_DIR = Path(
    r"C:\Users\lhudson\Desktop\Projects\Digital Twin\human-digital-twin"
    r"\model_correspondance\ik_check\febio_bin"
)

# Same fallback list ik_check/build_hip_flexion_sim.py uses, kept in sync.
_FEBIO_EXECUTABLE_CANDIDATES = [
    Path(r"C:\Program Files\FEBioStudio\bin\febio4.exe"),
    Path(r"C:\Program Files\FEBioStudio2\bin\febio4.exe"),
    Path(r"C:\Program Files\FEBio\bin\febio4.exe"),
    Path("/usr/local/bin/febio4"),
    Path("/opt/febio/bin/febio4"),
]


def _febio_bin_dirs() -> list:
    """Every febio_bin/-style directory to try, in priority order."""
    override = os.environ.get("FEBIO_BIN_ROOT")
    dirs = []
    if override:
        dirs.append(Path(override))
    dirs.append(_LOCAL_FEBIO_BIN_DIR)
    dirs.append(_FALLBACK_FEBIO_BIN_DIR)
    return dirs


def find_febio_executable() -> Optional[Path]:
    """Return a usable ``febio4`` executable path, or ``None`` if not found.

    Resolution order:
        1. ``$FEBIO_BIN_ROOT/febio4.exe``, if that env var is set.
        2. This repo's own bundled ``febio_bin/febio4.exe``.
        3. The Digital Twin project's ``ik_check/febio_bin/febio4.exe``
           (fallback, in case this repo's copy hasn't been populated).
        4. ``febio4`` on PATH.
        5. Common FEBioStudio install locations.

    Does not raise -- callers that need FEBio (the smoke test) should turn a
    ``None`` result into a clear skip/error themselves, since "FEBio isn't
    installed on this machine" is a different failure mode than "FEBio
    rejected this mesh".
    """
    for bin_dir in _febio_bin_dirs():
        candidate = bin_dir / "febio4.exe"
        if candidate.is_file():
            return candidate

    which_result = shutil.which("febio4")
    if which_result:
        return Path(which_result)

    for candidate in _FEBIO_EXECUTABLE_CANDIDATES:
        if candidate.is_file():
            return candidate

    return None
