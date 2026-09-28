"""Multi-format mesh I/O for volume and surface meshes.

Provides one entry point -- ``load_mesh`` / ``save_mesh`` -- that dispatches
by file extension across every format this project needs: LS-DYNA ``.k``,
Abaqus ``.inp``, FEBio ``.feb``, and STL/OBJ surfaces (AGENTS.md section 1.4).

--------------------------------------------------------------------------
Vendored from: C:\\Users\\lhudson\\Documents\\Morphing\\fe_personalization\\mesh_io\\__init__.py
Vendored on:   2026-09-18.
Local changes: dropped ``compact`` from ``__all__`` (not vendored -- no
               caller in this project's design references it; add it back
               if that changes). Everything else is unmodified.
See AGENTS.md "Vendoring policy".
--------------------------------------------------------------------------
"""

from pathlib import Path

from .base import Mesh, SurfaceMesh, VolumeMesh
from .febio import load_febio, save_febio
from .lsdyna import load_lsdyna, save_lsdyna
from .abaqus import load_abaqus, save_abaqus
from .surface import load_stl, load_obj, save_stl, save_obj

_FORMAT_BY_EXT = {
    ".feb": "febio",
    ".k": "lsdyna",
    ".inp": "abaqus",
    ".stl": "stl",
    ".obj": "obj",
}

_LOADERS = {
    "febio": load_febio,
    "lsdyna": load_lsdyna,
    "abaqus": load_abaqus,
    "stl": load_stl,
    "obj": load_obj,
}

_SAVERS = {
    "febio": save_febio,
    "lsdyna": save_lsdyna,
    "abaqus": save_abaqus,
    "stl": save_stl,
    "obj": save_obj,
}


def _resolve_format(filepath, format: str = None) -> str:
    if format is not None:
        return format
    ext = Path(filepath).suffix.lower()
    resolved = _FORMAT_BY_EXT.get(ext)
    if resolved is None:
        raise ValueError(f"Unknown file extension: {ext}")
    return resolved


def load_mesh(filepath: str, format: str = None):
    """Load a mesh from file, auto-detecting format from the extension.

    Args:
        filepath: Path to mesh file.
        format: Optional format override ('febio', 'lsdyna', 'abaqus',
            'stl', 'obj').

    Returns:
        Mesh object (VolumeMesh or SurfaceMesh depending on format).
    """
    fmt = _resolve_format(filepath, format)
    return _LOADERS[fmt](filepath)


def save_mesh(mesh, filepath: str, format: str = None, source_filepath: str = None):
    """Save a mesh to file, auto-detecting format from the extension.

    Args:
        mesh: Mesh object to save.
        filepath: Path to output file.
        format: Optional format override.
        source_filepath: Original file to use as template (preserves
            structure for FEBio/LS-DYNA saves).
    """
    fmt = _resolve_format(filepath, format)
    saver = _SAVERS[fmt]

    if source_filepath and fmt == "febio":
        if Path(source_filepath).suffix.lower() in (".feb", ".febio"):
            return saver(mesh, filepath, source_filepath=source_filepath)
    if source_filepath and fmt == "lsdyna":
        if Path(source_filepath).suffix.lower() == ".k":
            return saver(mesh, filepath, source_filepath=source_filepath)

    return saver(mesh, filepath)


__all__ = [
    "Mesh",
    "SurfaceMesh",
    "VolumeMesh",
    "load_mesh",
    "save_mesh",
    "load_febio",
    "save_febio",
    "load_lsdyna",
    "save_lsdyna",
    "load_abaqus",
    "save_abaqus",
    "load_stl",
    "load_obj",
    "save_stl",
    "save_obj",
]
