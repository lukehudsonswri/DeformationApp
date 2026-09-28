"""PPE (personal protective equipment) registry: multi-format mesh loading
and a small metadata wrapper around each PPE item.

**Rigid armored plate only, per explicit instruction** — this project is
scoped to the torso + rigid plate case. The design supports other formats
and future PPE (see below) but nothing here assumes more than one PPE item
exists.

Backed directly by ``core.mesh`` (vendored multi-format mesh I/O -- AGENTS.md
section 1.4): ``.k`` (LS-DYNA), ``.inp`` (Abaqus), ``.feb`` (FEBio) all load
as volume meshes with no extra work. ``.stl`` is surface-only and would need
tetrahedralizing into a solid before it could be treated the same way as the
others (per the decided design) -- **not implemented yet**, since the one
real PPE this project has (``PPE/plate.inp``) is already a volume mesh and
no STL PPE exists to build/verify that path against. Raises
``NotImplementedError`` with a clear message rather than silently mishandling
an `.stl` if one is ever supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.mesh import load_mesh
from core.mesh.base import VolumeMesh

_SURFACE_ONLY_FORMATS = {".stl", ".obj"}


@dataclass(frozen=True)
class PpeSpec:
    """One PPE item's identity and mesh source.

    Args:
        key: short identifier, e.g. ``"armored_plate"``.
        display_name: human-readable name for the GUI.
        mesh_path: path to the PPE's mesh file.
        is_rigid: whether this PPE is treated as a rigid body in the FEBio
            case (AGENTS.md section 2.3: a rigid PPE gets an exact rigid
            transform only, never RBF/deformable morphing). The armored
            plate is rigid.
    """

    key: str
    display_name: str
    mesh_path: Path
    is_rigid: bool = True

    def load(self) -> VolumeMesh:
        """Load this PPE's mesh via the multi-format ``core.mesh`` loader.

        Raises:
            NotImplementedError: for ``.stl``/``.obj`` sources -- surface
                tetrahedralization into a solid is designed (AGENTS.md
                section 1.4) but not implemented, since no STL PPE exists
                yet to build/verify that path against. Implement this only
                when an actual STL PPE needs it, using ``tetgen``
                constrained-Delaunay per the decided design, not before.
        """
        ext = self.mesh_path.suffix.lower()
        if ext in _SURFACE_ONLY_FORMATS:
            raise NotImplementedError(
                f"'{self.key}': {ext} PPE sources need tetrahedralization into a solid "
                "(AGENTS.md section 1.4) -- not implemented; no STL/OBJ PPE exists yet to "
                "build this against. Supply .k/.inp/.feb, or implement the tetgen path first."
            )
        return load_mesh(str(self.mesh_path))


# The one PPE this project is scoped to today. A registry keyed by `key`
# rather than a bare constant so a second PPE is a one-line addition, not a
# structural change -- but nothing downstream should assume more than one
# entry exists yet.
_PPE_REGISTRY = {
    "armored_plate": PpeSpec(
        key="armored_plate",
        display_name="Armored Plate",
        mesh_path=Path(__file__).resolve().parent.parent / "PPE" / "plate.inp",
        is_rigid=True,
    ),
}


def get_ppe(key: str) -> PpeSpec:
    """Look up a registered PPE by key.

    Raises:
        KeyError: if ``key`` isn't registered.
    """
    if key not in _PPE_REGISTRY:
        raise KeyError(f"Unknown PPE '{key}' -- known: {sorted(_PPE_REGISTRY)}")
    return _PPE_REGISTRY[key]


def known_ppe() -> tuple:
    """Every PPE key currently registered."""
    return tuple(_PPE_REGISTRY.keys())
