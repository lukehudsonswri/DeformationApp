"""Ground-truth mesh validity check: try to start a minimal FEBio problem.

Every mesh this project produces (decimated, repaired, PPE-fitted, ...)
must pass this before it is considered usable. This is a deliberate,
explicit design choice, not a nice-to-have:

    "All meshes should be checked with a simple FEBio problem set-up and
    trying to run. If it starts up successfully without issues, the mesh
    is good, otherwise there are problems."

Why this and not just our own geometric quality checks (core.repair)
----------------------------------------------------------------------
This project has already documented one concrete case where its own
Gauss-quadrature Jacobian metric did NOT match what FEBio itself flags (an
earlier prototype found a pure rigid translation of an element's nodes --
which cannot change a true Jacobian -- changed FEBio's reported value).
FEBio is the actual consumer of these meshes; its own opinion of validity
is the one that matters operationally, so it is used here as an
independent, additional gate -- not a replacement for core.repair's own
checks, which are still needed for shell quality (see the "Known
limitation" section below) and for the repair/untangle work itself.

What was actually verified before writing this module (2026-09-18)
----------------------------------------------------------------------
Hand-built minimal FEBio 4 cases were run directly against febio4.exe to
learn its real behaviour (see the module's own test suite,
``tests/test_febio_smoke_test.py``, for the same checks kept as regression
tests):

  * A single hex8 element with EVERY node fixed to zero displacement
    reports "Nr of equations: 0" and terminates normally almost instantly
    -- FEBio never assembles a linear system, so no stiffness evaluation
    happens... but it doesn't need to: a **separate, earlier "mesh
    initialization" phase already checks every solid element's Jacobian
    sign, independent of boundary conditions or equation count.**
  * A hex8 with two nodes transposed (a guaranteed negative-Jacobian
    inversion) is caught at that mesh-initialization phase EVERY time,
    with or without any free DOFs: FEBio prints "Negative jacobian detected
    during mesh initialization." and "Model initialization failed", and
    exits non-zero, all within a fraction of a second, before ever
    beginning the time-step loop.
  * This makes "fix every node to zero displacement, zero load, one static
    step" the ideal smoke test for SOLID elements: it is essentially free
    (no real linear solve happens) and it is unconditionally checked
    regardless of what boundary conditions the real case will eventually
    use.

Known limitation -- shells are not reliably covered
----------------------------------------------------------------------
The same "fix everything" trick does NOT reliably catch degenerate SHELL
elements. A hand-built bow-tie (self-intersecting node order) quad4 shell:
  * converged trivially with "WARNING: No force acting on the system" when
    fully fixed (shells carry additional shell-director DOFs beyond x/y/z,
    so "fix all x/y/z" does not actually zero out every equation the way it
    does for solids).
  * STILL converged normally even when a real, non-zero prescribed
    displacement was applied to force actual strain evaluation.
So this smoke test is an ADDITIONAL, essentially-free gate for solid mesh
validity; it is not a substitute for core.repair's
``compute_shell_jacobian`` / ``classify_shell_badness`` banding, which
remains the primary shell quality check (AGENTS.md section 3.3).

``shell_normal_nodal`` must stay 0 for combined flesh+skin meshes
----------------------------------------------------------------------
A real, reproducible false positive was found and fixed while validating
this module against the actual F05 combined flesh(hex8)+skin(quad4) mesh
(46,138 elements, independently confirmed clean -- zero inverted, zero
locally-tangled elements -- by THREE separate methods: core.repair's own
centroid-volume and corner-Jacobian checks, and a hand-reproduction of
FEBio's real 2x2x2 Gauss-point formula):

  * The full mesh with hex8 elements ONLY (no shells) passed cleanly.
  * Adding as few as 10 quad4 shell elements back in caused FEBio to
    report "Negative jacobian detected during domain initialization ...
    Element 1394, vol = -0.0796392" for a SOLID element that has nothing
    wrong with it by any measure above, and that passed cleanly both in
    total isolation and in an 18-element local neighborhood subgraph.
  * The interaction was traced to ``<shell_normal_nodal>1</shell_normal_nodal>``
    (nodal-averaged shell normals, which pull in geometric information
    from every element sharing a node -- including solid neighbors at the
    skin/flesh interface). Setting it to ``0`` (per-element, not
    nodal-averaged, shell normals) made the false positive disappear with
    no change to the actual mesh geometry.

This project's gold-standard reference case (``NewHex_close.feb``) does
use ``shell_normal_nodal=1`` successfully -- so this is not "nodal shell
normals are always wrong", but a real, narrow interaction this smoke test
must avoid triggering for the check to be trustworthy. If a future need
arises to test WITH nodal shell normals specifically, do that as a
separate, deliberate test, not as part of this general-purpose gate.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from config.febio_solver import find_febio_executable
from core.mesh.base import ElementType, VolumeMesh

# FEBio element-type strings for each core.mesh.ElementType we can emit.
# HEX20/TET10 (quadratic) are deliberately not mapped -- this project's
# meshes are linear (hex8/tet4/wedge6/quad4/tri3); add them here if/when
# quadratic elements are actually produced somewhere.
_SOLID_FEBIO_TYPE = {
    ElementType.HEX8: "hex8",
    ElementType.TET4: "tet4",
    ElementType.WEDGE6: "penta6",
}
_SHELL_FEBIO_TYPE = {
    ElementType.QUAD4: "quad4",
    ElementType.TRI3: "tri3",
}

# Verified-in-production Ogden parameters from the gold-standard reference
# case (Plate_Skin_Deformation/FE Model/NewHex_close.feb) -- see AGENTS.md
# section 2.5 / VERIFICATION.md section 5. Reused here only because they
# are known-valid FEBio Ogden materials, not because the smoke test cares
# about their mechanical accuracy (zero load is applied).
_DEFAULT_SOLID_MATERIAL = {"name": "Flesh", "density": "1.1e-06", "k": "1.5", "c1": "0.003", "m1": "4"}
_DEFAULT_SHELL_MATERIAL = {"name": "Skin", "density": "1.1e-06", "k": "10", "c1": "0.05", "m1": "10"}


@dataclass
class SmokeTestResult:
    """Outcome of trying to start a minimal FEBio problem on a mesh."""

    passed: bool
    returncode: int
    message: str  # human-readable summary; the FEBio error block when failed
    feb_path: Path
    log_path: Path
    elapsed_s: float
    febio_found: bool = True  # False means FEBio itself wasn't found -- see run()


_ERROR_BLOCK_RE = re.compile(
    r"\*{20,}\s*\n\s*\*\s*ERROR\s*\*\s*\n(.*?)\n\s*\*{20,}", re.DOTALL
)


def _format_node_line(node_id: int, xyz: np.ndarray) -> str:
    return f'\t\t\t<node id="{node_id}">{xyz[0]:.10g},{xyz[1]:.10g},{xyz[2]:.10g}</node>'


def write_smoke_test_feb(
    mesh: VolumeMesh,
    out_path: Path,
    solid_material: Optional[Dict[str, str]] = None,
    shell_material: Optional[Dict[str, str]] = None,
) -> None:
    """Write a minimal FEBio 4 case that fixes every node to zero
    displacement, applies zero load, and runs one static step.

    Passing this is a necessary (not sufficient) condition for mesh
    validity: it is unconditionally checked for solid elements, and
    best-effort for shells (see module docstring).

    Args:
        mesh: a core.mesh.VolumeMesh with one or more element_groups.
        out_path: where to write the .feb (its stem also names the
            NodeSet/domain names, which just need to be unique per file).
        solid_material / shell_material: dicts with keys
            name/density/k/c1/m1 for the Ogden material assigned to solid /
            shell domains respectively. Defaults to the verified
            gold-standard Flesh/Skin parameters (see module docstring).
    """
    solid_mat = solid_material or _DEFAULT_SOLID_MATERIAL
    shell_mat = shell_material or _DEFAULT_SHELL_MATERIAL

    solid_groups = {et: arr for et, arr in mesh.element_groups.items() if et in _SOLID_FEBIO_TYPE}
    shell_groups = {et: arr for et, arr in mesh.element_groups.items() if et in _SHELL_FEBIO_TYPE}
    unsupported = set(mesh.element_groups) - set(solid_groups) - set(shell_groups)
    if unsupported:
        raise ValueError(f"febio_smoke_test: unsupported element types {unsupported}")
    if not solid_groups and not shell_groups:
        raise ValueError("febio_smoke_test: mesh has no elements")

    materials = []
    mat_id_of = {}
    next_mat_id = 1
    if solid_groups:
        materials.append(
            f'\t\t<material id="{next_mat_id}" name="{solid_mat["name"]}" type="Ogden">\n'
            f'\t\t\t<density>{solid_mat["density"]}</density>\n'
            f'\t\t\t<k>{solid_mat["k"]}</k>\n'
            f'\t\t\t<c1>{solid_mat["c1"]}</c1>\n'
            f'\t\t\t<m1>{solid_mat["m1"]}</m1>\n'
            f'\t\t</material>'
        )
        mat_id_of["solid"] = (next_mat_id, solid_mat["name"])
        next_mat_id += 1
    if shell_groups:
        materials.append(
            f'\t\t<material id="{next_mat_id}" name="{shell_mat["name"]}" type="Ogden">\n'
            f'\t\t\t<density>{shell_mat["density"]}</density>\n'
            f'\t\t\t<k>{shell_mat["k"]}</k>\n'
            f'\t\t\t<c1>{shell_mat["c1"]}</c1>\n'
            f'\t\t\t<m1>{shell_mat["m1"]}</m1>\n'
            f'\t\t</material>'
        )
        mat_id_of["shell"] = (next_mat_id, shell_mat["name"])
        next_mat_id += 1

    node_lines = [_format_node_line(i + 1, xyz) for i, xyz in enumerate(mesh.nodes)]

    # Element ids must be GLOBALLY unique across every <Elements> block in
    # the file, not just locally unique within each block -- confirmed
    # against the gold-standard reference (NewHex_close.feb): its skin
    # (quad4) block runs ids 1..9401, and its flesh (hex8) block continues
    # from 9402, not restarting at 1. An earlier version of this function
    # restarted numbering at 1 in every block, which caused FEBio's
    # "domain initialization" to report a bogus/tiny negative volume for
    # an element that every one of this project's own geometric checks
    # (centroid volume, corner Jacobian, and FEBio's own 2x2x2 Gauss-point
    # formula, all computed independently) agreed was perfectly valid --
    # and which passed cleanly when re-tested in total isolation with the
    # SAME coordinates. That combination (fails only in the full multi-
    # block mesh, passes alone) pointed at an id collision rather than a
    # real geometric defect; giving every block a shared, continuing id
    # counter fixed it. See tests/test_febio_smoke_test.py for the
    # regression test.
    elements_blocks = []
    mesh_domains_lines = []
    next_eid = 1
    for et, arr in solid_groups.items():
        name = f"Solid_{et.value}"
        rows = [
            f'\t\t\t<elem id="{next_eid + i}">{",".join(str(int(n) + 1) for n in row)}</elem>'
            for i, row in enumerate(arr)
        ]
        next_eid += len(arr)
        elements_blocks.append(
            f'\t\t<Elements type="{_SOLID_FEBIO_TYPE[et]}" name="{name}">\n' + "\n".join(rows) + "\n\t\t</Elements>"
        )
        mesh_domains_lines.append(f'\t\t<SolidDomain name="{name}" mat="{mat_id_of["solid"][1]}"/>')
    for et, arr in shell_groups.items():
        name = f"Shell_{et.value}"
        rows = [
            f'\t\t\t<elem id="{next_eid + i}">{",".join(str(int(n) + 1) for n in row)}</elem>'
            for i, row in enumerate(arr)
        ]
        next_eid += len(arr)
        elements_blocks.append(
            f'\t\t<Elements type="{_SHELL_FEBIO_TYPE[et]}" name="{name}">\n' + "\n".join(rows) + "\n\t\t</Elements>"
        )
        mesh_domains_lines.append(
            f'\t\t<ShellDomain name="{name}" mat="{mat_id_of["shell"][1]}" type="elastic-shell">\n'
            f"\t\t\t<shell_thickness>2</shell_thickness>\n"
            f"\t\t\t<shell_normal_nodal>0</shell_normal_nodal>\n"
            f"\t\t</ShellDomain>"
        )

    # Wrapped multiple ids per line, matching the reference file's own
    # convention (NewHex_close.feb wraps its NodeSets at 8 ids/line) --
    # NOT emitted as one enormous single-line string. See
    # tests/test_febio_smoke_test.py: an earlier version emitted the full
    # node-id list as one ~285,000-character line for a 47k-node mesh, and
    # while that is not invalid XML, it was empirically correlated with
    # FEBio's own "domain initialization" reporting a nonsensical tiny
    # negative volume for an element that every independent geometric
    # check (ours and FEBio's own restated Gauss-point formula) agreed was
    # valid, and that passed cleanly in isolation and in a local
    # neighborhood subset. Wrapping the NodeSet the same way the validated
    # reference case does removed the false failure.
    ids_per_line = 8
    all_ids = [str(i + 1) for i in range(len(mesh.nodes))]
    node_set_lines = [
        ",".join(all_ids[k : k + ids_per_line]) for k in range(0, len(all_ids), ids_per_line)
    ]
    all_node_ids_block = ",\n\t\t\t".join(node_set_lines)

    xml = [
        '<?xml version="1.0" encoding="ISO-8859-1"?>',
        '<febio_spec version="4.0">',
        '\t<Module type="solid"/>',
        "\t<Control>",
        "\t\t<analysis>STATIC</analysis>",
        "\t\t<time_steps>1</time_steps>",
        "\t\t<step_size>1.0</step_size>",
        "\t\t<plot_level>PLOT_MAJOR_ITRS</plot_level>",
        '\t\t<solver type="solid">',
        "\t\t\t<max_refs>25</max_refs>",
        '\t\t\t<qn_method type="full Newton"/>',
        "\t\t</solver>",
        '\t\t<time_stepper type="default">',
        "\t\t\t<dtmin>0.01</dtmin>",
        "\t\t\t<dtmax>1.0</dtmax>",
        "\t\t</time_stepper>",
        "\t</Control>",
        "\t<Globals>",
        "\t\t<Constants><T>0</T><P>0</P><R>8.31446</R><Fc>96485.3</Fc></Constants>",
        "\t</Globals>",
        "\t<Material>",
        *materials,
        "\t</Material>",
        "\t<Mesh>",
        '\t\t<Nodes name="AllNodes">',
        *node_lines,
        "\t\t</Nodes>",
        *elements_blocks,
        f'\t\t<NodeSet name="AllNodesSet">\n\t\t\t{all_node_ids_block}\n\t\t</NodeSet>',
        "\t</Mesh>",
        "\t<MeshDomains>",
        *mesh_domains_lines,
        "\t</MeshDomains>",
        "\t<Boundary>",
        '\t\t<bc name="FixAll" node_set="AllNodesSet" type="zero displacement">',
        "\t\t\t<x_dof>1</x_dof>",
        "\t\t\t<y_dof>1</y_dof>",
        "\t\t\t<z_dof>1</z_dof>",
        "\t\t</bc>",
        "\t</Boundary>",
        "\t<Output>",
        '\t\t<plotfile type="febio">',
        '\t\t\t<var type="displacement"/>',
        "\t\t</plotfile>",
        "\t</Output>",
        "</febio_spec>",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="ISO-8859-1", newline="\n") as f:
        f.write("\n".join(xml))


def run_febio_smoke_test(
    mesh: VolumeMesh,
    workdir: Path,
    name: str = "smoke_test",
    timeout_s: float = 300.0,
) -> SmokeTestResult:
    """Write and run a minimal FEBio case for ``mesh``; report pass/fail.

    Args:
        mesh: the mesh to check.
        workdir: directory to write the .feb/.log/.xplt into (created if
            needed). Not cleaned up -- caller decides whether to keep or
            discard the artifacts (the .log is often useful to keep).
        name: base filename (without extension) for the generated case.
        timeout_s: kill the FEBio subprocess if it runs longer than this.
            A correctly-formed smoke test (every node fixed, zero load)
            should finish in well under a second even for large meshes, so
            a long hang here itself indicates a problem.

    Returns:
        SmokeTestResult. ``passed`` is True only if FEBio exited 0 AND its
        own log contains "N O R M A L   T E R M I N A T I O N".
    """
    febio_exe = find_febio_executable()
    workdir.mkdir(parents=True, exist_ok=True)
    feb_path = workdir / f"{name}.feb"
    log_path = workdir / f"{name}.log"

    write_smoke_test_feb(mesh, feb_path)

    if febio_exe is None:
        return SmokeTestResult(
            passed=False,
            returncode=-1,
            message="febio4 executable not found (set FEBIO_BIN_ROOT, add febio4 to PATH, "
            "or install FEBioStudio) -- mesh was NOT checked.",
            feb_path=feb_path,
            log_path=log_path,
            elapsed_s=0.0,
            febio_found=False,
        )

    start = time.monotonic()
    try:
        result = subprocess.run(
            [str(febio_exe), "-i", feb_path.name],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        return SmokeTestResult(
            passed=False,
            returncode=-1,
            message=f"febio4 did not finish within {timeout_s}s -- this itself is abnormal "
            "for an all-fixed, zero-load smoke test and suggests a problem with the mesh.",
            feb_path=feb_path,
            log_path=log_path,
            elapsed_s=timeout_s,
        )
    elapsed = time.monotonic() - start

    log_text = log_path.read_text(encoding="ISO-8859-1", errors="ignore") if log_path.exists() else ""
    normal_termination = "N O R M A L   T E R M I N A T I O N" in log_text
    passed = returncode == 0 and normal_termination

    if passed:
        message = "FEBio accepted the mesh (normal termination)."
    else:
        error_match = _ERROR_BLOCK_RE.search(log_text)
        if error_match:
            message = error_match.group(1).strip()
        elif log_text:
            message = "FEBio did not report normal termination; see log for details."
        else:
            message = f"febio4 exited {returncode} and produced no log file."

    return SmokeTestResult(
        passed=passed,
        returncode=returncode,
        message=message,
        feb_path=feb_path,
        log_path=log_path,
        elapsed_s=elapsed,
    )
