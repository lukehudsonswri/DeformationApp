"""Assemble a runnable FEBio case from a seated plate + torso volume
(AGENTS.md section 2.5).

Removes the manual FEBio Studio setup that the client's own
``apply_control_and_materials.py`` explicitly punts on ("Contact ... the
plate's rigid body prescribed-motion BC, and the load curve are
intentionally NOT touched here"). Every material/contact/control parameter
below is ported verbatim from the gold-standard reference case, per
AGENTS.md section 2.5 -- this module only assembles the *geometry* (from
``fitting.seat_plate`` and ``fitting.body_volume``) into that same
structure, it does not invent new physics parameters.

**Output includes a plain-text node displacement log, not just the binary
.xplt.** Every case gets an ``<Output><logfile><node_data ...>`` block
(``<out_path.stem>_node_displacement.txt``, one ``x;y;z;ux;uy;uz`` row per
node per solved step) alongside the usual ``.xplt``. This is deliberate:
the installed ``febio-python`` xplt reader cannot parse FEBio 4.12's plot
file version (53) -- it only recognizes versions up to 52, and even
patching that check to treat 53 as 52 does not correctly locate the
(per-state-compressed) STATE section, a real, unresolved bug/format-drift
in that third-party library, not something in our control. The plain-text
logfile is FEBio's own first-class, documented, uncompressed output
mechanism and sidesteps this entirely -- ``main_2.py`` reads it directly
instead of the ``.xplt``.

**A real bug found and fixed here: the log was scoped to the whole model,
not just the torso.** An earlier version's ``node_data`` block had no
``node_set`` attribute, so FEBio logged *every* node in the model --
including the rigid plate's own ~619k nodes -- at every solved step. Since
``postprocess.project_displacement`` only ever reads the torso's ~47.5k
nodes (the plate's own displacement is discarded entirely -- it's a rigid
body, its deformation isn't meaningful to project onto the HBM), this was
pure waste: a real solved case produced a ~780MB text file, ~93% of it
plate-node rows nothing downstream ever reads. Fixed by adding a
``TorsoNodes`` ``<NodeSet>`` (ids ``1..n_torso``, the same ids
``fitting.body_volume.TorsoVolume``'s node order already establishes) and
referencing it via ``node_set="TorsoNodes"`` on the ``node_data`` block --
confirmed directly (a tiny synthetic multi-domain case) that FEBio 4.12
honors this and logs only the referenced node set. Cuts the log to
~47.5k/666.6k ≈ 7% of its previous size.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from febio.loadcurve import LoadCurveSpec
from fitting.body_volume import TorsoVolume, hex8_boundary_faces

_DEFAULT_SKIN_MATERIAL = {"name": "Skin", "density": "1.1e-06", "k": "10", "c1": "0.05", "m1": "10"}
_DEFAULT_FLESH_MATERIAL = {"name": "Flesh", "density": "1.1e-06", "k": "1.5", "c1": "0.003", "m1": "4"}
_DEFAULT_RIGID_MATERIAL = {"name": "PlateRigid", "density": "1"}


def _node_line(node_id: int, xyz: np.ndarray) -> str:
    return f'\t\t\t<node id="{node_id}">{xyz[0]:.10g},{xyz[1]:.10g},{xyz[2]:.10g}</node>'


def _wrapped_id_block(ids, per_line: int = 8) -> str:
    """Format a NodeSet's id list wrapped at ``per_line`` ids/line --
    matches the gold-standard reference's own convention. See
    ``remesh/febio_smoke_test.py`` for why this isn't emitted as one huge
    single-line string (empirically correlated with a false-positive
    "domain initialization" failure on at least one FEBio version).
    """
    strs = [str(i) for i in ids]
    lines = [",".join(strs[k : k + per_line]) for k in range(0, len(strs), per_line)]
    return ",\n\t\t\t".join(lines)


def assemble_torso_plate_case(
    torso: TorsoVolume,
    plate_nodes: np.ndarray,
    plate_hexes: np.ndarray,
    plate_concave_face_node_ids: np.ndarray,
    load_curve: LoadCurveSpec,
    into_body_direction: np.ndarray,
    out_path: Path,
    skin_material: Optional[Dict[str, str]] = None,
    flesh_material: Optional[Dict[str, str]] = None,
    rigid_material: Optional[Dict[str, str]] = None,
) -> None:
    """Write a full torso(skin+flesh)-vs-rigid-plate FEBio 4 case.

    Args:
        torso: from ``fitting.body_volume.extract_torso_volume``.
        plate_nodes: the plate's full node array, already rigidly
            transformed by ``fitting.seat_plate.seat_plate`` (i.e.
            ``SeatingResult.transformed_plate_nodes``).
        plate_hexes: the plate's hex8 connectivity, 0-based indices into
            ``plate_nodes`` (i.e. the loaded ``VolumeMesh``'s own
            ``element_groups[ElementType.HEX8]`` -- unchanged by seating,
            which only moves node coordinates).
        plate_concave_face_node_ids: from
            ``fitting.plate_signature.PlateOrientation`` -- used to build
            the contact "secondary" surface from only the plate's
            body-facing boundary faces, not its whole exterior.
        load_curve: from ``febio.loadcurve.derive_load_curve``.
        into_body_direction: unit vector, the plate's *current* (post-
            seating) concave-normal direction -- the rigid body is
            prescribed to translate along this direction, driven by
            ``load_curve``.
        out_path: where to write the ``.feb``.
    """
    out_path = Path(out_path)
    skin_mat = skin_material or _DEFAULT_SKIN_MATERIAL
    flesh_mat = flesh_material or _DEFAULT_FLESH_MATERIAL
    rigid_mat = rigid_material or _DEFAULT_RIGID_MATERIAL

    n_torso = len(torso.nodes)
    all_nodes = np.concatenate([torso.nodes, plate_nodes], axis=0)
    # torso nodes: ids 1..n_torso; plate nodes: ids n_torso+1..n_torso+n_plate
    plate_id_offset = n_torso
    node_lines = [_node_line(i + 1, xyz) for i, xyz in enumerate(all_nodes)]

    # --- elements: skin quad4, flesh hex8, plate hex8 -- globally unique ids ---
    next_eid = 1
    skin_rows = []
    for i, quad in enumerate(torso.skin_quads):
        skin_rows.append(f'\t\t\t<elem id="{next_eid + i}">{",".join(str(int(n) + 1) for n in quad)}</elem>')
    next_eid += len(torso.skin_quads)

    flesh_rows = []
    for i, hexa in enumerate(torso.flesh_hexes):
        flesh_rows.append(f'\t\t\t<elem id="{next_eid + i}">{",".join(str(int(n) + 1) for n in hexa)}</elem>')
    next_eid += len(torso.flesh_hexes)

    plate_rows = []
    for i, hexa in enumerate(plate_hexes):
        ids = ",".join(str(int(n) + 1 + plate_id_offset) for n in hexa)
        plate_rows.append(f'\t\t\t<elem id="{next_eid + i}">{ids}</elem>')
    next_eid += len(plate_hexes)

    # --- contact surfaces: primary = full skin (shell) surface, secondary
    # = the plate's concave (body-facing) boundary faces only ---
    primary_rows = []
    for i, quad in enumerate(torso.skin_quads):
        ids = ",".join(str(int(n) + 1) for n in quad)
        primary_rows.append(f'\t\t\t<quad4 id="{i + 1}">{ids}</quad4>')

    concave_set = set(int(x) for x in plate_concave_face_node_ids)
    plate_boundary_quads = hex8_boundary_faces(plate_nodes, plate_hexes)
    concave_quads = [q for q in plate_boundary_quads if all(int(n) in concave_set for n in q)]
    secondary_rows = []
    for i, quad in enumerate(concave_quads):
        ids = ",".join(str(int(n) + 1 + plate_id_offset) for n in quad)
        secondary_rows.append(f'\t\t\t<quad4 id="{i + 1}">{ids}</quad4>')

    # --- boundary: fix the torso's artificial cut surface (free boundary) ---
    free_ids_1based = [int(i) + 1 for i in torso.free_boundary_node_ids]
    free_and_skin = set(free_ids_1based) & set(int(n) + 1 for n in np.unique(torso.skin_quads))
    free_boundary_block = _wrapped_id_block(sorted(free_ids_1based))
    shell_rotation_block = _wrapped_id_block(sorted(free_and_skin)) if free_and_skin else None

    # --- torso-only node set, for a size-scoped Output/logfile node_data ---
    torso_node_set_block = _wrapped_id_block(list(range(1, n_torso + 1)))

    # --- rigid body prescribed displacement: 3 axis dofs, each value = the
    # unit direction's component negated (see febio/assemble_case.py
    # module docstring derivation: load_curve.points already carries
    # -total_travel_mm, so value=-direction_component gives a net
    # +total_travel_mm * direction_component displacement at load factor 1,
    # i.e. moving INTO the body as into_body_direction specifies) ---
    dv = -np.asarray(into_body_direction, dtype=np.float64)
    dv = dv / np.linalg.norm(dv)

    points_block = "\n".join(f"\t\t\t\t<pt>{t:.10g},{v:.10g}</pt>" for t, v in load_curve.points)

    xml = [
        '<?xml version="1.0" encoding="ISO-8859-1"?>',
        '<febio_spec version="4.0">',
        "\t<Module type=\"solid\"/>",
        "\t<Control>",
        "\t\t<analysis>STATIC</analysis>",
        f"\t\t<time_steps>{load_curve.time_steps}</time_steps>",
        f"\t\t<step_size>{load_curve.step_size:.10g}</step_size>",
        "\t\t<plot_zero_state>1</plot_zero_state>",
        "\t\t<plot_range>0,-1</plot_range>",
        "\t\t<plot_level>PLOT_MAJOR_ITRS</plot_level>",
        "\t\t<output_level>OUTPUT_MAJOR_ITRS</output_level>",
        "\t\t<time_stepper type=\"default\">",
        "\t\t\t<max_retries>15</max_retries>",
        "\t\t\t<opt_iter>8</opt_iter>",
        "\t\t\t<dtmin>0.0001</dtmin>",
        "\t\t\t<dtmax>0.3</dtmax>",
        "\t\t\t<cutback>0.5</cutback>",
        "\t\t</time_stepper>",
        "\t\t<solver type=\"solid\">",
        "\t\t\t<symmetric_stiffness>non-symmetric</symmetric_stiffness>",
        "\t\t\t<equation_scheme>staggered</equation_scheme>",
        "\t\t\t<lstol>0.9</lstol>",
        "\t\t\t<lsmin>0.01</lsmin>",
        "\t\t\t<lsiter>5</lsiter>",
        "\t\t\t<ls_check_jacobians>1</ls_check_jacobians>",
        "\t\t\t<max_refs>15</max_refs>",
        "\t\t\t<dtol>0.01</dtol>",
        "\t\t\t<etol>0.01</etol>",
        "\t\t\t<qn_method type=\"full Newton\"/>",
        "\t\t\t<linear_solver type=\"mkl_dss\"/>",
        "\t\t</solver>",
        "\t</Control>",
        "\t<Globals>",
        "\t\t<Constants><T>0</T><P>0</P><R>8.31446</R><Fc>96485.3</Fc></Constants>",
        "\t</Globals>",
        "\t<Material>",
        f'\t\t<material id="1" name="{skin_mat["name"]}" type="Ogden">',
        f'\t\t\t<density>{skin_mat["density"]}</density>',
        f'\t\t\t<k>{skin_mat["k"]}</k>',
        f'\t\t\t<c1>{skin_mat["c1"]}</c1>',
        f'\t\t\t<m1>{skin_mat["m1"]}</m1>',
        "\t\t</material>",
        f'\t\t<material id="2" name="{flesh_mat["name"]}" type="Ogden">',
        f'\t\t\t<density>{flesh_mat["density"]}</density>',
        f'\t\t\t<k>{flesh_mat["k"]}</k>',
        f'\t\t\t<c1>{flesh_mat["c1"]}</c1>',
        f'\t\t\t<m1>{flesh_mat["m1"]}</m1>',
        "\t\t</material>",
        f'\t\t<material id="3" name="{rigid_mat["name"]}" type="rigid body">',
        f'\t\t\t<density>{rigid_mat["density"]}</density>',
        "\t\t\t<E>1</E>",
        "\t\t\t<v>0</v>",
        "\t\t</material>",
        "\t</Material>",
        "\t<Mesh>",
        '\t\t<Nodes name="AllNodes">',
        *node_lines,
        "\t\t</Nodes>",
        '\t\t<Elements type="quad4" name="TorsoSkin">',
        *skin_rows,
        "\t\t</Elements>",
        '\t\t<Elements type="hex8" name="TorsoFlesh">',
        *flesh_rows,
        "\t\t</Elements>",
        '\t\t<Elements type="hex8" name="Plate">',
        *plate_rows,
        "\t\t</Elements>",
        f'\t\t<NodeSet name="TorsoFreeBoundary">\n\t\t\t{free_boundary_block}\n\t\t</NodeSet>',
        *(
            [f'\t\t<NodeSet name="TorsoFreeBoundaryShell">\n\t\t\t{shell_rotation_block}\n\t\t</NodeSet>']
            if shell_rotation_block
            else []
        ),
        # torso-only node set (ids 1..n_torso), so the Output/logfile
        # node_data block below can log just the torso -- not the plate's
        # own 619k nodes, which main_2.py's projection step never reads
        # anyway (see postprocess/project_displacement.py). Logging every
        # node in the model at every solved step produced a ~780MB text
        # file; scoping to torso-only cuts that to ~93% smaller.
        f'\t\t<NodeSet name="TorsoNodes">\n\t\t\t{torso_node_set_block}\n\t\t</NodeSet>',
        '\t\t<Surface name="SlidingElastic1Primary">',
        *primary_rows,
        "\t\t</Surface>",
        '\t\t<Surface name="SlidingElastic1Secondary">',
        *secondary_rows,
        "\t\t</Surface>",
        '\t\t<SurfacePair name="SlidingElastic1">',
        "\t\t\t<primary>SlidingElastic1Primary</primary>",
        "\t\t\t<secondary>SlidingElastic1Secondary</secondary>",
        "\t\t</SurfacePair>",
        "\t</Mesh>",
        "\t<MeshDomains>",
        f'\t\t<ShellDomain name="TorsoSkin" mat="{skin_mat["name"]}" type="elastic-shell">',
        "\t\t\t<shell_thickness>2</shell_thickness>",
        "\t\t\t<shell_normal_nodal>0</shell_normal_nodal>",
        "\t\t</ShellDomain>",
        f'\t\t<SolidDomain name="TorsoFlesh" mat="{flesh_mat["name"]}"/>',
        f'\t\t<SolidDomain name="Plate" mat="{rigid_mat["name"]}"/>',
        "\t</MeshDomains>",
        "\t<Boundary>",
        '\t\t<bc name="TorsoFreeBoundary_ZeroDisplacement" node_set="TorsoFreeBoundary" type="zero displacement">',
        "\t\t\t<x_dof>1</x_dof>",
        "\t\t\t<y_dof>1</y_dof>",
        "\t\t\t<z_dof>1</z_dof>",
        "\t\t</bc>",
        *(
            [
                '\t\t<bc name="TorsoFreeBoundary_ZeroRotation" node_set="TorsoFreeBoundaryShell" type="zero rotation">',
                "\t\t\t<u_dof>1</u_dof>",
                "\t\t\t<v_dof>1</v_dof>",
                "\t\t\t<w_dof>1</w_dof>",
                "\t\t</bc>",
            ]
            if shell_rotation_block
            else []
        ),
        "\t</Boundary>",
        "\t<Rigid>",
        f'\t\t<rigid_bc name="PlateDriveX" type="rigid_displacement">',
        f"\t\t\t<rb>{rigid_mat['name']}</rb>",
        "\t\t\t<dof>x</dof>",
        f'\t\t\t<value lc="1">{dv[0]:.10g}</value>',
        "\t\t\t<relative>0</relative>",
        "\t\t</rigid_bc>",
        f'\t\t<rigid_bc name="PlateDriveY" type="rigid_displacement">',
        f"\t\t\t<rb>{rigid_mat['name']}</rb>",
        "\t\t\t<dof>y</dof>",
        f'\t\t\t<value lc="1">{dv[1]:.10g}</value>',
        "\t\t\t<relative>0</relative>",
        "\t\t</rigid_bc>",
        f'\t\t<rigid_bc name="PlateDriveZ" type="rigid_displacement">',
        f"\t\t\t<rb>{rigid_mat['name']}</rb>",
        "\t\t\t<dof>z</dof>",
        f'\t\t\t<value lc="1">{dv[2]:.10g}</value>',
        "\t\t\t<relative>0</relative>",
        "\t\t</rigid_bc>",
        '\t\t<rigid_bc name="PlateFixRotation" type="rigid_fixed">',
        f"\t\t\t<rb>{rigid_mat['name']}</rb>",
        "\t\t\t<Ru_dof>1</Ru_dof>",
        "\t\t\t<Rv_dof>1</Rv_dof>",
        "\t\t\t<Rw_dof>1</Rw_dof>",
        "\t\t</rigid_bc>",
        "\t</Rigid>",
        "\t<Contact>",
        '\t\t<contact name="SlidingElastic1" surface_pair="SlidingElastic1" type="sliding-elastic">',
        "\t\t\t<laugon>AUGLAG</laugon>",
        "\t\t\t<tolerance>0.2</tolerance>",
        "\t\t\t<gaptol>0</gaptol>",
        "\t\t\t<penalty>1</penalty>",
        "\t\t\t<auto_penalty>1</auto_penalty>",
        "\t\t\t<update_penalty>1</update_penalty>",
        "\t\t\t<two_pass>0</two_pass>",
        "\t\t\t<search_tol>0.01</search_tol>",
        "\t\t\t<search_radius>3</search_radius>",
        "\t\t\t<minaug>2</minaug>",
        "\t\t\t<maxaug>10</maxaug>",
        "\t\t\t<fric_coeff>0</fric_coeff>",
        "\t\t</contact>",
        "\t</Contact>",
        "\t<LoadData>",
        f'\t\t<load_controller id="1" name="LC1" type="loadcurve">',
        "\t\t\t<interpolate>LINEAR</interpolate>",
        f"\t\t\t<extend>{load_curve.extend}</extend>",
        "\t\t\t<points>",
        points_block,
        "\t\t\t</points>",
        "\t\t</load_controller>",
        "\t</LoadData>",
        "\t<Output>",
        "\t\t<logfile>",
        (
            f'\t\t\t<node_data data="x;y;z;ux;uy;uz" '
            f'file="{out_path.stem}_node_displacement.txt" node_set="TorsoNodes"></node_data>'
        ),
        "\t\t</logfile>",
        '\t\t<plotfile type="febio">',
        '\t\t\t<var type="displacement"/>',
        '\t\t\t<var type="stress"/>',
        "\t\t</plotfile>",
        "\t</Output>",
        "</febio_spec>",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(xml), encoding="ISO-8859-1")
