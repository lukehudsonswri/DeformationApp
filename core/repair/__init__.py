"""Mesh quality detection and negative-Jacobian / shell-distortion repair.

Vendored from ForSarah/Sparse's ``lsdyna_helper_functions_sparse.py`` -- see
``_sparse_mesh_repair.py``'s own provenance header for exactly what changed
(nothing, functionally) and why it's copied wholesale rather than picked
apart (it's self-contained: stdlib + numpy + pandas only).

This is the escalation ladder's toolbox (AGENTS.md section 3.3); the ladder
itself (decimate -> detect -> repair -> untangle -> FEBio-guided -> fail
loudly) is assembled in ``remesh/quality.py``, not here.

Curated public API (a subset of ``_sparse_mesh_repair``'s full surface):

Detection:
    compute_element_volumes, compute_element_distortion,
    classify_solid_badness, compute_shell_jacobian, classify_shell_badness

Pocket splitting:
    group_bad_elements_into_pockets, group_bad_shells_into_pockets

Repair (sparse-Jacobian trust-constr optimization):
    repair_inverted_elements, repair_distorted_shells

Connectivity (the "maintain connectivity while repairing" requirement):
    build_solid_node_adjacency, find_touching_solid_elements

Penetration (soft-tissue-vs-hard-part; relevant once straps / bone-adjacent
PPE arrive):
    build_hard_surface_index, detect_soft_hard_penetrations,
    repair_soft_hard_penetrations, classify_priority_tiers,
    find_overlapping_pid_pairs, detect_all_pid_penetrations

Reporting:
    compute_pid_reference_volumes, flag_large_volume_outliers

Mesh I/O (LS-DYNA .k -- used here mainly for the test suite; production
code should prefer core.mesh.load_mesh/save_mesh for the multi-format
dispatch, unless a Sparse-specific dataframe shape is needed):
    read_nodes_from_kfile, read_elements_from_kfile,
    read_all_elements_from_kfile, read_multi_part_mesh,
    write_nodes_to_kfile, write_mesh_to_kfile, write_multi_part_mesh
"""

from ._sparse_mesh_repair import (
    # detection
    compute_element_volumes,
    compute_element_distortion,
    classify_solid_badness,
    compute_shell_jacobian,
    classify_shell_badness,
    # pocket splitting
    group_bad_elements_into_pockets,
    group_bad_shells_into_pockets,
    # repair
    repair_inverted_elements,
    repair_distorted_shells,
    # connectivity
    build_solid_node_adjacency,
    find_touching_solid_elements,
    # penetration
    build_hard_surface_index,
    detect_soft_hard_penetrations,
    repair_soft_hard_penetrations,
    classify_priority_tiers,
    find_overlapping_pid_pairs,
    detect_all_pid_penetrations,
    # reporting
    compute_pid_reference_volumes,
    flag_large_volume_outliers,
    # mesh I/O (LS-DYNA specific dataframe shape)
    read_nodes_from_kfile,
    read_elements_from_kfile,
    read_all_elements_from_kfile,
    read_multi_part_mesh,
    write_nodes_to_kfile,
    write_mesh_to_kfile,
    write_multi_part_mesh,
)

__all__ = [
    "compute_element_volumes",
    "compute_element_distortion",
    "classify_solid_badness",
    "compute_shell_jacobian",
    "classify_shell_badness",
    "group_bad_elements_into_pockets",
    "group_bad_shells_into_pockets",
    "repair_inverted_elements",
    "repair_distorted_shells",
    "build_solid_node_adjacency",
    "find_touching_solid_elements",
    "build_hard_surface_index",
    "detect_soft_hard_penetrations",
    "repair_soft_hard_penetrations",
    "classify_priority_tiers",
    "find_overlapping_pid_pairs",
    "detect_all_pid_penetrations",
    "compute_pid_reference_volumes",
    "flag_large_volume_outliers",
    "read_nodes_from_kfile",
    "read_elements_from_kfile",
    "read_all_elements_from_kfile",
    "read_multi_part_mesh",
    "write_nodes_to_kfile",
    "write_mesh_to_kfile",
    "write_multi_part_mesh",
]
