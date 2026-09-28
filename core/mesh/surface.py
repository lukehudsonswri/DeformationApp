"""
Surface mesh loaders and savers for STL and OBJ formats.

Uses trimesh for robust file handling with fallback implementations.

--------------------------------------------------------------------------
Vendored from: C:\\Users\\lhudson\\Documents\\Morphing\\fe_personalization\\mesh_io\\surface.py
Vendored on:   2026-09-18, unmodified.
Local changes: none yet.
See AGENTS.md "Vendoring policy" and section 1.4 -- STL PPE is
tetrahedralised into a solid on import (decided), so load_stl() here is
one step in that pipeline, not a terminal rigid-surface representation.
--------------------------------------------------------------------------
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import re
import logging

from .base import SurfaceMesh, ElementType


def load_stl(filepath: str) -> SurfaceMesh:
    """
    Load a surface mesh from an STL file.

    Args:
        filepath: Path to the .stl file

    Returns:
        SurfaceMesh object
    """
    filepath = Path(filepath)

    try:
        import trimesh
        mesh = trimesh.load(str(filepath), force='mesh')
        surface = SurfaceMesh(
            nodes=np.asarray(mesh.vertices, dtype=np.float64),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            face_type=ElementType.TRI3,
            name=filepath.stem,
            metadata={'source_file': str(filepath)}
        )
        surface.compute_normals()
        return surface
    except ImportError:
        # Fallback to manual parsing
        return _load_stl_manual(filepath)


def _load_stl_manual(filepath: Path) -> SurfaceMesh:
    """Manual STL parser for when trimesh is not available."""
    with open(filepath, 'rb') as f:
        # Check if binary or ASCII
        header = f.read(80)
        try:
            header.decode('ascii')
            is_ascii = b'solid' in header.lower()
        except UnicodeDecodeError:
            is_ascii = False

    if is_ascii:
        return _load_stl_ascii(filepath)
    else:
        return _load_stl_binary(filepath)


def _load_stl_ascii(filepath: Path) -> SurfaceMesh:
    """Load ASCII STL file."""
    vertices = []
    faces = []
    vertex_map = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse facets
    facet_pattern = r'vertex\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    matches = re.findall(facet_pattern, content)

    for i in range(0, len(matches), 3):
        face = []
        for j in range(3):
            if i + j < len(matches):
                v = tuple(float(x) for x in matches[i + j])
                if v not in vertex_map:
                    vertex_map[v] = len(vertices)
                    vertices.append(v)
                face.append(vertex_map[v])
        if len(face) == 3:
            faces.append(face)

    surface = SurfaceMesh(
        nodes=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int64),
        face_type=ElementType.TRI3,
        name=filepath.stem,
        metadata={'source_file': str(filepath)}
    )
    surface.compute_normals()
    return surface


def _load_stl_binary(filepath: Path) -> SurfaceMesh:
    """Load binary STL file."""
    with open(filepath, 'rb') as f:
        # Skip 80-byte header
        f.read(80)
        # Read number of triangles
        num_triangles = np.frombuffer(f.read(4), dtype=np.uint32)[0]

        vertices = []
        faces = []
        vertex_map = {}

        for _ in range(num_triangles):
            # Skip normal (12 bytes)
            f.read(12)
            # Read 3 vertices (36 bytes)
            face = []
            for _ in range(3):
                v = tuple(np.frombuffer(f.read(12), dtype=np.float32))
                if v not in vertex_map:
                    vertex_map[v] = len(vertices)
                    vertices.append(v)
                face.append(vertex_map[v])
            faces.append(face)
            # Skip attribute byte count (2 bytes)
            f.read(2)

    surface = SurfaceMesh(
        nodes=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int64),
        face_type=ElementType.TRI3,
        name=filepath.stem,
        metadata={'source_file': str(filepath)}
    )
    surface.compute_normals()
    return surface


def load_obj(filepath: str) -> SurfaceMesh:
    """
    Load a surface mesh from an OBJ file.

    Args:
        filepath: Path to the .obj file

    Returns:
        SurfaceMesh object
    """
    filepath = Path(filepath)

    try:
        import trimesh
        mesh = trimesh.load(str(filepath), force='mesh')
        surface = SurfaceMesh(
            nodes=np.asarray(mesh.vertices, dtype=np.float64),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            face_type=ElementType.TRI3,
            name=filepath.stem,
            metadata={'source_file': str(filepath)}
        )
        surface.compute_normals()
        return surface
    except ImportError:
        # Fallback to manual parsing
        return _load_obj_manual(filepath)


def _load_obj_manual(filepath: Path) -> SurfaceMesh:
    """Manual OBJ parser for when trimesh is not available."""
    vertices = []
    faces = []
    normals = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'v':
                # Vertex
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vn':
                # Vertex normal
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                # Face - handle various formats (v, v/vt, v/vt/vn, v//vn)
                face = []
                for p in parts[1:]:
                    # Parse vertex index (first number before any /)
                    v_idx = int(p.split('/')[0])
                    # OBJ uses 1-based indexing, convert to 0-based
                    if v_idx > 0:
                        face.append(v_idx - 1)
                    else:
                        # Negative indices are relative to end
                        face.append(len(vertices) + v_idx)

                # Triangulate if more than 3 vertices
                if len(face) == 3:
                    faces.append(face)
                elif len(face) > 3:
                    # Fan triangulation
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i + 1]])

    surface = SurfaceMesh(
        nodes=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int64),
        face_type=ElementType.TRI3,
        name=filepath.stem,
        metadata={'source_file': str(filepath)}
    )
    surface.compute_normals()
    return surface


def save_stl(mesh: SurfaceMesh, filepath: str, binary: bool = True) -> None:
    """
    Save a surface mesh to STL format.

    Args:
        mesh: SurfaceMesh to save
        filepath: Output file path
        binary: If True, save as binary STL (smaller file size)
    """
    filepath = Path(filepath)

    try:
        import trimesh
        tm = mesh.to_trimesh()
        tm.export(str(filepath), file_type='stl')
        return
    except ImportError:
        pass

    if binary:
        _save_stl_binary(mesh, filepath)
    else:
        _save_stl_ascii(mesh, filepath)


def _save_stl_ascii(mesh: SurfaceMesh, filepath: Path) -> None:
    """Save as ASCII STL."""
    mesh.compute_normals()

    with open(filepath, 'w') as f:
        f.write(f'solid {mesh.name}\n')

        for i, face in enumerate(mesh.faces):
            normal = mesh.face_normals[i] if mesh.face_normals is not None else [0, 0, 1]
            f.write(f'  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n')
            f.write('    outer loop\n')
            for vid in face:
                v = mesh.nodes[vid]
                f.write(f'      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n')
            f.write('    endloop\n')
            f.write('  endfacet\n')

        f.write(f'endsolid {mesh.name}\n')


def _save_stl_binary(mesh: SurfaceMesh, filepath: Path) -> None:
    """Save as binary STL."""
    mesh.compute_normals()

    with open(filepath, 'wb') as f:
        # 80-byte header
        header = f'{mesh.name}'.encode('ascii')[:80].ljust(80, b'\0')
        f.write(header)

        # Number of triangles
        f.write(np.uint32(len(mesh.faces)).tobytes())

        # Write triangles
        for i, face in enumerate(mesh.faces):
            normal = mesh.face_normals[i] if mesh.face_normals is not None else [0, 0, 1]
            f.write(np.float32(normal).tobytes())
            for vid in face:
                v = mesh.nodes[vid]
                f.write(np.float32(v).tobytes())
            f.write(np.uint16(0).tobytes())  # Attribute byte count


def save_obj(mesh: SurfaceMesh, filepath: str) -> None:
    """
    Save a surface mesh to OBJ format.

    Args:
        mesh: SurfaceMesh to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    mesh.compute_normals()

    with open(filepath, 'w') as f:
        f.write(f'# OBJ file: {mesh.name}\n')
        f.write(f'# Vertices: {mesh.num_nodes}\n')
        f.write(f'# Faces: {mesh.num_faces}\n\n')

        # Write vertices
        for v in mesh.nodes:
            f.write(f'v {v[0]:.10g} {v[1]:.10g} {v[2]:.10g}\n')

        # Write normals
        if mesh.vertex_normals is not None:
            f.write('\n')
            for n in mesh.vertex_normals:
                f.write(f'vn {n[0]:.6g} {n[1]:.6g} {n[2]:.6g}\n')

        # Write faces (1-based indexing)
        f.write('\n')
        if mesh.vertex_normals is not None:
            for face in mesh.faces:
                f.write('f')
                for vid in face:
                    f.write(f' {vid + 1}//{vid + 1}')
                f.write('\n')
        else:
            for face in mesh.faces:
                f.write('f')
                for vid in face:
                    f.write(f' {vid + 1}')
                f.write('\n')


def extract_unique_vertices(filepath: str) -> np.ndarray:
    """
    Extract unique vertices from an STL or OBJ file.
    Useful for point cloud registration.

    Args:
        filepath: Path to STL or OBJ file

    Returns:
        (N, 3) array of unique vertex coordinates
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext == '.stl':
        mesh = load_stl(str(filepath))
    elif ext == '.obj':
        mesh = load_obj(str(filepath))
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return mesh.nodes


def _ray_hits_any_bruteforce(
    origin: np.ndarray,
    direction: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    eps: float,
) -> bool:
    """
    Brute-force ray-triangle intersection (Moller-Trumbore), vectorized over all triangles.

    Returns True if the ray intersects any triangle at distance > eps.
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    a = np.einsum("ij,ij->i", edge1, h)
    mask = np.abs(a) > eps
    if not np.any(mask):
        return False
    f = np.zeros_like(a)
    f[mask] = 1.0 / a[mask]
    s = origin - v0
    u = f * np.einsum("ij,ij->i", s, h)
    mask_u = (u >= 0.0) & (u <= 1.0) & mask
    if not np.any(mask_u):
        return False
    q = np.cross(s, edge1)
    v = f * np.einsum("ij,j->i", q, direction)
    mask_v = (v >= 0.0) & ((u + v) <= 1.0) & mask_u
    if not np.any(mask_v):
        return False
    t = f * np.einsum("ij,ij->i", edge2, q)
    return np.any(t[mask_v] > eps)


def _external_face_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_bruteforce_faces: int = 5000,
) -> np.ndarray:
    """
    Identify external faces using a ray-escape test.

    A face is considered external if a ray cast from a point slightly offset
    along its normal escapes in at least one direction.
    """
    logger = logging.getLogger(__name__)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    normals = normals / norm

    centroids = (v0 + v1 + v2) / 3.0

    bbox = np.ptp(vertices, axis=0)
    bbox_diag = float(np.linalg.norm(bbox))
    eps = max(bbox_diag * 1e-3, 1e-4)

    # Try trimesh ray intersector if rtree is available
    use_trimesh = False
    try:
        import trimesh  # noqa: F401
        import rtree  # noqa: F401
        use_trimesh = True
    except Exception:
        use_trimesh = False

    if use_trimesh:
        import trimesh
        tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(tmesh)
        origins = np.vstack([centroids + eps * normals, centroids - eps * normals])
        directions = np.vstack([normals, -normals])
        if hasattr(intersector, "intersects_any"):
            hits = intersector.intersects_any(origins, directions)
        else:
            locs, ray_idx, _ = intersector.intersects_location(
                origins, directions, multiple_hits=False
            )
            hits = np.zeros(len(origins), dtype=bool)
            hits[ray_idx] = True
        hits = hits.reshape(2, -1)
        return ~(hits[0] & hits[1])

    # Brute-force fallback (O(N^2) - avoid for very large surfaces)
    if faces.shape[0] > max_bruteforce_faces:
        logger.warning(
            "External surface filter skipped (faces=%d > %d) due to missing rtree.",
            faces.shape[0],
            max_bruteforce_faces,
        )
        return np.ones(len(faces), dtype=bool)

    logger.warning("rtree not available; using brute-force ray tests")
    external_mask = np.zeros(len(faces), dtype=bool)
    for i in range(len(faces)):
        n = normals[i]
        c = centroids[i]
        hits_pos = _ray_hits_any_bruteforce(c + eps * n, n, v0, v1, v2, eps)
        hits_neg = _ray_hits_any_bruteforce(c - eps * n, -n, v0, v1, v2, eps)
        external_mask[i] = not (hits_pos and hits_neg)
    return external_mask


def extract_outer_surface(
    volume_mesh,
    angle_threshold: float = 95.0,
    external_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract outer surface nodes from a volume mesh using flood-fill
    with normal continuity.

    This function identifies the outer surface of a volume mesh (e.g., bone)
    by starting from the farthest point from the centroid and expanding
    to connected surface nodes whose normals are within angle_threshold degrees.

    This effectively separates the outer surface from inner surfaces
    (e.g., medullary cavity in bone).

    Args:
        volume_mesh: A VolumeMesh object with extract_surface() and get_surface_nodes() methods
        angle_threshold: Maximum angle (degrees) between adjacent normals to include
                        in the same surface. Default 90 degrees.
        external_only: If True, filter boundary faces using a ray-escape test to
                       remove internal surfaces before flood-fill.

    Returns:
        Tuple of:
        - outer_points: (N, 3) array of outer surface vertex coordinates
        - outer_faces: (M, 3) array of outer surface face indices (0-based, remapped)
        - outer_node_ids: (N,) array of original node IDs in the volume mesh
    """
    from collections import deque

    # Extract full surface and optionally filter to external faces only
    surface_mesh = volume_mesh.extract_surface()
    nodes = volume_mesh.nodes
    faces = surface_mesh.faces

    if external_only:
        ext_mask = _external_face_mask(nodes, faces)
        faces = faces[ext_mask]
        # Recompute normals for filtered surface to avoid internal-face influence
        filtered_surface = SurfaceMesh(
            nodes=nodes.copy(),
            faces=faces,
            face_type=ElementType.TRI3,
            name=f"{volume_mesh.name}_surface_filtered",
        )
        filtered_surface.compute_normals()
        vertex_normals = filtered_surface.vertex_normals
        surface_node_ids = np.unique(faces).astype(np.int64)
    else:
        vertex_normals = surface_mesh.vertex_normals
        surface_node_ids = volume_mesh.get_surface_nodes()

    surface_set = set(surface_node_ids.tolist())

    # Build adjacency graph for surface nodes using shared edges (not just shared nodes)
    # This avoids incorrectly connecting disjoint surfaces that only touch at a point.
    adjacency = {nid: set() for nid in surface_node_ids}
    for face in faces:
        face_nodes = [int(n) for n in face if n in surface_set]
        if len(face_nodes) < 2:
            continue
        # Connect along edges only (triangles => 3 edges)
        for i in range(len(face_nodes)):
            n1 = face_nodes[i]
            n2 = face_nodes[(i + 1) % len(face_nodes)]
            adjacency[n1].add(n2)
            adjacency[n2].add(n1)

    # Find seed: farthest surface node from centroid (guaranteed to be outer surface)
    surface_positions = nodes[surface_node_ids]
    centroid = surface_positions.mean(axis=0)
    distances = np.linalg.norm(surface_positions - centroid, axis=1)
    seed_node = surface_node_ids[np.argmax(distances)]

    def get_normal(nid):
        """Get normalized vertex normal."""
        n = vertex_normals[nid]
        norm = np.linalg.norm(n)
        return n / norm if norm > 1e-10 else np.array([0, 0, 1])

    # Flood-fill from seed, only expanding to nodes with similar normals
    cos_threshold = np.cos(np.radians(angle_threshold))
    outer_nodes = set()
    visited = set()
    queue = deque([seed_node])

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        outer_nodes.add(current)

        current_normal = get_normal(current)
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                # Check if normals are within threshold
                if np.dot(current_normal, get_normal(neighbor)) >= cos_threshold:
                    queue.append(neighbor)

    outer_ids = np.array(sorted(outer_nodes), dtype=np.int64)
    outer_points = nodes[outer_ids]

    # Extract faces that only use outer surface nodes
    outer_set = set(outer_ids.tolist())
    outer_faces = []
    for face in faces:
        if all(v in outer_set for v in face):
            outer_faces.append(face)
    outer_faces = np.array(outer_faces, dtype=np.int64)

    # Remap faces to use 0-based indices for the outer surface
    id_to_idx = {nid: idx for idx, nid in enumerate(outer_ids)}
    remapped_faces = np.array([[id_to_idx[v] for v in face] for face in outer_faces])

    return outer_points, remapped_faces, outer_ids


def extract_boundary_surface(volume_mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the complete boundary surface from a volume mesh.

    Unlike :func:`extract_outer_surface`, this does not run the external ray
    filter or normal-continuity flood fill. It is appropriate for solid
    single-shell structures such as muscle templates, where every boundary
    face should be available to the registration.
    """
    surface_mesh = volume_mesh.extract_surface()
    faces = np.asarray(surface_mesh.faces, dtype=np.int64)

    if faces.size == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    boundary_ids = np.unique(faces).astype(np.int64)
    id_to_idx = {int(nid): idx for idx, nid in enumerate(boundary_ids)}
    remapped_faces = np.array(
        [[id_to_idx[int(v)] for v in face] for face in faces],
        dtype=np.int64,
    )

    return volume_mesh.nodes[boundary_ids], remapped_faces, boundary_ids


def fix_mesh_normals(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fix inverted faces by ensuring consistent outward-facing normals.

    Uses trimesh's fix_normals() which orients all faces consistently
    and ensures normals point outward.

    Args:
        vertices: (N, 3) vertex coordinates
        faces: (M, 3) face indices

    Returns:
        Tuple of (fixed_vertices, fixed_faces) as float64/int64 arrays
    """
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.fix_normals()
        return np.asarray(mesh.vertices, dtype=np.float64), mesh.faces.copy()
    except ImportError:
        # If trimesh not available, return as-is
        return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
