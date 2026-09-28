"""
Base mesh classes for FE Personalization.

Provides unified data structures for volumetric and surface meshes.

--------------------------------------------------------------------------
Vendored from: C:\\Users\\lhudson\\Documents\\Morphing\\fe_personalization\\mesh_io\\base.py
Vendored on:   2026-09-18, unmodified.
Local changes: none yet. If this file is edited locally, record the change
               here rather than in the upstream project.
See AGENTS.md "Vendoring policy" for why this is a copy, not an import.
--------------------------------------------------------------------------
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, List, Optional, Tuple, Any
import numpy as np

# Optional PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ElementType(Enum):
    """Supported finite element types."""
    HEX8 = "hex8"       # 8-node hexahedron
    HEX20 = "hex20"     # 20-node hexahedron
    TET4 = "tet4"       # 4-node tetrahedron
    TET10 = "tet10"     # 10-node tetrahedron
    WEDGE6 = "wedge6"   # 6-node wedge/prism
    QUAD4 = "quad4"     # 4-node quadrilateral (surface)
    TRI3 = "tri3"       # 3-node triangle (surface)


# Number of nodes per element type
ELEMENT_NODES = {
    ElementType.HEX8: 8,
    ElementType.HEX20: 20,
    ElementType.TET4: 4,
    ElementType.TET10: 10,
    ElementType.WEDGE6: 6,
    ElementType.QUAD4: 4,
    ElementType.TRI3: 3,
}

# 2D surface ("shell") element types. A mesh made only of these is a shell mesh:
# the elements *are* the surface, so triage skips boundary extraction and uses
# 2D shell quality metrics instead of the solid (hex/tet) scaled Jacobian.
SHELL_TYPES = frozenset({ElementType.QUAD4, ElementType.TRI3})


def shell_unit_normals(P: np.ndarray) -> np.ndarray:
    """Per-element unit normal for shell elements ``P`` of shape ``(n, k, 3)``.

    Built from the sum of the corner cross products so it stays well-defined for
    a warped (non-planar) quad. For a consistently CCW-wound element the normal
    points to the element's "outward" (right-hand-rule) side.
    """
    k = P.shape[1]
    acc = np.zeros((len(P), 3), dtype=np.float64)
    for i0 in range(k):
        nxt = P[:, (i0 + 1) % k] - P[:, i0]
        prv = P[:, (i0 - 1) % k] - P[:, i0]
        acc += np.cross(nxt, prv)
    norms = np.linalg.norm(acc, axis=1, keepdims=True)
    norms[norms < 1e-30] = 1.0
    return acc / norms


def shell_corner_min_jacobian(P: np.ndarray) -> np.ndarray:
    """Raw (un-normalized) min corner Jacobian per shell element ``P`` (n, k, 3).

    At each corner the cross product of the two adjacent edges is projected onto
    the element's unit normal -- the signed area of that corner's parallelogram.
    The per-element value is the minimum over corners: positive for a valid
    element, ``<= 0`` for a folded/degenerate quad. (A single triangle is never
    "inverted" on its own, so its value is just twice its area.)
    """
    if len(P) == 0:
        return np.array([])
    n = shell_unit_normals(P)
    k = P.shape[1]
    jac = np.full(len(P), np.inf)
    for i0 in range(k):
        nxt = P[:, (i0 + 1) % k] - P[:, i0]
        prv = P[:, (i0 - 1) % k] - P[:, i0]
        cn = np.cross(nxt, prv)
        jac = np.minimum(jac, np.einsum('ij,ij->i', cn, n))
    return jac


@dataclass
class NodeSet:
    """A named set of node indices."""
    name: str
    node_ids: np.ndarray  # 0-indexed node IDs

    def __post_init__(self):
        self.node_ids = np.asarray(self.node_ids, dtype=np.int64)


@dataclass
class ElementSet:
    """A named set of element indices."""
    name: str
    element_ids: np.ndarray  # 0-indexed element IDs

    def __post_init__(self):
        self.element_ids = np.asarray(self.element_ids, dtype=np.int64)


@dataclass
class Material:
    """Material definition for FE analysis."""
    name: str
    material_type: str  # e.g., "neo-Hookean", "linear elastic"
    properties: Dict[str, float] = field(default_factory=dict)
    element_ids: Optional[np.ndarray] = None  # Elements using this material


@dataclass
class Mesh:
    """Base class for all mesh types."""
    nodes: np.ndarray  # (N, 3) array of node coordinates
    name: str = "unnamed"
    node_sets: Dict[str, NodeSet] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=np.float64)
        if self.nodes.ndim == 1:
            self.nodes = self.nodes.reshape(-1, 3)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_coords, max_coords) bounding box."""
        return self.nodes.min(axis=0), self.nodes.max(axis=0)

    @property
    def centroid(self) -> np.ndarray:
        """Return the geometric centroid."""
        return self.nodes.mean(axis=0)

    def translate(self, offset: np.ndarray) -> 'Mesh':
        """Translate all nodes by offset."""
        self.nodes += np.asarray(offset)
        return self

    def scale(self, factor: float, center: Optional[np.ndarray] = None) -> 'Mesh':
        """Scale mesh around center point (default: centroid)."""
        if center is None:
            center = self.centroid
        self.nodes = center + (self.nodes - center) * factor
        return self

    def rotate(self, rotation_matrix: np.ndarray, center: Optional[np.ndarray] = None) -> 'Mesh':
        """Rotate mesh around center point."""
        if center is None:
            center = self.centroid
        centered = self.nodes - center
        self.nodes = (rotation_matrix @ centered.T).T + center
        return self


@dataclass
class SurfaceMesh(Mesh):
    """Surface mesh (triangles or quads) for visualization and registration."""
    faces: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    face_type: ElementType = ElementType.TRI3
    vertex_normals: Optional[np.ndarray] = None
    face_normals: Optional[np.ndarray] = None

    def __post_init__(self):
        super().__post_init__()
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.faces.ndim == 1 and len(self.faces) > 0:
            nodes_per_face = ELEMENT_NODES[self.face_type]
            self.faces = self.faces.reshape(-1, nodes_per_face)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def compute_normals(self) -> None:
        """Compute face and vertex normals."""
        if self.face_type == ElementType.TRI3:
            v0 = self.nodes[self.faces[:, 0]]
            v1 = self.nodes[self.faces[:, 1]]
            v2 = self.nodes[self.faces[:, 2]]
            self.face_normals = np.cross(v1 - v0, v2 - v0)
            norms = np.linalg.norm(self.face_normals, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            self.face_normals /= norms

            # Compute vertex normals by averaging adjacent face normals
            self.vertex_normals = np.zeros_like(self.nodes)
            for i, face in enumerate(self.faces):
                for vid in face:
                    self.vertex_normals[vid] += self.face_normals[i]
            norms = np.linalg.norm(self.vertex_normals, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            self.vertex_normals /= norms

    def to_trimesh(self):
        """Convert to trimesh.Trimesh object."""
        import trimesh
        return trimesh.Trimesh(vertices=self.nodes, faces=self.faces, process=False)

    @classmethod
    def from_trimesh(cls, mesh, name: str = "unnamed") -> 'SurfaceMesh':
        """Create from trimesh.Trimesh object."""
        return cls(
            nodes=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.faces),
            face_type=ElementType.TRI3,
            name=name,
        )


@dataclass
class VolumeMesh(Mesh):
    """Volumetric finite element mesh.

    Supports both legacy single element type and new mixed element types:
    - Legacy: Use `elements` and `element_type` parameters
    - New: Use `element_groups` parameter (Dict[ElementType, np.ndarray])

    The class automatically converts between formats for backward compatibility.
    """
    # NEW: Support multiple element types via element_groups
    element_groups: Dict[ElementType, np.ndarray] = field(default_factory=dict)

    # Legacy fields - still accepted for backward compatibility
    elements: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    element_type: ElementType = ElementType.HEX8

    element_sets: Dict[str, ElementSet] = field(default_factory=dict)
    materials: Dict[str, Material] = field(default_factory=dict)
    surface_node_ids: Optional[np.ndarray] = None  # Cached surface node indices

    def __post_init__(self):
        super().__post_init__()

        # Convert elements to numpy array
        self.elements = np.asarray(self.elements, dtype=np.int64)

        # Reshape if needed
        if self.elements.ndim == 1 and len(self.elements) > 0:
            nodes_per_elem = ELEMENT_NODES[self.element_type]
            self.elements = self.elements.reshape(-1, nodes_per_elem)

        # Handle backward compatibility: if elements is provided but not element_groups,
        # convert elements to element_groups
        if len(self.elements) > 0 and not self.element_groups:
            self.element_groups[self.element_type] = self.elements

        # Ensure all element_groups arrays are proper numpy arrays
        for elem_type, elems in list(self.element_groups.items()):
            self.element_groups[elem_type] = np.asarray(elems, dtype=np.int64)

        # If element_groups is provided, sync the legacy fields for backward compat
        if self.element_groups and len(self.elements) == 0:
            # Set element_type to the primary type (most elements)
            self.element_type = max(self.element_groups.keys(),
                                   key=lambda t: len(self.element_groups[t]))
            # Set elements to primary group
            self.elements = self.element_groups[self.element_type]

    @property
    def num_elements(self) -> int:
        """Total number of elements across all groups."""
        if self.element_groups:
            return sum(len(elems) for elems in self.element_groups.values())
        return len(self.elements)

    @property
    def is_shell(self) -> bool:
        """True when this mesh is purely 2D shell elements (quad4/tri3).

        A shell mesh has no interior volume -- the elements themselves form the
        surface -- so triage treats it differently from a solid: no boundary
        extraction (the mesh *is* the boundary) and 2D shell quality metrics
        rather than the hex/tet scaled Jacobian. Returns ``False`` for a mixed
        solid+shell mesh (those are handled on the solid path for now).
        """
        groups = self.element_groups
        if not groups:
            groups = {self.element_type: self.elements} if len(self.elements) else {}
        has_shell = any(t in SHELL_TYPES and len(e) > 0 for t, e in groups.items())
        has_solid = any(t not in SHELL_TYPES and len(e) > 0 for t, e in groups.items())
        return has_shell and not has_solid

    def get_surface_nodes(self) -> np.ndarray:
        """Get indices of nodes on the surface (boundary)."""
        if self.surface_node_ids is not None:
            return self.surface_node_ids

        # Check for pre-defined SurfaceNodes NodeSet (preferred - handles multi-part correctly)
        if "SurfaceNodes" in self.node_sets:
            self.surface_node_ids = self.node_sets["SurfaceNodes"].node_ids
            return self.surface_node_ids

        # Fallback: find boundary faces by counting face occurrences
        # Iterate over all element groups
        face_counts = {}

        if self.element_groups:
            for elem_type, elements in self.element_groups.items():
                try:
                    face_map = self._get_element_faces(elem_type)
                    for elem in elements:
                        for face_nodes in face_map(elem):
                            face_key = tuple(sorted(face_nodes))
                            face_counts[face_key] = face_counts.get(face_key, 0) + 1
                except NotImplementedError:
                    # Skip element types without face definitions
                    continue
        else:
            # Legacy single-type path
            face_map = self._get_element_faces()
            for elem in self.elements:
                for face_nodes in face_map(elem):
                    face_key = tuple(sorted(face_nodes))
                    face_counts[face_key] = face_counts.get(face_key, 0) + 1

        # Surface faces appear exactly once
        surface_nodes = set()
        for face_key, count in face_counts.items():
            if count == 1:
                surface_nodes.update(face_key)

        self.surface_node_ids = np.array(sorted(surface_nodes), dtype=np.int64)
        return self.surface_node_ids

    def _get_element_faces(self, element_type: Optional[ElementType] = None):
        """Return function to get faces for element type.

        Args:
            element_type: The element type to get faces for. If None, uses primary type.
        """
        if element_type is None:
            element_type = self.element_type

        if element_type == ElementType.HEX8:
            def faces(elem):
                return [
                    [elem[0], elem[1], elem[2], elem[3]],  # bottom
                    [elem[4], elem[5], elem[6], elem[7]],  # top
                    [elem[0], elem[1], elem[5], elem[4]],  # front
                    [elem[2], elem[3], elem[7], elem[6]],  # back
                    [elem[0], elem[3], elem[7], elem[4]],  # left
                    [elem[1], elem[2], elem[6], elem[5]],  # right
                ]
            return faces
        elif element_type == ElementType.TET4:
            def faces(elem):
                return [
                    [elem[0], elem[1], elem[2]],
                    [elem[0], elem[1], elem[3]],
                    [elem[1], elem[2], elem[3]],
                    [elem[0], elem[2], elem[3]],
                ]
            return faces
        elif element_type == ElementType.WEDGE6:
            def faces(elem):
                return [
                    [elem[0], elem[1], elem[2]],           # bottom triangle
                    [elem[3], elem[4], elem[5]],           # top triangle
                    [elem[0], elem[1], elem[4], elem[3]],  # quad face 1
                    [elem[1], elem[2], elem[5], elem[4]],  # quad face 2
                    [elem[2], elem[0], elem[3], elem[5]],  # quad face 3
                ]
            return faces
        elif element_type == ElementType.QUAD4:
            # A shell element *is* its own face -- no boundary to extract.
            def faces(elem):
                return [[elem[0], elem[1], elem[2], elem[3]]]
            return faces
        elif element_type == ElementType.TRI3:
            def faces(elem):
                return [[elem[0], elem[1], elem[2]]]
            return faces
        else:
            raise NotImplementedError(f"Face extraction not implemented for {element_type}")

    def _extract_shell_surface(self) -> SurfaceMesh:
        """Surface of a pure shell mesh: the elements themselves, triangulated.

        Quads split into two triangles ([0,1,2] + [0,2,3]); tris pass through.
        Winding is the element's own (shells render double-sided), so no
        centroid-based orientation pass is needed.
        """
        triangles: List[np.ndarray] = []
        for etype in (ElementType.QUAD4, ElementType.TRI3):
            elems = self.element_groups.get(etype)
            if elems is None or len(elems) == 0:
                continue
            if etype == ElementType.QUAD4:
                triangles.append(elems[:, [0, 1, 2]])
                triangles.append(elems[:, [0, 2, 3]])
            else:
                triangles.append(elems[:, [0, 1, 2]])
        faces_array = (
            np.concatenate(triangles) if triangles
            else np.zeros((0, 3), dtype=np.int64)
        )
        surface = SurfaceMesh(
            nodes=self.nodes.copy(),
            faces=faces_array,
            face_type=ElementType.TRI3,
            name=f"{self.name}_surface",
        )
        surface.compute_normals()
        return surface

    def extract_surface(self) -> SurfaceMesh:
        """Extract the surface mesh from this volume mesh."""
        # A pure shell mesh has no interior: every element is already a surface
        # facet, so boundary extraction (keep faces seen once) would be both
        # wrong and unnecessary. Build the surface directly from the shells.
        if self.element_groups and self.is_shell:
            return self._extract_shell_surface()

        # Fast path: numpy-only boundary extraction for the element types whose
        # face indexing is statically known (TET4 / HEX8 / WEDGE6). On large
        # meshes the per-element Python loop with ``tuple(sorted)`` face keys
        # used to dominate triage start; the vectorized version below does the
        # same work in a few hundred ms by deduping with np.unique. Falls back
        # to the legacy loop only when no supported types are present (e.g. a
        # mesh of HEX20 or TET10 elements, which the legacy loop also can't
        # actually extract -- it'd skip them via NotImplementedError).
        if self.element_groups:
            return self._extract_surface_vectorized()

        face_counts = {}
        all_faces = []  # List of (face_nodes, is_quad, elem_centroid) tuples

        # Iterate over all element groups
        if self.element_groups:
            for elem_type, elements in self.element_groups.items():
                try:
                    face_map = self._get_element_faces(elem_type)
                    for elem in elements:
                        elem_centroid = self.nodes[elem].mean(axis=0)
                        for face_nodes in face_map(elem):
                            face_key = tuple(sorted(face_nodes))
                            is_quad = len(face_nodes) == 4
                            if face_key not in face_counts:
                                face_counts[face_key] = 0
                                all_faces.append((face_nodes, is_quad, elem_centroid))
                            face_counts[face_key] += 1
                except NotImplementedError:
                    continue
        else:
            # Legacy single-type path
            face_map = self._get_element_faces()
            for elem in self.elements:
                elem_centroid = self.nodes[elem].mean(axis=0)
                for face_nodes in face_map(elem):
                    face_key = tuple(sorted(face_nodes))
                    is_quad = len(face_nodes) == 4
                    if face_key not in face_counts:
                        face_counts[face_key] = 0
                        all_faces.append((face_nodes, is_quad, elem_centroid))
                    face_counts[face_key] += 1

        # Keep only boundary faces (count == 1) and triangulate
        triangles = []
        for face_nodes, is_quad, elem_centroid in all_faces:
            face_key = tuple(sorted(face_nodes))
            if face_counts[face_key] == 1:
                # Ensure outward orientation (normal points away from element centroid)
                coords = self.nodes[np.array(face_nodes, dtype=np.int64)]
                face_center = coords.mean(axis=0)
                # Use first 3 nodes for normal (quad or tri)
                v1 = coords[1] - coords[0]
                v2 = coords[2] - coords[0]
                normal = np.cross(v1, v2)
                if np.dot(normal, elem_centroid - face_center) > 0:
                    face_nodes = list(reversed(face_nodes))

                if is_quad:
                    # Triangulate quad face
                    triangles.append([face_nodes[0], face_nodes[1], face_nodes[2]])
                    triangles.append([face_nodes[0], face_nodes[2], face_nodes[3]])
                else:
                    # Already a triangle
                    triangles.append(face_nodes)

        # ``np.array([], dtype=np.int64)`` is 1-D (shape (0,)), which then
        # breaks ``compute_normals`` -- force the (n, 3) shape even when no
        # boundary faces exist.
        faces_array = (
            np.array(triangles, dtype=np.int64) if triangles
            else np.zeros((0, 3), dtype=np.int64)
        )

        surface = SurfaceMesh(
            nodes=self.nodes.copy(),
            faces=faces_array,
            face_type=ElementType.TRI3,
            name=f"{self.name}_surface",
        )
        surface.compute_normals()
        return surface

    # Face indexing per element type, matching ``_get_element_faces``. Each
    # entry is a list of (face_size, face_index_array) -- WEDGE6 has both
    # triangle and quad faces. Types absent from this map fall back to the
    # legacy loop (which also can't actually extract their surfaces today).
    _SURFACE_FACE_SPECS: ClassVar[Dict[ElementType, List[Tuple[int, np.ndarray]]]] = {
        ElementType.TET4: [
            (3, np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]], dtype=np.int64)),
        ],
        ElementType.HEX8: [
            (4, np.array([
                [0, 1, 2, 3],   # bottom
                [4, 5, 6, 7],   # top
                [0, 1, 5, 4],   # front
                [2, 3, 7, 6],   # back
                [0, 3, 7, 4],   # left
                [1, 2, 6, 5],   # right
            ], dtype=np.int64)),
        ],
        ElementType.WEDGE6: [
            (3, np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)),
            (4, np.array([
                [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5],
            ], dtype=np.int64)),
        ],
    }

    def _extract_surface_vectorized(self) -> SurfaceMesh:
        """Numpy-only boundary extraction for TET4/HEX8/WEDGE6 meshes (any
        combination). Returns an empty ``SurfaceMesh`` if no element type in
        this mesh has a known face spec -- matches the legacy loop's behaviour
        for unsupported-only meshes (e.g. pure HEX20/TET10), without exercising
        the legacy loop's empty-shape footgun.
        """
        specs = type(self)._SURFACE_FACE_SPECS
        supported = [(t, e) for t, e in self.element_groups.items() if t in specs]
        if not supported:
            return SurfaceMesh(
                nodes=self.nodes.copy(),
                faces=np.zeros((0, 3), dtype=np.int64),
                face_type=ElementType.TRI3,
                name=f"{self.name}_surface",
            )

        # Concatenated element-centroid array indexed by a global element id.
        # Storing the offset per element type lets us tag each face with the
        # global id of its owning element so orientation can index into one
        # array regardless of which type the face came from.
        centroids_per_type = [self.nodes[e].mean(axis=1) for _, e in supported]
        elem_centroids = (
            np.concatenate(centroids_per_type) if centroids_per_type
            else np.zeros((0, 3), dtype=np.float64)
        )
        offsets = np.cumsum([0] + [len(c) for c in centroids_per_type])

        # Faces grouped by face size so same-size faces from different element
        # types (e.g. a tet tri and a wedge tri) dedupe against each other.
        by_size: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
        for tid, (etype, elements) in enumerate(supported):
            n_elems = len(elements)
            if n_elems == 0:
                continue
            global_ids = np.arange(offsets[tid], offsets[tid + 1], dtype=np.int64)
            for face_size, face_idx in specs[etype]:
                faces = elements[:, face_idx].reshape(-1, face_size)
                owners = np.repeat(global_ids, len(face_idx))
                bucket = by_size.setdefault(face_size, ([], []))
                bucket[0].append(faces)
                bucket[1].append(owners)

        triangles_out: List[np.ndarray] = []
        for face_size, (face_arrs, owner_arrs) in by_size.items():
            faces = np.concatenate(face_arrs)
            owners = np.concatenate(owner_arrs)
            keys = np.sort(faces, axis=1)
            _uniq, inverse, counts = np.unique(
                keys, axis=0, return_inverse=True, return_counts=True,
            )
            mask = counts[inverse] == 1
            if not mask.any():
                continue
            b_faces = faces[mask].copy()
            b_owners = owners[mask]

            # Outward orientation: normal from the first two edges; flip if
            # it points toward the owning element's centroid (same test as
            # the legacy loop). For a quad this is the normal of the first
            # three vertices; the slow path uses the same approximation.
            P = self.nodes[b_faces]
            normals = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            face_centers = P.mean(axis=1)
            flip = (normals * (elem_centroids[b_owners] - face_centers)).sum(axis=1) > 0
            if flip.any():
                b_faces[flip] = b_faces[flip, ::-1]

            if face_size == 3:
                triangles_out.append(b_faces)
            elif face_size == 4:
                # Triangulate quads as [a,b,c] + [a,c,d], matching the legacy
                # loop. Reversed winding from the flip propagates correctly:
                # a reversed [a,b,c,d] = [d,c,b,a] still triangulates outward.
                triangles_out.append(b_faces[:, [0, 1, 2]])
                triangles_out.append(b_faces[:, [0, 2, 3]])
            # Larger face sizes (e.g. higher-order quads) are not supported by
            # this fast path; types that produce them aren't in the spec map.

        faces_array = (
            np.concatenate(triangles_out) if triangles_out
            else np.zeros((0, 3), dtype=np.int64)
        )
        surface = SurfaceMesh(
            nodes=self.nodes.copy(),
            faces=faces_array,
            face_type=ElementType.TRI3,
            name=f"{self.name}_surface",
        )
        surface.compute_normals()
        return surface

    def compute_element_jacobians(self, check_points: str = "center", use_gpu: Optional[bool] = None) -> np.ndarray:
        """
        Compute Jacobian determinants for all elements.

        Args:
            check_points: Where to evaluate ('center', 'corners', 'gauss', 'all')
            use_gpu: Whether to use GPU acceleration (None = auto-detect)

        Returns:
            Array of minimum Jacobian determinants per element.
            For mixed meshes, returns Jacobians for supported element types only (HEX8).
            Unsupported types (e.g., WEDGE6) are skipped and not included in the output.
        """
        # Auto-detect: use PyTorch if available (GPU or CPU vectorized is faster)
        if use_gpu is None:
            use_gpu = TORCH_AVAILABLE

        all_jacobians = []

        # Handle element_groups (new format)
        if self.element_groups:
            skipped = {}
            for elem_type, elements in self.element_groups.items():
                if elem_type == ElementType.HEX8:
                    if use_gpu and TORCH_AVAILABLE:
                        jacs = self._compute_hex8_jacobians_gpu_for_elements(elements, check_points)
                    else:
                        jacs = np.zeros(len(elements))
                        for i, elem in enumerate(elements):
                            elem_nodes = self.nodes[elem]
                            jacs[i] = self._compute_hex8_jacobian(elem_nodes, check_points)
                    all_jacobians.append(jacs)
                elif elem_type == ElementType.TET4:
                    all_jacobians.append(self._compute_tet4_jacobians(elements))
                elif elem_type in SHELL_TYPES:
                    # 2D shells: raw min corner Jacobian (signed parallelogram
                    # area). check_points is irrelevant for linear shells.
                    all_jacobians.append(shell_corner_min_jacobian(self.nodes[elements]))
                else:
                    # Jacobian not implemented for this type (WEDGE6, HEX20, TET10, ...)
                    skipped[elem_type] = len(elements)

            if skipped:
                import warnings
                desc = ", ".join(f"{t.value}: {n}" for t, n in skipped.items())
                warnings.warn(
                    f"compute_element_jacobians skipped unsupported element types ({desc}); "
                    f"their validity is NOT checked.",
                    stacklevel=2,
                )

            if all_jacobians:
                return np.concatenate(all_jacobians)
            return np.array([])

        # Legacy single-type path
        if self.element_type == ElementType.TET4:
            return self._compute_tet4_jacobians(self.elements)
        if self.element_type in SHELL_TYPES:
            return shell_corner_min_jacobian(self.nodes[self.elements])
        if self.element_type != ElementType.HEX8:
            raise NotImplementedError(f"Jacobian computation not implemented for {self.element_type}")

        # Use PyTorch-accelerated version if available (works on GPU or CPU)
        if use_gpu and TORCH_AVAILABLE:
            return self._compute_hex8_jacobians_gpu(check_points)

        # Fall back to CPU version
        jacobians = np.zeros(len(self.elements))

        for i, elem in enumerate(self.elements):
            elem_nodes = self.nodes[elem]
            jacobians[i] = self._compute_hex8_jacobian(elem_nodes, check_points)

        return jacobians

    def _compute_tet4_jacobians(self, elements: np.ndarray) -> np.ndarray:
        """Jacobian determinant for linear tets: det([n1-n0, n2-n0, n3-n0]).

        For a constant-strain TET4 the Jacobian is constant over the element, so
        ``check_points`` is irrelevant. The determinant equals 6x the signed
        volume; a positive value means correct (right-handed) orientation, so the
        ``> threshold`` invalid-element test matches the HEX8 sign convention.
        """
        if len(elements) == 0:
            return np.array([])
        P = self.nodes[np.asarray(elements)]          # (n, 4, 3)
        v1 = P[:, 1] - P[:, 0]
        v2 = P[:, 2] - P[:, 0]
        v3 = P[:, 3] - P[:, 0]
        return np.einsum('ij,ij->i', np.cross(v1, v2), v3)

    def _compute_hex8_jacobians_gpu_for_elements(self, elements: np.ndarray, check_points: str) -> np.ndarray:
        """GPU-accelerated Jacobian computation for a specific set of hex8 elements."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for GPU acceleration")

        # Select device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get check points in natural coordinates
        if check_points == "center":
            xi_points = np.array([[0.0, 0.0, 0.0]])
        elif check_points == "corners":
            xi_points = np.array([
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
            ], dtype=np.float64)
        else:  # gauss
            g = 1.0 / np.sqrt(3)
            xi_points = np.array([
                [-g, -g, -g], [g, -g, -g], [g, g, -g], [-g, g, -g],
                [-g, -g, g], [g, -g, g], [g, g, g], [-g, g, g],
            ], dtype=np.float64)

        n_elem = len(elements)

        # Transfer data to device
        nodes_t = torch.tensor(self.nodes, dtype=torch.float64, device=device)
        elements_t = torch.tensor(elements, dtype=torch.int64, device=device)
        xi_t = torch.tensor(xi_points, dtype=torch.float64, device=device)

        # Get element node coordinates: [n_elem, 8, 3]
        elem_coords = nodes_t[elements_t]

        # Compute shape function derivatives for all check points
        xi = xi_t[:, 0:1]
        eta = xi_t[:, 1:2]
        zeta = xi_t[:, 2:3]

        # Shape function derivatives for hex8 (vectorized over points)
        dN_dxi = torch.stack([
            torch.stack([-(1-eta)*(1-zeta), -(1-xi)*(1-zeta), -(1-xi)*(1-eta)], dim=-1),
            torch.stack([(1-eta)*(1-zeta), -(1+xi)*(1-zeta), -(1+xi)*(1-eta)], dim=-1),
            torch.stack([(1+eta)*(1-zeta), (1+xi)*(1-zeta), -(1+xi)*(1+eta)], dim=-1),
            torch.stack([-(1+eta)*(1-zeta), (1-xi)*(1-zeta), -(1-xi)*(1+eta)], dim=-1),
            torch.stack([-(1-eta)*(1+zeta), -(1-xi)*(1+zeta), (1-xi)*(1-eta)], dim=-1),
            torch.stack([(1-eta)*(1+zeta), -(1+xi)*(1+zeta), (1+xi)*(1-eta)], dim=-1),
            torch.stack([(1+eta)*(1+zeta), (1+xi)*(1+zeta), (1+xi)*(1+eta)], dim=-1),
            torch.stack([-(1+eta)*(1+zeta), (1-xi)*(1+zeta), (1-xi)*(1+eta)], dim=-1),
        ], dim=1) / 8.0

        dN_dxi = dN_dxi.squeeze(-2)

        # Compute Jacobian matrices for all elements and all points
        J = torch.einsum('pki,ekj->epij', dN_dxi, elem_coords)

        # Compute determinants
        det_J = torch.linalg.det(J)

        # Get minimum Jacobian per element
        min_jac_per_elem = det_J.min(dim=1).values

        return min_jac_per_elem.cpu().numpy()

    def _compute_hex8_jacobian(self, nodes: np.ndarray, check_points: str) -> float:
        """Compute minimum Jacobian for a single hex8 element."""
        if check_points == "center":
            points = [(0, 0, 0)]
        elif check_points == "corners":
            points = [
                (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
            ]
        else:  # gauss or all
            g = 1.0 / np.sqrt(3)
            points = [
                (-g, -g, -g), (g, -g, -g), (g, g, -g), (-g, g, -g),
                (-g, -g, g), (g, -g, g), (g, g, g), (-g, g, g),
            ]

        min_jac = float('inf')
        for xi, eta, zeta in points:
            jac = self._eval_hex8_jacobian_at_point(nodes, xi, eta, zeta)
            min_jac = min(min_jac, jac)

        return min_jac

    def _eval_hex8_jacobian_at_point(self, nodes: np.ndarray, xi: float, eta: float, zeta: float) -> float:
        """Evaluate Jacobian determinant at a point in natural coordinates."""
        # Shape function derivatives for hex8
        dN_dxi = np.array([
            [-(1-eta)*(1-zeta), -(1-xi)*(1-zeta), -(1-xi)*(1-eta)],
            [(1-eta)*(1-zeta), -(1+xi)*(1-zeta), -(1+xi)*(1-eta)],
            [(1+eta)*(1-zeta), (1+xi)*(1-zeta), -(1+xi)*(1+eta)],
            [-(1+eta)*(1-zeta), (1-xi)*(1-zeta), -(1-xi)*(1+eta)],
            [-(1-eta)*(1+zeta), -(1-xi)*(1+zeta), (1-xi)*(1-eta)],
            [(1-eta)*(1+zeta), -(1+xi)*(1+zeta), (1+xi)*(1-eta)],
            [(1+eta)*(1+zeta), (1+xi)*(1+zeta), (1+xi)*(1+eta)],
            [-(1+eta)*(1+zeta), (1-xi)*(1+zeta), (1-xi)*(1+eta)],
        ]) / 8.0

        # Jacobian matrix: J = dN/dxi * nodes
        J = dN_dxi.T @ nodes

        return np.linalg.det(J)

    def _compute_hex8_jacobians_gpu(self, check_points: str) -> np.ndarray:
        """
        GPU-accelerated Jacobian computation for all hex8 elements.

        Uses PyTorch for parallel computation across all elements and check points.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for GPU acceleration")

        # Select device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get check points in natural coordinates
        if check_points == "center":
            xi_points = np.array([[0.0, 0.0, 0.0]])
        elif check_points == "corners":
            xi_points = np.array([
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
            ], dtype=np.float64)
        else:  # gauss
            g = 1.0 / np.sqrt(3)
            xi_points = np.array([
                [-g, -g, -g], [g, -g, -g], [g, g, -g], [-g, g, -g],
                [-g, -g, g], [g, -g, g], [g, g, g], [-g, g, g],
            ], dtype=np.float64)

        n_points = len(xi_points)
        n_elem = self.num_elements

        # Transfer data to device
        nodes_t = torch.tensor(self.nodes, dtype=torch.float64, device=device)
        elements_t = torch.tensor(self.elements, dtype=torch.int64, device=device)
        xi_t = torch.tensor(xi_points, dtype=torch.float64, device=device)

        # Get element node coordinates: [n_elem, 8, 3]
        elem_coords = nodes_t[elements_t]

        # Compute shape function derivatives for all check points
        # dN_dxi[point, node, dim]: [n_points, 8, 3]
        xi = xi_t[:, 0:1]  # [n_points, 1]
        eta = xi_t[:, 1:2]
        zeta = xi_t[:, 2:3]

        # Shape function derivatives for hex8 (vectorized over points)
        dN_dxi = torch.stack([
            torch.stack([-(1-eta)*(1-zeta), -(1-xi)*(1-zeta), -(1-xi)*(1-eta)], dim=-1),
            torch.stack([(1-eta)*(1-zeta), -(1+xi)*(1-zeta), -(1+xi)*(1-eta)], dim=-1),
            torch.stack([(1+eta)*(1-zeta), (1+xi)*(1-zeta), -(1+xi)*(1+eta)], dim=-1),
            torch.stack([-(1+eta)*(1-zeta), (1-xi)*(1-zeta), -(1-xi)*(1+eta)], dim=-1),
            torch.stack([-(1-eta)*(1+zeta), -(1-xi)*(1+zeta), (1-xi)*(1-eta)], dim=-1),
            torch.stack([(1-eta)*(1+zeta), -(1+xi)*(1+zeta), (1+xi)*(1-eta)], dim=-1),
            torch.stack([(1+eta)*(1+zeta), (1+xi)*(1+zeta), (1+xi)*(1+eta)], dim=-1),
            torch.stack([-(1+eta)*(1+zeta), (1-xi)*(1+zeta), (1-xi)*(1+eta)], dim=-1),
        ], dim=1) / 8.0  # [n_points, 8, 3] after squeeze

        # Squeeze out the extra dimension from broadcasting
        dN_dxi = dN_dxi.squeeze(-2)  # [n_points, 8, 3]

        # Compute Jacobian matrices for all elements and all points
        # J[elem, point] = dN_dxi[point].T @ elem_coords[elem]
        # J shape: [n_elem, n_points, 3, 3]
        # Using einsum: J_ij = sum_k dN_dxi[p,k,i] * coords[e,k,j]
        # dN_dxi: [p, k, i=3] -> [p, 8, 3]
        # elem_coords: [e, k, j=3] -> [n_elem, 8, 3]
        # J: [e, p, i, j] -> [n_elem, n_points, 3, 3]
        J = torch.einsum('pki,ekj->epij', dN_dxi, elem_coords)

        # Compute determinants: [n_elem, n_points]
        det_J = torch.linalg.det(J)

        # Get minimum Jacobian per element
        min_jac_per_elem = det_J.min(dim=1).values

        return min_jac_per_elem.cpu().numpy()

    def has_valid_jacobians(self, threshold: float = 1e-6) -> bool:
        """Check if all elements have positive Jacobians."""
        jacobians = self.compute_element_jacobians(check_points="corners")
        return np.all(jacobians > threshold)

    def get_invalid_elements(self, threshold: float = 1e-6) -> np.ndarray:
        """Get indices of elements with invalid (non-positive) Jacobians."""
        jacobians = self.compute_element_jacobians(check_points="corners")
        return np.where(jacobians <= threshold)[0]
