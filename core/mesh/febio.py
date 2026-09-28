"""
FEBio (.feb) mesh file loader and saver.

Parses FEBio XML files to extract mesh data and write morphed meshes.
Supports FEBio spec versions 2.x, 3.x, and 4.x.

--------------------------------------------------------------------------
Vendored from: C:\\Users\\lhudson\\Documents\\Morphing\\fe_personalization\\mesh_io\\febio.py
Vendored on:   2026-09-18, unmodified.
Local changes: none yet.
See AGENTS.md "Vendoring policy" and section 2.5 (FEBio case assembly) --
this reader/writer handles mesh geometry; febio/ (this project's own
package) is responsible for Control/Material/Contact/Boundary/LoadData,
which is written from the gold-standard reference case, not from this file.
--------------------------------------------------------------------------
"""

import logging
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .base import VolumeMesh, SurfaceMesh, NodeSet, ElementSet, Material, ElementType

logger = logging.getLogger(__name__)


def load_febio(filepath: str, element_set: Optional[str] = None) -> VolumeMesh:
    """
    Load a mesh from an FEBio .feb file.

    Args:
        filepath: Path to the .feb file
        element_set: Optional name of specific element set to load.
                    If None, loads all elements.

    Returns:
        VolumeMesh object

    Raises:
        ValueError: If no suitable elements found
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    tree = ET.parse(filepath)
    root = tree.getroot()

    # Get FEBio version
    version = root.get('version', '2.0')

    # Handle different FEBio spec versions
    mesh_section = root.find('.//Mesh')
    if mesh_section is None:
        mesh_section = root.find('.//Geometry')
    if mesh_section is None:
        raise ValueError("Could not find Mesh or Geometry section in .feb file")

    # Parse nodes. Some FEB templates split the mesh across multiple <Nodes>
    # blocks. We collect ids in declaration order and compact to a dense
    # (N, 3) array -- a file with sparse ids (e.g. 152k nodes whose max id
    # is 8M, common in some lower-limb templates) used to inflate
    # ``nodes`` to (max_id, 3) and poison every downstream consumer sized by
    # num_nodes (neighbor lists, JSON viewer payloads, bounding-box centers).
    # The original ids are kept in metadata so savers can write them back.
    nodes_sections = list(mesh_section.findall('Nodes'))
    if not nodes_sections:
        nodes_sections = list(mesh_section.findall('.//Nodes'))

    if not nodes_sections:
        raise ValueError("Could not find Nodes section in .feb file")

    node_order: List[int] = []           # original ids in declaration order
    node_coords_list: List[List[float]] = []
    seen_id_to_index: Dict[int, int] = {}  # orig id -> position in node_order
    for nodes_section in nodes_sections:
        for node in nodes_section.findall('node'):
            nid = int(node.get('id'))
            coords = [float(x) for x in node.text.strip().split(',')]
            if nid in seen_id_to_index:
                node_coords_list[seen_id_to_index[nid]] = coords  # last wins
            else:
                seen_id_to_index[nid] = len(node_order)
                node_order.append(nid)
                node_coords_list.append(coords)

    if not node_order:
        raise ValueError("No nodes parsed from .feb file")

    nodes = np.asarray(node_coords_list, dtype=np.float64)
    orig_to_compact: Dict[int, int] = {oid: i for i, oid in enumerate(node_order)}
    original_node_ids = node_order

    # Parse elements - support multiple element types via element_groups
    element_groups = {}  # Dict[ElementType, np.ndarray]
    element_sets_dict = {}
    element_count_by_type = {}  # For tracking element set indices per type

    for elements_section in mesh_section.findall('Elements'):
        elem_type_str = elements_section.get('type', '').lower()
        elem_name = elements_section.get('name', 'default')

        # Filter by element set if specified
        if element_set is not None and elem_name != element_set:
            continue

        # Determine element type
        current_type = _parse_element_type(elem_type_str)
        if current_type is None:
            print(f"Warning: Unknown element type '{elem_type_str}' in section '{elem_name}', skipping.")
            continue

        # Initialize tracking for this element type if needed
        if current_type not in element_count_by_type:
            element_count_by_type[current_type] = 0

        # Collect elements of this type
        elements_list = []
        elem_ids = []  # Element IDs for this section (relative to total of this type)

        for elem in elements_section.findall('elem'):
            # Parse connectivity (1-based original ids in FEB) and remap to the
            # dense compact indices we assigned during node parsing.
            try:
                node_ids = [orig_to_compact[int(x)] for x in elem.text.strip().split(',')]
            except KeyError as exc:
                raise ValueError(
                    f"Element references unknown node id {exc.args[0]} in {filepath.name}"
                )
            elements_list.append(node_ids)
            elem_ids.append(element_count_by_type[current_type] + len(elements_list) - 1)

        if elements_list:
            elements_array = np.array(elements_list, dtype=np.int64)

            # Add to or create element group for this type
            if current_type in element_groups:
                element_groups[current_type] = np.vstack([
                    element_groups[current_type],
                    elements_array
                ])
            else:
                element_groups[current_type] = elements_array

            # Track element set
            if elem_ids:
                element_sets_dict[elem_name] = ElementSet(
                    name=elem_name,
                    element_ids=np.array(elem_ids, dtype=np.int64)
                )

            # Update count for this type
            element_count_by_type[current_type] = len(element_groups[current_type])

    if not element_groups:
        raise ValueError("No supported elements found in .feb file")

    # Log element group info
    total_elements = sum(len(elems) for elems in element_groups.values())
    print(f"Loaded {total_elements} elements in {len(element_groups)} group(s):")
    for elem_type, elems in element_groups.items():
        print(f"  {elem_type.value}: {len(elems)} elements")

    # Parse node sets. Original ids are remapped to compact indices; ids that
    # don't resolve to a declared node (orphans) are dropped silently -- the
    # set still loads but references only real nodes.
    def _map_node_id(raw: int) -> Optional[int]:
        return orig_to_compact.get(raw)

    node_sets_dict = {}
    for nodeset in mesh_section.findall('.//NodeSet'):
        ns_name = nodeset.get('name', 'unnamed')
        raw_ids: List[int] = []
        for node in nodeset.findall('n'):  # FEBio 2.x/3.x format
            raw_ids.append(int(node.get('id')))
        for node in nodeset.findall('node'):  # Alternative format
            raw_ids.append(int(node.get('id')))
        # Also check for comma-separated format
        if not raw_ids and nodeset.text:
            try:
                raw_ids = [int(x) for x in nodeset.text.strip().split(',') if x.strip()]
            except ValueError:
                pass
        node_ids = [c for c in (_map_node_id(r) for r in raw_ids) if c is not None]
        if node_ids:
            node_sets_dict[ns_name] = NodeSet(
                name=ns_name,
                node_ids=np.array(node_ids, dtype=np.int64)
            )

    # Parse materials
    materials_dict = {}
    material_section = root.find('.//Material')
    if material_section is not None:
        for mat in material_section.findall('material'):
            mat_id = mat.get('id', '1')
            mat_name = mat.get('name', f'Material_{mat_id}')
            mat_type = mat.get('type', 'unknown')

            props = {}
            for prop in mat:
                if prop.text:
                    try:
                        props[prop.tag] = float(prop.text)
                    except ValueError:
                        props[prop.tag] = prop.text

            materials_dict[mat_name] = Material(
                name=mat_name,
                material_type=mat_type,
                properties=props
            )

    # Create mesh
    metadata = {
        'source_file': str(filepath),
        'febio_version': version,
        'original_node_ids': original_node_ids,  # For non-sequential node numbering
    }

    mesh = VolumeMesh(
        nodes=nodes,
        element_groups=element_groups,
        name=filepath.stem,
        node_sets=node_sets_dict,
        element_sets=element_sets_dict,
        materials=materials_dict,
        metadata=metadata
    )

    return mesh


def extract_element_sets(filepath: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract element sets with connectivity and element type from an FEBio file.

    Returns:
        Dict mapping set name -> {"type": ElementType, "elements": np.ndarray}
    """
    filepath = Path(filepath)
    tree = ET.parse(filepath)
    root = tree.getroot()

    mesh_section = root.find('.//Mesh')
    if mesh_section is None:
        mesh_section = root.find('.//Geometry')
    if mesh_section is None:
        raise ValueError("Could not find Mesh or Geometry section in .feb file")

    sets: Dict[str, Dict[str, np.ndarray]] = {}
    for elements_section in mesh_section.findall('Elements'):
        name = elements_section.get('name', 'default')
        type_str = elements_section.get('type', '').lower()
        elem_type = _parse_element_type(type_str)
        if elem_type is None:
            continue
        elems = []
        for elem in elements_section.findall('elem'):
            node_ids = [int(x) - 1 for x in elem.text.strip().split(',')]
            elems.append(node_ids)
        if elems:
            sets[name] = {
                "type": elem_type,
                "elements": np.array(elems, dtype=np.int64),
            }
    return sets


def _parse_element_type(type_str: str) -> Optional[ElementType]:
    """Parse element type string to ElementType enum."""
    type_map = {
        'hex8': ElementType.HEX8,
        'hex': ElementType.HEX8,
        'hexahedron': ElementType.HEX8,
        'hex20': ElementType.HEX20,
        'tet4': ElementType.TET4,
        'tet': ElementType.TET4,
        'tetrahedron': ElementType.TET4,
        'tet10': ElementType.TET10,
        'penta6': ElementType.WEDGE6,
        'wedge': ElementType.WEDGE6,
        'wedge6': ElementType.WEDGE6,
        'quad4': ElementType.QUAD4,
        'quad': ElementType.QUAD4,
        'tri3': ElementType.TRI3,
        'tri': ElementType.TRI3,
    }
    return type_map.get(type_str.lower())


def save_febio(mesh: VolumeMesh, filepath: str,
               source_filepath: Optional[str] = None) -> None:
    """
    Save a mesh to FEBio .feb format.

    If source_filepath is provided, preserves the original file structure
    and only updates node coordinates. Otherwise creates a new FEB file.

    Args:
        mesh: The VolumeMesh to save
        filepath: Output file path
        source_filepath: Original .feb file to use as template
    """
    if source_filepath is not None:
        _save_febio_from_template(mesh, filepath, source_filepath)
    else:
        _save_febio_minimal(mesh, filepath)


def _save_febio_from_template(mesh: VolumeMesh, filepath: str,
                               source_filepath: str) -> None:
    """
    Save mesh by updating node coordinates in an existing FEB file.

    Preferred path is the byte-for-byte minimal-diff patcher, which rewrites
    only the coordinate fields of nodes that actually moved and preserves every
    other byte (encoding, line endings, ids, element blocks, materials, BCs).
    Falls back to the re-serializing path below when the ``<Nodes>`` block isn't
    the simple one-node-per-line shape the patcher requires.
    """
    from .template_patch import patch_febio_nodes

    try:
        if patch_febio_nodes(mesh, source_filepath, filepath):
            return
    except Exception:
        logger.exception(
            "Byte-for-byte FEB patch failed for %s; falling back to "
            "re-serializing saver.", source_filepath,
        )
    _save_febio_from_template_reserialize(mesh, filepath, source_filepath)


def _save_febio_from_template_reserialize(mesh: VolumeMesh, filepath: str,
                                          source_filepath: str) -> None:
    """
    Save mesh by updating node coordinates in an existing FEB file.
    Preserves all other data (materials, boundary conditions, etc.).

    This re-serializes via ElementTree (the XML declaration encoding and
    whitespace may shift), so it is correct but not byte-for-byte. Used as the
    fallback when ``patch_febio_nodes`` can't handle the node-block shape.
    """
    tree = ET.parse(source_filepath)
    root = tree.getroot()

    # Find mesh section
    mesh_section = root.find('.//Mesh')
    if mesh_section is None:
        mesh_section = root.find('.//Geometry')
    if mesh_section is None:
        raise ValueError("Could not find Mesh or Geometry section in source .feb file")

    # Find all node sections. Some FEB templates split nodes across multiple blocks.
    nodes_sections = list(mesh_section.findall('Nodes'))
    if not nodes_sections:
        nodes_sections = list(mesh_section.findall('.//Nodes'))
    if not nodes_sections:
        raise ValueError("Could not find Nodes section in source .feb file")

    # Update node coordinates. The loader now compacts sparse ids into a dense
    # array, so look each source <node id="N"/> up through the
    # ``original_node_ids`` mapping in metadata. Falls back to ``node_id - 1``
    # for callers building a VolumeMesh by hand with dense 1-based ids.
    orig_ids = mesh.metadata.get("original_node_ids") if mesh.metadata else None
    orig_to_compact: Optional[Dict[int, int]] = (
        {oid: i for i, oid in enumerate(orig_ids)} if orig_ids else None
    )
    nodes_updated = 0
    for nodes_section in nodes_sections:
        for node_elem in nodes_section.findall('node'):
            node_id = int(node_elem.get('id'))
            if orig_to_compact is not None:
                mesh_idx = orig_to_compact.get(node_id, -1)
            else:
                mesh_idx = node_id - 1

            if 0 <= mesh_idx < len(mesh.nodes):
                coords = mesh.nodes[mesh_idx]
                node_elem.text = f"{coords[0]:.10g},{coords[1]:.10g},{coords[2]:.10g}"
                nodes_updated += 1

    # Write to file
    tree.write(filepath, encoding='utf-8', xml_declaration=True)

    print(f"Saved mesh to {filepath}:")
    print(f"  Updated {nodes_updated} node coordinates")


def _save_febio_minimal(mesh: VolumeMesh, filepath: str) -> None:
    """
    Save mesh to a FEB file with proper structure for FEBio import.
    Creates a valid FEBio 4.0 file with required sections.
    """
    # Build XML structure
    root = ET.Element('febio_spec', version="4.0")

    # Add Module section (required for FEBio to recognize the analysis type)
    ET.SubElement(root, 'Module', type="solid")

    # Add Globals section with default constants
    globals_section = ET.SubElement(root, 'Globals')
    constants = ET.SubElement(globals_section, 'Constants')
    ET.SubElement(constants, 'T').text = '0'
    ET.SubElement(constants, 'P').text = '0'
    ET.SubElement(constants, 'R').text = '8.31446'
    ET.SubElement(constants, 'Fc').text = '96485.3'

    # Add Mesh section
    mesh_section = ET.SubElement(root, 'Mesh')

    # Use mesh name or default part name
    part_name = mesh.name if mesh.name else "Part1"

    # Add Nodes section. ``original_node_ids`` (from metadata) maps each compact
    # row of mesh.nodes back to its original ID for writing. Callers that built
    # the mesh by hand fall back to sequential 1..N IDs.
    nodes_section = ET.SubElement(mesh_section, 'Nodes', name=part_name)
    metadata_ids = mesh.metadata.get('original_node_ids') if mesh.metadata else None
    if metadata_ids is not None and len(metadata_ids) == len(mesh.nodes):
        effective_ids = list(metadata_ids)
    else:
        effective_ids = list(range(1, len(mesh.nodes) + 1))

    for i, node in enumerate(mesh.nodes):
        node_elem = ET.SubElement(nodes_section, 'node', id=str(effective_ids[i]))
        node_elem.text = f"{node[0]:.10g}, {node[1]:.10g}, {node[2]:.10g}"

    # Element type to FEBio type string mapping
    type_map = {
        ElementType.HEX8: 'hex8',
        ElementType.HEX20: 'hex20',
        ElementType.TET4: 'tet4',
        ElementType.TET10: 'tet10',
        ElementType.WEDGE6: 'penta6',
        ElementType.QUAD4: 'quad4',
        ElementType.TRI3: 'tri3',
    }

    # Add Elements sections for each element group
    elem_id = 1  # Global element ID counter (1-based)
    if mesh.element_groups:
        for elem_type, elements in mesh.element_groups.items():
            elem_type_str = type_map.get(elem_type, 'hex8')
            section_name = f"{part_name}_{elem_type.value}" if len(mesh.element_groups) > 1 else part_name

            elements_section = ET.SubElement(mesh_section, 'Elements',
                                              type=elem_type_str,
                                              name=section_name)
            for elem in elements:
                elem_node = ET.SubElement(elements_section, 'elem', id=str(elem_id))
                # Map each compact node index back to its original id.
                elem_node.text = ','.join(str(effective_ids[n]) for n in elem)
                elem_id += 1
    else:
        # Legacy single-element-type path
        elem_type_str = type_map.get(mesh.element_type, 'hex8')
        elements_section = ET.SubElement(mesh_section, 'Elements',
                                          type=elem_type_str,
                                          name=part_name)
        for i, elem in enumerate(mesh.elements):
            elem_node = ET.SubElement(elements_section, 'elem', id=str(i + 1))
            elem_node.text = ','.join(str(effective_ids[n]) for n in elem)

    # Add node sets. node_set.node_ids are compact -> map back to original ids.
    for ns_name, node_set in mesh.node_sets.items():
        ns_elem = ET.SubElement(mesh_section, 'NodeSet', name=ns_name)
        for nid in node_set.node_ids:
            ET.SubElement(ns_elem, 'n', id=str(effective_ids[int(nid)]))

    # Add MeshDomains section (required for FEBio to link elements to domains)
    domains_section = ET.SubElement(root, 'MeshDomains')
    if mesh.element_groups and len(mesh.element_groups) > 1:
        # Create a domain for each element group
        for elem_type in mesh.element_groups.keys():
            section_name = f"{part_name}_{elem_type.value}"
            ET.SubElement(domains_section, 'SolidDomain', name=section_name, mat="")
    else:
        ET.SubElement(domains_section, 'SolidDomain', name=part_name, mat="")

    # Add empty Step section (required structure)
    ET.SubElement(root, 'Step')

    # Add Output section with default plot variables
    output_section = ET.SubElement(root, 'Output')
    plotfile = ET.SubElement(output_section, 'plotfile', type="febio")
    ET.SubElement(plotfile, 'var', type="displacement")
    ET.SubElement(plotfile, 'var', type="stress")
    ET.SubElement(plotfile, 'var', type="relative volume")

    # Write to file with nice formatting
    _indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(filepath, encoding='utf-8', xml_declaration=True)

    print(f"Saved mesh to {filepath}:")
    print(f"  Nodes: {mesh.num_nodes}")
    print(f"  Elements: {mesh.num_elements}")


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Add indentation to XML elements for pretty printing."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def get_febio_info(filepath: str) -> Dict[str, Any]:
    """
    Get information about a .feb file without fully loading it.

    Args:
        filepath: Path to the .feb file

    Returns:
        Dictionary with file information
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    info = {
        'version': root.get('version', 'unknown'),
        'node_count': 0,
        'element_sets': [],
        'node_sets': [],
        'materials': [],
    }

    # Find mesh section
    mesh_section = root.find('.//Mesh')
    if mesh_section is None:
        mesh_section = root.find('.//Geometry')

    if mesh_section is not None:
        # Count nodes
        nodes_section = mesh_section.find('Nodes')
        if nodes_section is not None:
            info['node_count'] = len(nodes_section.findall('node'))

        # List element sets
        for elements_section in mesh_section.findall('Elements'):
            elem_type = elements_section.get('type', 'unknown')
            elem_name = elements_section.get('name', 'unnamed')
            elem_count = len(elements_section.findall('elem'))
            info['element_sets'].append({
                'name': elem_name,
                'type': elem_type,
                'count': elem_count
            })

        # List node sets
        for nodeset in mesh_section.findall('.//NodeSet'):
            ns_name = nodeset.get('name', 'unnamed')
            info['node_sets'].append(ns_name)

    # List materials
    material_section = root.find('.//Material')
    if material_section is not None:
        for mat in material_section.findall('material'):
            mat_name = mat.get('name', 'unnamed')
            mat_type = mat.get('type', 'unknown')
            info['materials'].append({
                'name': mat_name,
                'type': mat_type
            })

    return info


def compact_febio_nodes(src_path: str, dst_path: str) -> Dict[str, int]:
    """Rewrite an FEB file so node IDs are sequential and orphan nodes are dropped.

    Some authoring tools (and a number of canonical lower-limb templates) emit
    FEB files with very gappy node IDs — for example ``Lower_Limb_Model_Surfaces.feb``
    declares 151,666 ``<node>`` entries but its max id is 8,191,884. ``load_febio``
    allocates ``np.zeros((max_id, 3))`` to accommodate that and the resulting
    8M-row node array poisons every downstream consumer: viewer bbox centers on
    the 98% zero rows, the morph runner copies it as the "combined output"
    template, output FEBs are 20MB+, and 3D rendering pans/zooms wrong.

    This function rewrites the file with:
      - ``<Nodes>`` containing only referenced nodes, renumbered 1..N
      - ``<elem>`` text rewritten so connectivity points at the new IDs
      - ``<NodeSet><n id=…>`` entries remapped to the new IDs (dropped if orphan)
      - Everything else (element-set names, materials, BCs, modules) copied
        through unchanged — this is a node-ID transformation, not a content
        rewrite.

    Returns a small stats dict: ``{"nodes_before": …, "nodes_after": …, "orphans_removed": …}``.

    Idempotent: rewriting an already-compact FEB yields the same file.
    """
    tree = ET.parse(src_path)
    root = tree.getroot()

    mesh_section = root.find('.//Mesh')
    if mesh_section is None:
        mesh_section = root.find('.//Geometry')
    if mesh_section is None:
        raise ValueError(f"Could not find Mesh or Geometry section in {src_path}")

    # 1) Collect all node ids that are *referenced* by any element. Unreferenced
    #    declared nodes are orphans and we'll drop them.
    referenced: set = set()
    for elements_section in mesh_section.findall('.//Elements'):
        for elem in elements_section.findall('elem'):
            if not elem.text:
                continue
            for tok in elem.text.replace('\n', ',').split(','):
                tok = tok.strip()
                if tok:
                    try:
                        referenced.add(int(tok))
                    except ValueError:
                        pass

    # 2) Walk every <Nodes> block and gather (orig_id, coords) for referenced
    #    ones in declaration order, building a stable remap.
    nodes_sections = list(mesh_section.findall('Nodes')) or list(
        mesh_section.findall('.//Nodes')
    )

    nodes_before = 0
    kept: List[Tuple[int, str, ET.Element]] = []  # (orig_id, coord_text, parent_section)
    remap: Dict[int, int] = {}
    next_id = 1
    for section in nodes_sections:
        for node in list(section.findall('node')):
            nodes_before += 1
            try:
                orig_id = int(node.get('id'))
            except (TypeError, ValueError):
                continue
            if orig_id not in referenced:
                # Orphan — will be removed.
                section.remove(node)
                continue
            remap[orig_id] = next_id
            kept.append((orig_id, node.text or '', section))
            node.set('id', str(next_id))
            next_id += 1

    # 3) Rewrite element connectivity through remap.
    for elements_section in mesh_section.findall('.//Elements'):
        for elem in elements_section.findall('elem'):
            if not elem.text:
                continue
            new_tokens = []
            for tok in elem.text.replace('\n', ',').split(','):
                t = tok.strip()
                if not t:
                    continue
                try:
                    new_tokens.append(str(remap[int(t)]))
                except (KeyError, ValueError):
                    # Connectivity refers to an unknown node — preserve the raw
                    # token rather than silently rewriting; this surfaces broken
                    # source files instead of corrupting them.
                    new_tokens.append(t)
            elem.text = ','.join(new_tokens)

    # 4) Drop orphan NodeSets and canonicalize the rest.
    #
    #    Two problems collide here:
    #      (a) Some authoring tools emit <NodeSet> with comma-separated text
    #          content (a FEBio 2.x style). FEBio 4.0 rejects that with a
    #          misleading "tag 'Elements' (line N) : unrecognized tag" error
    #          pointing at the *next* sibling — i.e. the parser blames the
    #          following Elements block.
    #      (b) FEBio 4.0 also refuses to construct a model when the file
    #          declares NodeSets that no BC / contact / load / surface ever
    #          references. The error there is "FATAL ERROR: Failed
    #          constructing model" — even after the XML parses successfully.
    #
    #    The canonical Lower_Limb_Model_Surfaces.feb the user uploaded had
    #    both: six orphan NodeSets in comma-text format. Cleaning is:
    #      - find every `node_set="..."` attribute anywhere in the file
    #      - drop NodeSets whose name doesn't appear in that referenced set
    #      - for the rest, rewrite to <n id="N"/> children with remapped IDs
    referenced_node_set_names: set = set()
    for elem in root.iter():
        # FEBio uses `node_set="…"` on BCs, contacts, loads, surfaces.
        for attr in ('node_set', 'nodeset', 'surface'):
            val = elem.get(attr)
            if val:
                referenced_node_set_names.add(val)

    for ns in list(mesh_section.findall('.//NodeSet')):
        name = ns.get('name', '')

        # Orphan? Drop the whole block. FEBio won't construct a model otherwise.
        if name and name not in referenced_node_set_names:
            mesh_section.remove(ns)
            continue

        collected_old_ids: List[int] = []

        # <n id="N"/> style — already canonical, collect for remap.
        for n_elem in list(ns.findall('n')):
            try:
                collected_old_ids.append(int(n_elem.get('id')))
            except (TypeError, ValueError):
                pass
            ns.remove(n_elem)

        # <node id="N"/> style — same deal.
        for n_elem in list(ns.findall('node')):
            try:
                collected_old_ids.append(int(n_elem.get('id')))
            except (TypeError, ValueError):
                pass
            ns.remove(n_elem)

        # Comma-separated text content — the format FEBio 4.0 chokes on.
        if ns.text and ns.text.strip():
            for tok in ns.text.replace('\n', ',').split(','):
                t = tok.strip()
                if not t:
                    continue
                try:
                    collected_old_ids.append(int(t))
                except ValueError:
                    pass
        ns.text = None

        # Rewrite as canonical <n id="N"/> children with remapped IDs.
        # Drop references to orphan nodes that we already removed above.
        for old in collected_old_ids:
            new = remap.get(old)
            if new is None:
                continue
            ET.SubElement(ns, 'n', id=str(new))

    tree.write(dst_path, encoding='utf-8', xml_declaration=True)

    # Original max id matters for the "load_febio over-allocates np.zeros((max_id, 3))"
    # diagnostic — without this you can't tell why a 152K-node file was getting
    # an 8M-row in-memory mesh.
    orig_max_id = max(remap.keys()) if remap else 0
    return {
        "nodes_before": nodes_before,
        "nodes_after": len(remap),
        "orphans_removed": nodes_before - len(remap),
        "orig_max_id": orig_max_id,
        "id_collapse_ratio": (
            len(remap) / orig_max_id if orig_max_id > 0 else 1.0
        ),
    }
