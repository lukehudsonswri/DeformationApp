"""
Abaqus (.inp) mesh file loader and saver.

Parses Abaqus input files to extract mesh data.

--------------------------------------------------------------------------
Vendored from: C:\\Users\\lhudson\\Documents\\Morphing\\fe_personalization\\mesh_io\\abaqus.py
Vendored on:   2026-09-18, unmodified.
Local changes: none yet.
See AGENTS.md "Vendoring policy".
--------------------------------------------------------------------------
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

from .base import VolumeMesh, NodeSet, ElementSet, Material, ElementType


def load_abaqus(filepath: str) -> VolumeMesh:
    """
    Load a mesh from an Abaqus .inp file.

    Args:
        filepath: Path to the .inp file

    Returns:
        VolumeMesh object
    """
    filepath = Path(filepath)

    nodes = {}
    elements = []
    element_type = None
    current_section = None
    node_sets = {}
    element_sets = {}
    current_set_name = None
    current_set_nodes = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and comments
        if not line or line.startswith('**'):
            i += 1
            continue

        # Detect keyword sections
        if line.startswith('*'):
            keyword_line = line.upper()

            # Save previous node set if any
            if current_set_name and current_set_nodes:
                node_sets[current_set_name] = NodeSet(
                    name=current_set_name,
                    node_ids=np.array(current_set_nodes, dtype=np.int64)
                )
                current_set_nodes = []
                current_set_name = None

            if '*NODE' in keyword_line and '*NSET' not in keyword_line:
                current_section = 'NODE'
            elif '*ELEMENT' in keyword_line and '*ELSET' not in keyword_line:
                current_section = 'ELEMENT'
                # Parse element type
                match = re.search(r'TYPE\s*=\s*(\w+)', keyword_line)
                if match:
                    elem_type_str = match.group(1).upper()
                    element_type = _parse_element_type(elem_type_str)
            elif '*NSET' in keyword_line:
                current_section = 'NSET'
                match = re.search(r'NSET\s*=\s*(\w+)', keyword_line)
                if match:
                    current_set_name = match.group(1)
            elif '*ELSET' in keyword_line:
                current_section = 'ELSET'
            elif '*END' in keyword_line:
                break
            else:
                current_section = None
            i += 1
            continue

        # Parse data based on current section
        if current_section == 'NODE':
            parts = line.replace(',', ' ').split()
            if len(parts) >= 4:
                try:
                    node_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    nodes[node_id] = [x, y, z]
                except (ValueError, IndexError):
                    pass

        elif current_section == 'ELEMENT':
            parts = line.replace(',', ' ').split()
            if len(parts) >= 2:
                try:
                    # First number is element ID, rest are node IDs
                    # Abaqus uses 1-based indexing
                    node_ids = [int(p) - 1 for p in parts[1:]]
                    if len(node_ids) == 8:
                        elements.append(node_ids)
                    elif len(node_ids) == 4:
                        elements.append(node_ids)
                        if element_type is None:
                            element_type = ElementType.TET4
                except (ValueError, IndexError):
                    pass

        elif current_section == 'NSET':
            parts = line.replace(',', ' ').split()
            for p in parts:
                try:
                    node_id = int(p) - 1  # Convert to 0-based
                    current_set_nodes.append(node_id)
                except ValueError:
                    pass

        i += 1

    # Save final node set if any
    if current_set_name and current_set_nodes:
        node_sets[current_set_name] = NodeSet(
            name=current_set_name,
            node_ids=np.array(current_set_nodes, dtype=np.int64)
        )

    if not nodes:
        raise ValueError("No nodes found in .inp file")

    if not elements:
        raise ValueError("No elements found in .inp file")

    # Convert nodes dict to array
    max_node_id = max(nodes.keys())
    nodes_array = np.zeros((max_node_id, 3), dtype=np.float64)
    for node_id, coords in nodes.items():
        nodes_array[node_id - 1] = coords

    elements_array = np.array(elements, dtype=np.int64)

    mesh = VolumeMesh(
        nodes=nodes_array,
        elements=elements_array,
        element_type=element_type or ElementType.HEX8,
        name=filepath.stem,
        node_sets=node_sets,
        element_sets=element_sets,
        metadata={'source_file': str(filepath)}
    )

    return mesh


def _parse_element_type(type_str: str) -> Optional[ElementType]:
    """Parse Abaqus element type string to ElementType enum."""
    type_str = type_str.upper()
    type_map = {
        'C3D8': ElementType.HEX8,
        'C3D8R': ElementType.HEX8,
        'C3D8I': ElementType.HEX8,
        'C3D20': ElementType.HEX20,
        'C3D20R': ElementType.HEX20,
        'C3D4': ElementType.TET4,
        'C3D10': ElementType.TET10,
        'C3D6': ElementType.WEDGE6,
    }
    return type_map.get(type_str)


def save_abaqus(mesh: VolumeMesh, filepath: str) -> None:
    """
    Save a mesh to Abaqus .inp format.

    Args:
        mesh: VolumeMesh to save
        filepath: Output file path
    """
    filepath = Path(filepath)

    # Map element type to Abaqus type string
    type_map = {
        ElementType.HEX8: 'C3D8',
        ElementType.HEX20: 'C3D20',
        ElementType.TET4: 'C3D4',
        ElementType.TET10: 'C3D10',
        ElementType.WEDGE6: 'C3D6',
    }
    elem_type_str = type_map.get(mesh.element_type, 'C3D8')

    with open(filepath, 'w') as f:
        f.write('** Abaqus input file\n')
        f.write(f'** Generated by FE Personalization\n')
        f.write(f'** Mesh: {mesh.name}\n')
        f.write('**\n')

        # Write nodes
        f.write('*NODE\n')
        for i, node in enumerate(mesh.nodes):
            # Format: node_id, x, y, z (1-based)
            f.write(f'{i+1}, {node[0]:.10g}, {node[1]:.10g}, {node[2]:.10g}\n')

        # Write elements
        f.write(f'*ELEMENT, TYPE={elem_type_str}\n')
        for i, elem in enumerate(mesh.elements):
            # Format: elem_id, n1, n2, ... (1-based)
            nodes_str = ', '.join(str(n + 1) for n in elem)
            f.write(f'{i+1}, {nodes_str}\n')

        # Write node sets
        for ns_name, node_set in mesh.node_sets.items():
            f.write(f'*NSET, NSET={ns_name}\n')
            # Write node IDs in groups of 8 per line
            for j in range(0, len(node_set.node_ids), 8):
                chunk = node_set.node_ids[j:j+8]
                nodes_str = ', '.join(str(n + 1) for n in chunk)
                if j + 8 < len(node_set.node_ids):
                    f.write(f'{nodes_str},\n')
                else:
                    f.write(f'{nodes_str}\n')

        # Write element sets
        for es_name, elem_set in mesh.element_sets.items():
            f.write(f'*ELSET, ELSET={es_name}\n')
            for j in range(0, len(elem_set.element_ids), 8):
                chunk = elem_set.element_ids[j:j+8]
                elems_str = ', '.join(str(e + 1) for e in chunk)
                if j + 8 < len(elem_set.element_ids):
                    f.write(f'{elems_str},\n')
                else:
                    f.write(f'{elems_str}\n')

    print(f"Saved mesh to {filepath}:")
    print(f"  Nodes: {mesh.num_nodes}")
    print(f"  Elements: {mesh.num_elements}")
