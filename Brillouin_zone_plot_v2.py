#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import re
import sys

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Voronoi

import seekpath
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.groups import SpaceGroup
from ase.io import read as ase_read



def make_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]

    return obj

# ============================================================
# Basic structure loading
# ============================================================

def load_structure(filename: str) -> Structure:
    path = Path(filename)
    suffix = path.suffix.lower()

    if suffix == ".cell":
        atoms = ase_read(str(path), format="castep-cell")
        return AseAtomsAdaptor.get_structure(atoms)

    return Structure.from_file(str(path))


def to_seekpath_input(structure: Structure):
    lattice = structure.lattice.matrix
    positions = structure.frac_coords
    numbers = [site.specie.Z for site in structure]
    return lattice, positions, numbers


# ============================================================
# Lattice / reciprocal-lattice utilities
# ============================================================

def reciprocal_lattice_from_real(real_lattice):
    """
    Input real_lattice is a 3x3 matrix whose rows are real-space lattice vectors.
    Output reciprocal_lattice is a 3x3 matrix whose rows are reciprocal vectors,
    using the 2*pi/Angstrom convention.
    """
    real_lattice = np.array(real_lattice, dtype=float)
    return np.linalg.inv(real_lattice).T * 2 * np.pi


def lattice_from_abc(a, b, c, alpha, beta, gamma):
    """
    Build conventional real-space lattice matrix from lengths and angles.
    Angles are in degrees.
    Rows are lattice vectors.
    """
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])

    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz_sq = c**2 - cx**2 - cy**2

    if cz_sq < -1e-8:
        raise ValueError("Invalid lattice parameters: computed c_z^2 is negative.")

    vc = np.array([cx, cy, np.sqrt(max(cz_sq, 0.0))])
    return np.vstack([va, vb, vc])


def primitive_from_centering(conventional, centering):
    """
    Convert a conventional cell to a primitive cell based on centering.
    Rows are lattice vectors.
    """
    A, B, C = np.array(conventional, dtype=float)
    centering = centering.upper()

    if centering == "P":
        return np.vstack([A, B, C])

    if centering == "I":
        return np.vstack([
            (A + B - C) / 2,
            (A - B + C) / 2,
            (-A + B + C) / 2,
        ])

    if centering == "F":
        return np.vstack([
            (B + C) / 2,
            (A + C) / 2,
            (A + B) / 2,
        ])

    if centering == "C":
        return np.vstack([
            (A + B) / 2,
            (-A + B) / 2,
            C,
        ])

    if centering == "A":
        return np.vstack([
            A,
            (B + C) / 2,
            (-B + C) / 2,
        ])

    if centering == "B":
        return np.vstack([
            (A + C) / 2,
            B,
            (-A + C) / 2,
        ])

    if centering == "R":
        # Rhombohedral lattice in the hexagonal conventional setting.
        # This assumes the input conventional cell is given in hexagonal axes.
        transform = np.array([
            [ 2/3,  1/3, 1/3],
            [-1/3,  1/3, 1/3],
            [-1/3, -2/3, 1/3],
        ])
        return transform @ np.vstack([A, B, C])

    raise ValueError(f"Unsupported centering: {centering}")


# ============================================================
# Space-group / Bravais-lattice identification
# ============================================================

def crystal_system_from_number(n):
    if 1 <= n <= 2:
        return "triclinic"
    if 3 <= n <= 15:
        return "monoclinic"
    if 16 <= n <= 74:
        return "orthorhombic"
    if 75 <= n <= 142:
        return "tetragonal"
    if 143 <= n <= 167:
        return "trigonal"
    if 168 <= n <= 194:
        return "hexagonal"
    if 195 <= n <= 230:
        return "cubic"
    raise ValueError("Space-group number must be between 1 and 230.")


def bravais_from_space_group(sg_input):
    """
    Return space-group number, international symbol, crystal system,
    centering letter, and SeeK-path-like Bravais label.
    """
    sg_text = str(sg_input).strip()

    if sg_text.isdigit():
        sg = SpaceGroup.from_int_number(int(sg_text))
    else:
        sg = SpaceGroup(sg_text)

    number = sg.int_number
    symbol = sg.symbol
    system = crystal_system_from_number(number)

    compact_symbol = symbol.replace(" ", "")
    centering = compact_symbol[0].upper()

    if system == "triclinic":
        bravais = "aP"

    elif system == "monoclinic":
        if centering == "P":
            bravais = "mP"
        elif centering in {"A", "B", "C", "I"}:
            bravais = "mC"
        else:
            raise ValueError(f"Unexpected monoclinic centering: {centering}")

    elif system == "orthorhombic":
        if centering == "P":
            bravais = "oP"
        elif centering in {"A", "B", "C"}:
            bravais = "oC"
        elif centering == "I":
            bravais = "oI"
        elif centering == "F":
            bravais = "oF"
        else:
            raise ValueError(f"Unexpected orthorhombic centering: {centering}")

    elif system == "tetragonal":
        if centering == "P":
            bravais = "tP"
        elif centering == "I":
            bravais = "tI"
        else:
            raise ValueError(f"Unexpected tetragonal centering: {centering}")

    elif system == "trigonal":
        if centering == "R":
            bravais = "hR"
        else:
            bravais = "hP"

    elif system == "hexagonal":
        bravais = "hP"

    elif system == "cubic":
        if centering == "P":
            bravais = "cP"
        elif centering == "I":
            bravais = "cI"
        elif centering == "F":
            bravais = "cF"
        else:
            raise ValueError(f"Unexpected cubic centering: {centering}")

    else:
        raise ValueError(f"Unsupported crystal system: {system}")

    return {
        "spacegroup_number": number,
        "spacegroup_symbol": symbol,
        "crystal_system": system,
        "centering": centering,
        "bravais_lattice": bravais,
    }


def default_cell_for_bravais(bravais):
    """
    Schematic cell only.
    This is NOT guaranteed to represent your real material.
    Use --cell for a physically meaningful BZ.
    """
    if bravais.startswith("c"):
        return [1.0, 1.0, 1.0, 90, 90, 90]
    if bravais.startswith("t"):
        return [1.0, 1.0, 1.4, 90, 90, 90]
    if bravais.startswith("o"):
        return [1.0, 1.3, 1.7, 90, 90, 90]
    if bravais == "hP":
        return [1.0, 1.0, 1.6, 90, 90, 120]
    if bravais == "hR":
        return [1.0, 1.0, 2.4, 90, 90, 120]
    if bravais.startswith("m"):
        return [1.0, 1.3, 1.6, 90, 105, 90]
    if bravais == "aP":
        return [1.0, 1.2, 1.4, 75, 80, 70]

    raise ValueError(f"No default cell for Bravais lattice {bravais}")


# ============================================================
# High-symmetry label/path handling
# ============================================================

def display_label(label):
    if label.upper() in {"G", "GAMMA", "Γ"}:
        return "Γ"
    return label


def canonical_label(label, available_labels):
    """
    Allows the user to type G, Gamma, or Γ for GAMMA.
    Otherwise uses exact matching first.
    """
    label = label.strip()

    if label in available_labels:
        return label

    gamma_aliases = {"G", "GAMMA", "Gamma", "gamma", "Γ"}
    if label in gamma_aliases:
        for candidate in ["GAMMA", "Γ", "G"]:
            if candidate in available_labels:
                return candidate

    # Case-insensitive fallback.
    for existing in available_labels:
        if existing.upper() == label.upper():
            return existing

    raise KeyError(f"Label {label!r} not found in available high-symmetry points.")


def load_points_json(filename):
    """
    JSON format example:
    {
      "GAMMA": [0, 0, 0],
      "X": [0.5, 0, 0],
      "M": [0.5, 0.5, 0]
    }
    """
    with open(filename, "r") as f:
        data = json.load(f)

    return {str(k): np.array(v, dtype=float) for k, v in data.items()}


def parse_path_string(path_string, available_labels):
    """
    Examples:
      "G-X-M-G"
      "GAMMA-X-M-GAMMA,Z-R-A-Z"
      "T-G-Y-Z-L-R-G"

    Commas create visual breaks.
    """
    sequences = []

    for block in re.split(r"\s*,\s*", path_string.strip()):
        if not block:
            continue

        labels = [x for x in re.split(r"\s*-\s*", block) if x]
        labels = [canonical_label(x, available_labels) for x in labels]

        if len(labels) < 2:
            raise ValueError(f"Path block {block!r} has fewer than two labels.")

        sequences.append(labels)

    return sequences


def seekpath_pairs_to_sequences(path_pairs):
    """
    Keep SeeK-path segments as individual line pieces.
    This avoids accidentally joining separate path branches.
    """
    return [[a, b] for a, b in path_pairs]


# ============================================================
# Brillouin-zone plotting
# ============================================================

def bz_faces_from_reciprocal(reciprocal_lattice, search_range=3):
    """
    Build the first Brillouin zone as the Voronoi cell around Gamma.
    """
    B = np.array(reciprocal_lattice, dtype=float)

    points = []
    origin_index = None

    for i in range(-search_range, search_range + 1):
        for j in range(-search_range, search_range + 1):
            for k in range(-search_range, search_range + 1):
                vec = i * B[0] + j * B[1] + k * B[2]
                if i == 0 and j == 0 and k == 0:
                    origin_index = len(points)
                points.append(vec)

    points = np.array(points)
    vor = Voronoi(points)

    faces = []

    for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        if origin_index not in ridge_points:
            continue

        if any(v < 0 for v in ridge_vertices):
            continue

        face = vor.vertices[ridge_vertices]
        face = order_face_vertices(face)
        faces.append(face)

    return faces


def order_face_vertices(face):
    """
    Order vertices around a polygonal face.
    """
    face = np.array(face, dtype=float)
    center = face.mean(axis=0)

    if len(face) <= 2:
        return face

    # Estimate normal from the first three non-collinear points.
    normal = None
    for i in range(1, len(face) - 1):
        n = np.cross(face[i] - face[0], face[i + 1] - face[0])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            normal = n / norm
            break

    if normal is None:
        return face

    u = face[0] - center
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    angles = []
    for p in face:
        r = p - center
        angles.append(np.arctan2(np.dot(r, v), np.dot(r, u)))

    return face[np.argsort(angles)]


def reduced_to_cartesian_k(coord, reciprocal_lattice):
    coord = np.array(coord, dtype=float)
    B = np.array(reciprocal_lattice, dtype=float)
    return coord @ B


def make_bz_plot(
    reciprocal_lattice,
    point_coords=None,
    path_sequences=None,
    title="Brillouin zone",
    output_html="brillouin_zone.html",
):
    point_coords = point_coords or {}
    path_sequences = path_sequences or []

    faces = bz_faces_from_reciprocal(reciprocal_lattice)

    fig = go.Figure()

    # BZ faces and edges.
    for face in faces:
        closed = np.vstack([face, face[0]])

        fig.add_trace(go.Scatter3d(
            x=closed[:, 0],
            y=closed[:, 1],
            z=closed[:, 2],
            mode="lines",
            line=dict(width=4),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Transparent filled face.
        if len(face) >= 3:
            center = face.mean(axis=0)
            vertices = np.vstack([center, face])
            i = []
            j = []
            k = []

            for n in range(1, len(face)):
                i.append(0)
                j.append(n)
                k.append(n + 1)

            i.append(0)
            j.append(len(face))
            k.append(1)

            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=i,
                j=j,
                k=k,
                opacity=0.12,
                showlegend=False,
                hoverinfo="skip",
            ))

    # Reciprocal basis vectors.
    origin = np.zeros(3)
    for idx, bvec in enumerate(reciprocal_lattice, start=1):
        fig.add_trace(go.Scatter3d(
            x=[origin[0], bvec[0]],
            y=[origin[1], bvec[1]],
            z=[origin[2], bvec[2]],
            mode="lines+text",
            text=["", f"b{idx}"],
            textposition="top center",
            line=dict(width=5, dash="dash"),
            name=f"b{idx}",
        ))

    # High-symmetry points.
    if point_coords:
        xs, ys, zs, texts = [], [], [], []

        for label, coord in point_coords.items():
            cart = reduced_to_cartesian_k(coord, reciprocal_lattice)
            xs.append(cart[0])
            ys.append(cart[1])
            zs.append(cart[2])
            texts.append(display_label(label))

        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            text=texts,
            textposition="top center",
            marker=dict(size=5),
            name="High-symmetry points",
        ))

    # User-selected or recommended path.
    for seq in path_sequences:
        coords = []
        labels = []

        for label in seq:
            coord = point_coords[label]
            coords.append(reduced_to_cartesian_k(coord, reciprocal_lattice))
            labels.append(display_label(label))

        coords = np.array(coords)

        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="lines+markers+text",
            text=labels,
            textposition="bottom center",
            line=dict(width=7),
            marker=dict(size=4),
            name="-".join(labels),
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="kx / Å⁻¹",
            yaxis_title="ky / Å⁻¹",
            zaxis_title="kz / Å⁻¹",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    fig.write_html(output_html)
    print(f"\nWrote interactive BZ plot to: {output_html}")


# ============================================================
# Main modes
# ============================================================

def run_structure_mode(args):
    path = Path(args.structure)
    structure = load_structure(args.structure)

    sp_input = to_seekpath_input(structure)
    sp = seekpath.get_path(
        sp_input,
        with_time_reversal=not args.no_time_reversal,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
    )

    primitive_lattice = np.array(sp["primitive_lattice"], dtype=float)
    reciprocal_lattice = reciprocal_lattice_from_real(primitive_lattice)

    point_coords = {
        label: np.array(coord, dtype=float)
        for label, coord in sp["point_coords"].items()
    }

    if args.points:
        point_coords.update(load_points_json(args.points))

    print("\n=== Symmetry / lattice information from SeeK-path ===")
    print("Space group:", sp["spacegroup_international"], sp["spacegroup_number"])
    print("Bravais lattice:", sp["bravais_lattice"])
    print("Extended Bravais lattice:", sp["bravais_lattice_extended"])

    print("\n=== Standardized primitive real-space lattice / Å ===")
    print(primitive_lattice)

    print("\n=== Standardized primitive reciprocal lattice / Å^-1, 2π convention ===")
    print(reciprocal_lattice)

    print("\n=== High-symmetry points ===")
    print("Coordinates are reduced coordinates in the standardized primitive reciprocal basis.")
    for label, coord in point_coords.items():
        print(f"{display_label(label):8s} {coord}")

    print("\n=== SeeK-path recommended path ===")
    print(sp["path"])

    if args.path:
        path_sequences = parse_path_string(args.path, point_coords.keys())
    elif args.interactive:
        answer = input("\nUse SeeK-path recommended path? [Y/n]: ").strip().lower()

        if answer in {"", "y", "yes"}:
            path_sequences = seekpath_pairs_to_sequences(sp["path"])
        else:
            manual = input("Enter manual path, e.g. G-X-M-G or T-G-Y-Z-L-R-G: ")
            path_sequences = parse_path_string(manual, point_coords.keys())
    else:
        path_sequences = seekpath_pairs_to_sequences(sp["path"])

    html_name = args.output or f"{path.with_suffix('').name}_bz.html"
    json_name = f"{Path(html_name).with_suffix('').name}_data.json"

    make_bz_plot(
        reciprocal_lattice=reciprocal_lattice,
        point_coords=point_coords,
        path_sequences=path_sequences,
        title=f"{path.name} BZ",
        output_html=html_name,
    )

    output_data = {
        "mode": "structure",
        "spacegroup": [sp["spacegroup_international"], sp["spacegroup_number"]],
        "bravais_lattice": sp["bravais_lattice"],
        "bravais_lattice_extended": sp["bravais_lattice_extended"],
        "primitive_lattice_A": primitive_lattice.tolist(),
        "reciprocal_lattice_Ainv_2pi": reciprocal_lattice.tolist(),
        "point_coords_reduced": {
            k: np.array(v).tolist() for k, v in point_coords.items()
        },
        "path_sequences": path_sequences,
        "seekpath_raw": sp,
    }

    with open(json_name, "w") as f:
      # json.dump(output_data, f, indent=2)
        json.dump(make_json_safe(output_data), f, indent=2) 

    print(f"Wrote data to: {json_name}")


def run_spacegroup_mode(args):
    info = bravais_from_space_group(args.sg)

    if args.cell:
        cell = args.cell
        schematic = False
    elif args.ideal:
        cell = default_cell_for_bravais(info["bravais_lattice"])
        schematic = True
    else:
        raise SystemExit(
            "\nSpace-group mode needs lattice parameters for a correct BZ.\n"
            "Use for example:\n"
            "  --sg 62 --cell a b c alpha beta gamma\n"
            "or use --ideal for a schematic, non-material-specific BZ.\n"
        )

    conventional_lattice = lattice_from_abc(*cell)
    primitive_lattice = primitive_from_centering(
        conventional_lattice,
        info["centering"],
    )
    reciprocal_lattice = reciprocal_lattice_from_real(primitive_lattice)

    point_coords = {}
    if args.points:
        point_coords = load_points_json(args.points)

    print("\n=== Space-group-derived lattice information ===")
    print("Space group:", info["spacegroup_symbol"], info["spacegroup_number"])
    print("Crystal system:", info["crystal_system"])
    print("Centering:", info["centering"])
    print("Bravais lattice:", info["bravais_lattice"])

    if schematic:
        print("\nWARNING: using an ideal schematic cell, not a material-specific cell.")
        print("This is useful for checking topology, but not for a quantitatively correct BZ.")

    print("\n=== Conventional real-space lattice / Å ===")
    print(conventional_lattice)

    print("\n=== Primitive real-space lattice / Å ===")
    print(primitive_lattice)

    print("\n=== Primitive reciprocal lattice / Å^-1, 2π convention ===")
    print(reciprocal_lattice)

    if point_coords:
        print("\n=== User-provided high-symmetry points ===")
        print("Coordinates are assumed to be reduced coordinates in the primitive reciprocal basis.")
        for label, coord in point_coords.items():
            print(f"{display_label(label):8s} {coord}")
    else:
        print("\nNo high-symmetry point file was provided.")
        print("You can still plot the BZ, but no path will be drawn.")

    if args.path:
        if not point_coords:
            raise SystemExit("You gave --path but no --points JSON file.")
        path_sequences = parse_path_string(args.path, point_coords.keys())
    elif args.interactive and point_coords:
        manual = input("\nEnter manual path, e.g. G-X-M-G or T-G-Y-Z-L-R-G: ")
        path_sequences = parse_path_string(manual, point_coords.keys())
    else:
        path_sequences = []

    html_name = args.output or f"SG_{info['spacegroup_number']}_{info['bravais_lattice']}_bz.html"
    json_name = f"{Path(html_name).with_suffix('').name}_data.json"

    make_bz_plot(
        reciprocal_lattice=reciprocal_lattice,
        point_coords=point_coords,
        path_sequences=path_sequences,
        title=f"SG {info['spacegroup_number']} {info['spacegroup_symbol']} / {info['bravais_lattice']}",
        output_html=html_name,
    )

    output_data = {
        "mode": "spacegroup",
        "spacegroup_info": info,
        "schematic_cell": schematic,
        "cell_abc_angles": cell,
        "conventional_lattice_A": conventional_lattice.tolist(),
        "primitive_lattice_A": primitive_lattice.tolist(),
        "reciprocal_lattice_Ainv_2pi": reciprocal_lattice.tolist(),
        "point_coords_reduced": {
            k: np.array(v).tolist() for k, v in point_coords.items()
        },
        "path_sequences": path_sequences,
    }

    with open(json_name, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Wrote data to: {json_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Brillouin zone and selected high-symmetry paths."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--structure", help="Input .cell, .cif, POSCAR, etc.")
    mode.add_argument("--sg", help="Space-group number or symbol, e.g. 62 or Pnma.")

    parser.add_argument(
        "--cell",
        nargs=6,
        type=float,
        metavar=("a", "b", "c", "alpha", "beta", "gamma"),
        help="Conventional cell parameters for --sg mode.",
    )

    parser.add_argument(
        "--ideal",
        action="store_true",
        help="Use a schematic ideal cell in --sg mode. Not quantitatively correct.",
    )

    parser.add_argument(
        "--points",
        help="JSON file of high-symmetry point reduced coordinates.",
    )

    parser.add_argument(
        "--path",
        help="Manual path, e.g. 'G-X-M-G' or 'T-G-Y-Z-L-R-G'. Commas create breaks.",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask whether to use recommended path or enter a manual path.",
    )

    parser.add_argument(
        "--output",
        help="Output HTML filename.",
    )

    parser.add_argument(
        "--symprec",
        type=float,
        default=1e-5,
        help="Symmetry tolerance for SeeK-path structure mode.",
    )

    parser.add_argument(
        "--angle-tolerance",
        type=float,
        default=-1.0,
        help="Angle tolerance for SeeK-path structure mode.",
    )

    parser.add_argument(
        "--no-time-reversal",
        action="store_true",
        help="Use with_time_reversal=False in SeeK-path.",
    )

    args = parser.parse_args()

    if args.structure:
        run_structure_mode(args)
    else:
        run_spacegroup_mode(args)


if __name__ == "__main__":
    main()
