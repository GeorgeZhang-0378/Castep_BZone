#!/usr/bin/env python3

from pathlib import Path
from fractions import Fraction
import argparse
import json
import re

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Voronoi

import seekpath
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.groups import SpaceGroup
from ase.io import read as ase_read


# ============================================================
# JSON safety
# ============================================================

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


def reduced_to_cartesian_k(coord, reciprocal_lattice):
    """
    Convert reduced reciprocal coordinates to Cartesian k-vector.

    coord is written in the reciprocal basis represented by reciprocal_lattice rows.
    """
    coord = np.array(coord, dtype=float)
    B = np.array(reciprocal_lattice, dtype=float)
    return coord @ B


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
    label = str(label)
    if label.upper() in {"G", "GAMMA", "Γ"}:
        return "Γ"
    return label


def normalise_manual_label(label):
    """
    In fully manual mode, labels are display labels only.
    We normalise common Gamma aliases, but we do NOT map R, X, etc. to SeeK-path.
    """
    label = label.strip()
    label = label.replace("′", "'").replace("’", "'")

    if label.upper() in {"G", "GAMMA", "Γ"}:
        return "Γ"

    return label


def canonical_label(label, available_labels):
    """
    Automatic/SeeK-path mode only.
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
    Automatic/SeeK-path mode only.

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


def parse_fully_manual_path_string(path_string):
    """
    Fully manual mode.

    Labels are NOT interpreted as SeeK-path labels. They are display names only.
    Commas create visual breaks. Hyphens or whitespace separate points.

    Examples:
      "R-G-R'"
      "R Γ R'"
      "R-G-R', X-G-X'"
    """
    sequences = []

    s = path_string.strip()
    s = s.replace("→", "-").replace("->", "-")
    s = s.replace("—", "-").replace("–", "-")

    for block in re.split(r"\s*,\s*", s):
        if not block:
            continue

        if "-" in block:
            labels = [x.strip() for x in block.split("-") if x.strip()]
        else:
            labels = [x.strip() for x in block.split() if x.strip()]

        labels = [normalise_manual_label(x) for x in labels]

        if len(labels) < 2:
            raise ValueError(f"Path block {block!r} has fewer than two labels.")

        sequences.append(labels)

    return sequences


def ordered_unique_labels(path_sequences):
    seen = set()
    ordered = []
    for seq in path_sequences:
        for label in seq:
            if label not in seen:
                seen.add(label)
                ordered.append(label)
    return ordered


def parse_reduced_coordinate_string(raw):
    """
    Parse reduced coordinates such as:
        0 -1/2 1/2
        0, -0.5, 0.5
        [0, -1/2, 1/2]
    """
    raw = raw.replace("[", " ").replace("]", " ")
    raw = raw.replace(",", " ")
    parts = raw.split()

    if len(parts) != 3:
        raise ValueError("Please enter exactly three numbers: h k l")

    return np.array([float(Fraction(x)) for x in parts], dtype=float)


def ask_manual_coordinate_basis():
    print("\nWhich reciprocal basis are your manual coordinates written in?")
    print("  [1] Original input-cell reciprocal basis, i.e. the CASTEP .cell basis")
    print("  [2] SeeK-path standardized primitive reciprocal basis")

    while True:
        answer = input("Coordinate basis [1/2, default 1]: ").strip().lower()

        if answer in {"", "1", "input", "cell", "castep"}:
            return "input"

        if answer in {"2", "seekpath", "standard", "standardized", "primitive"}:
            return "seekpath"

        print("Please enter 1 or 2.")


def ask_fully_manual_points(path_sequences, reciprocal_lattice_for_coordinates):
    """
    Ask once for each unique label, then return both reduced and Cartesian coords.
    """
    labels = ordered_unique_labels(path_sequences)
    reduced_coords = {}
    cart_coords = {}

    print("\nEnter reduced coordinates for each manual point.")
    print("Fractions are allowed, e.g. 1/2 0 -1/2.")

    for label in labels:
        while True:
            raw = input(f"  {display_label(label)} = ")
            try:
                reduced = parse_reduced_coordinate_string(raw)
            except Exception as exc:
                print(f"Could not parse coordinates: {exc}")
                continue

            reduced_coords[label] = reduced
            cart_coords[label] = reduced_to_cartesian_k(
                reduced,
                reciprocal_lattice_for_coordinates,
            )
            break

    print("\n=== Fully manual path summary ===")
    for label in labels:
        r = reduced_coords[label]
        c = cart_coords[label]
        print(
            f"{display_label(label):8s} reduced = {r}    "
            f"cartesian / Å^-1 = [{c[0]: .6f}, {c[1]: .6f}, {c[2]: .6f}]"
        )

    confirm = input("\nProceed with this fully manual path? [Y/n]: ").strip().lower()
    if confirm not in {"", "y", "yes"}:
        raise RuntimeError("Manual path cancelled by user.")

    return reduced_coords, cart_coords


def seekpath_pairs_to_sequences(path_pairs):
    """
    Keep SeeK-path segments as individual line pieces.
    This avoids accidentally joining separate path branches.
    """
    return [[a, b] for a, b in path_pairs]


# ============================================================
# Brillouin-zone geometry
# ============================================================

def order_face_vertices(face):
    """
    Order vertices around a polygonal face.
    """
    face = np.array(face, dtype=float)
    center = face.mean(axis=0)

    if len(face) <= 2:
        return face

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


# ============================================================
# Brillouin-zone plotting
# ============================================================

def _colour_palette():
    """A small colour-safe palette used only for optional legend layers."""
    return [
        "#d62728",  # red
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#17becf",  # cyan
        "#bcbd22",  # olive
        "#7f7f7f",  # grey
    ]


def add_b_vector(
    fig,
    bvec,
    name,
    color="black",
    visible="legendonly",
    legendgroup=None,
    showlegend=True,
):
    """
    Add one reciprocal basis vector as a line + cone.

    The line trace carries the legend item. The cone is in the same
    legendgroup, so clicking the legend item hides/shows both together.
    """
    origin = np.zeros(3)
    bvec = np.array(bvec, dtype=float)
    norm = np.linalg.norm(bvec)
    legendgroup = legendgroup or name

    fig.add_trace(go.Scatter3d(
        x=[origin[0], bvec[0]],
        y=[origin[1], bvec[1]],
        z=[origin[2], bvec[2]],
        mode="lines+text",
        text=["", name],
        textposition="top center",
        line=dict(width=7, color=color),
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
        visible=visible,
        hovertemplate=(
            name + "<br>kx=%{x:.4f}<br>ky=%{y:.4f}<br>kz=%{z:.4f}<extra></extra>"
        ),
    ))

    # Small cone as arrow head. If plotly rendering is awkward, this trace can be removed safely.
    if norm > 1e-12:
        fig.add_trace(go.Cone(
            x=[bvec[0]],
            y=[bvec[1]],
            z=[bvec[2]],
            u=[bvec[0] / norm],
            v=[bvec[1] / norm],
            w=[bvec[2] / norm],
            anchor="tip",
            sizemode="absolute",
            sizeref=0.08 * norm,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=f"{name} arrow head",
            legendgroup=legendgroup,
            showlegend=False,
            visible=visible,
            hoverinfo="skip",
        ))


def _plotly_visible(value):
    """Convert user-facing visibility strings into Plotly's visibility values."""
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()
    if value in {"true", "show", "visible", "on", "yes", "y"}:
        return True
    if value in {"false", "hide", "hidden", "off", "no", "n"}:
        return False
    if value in {"legendonly", "legend", "toggle"}:
        return "legendonly"

    raise ValueError(
        "Visibility must be one of: true, false, legendonly. "
        f"Got {value!r}."
    )


def _cyclic_colour(index):
    palette = _colour_palette()
    return palette[index % len(palette)]


def make_bz_plot(
    bz_reciprocal_lattice,
    point_cart_coords=None,
    path_sequences=None,
    title="Brillouin zone",
    output_html="brillouin_zone.html",
    show_axes=False,
    include_b_vectors=True,
    b_vectors_visible="legendonly",
    all_points_visible="legendonly",
    path_visible=True,
    default_colour_code=False,
):
    point_cart_coords = point_cart_coords or {}
    path_sequences = path_sequences or []

    b_vectors_visible = _plotly_visible(b_vectors_visible)
    all_points_visible = _plotly_visible(all_points_visible)
    path_visible = _plotly_visible(path_visible)

    # Default style is black. The coloured layers are added as legend-only
    # alternatives, so the same HTML file can switch between clean black and
    # colour-coded views without rerunning the script.
    black_path_visible = path_visible if not default_colour_code else "legendonly"
    colour_path_visible = path_visible if default_colour_code else "legendonly"

    faces = bz_faces_from_reciprocal(bz_reciprocal_lattice)

    fig = go.Figure()

    # ------------------------------------------------------------
    # BZ faces and black edges.
    # Keep all BZ geometry in one legend group, so one legend click
    # can hide/show the whole zone without touching the path/points.
    # ------------------------------------------------------------
    first_bz_trace = True
    for face in faces:
        closed = np.vstack([face, face[0]])

        fig.add_trace(go.Scatter3d(
            x=closed[:, 0],
            y=closed[:, 1],
            z=closed[:, 2],
            mode="lines",
            line=dict(width=6, color="black"),
            name="Brillouin-zone boundary",
            legendgroup="BZ",
            showlegend=first_bz_trace,
            hoverinfo="skip",
        ))
        first_bz_trace = False

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
                color="lightgray",
                opacity=0.10,
                name="Brillouin-zone faces",
                legendgroup="BZ",
                showlegend=False,
                hoverinfo="skip",
            ))

    # ------------------------------------------------------------
    # Reciprocal basis vectors.
    # These are now available directly in the HTML legend. They are
    # legend-only by default, so the initial figure stays clean and black.
    # ------------------------------------------------------------
    if include_b_vectors:
        for idx, bvec in enumerate(bz_reciprocal_lattice, start=1):
            add_b_vector(
                fig,
                bvec,
                f"b{idx}",
                color="black",
                visible=b_vectors_visible,
                legendgroup=f"b{idx}-black",
                showlegend=True,
            )

        # Colour-coded alternative: b1/b2/b3 use different colours, but the
        # whole coloured set can be toggled with one legend click.
        b_colours = ["#d62728", "#2ca02c", "#1f77b4"]
        for idx, bvec in enumerate(bz_reciprocal_lattice, start=1):
            add_b_vector(
                fig,
                bvec,
                f"b{idx}",
                color=b_colours[idx - 1],
                visible="legendonly",
                legendgroup=f"b{idx}-colour-coded",
                showlegend=True,
            )

    # ------------------------------------------------------------
    # All high-symmetry points.
    # Separate from the selected path to avoid duplicated labels on open.
    # ------------------------------------------------------------
    if point_cart_coords:
        xs, ys, zs, texts = [], [], [], []
        for label, cart in point_cart_coords.items():
            cart = np.array(cart, dtype=float)
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
            marker=dict(size=7, color="black"),
            textfont=dict(size=16, color="#34495e"),
            name="All high-symmetry points",
            legendgroup="all-points-black",
            showlegend=True,
            visible=all_points_visible,
        ))

        colour_list = [_cyclic_colour(i) for i in range(len(xs))]
        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            text=texts,
            textposition="top center",
            marker=dict(size=7, color=colour_list),
            textfont=dict(size=16, color="#34495e"),
            name="All high-symmetry points",
            legendgroup="all-points-colour-coded",
            showlegend=True,
            visible="legendonly",
        ))

        # Individual point toggles: always present in the legend, but hidden
        # initially. This replaces the old --individual-point-toggles workflow.
        for point_idx, (label, cart) in enumerate(point_cart_coords.items()):
            cart = np.array(cart, dtype=float)
            shown_label = display_label(label)
            fig.add_trace(go.Scatter3d(
                x=[cart[0]],
                y=[cart[1]],
                z=[cart[2]],
                mode="markers+text",
                text=[shown_label],
                textposition="top center",
                marker=dict(size=8, color="black"),
                textfont=dict(size=16, color="#34495e"),
                name=f"Point {shown_label}",
                legendgroup=f"point-{label}",
                showlegend=True,
                visible="legendonly",
                hovertemplate=(
                    f"Point {shown_label}<br>kx=%{{x:.4f}}<br>ky=%{{y:.4f}}<br>kz=%{{z:.4f}}<extra></extra>"
                ),
            ))

    # ------------------------------------------------------------
    # Selected path: clean black default layer.
    # Draw path lines separately from path labels. This avoids repeated
    # labels when the recommended SeeK-path has disconnected segments.
    # ------------------------------------------------------------
    path_label_coords = {}

    first_black_path_trace = True
    for seq_idx, seq in enumerate(path_sequences):
        coords = []
        labels = []

        for label in seq:
            cart = np.array(point_cart_coords[label], dtype=float)
            coords.append(cart)
            labels.append(display_label(label))
            path_label_coords[label] = cart

        coords = np.array(coords)
        segment_name = "-".join(labels)

        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="lines+markers",
            line=dict(width=9, color="black"),
            marker=dict(size=6, color="black"),
            name="Selected path",
            legendgroup="selected-path-black",
            showlegend=first_black_path_trace,
            visible=black_path_visible,
            hovertemplate=(
                "Path segment: " + segment_name +
                "<br>kx=%{x:.4f}<br>ky=%{y:.4f}<br>kz=%{z:.4f}<extra></extra>"
            ),
        ))
        first_black_path_trace = False

        # Individual path segment toggles: always in the legend, hidden at
        # first. This replaces the old --individual-path-toggles workflow.
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="lines+markers+text",
            text=labels,
            textposition="bottom center",
            line=dict(width=7, color="black"),
            marker=dict(size=6, color="black"),
            textfont=dict(size=15, color="#34495e"),
            name=f"Path segment {segment_name}",
            legendgroup=f"path-segment-{seq_idx}",
            showlegend=True,
            visible="legendonly",
            hovertemplate=(
                "Path segment: " + segment_name +
                "<br>kx=%{x:.4f}<br>ky=%{y:.4f}<br>kz=%{z:.4f}<extra></extra>"
            ),
        ))

    # Unique labels for points that appear on the selected path.
    # This is grouped with the selected-path legend item, so one click hides
    # both the line and its labels. It also prevents duplicated Γ labels.
    if path_label_coords:
        ordered_path_labels = ordered_unique_labels(path_sequences)
        xs, ys, zs, texts = [], [], [], []
        for label in ordered_path_labels:
            cart = np.array(path_label_coords[label], dtype=float)
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
            textposition="bottom center",
            marker=dict(size=7, color="black"),
            textfont=dict(size=16, color="#34495e"),
            name="Selected path labels",
            legendgroup="selected-path-black",
            showlegend=False,
            visible=black_path_visible,
        ))

    # ------------------------------------------------------------
    # Colour-coded selected path alternative.
    # Default is legend-only unless --colour-code/--color-code is used.
    # ------------------------------------------------------------
    if path_sequences:
        first_colour_path_trace = True
        colour_label_coords = {}

        for seq_idx, seq in enumerate(path_sequences):
            coords = []
            labels = []
            for label in seq:
                cart = np.array(point_cart_coords[label], dtype=float)
                coords.append(cart)
                labels.append(display_label(label))
                colour_label_coords[label] = cart

            coords = np.array(coords)
            segment_name = "-".join(labels)
            colour = _cyclic_colour(seq_idx)

            fig.add_trace(go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="lines+markers",
                line=dict(width=8, color=colour),
                marker=dict(size=6, color=colour),
                name="Selected path",
                legendgroup="selected-path-colour-coded",
                showlegend=first_colour_path_trace,
                visible=colour_path_visible,
                hovertemplate=(
                    "Colour-coded path segment: " + segment_name +
                    "<br>kx=%{x:.4f}<br>ky=%{y:.4f}<br>kz=%{z:.4f}<extra></extra>"
                ),
            ))
            first_colour_path_trace = False

        ordered_path_labels = ordered_unique_labels(path_sequences)
        xs, ys, zs, texts = [], [], [], []
        for label in ordered_path_labels:
            cart = np.array(colour_label_coords[label], dtype=float)
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
            textposition="bottom center",
            marker=dict(size=7, color="black"),
            textfont=dict(size=16, color="#34495e"),
            name="Selected path labels",
            legendgroup="selected-path-colour-coded",
            showlegend=False,
            visible=colour_path_visible,
        ))

    if show_axes:
        axis_style = dict(
            title="",
            showticklabels=True,
            showgrid=True,
            zeroline=False,
            showbackground=True,
        )
    else:
        axis_style = dict(
            title="",
            showticklabels=False,
            ticks="",
            showgrid=False,
            zeroline=False,
            showbackground=False,
            visible=False,
        )

    fig.update_layout(
        title=(
            f"{title}<br>"
            "<sup>Click legend items to show/hide: BZ, selected path, individual path segments, "
            "high-symmetry points, b-vectors, and colour-coded alternatives.</sup>"
        ),
        scene=dict(
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            aspectmode="data",
        ),
        legend=dict(
            title="Click legend items to show/hide",
            groupclick="togglegroup",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=0, r=0, b=0, t=70),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    fig.write_html(output_html)
    print(f"\nWrote interactive BZ plot to: {output_html}")


# ============================================================
# Main modes
# ============================================================

def run_structure_mode(args):
    path = Path(args.structure)
    structure = load_structure(args.structure)

    input_lattice = np.array(structure.lattice.matrix, dtype=float)
    input_reciprocal_lattice = reciprocal_lattice_from_real(input_lattice)

    sp_input = to_seekpath_input(structure)
    sp = seekpath.get_path(
        sp_input,
        with_time_reversal=not args.no_time_reversal,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
    )

    primitive_lattice = np.array(sp["primitive_lattice"], dtype=float)
    seekpath_reciprocal_lattice = reciprocal_lattice_from_real(primitive_lattice)

    seekpath_point_coords = {
        label: np.array(coord, dtype=float)
        for label, coord in sp["point_coords"].items()
    }

    if args.points:
        seekpath_point_coords.update(load_points_json(args.points))

    print("\n=== Symmetry / lattice information from SeeK-path ===")
    print("Space group:", sp["spacegroup_international"], sp["spacegroup_number"])
    print("Bravais lattice:", sp["bravais_lattice"])
    print("Extended Bravais lattice:", sp["bravais_lattice_extended"])

    print("\n=== Original input-cell real-space lattice / Å ===")
    print(input_lattice)

    print("\n=== Original input-cell reciprocal lattice / Å^-1, 2π convention ===")
    print(input_reciprocal_lattice)

    print("\n=== SeeK-path standardized primitive real-space lattice / Å ===")
    print(primitive_lattice)

    print("\n=== SeeK-path standardized primitive reciprocal lattice / Å^-1, 2π convention ===")
    print(seekpath_reciprocal_lattice)

    print("\n=== SeeK-path high-symmetry points ===")
    print("Coordinates are reduced coordinates in the SeeK-path standardized primitive reciprocal basis.")
    for label, coord in seekpath_point_coords.items():
        print(f"{display_label(label):8s} {coord}")

    print("\n=== SeeK-path recommended path ===")
    print(sp["path"])

    # BZ shape should normally be the SeeK-path standardized primitive BZ.
    # The manual coordinate basis affects only how user points are converted into Cartesian k-space.
    bz_reciprocal_lattice = seekpath_reciprocal_lattice
    bz_basis_description = "SeeK-path standardized primitive reciprocal basis"

    point_source = "seekpath"
    manual_basis = None
    manual_reduced_coords = None

    if args.path:
        # Command-line --path still means SeeK-path labels.
        path_sequences = parse_path_string(args.path, seekpath_point_coords.keys())
        point_cart_coords = {
            label: reduced_to_cartesian_k(coord, seekpath_reciprocal_lattice)
            for label, coord in seekpath_point_coords.items()
        }
        point_source = "seekpath_path_argument"

    elif args.interactive:
        answer = input("\nUse SeeK-path recommended path? [Y/n]: ").strip().lower()

        if answer in {"", "y", "yes"}:
            path_sequences = seekpath_pairs_to_sequences(sp["path"])
            point_cart_coords = {
                label: reduced_to_cartesian_k(coord, seekpath_reciprocal_lattice)
                for label, coord in seekpath_point_coords.items()
            }
            point_source = "seekpath_recommended"

        else:
            print("\nFully manual path mode selected.")
            print("Manual labels will NOT be matched to SeeK-path labels.")
            print("The BZ shape will still be drawn using the SeeK-path standardized primitive cell.")
            print("Your coordinates will be converted to Cartesian k-space before plotting.")

            manual = input("Enter manual path labels only, e.g. R-G-R' or R G R': ")
            path_sequences = parse_fully_manual_path_string(manual)

            manual_basis = ask_manual_coordinate_basis()
            if manual_basis == "input":
                coord_reciprocal_lattice = input_reciprocal_lattice
                coord_basis_description = "original input-cell reciprocal basis"
            else:
                coord_reciprocal_lattice = seekpath_reciprocal_lattice
                coord_basis_description = "SeeK-path standardized primitive reciprocal basis"

            manual_reduced_coords, point_cart_coords = ask_fully_manual_points(
                path_sequences,
                coord_reciprocal_lattice,
            )
            point_source = "fully_manual"
            print(f"\nManual coordinates interpreted in: {coord_basis_description}")
            print(f"BZ shape drawn from: {bz_basis_description}")

    else:
        path_sequences = seekpath_pairs_to_sequences(sp["path"])
        point_cart_coords = {
            label: reduced_to_cartesian_k(coord, seekpath_reciprocal_lattice)
            for label, coord in seekpath_point_coords.items()
        }
        point_source = "seekpath_recommended_noninteractive"

    html_name = args.output or f"{path.with_suffix('').name}_bz.html"
    json_name = f"{Path(html_name).with_suffix('').name}_data.json"

    make_bz_plot(
        bz_reciprocal_lattice=bz_reciprocal_lattice,
        point_cart_coords=point_cart_coords,
        path_sequences=path_sequences,
        title=f"{path.name} BZ",
        output_html=html_name,
        show_axes=args.show_axes,
        include_b_vectors=not args.hide_b_vectors,
        b_vectors_visible=(True if args.show_b_vectors else "legendonly"),
        all_points_visible=args.all_points_visible,
        path_visible=args.path_visible,
        default_colour_code=args.colour_code,
    )

    output_data = {
        "mode": "structure",
        "spacegroup": [sp["spacegroup_international"], sp["spacegroup_number"]],
        "bravais_lattice": sp["bravais_lattice"],
        "bravais_lattice_extended": sp["bravais_lattice_extended"],
        "input_lattice_A": input_lattice.tolist(),
        "input_reciprocal_lattice_Ainv_2pi": input_reciprocal_lattice.tolist(),
        "seekpath_primitive_lattice_A": primitive_lattice.tolist(),
        "seekpath_reciprocal_lattice_Ainv_2pi": seekpath_reciprocal_lattice.tolist(),
        "bz_basis": bz_basis_description,
        "point_source": point_source,
        "manual_coordinate_basis": manual_basis,
        "seekpath_point_coords_reduced": {
            k: np.array(v).tolist() for k, v in seekpath_point_coords.items()
        },
        "manual_point_coords_reduced": (
            None if manual_reduced_coords is None
            else {k: np.array(v).tolist() for k, v in manual_reduced_coords.items()}
        ),
        "point_coords_cartesian_Ainv": {
            k: np.array(v).tolist() for k, v in point_cart_coords.items()
        },
        "path_sequences": path_sequences,
        "seekpath_raw": sp,
    }

    with open(json_name, "w") as f:
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

    point_cart_coords = {
        label: reduced_to_cartesian_k(coord, reciprocal_lattice)
        for label, coord in point_coords.items()
    }

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
    elif args.interactive:
        print("\nFully manual path mode selected for space-group mode.")
        manual = input("Enter manual path labels only, e.g. R-G-R' or R G R': ")
        path_sequences = parse_fully_manual_path_string(manual)
        manual_reduced_coords, point_cart_coords = ask_fully_manual_points(
            path_sequences,
            reciprocal_lattice,
        )
        point_coords = manual_reduced_coords
    else:
        path_sequences = []

    html_name = args.output or f"SG_{info['spacegroup_number']}_{info['bravais_lattice']}_bz.html"
    json_name = f"{Path(html_name).with_suffix('').name}_data.json"

    make_bz_plot(
        bz_reciprocal_lattice=reciprocal_lattice,
        point_cart_coords=point_cart_coords,
        path_sequences=path_sequences,
        title=f"SG {info['spacegroup_number']} {info['spacegroup_symbol']} / {info['bravais_lattice']}",
        output_html=html_name,
        show_axes=args.show_axes,
        include_b_vectors=not args.hide_b_vectors,
        b_vectors_visible=(True if args.show_b_vectors else "legendonly"),
        all_points_visible=args.all_points_visible,
        path_visible=args.path_visible,
        default_colour_code=args.colour_code,
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
        "point_coords_cartesian_Ainv": {
            k: np.array(v).tolist() for k, v in point_cart_coords.items()
        },
        "path_sequences": path_sequences,
    }

    with open(json_name, "w") as f:
        json.dump(make_json_safe(output_data), f, indent=2)

    print(f"Wrote data to: {json_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Brillouin zone and selected high-symmetry paths.",
        allow_abbrev=False,
    )

    # Extra alias, so `--h` prints help rather than conflicting with
    # `--hide-b-vectors`. Standard `-h` and `--help` still work.
    parser.add_argument("--h", action="help", help=argparse.SUPPRESS)

    parser.add_argument(
        "input_structure",
        nargs="?",
        help="Input structure file shortcut, e.g. Ca3Mn2O7.cell. Equivalent to --structure Ca3Mn2O7.cell.",
    )

    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--structure", help="Input .cell, .cif, POSCAR, etc. Optional if you give the file as the positional argument.")
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
        help="SeeK-path-label path, e.g. 'G-X-M-G'. In structure mode this still uses SeeK-path labels.",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        default=None,
        help="Ask whether to use recommended path or enter a fully manual path. This is now the default in structure-file mode.",
    )

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not ask interactive questions; use the SeeK-path recommended path automatically unless --path is supplied.",
    )

    parser.add_argument(
        "--output",
        help="Output HTML filename.",
    )

    parser.add_argument(
        "--show-axes",
        action="store_true",
        help="Show 3D axis ticks/grid/labels. By default they are hidden for a cleaner figure.",
    )

    parser.add_argument(
        "--show-b-vectors",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--hide-b-vectors",
        action="store_true",
        help="Backward-compatible option. If given, reciprocal basis vectors are hidden.",
    )

    parser.add_argument(
        "--all-points-visible",
        choices=["true", "false", "legendonly"],
        default="legendonly",
        help=(
            "Initial visibility of the full high-symmetry point layer. "
            "Default: legendonly, so it can be turned on from the HTML legend without duplicating path labels initially."
        ),
    )

    parser.add_argument(
        "--path-visible",
        choices=["true", "false", "legendonly"],
        default="true",
        help="Initial visibility of the selected path layer. Default: true.",
    )

    # Backward-compatible flags from the previous version. Individual point
    # and path-segment toggles are now always included in the HTML legend, so
    # these no longer need to be shown in --help.
    parser.add_argument(
        "--individual-point-toggles",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--individual-path-toggles",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--colour-code",
        "--color-code",
        dest="colour_code",
        action="store_true",
        help="Open the HTML with the colour-coded path visible by default. Without this, the default visible path is black.",
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

    # Convenience interface:
    #   python Brillouin_zone_plot_v3_final.py structure.cell
    # is equivalent to:
    #   python Brillouin_zone_plot_v3_final.py --structure structure.cell --interactive
    if args.input_structure:
        if args.structure:
            parser.error("Give the structure file either positionally or with --structure, not both.")
        if args.sg:
            parser.error("Do not give a positional structure file together with --sg.")
        args.structure = args.input_structure

    if not args.structure and not args.sg:
        parser.error("Give a structure file, e.g. python Brillouin_zone_plot_v3_final.py Ca3Mn2O7.cell, or use --sg for space-group mode.")

    if args.non_interactive:
        args.interactive = False
    elif args.interactive is None:
        # For the common structure-file workflow, interactive should be the default.
        # For --sg mode, keep the old behaviour unless --interactive is explicitly supplied.
        args.interactive = bool(args.structure)

    if args.structure:
        run_structure_mode(args)
    else:
        run_spacegroup_mode(args)


if __name__ == "__main__":
    main()
