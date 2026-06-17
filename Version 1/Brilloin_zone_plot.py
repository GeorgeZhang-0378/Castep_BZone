from pathlib import Path
import json
import numpy as np

import seekpath
from ase.io import read as ase_read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatviz import brillouin_zone_3d


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


def reciprocal_lattice_from_real(real_lattice):
    real_lattice = np.array(real_lattice)
    return np.linalg.inv(real_lattice).T * 2 * np.pi


def main(filename: str):
    path = Path(filename)
    structure = load_structure(filename)

    sp_input = to_seekpath_input(structure)
    sp = seekpath.get_path(sp_input)

    primitive_lattice = np.array(sp["primitive_lattice"])
    reciprocal_lattice = reciprocal_lattice_from_real(primitive_lattice)

    print("\n=== Symmetry / lattice information ===")
    print("Space group:", sp["spacegroup_international"], sp["spacegroup_number"])
    print("Bravais lattice:", sp["bravais_lattice"])
    print("Extended Bravais lattice:", sp["bravais_lattice_extended"])

    print("\n=== Standardized primitive real-space lattice, Å ===")
    print(primitive_lattice)

    print("\n=== Standardized primitive reciprocal lattice, 2π/Å convention ===")
    print(reciprocal_lattice)

    print("\n=== High-symmetry points, reduced coordinates in standardized primitive reciprocal basis ===")
    for label, coord in sp["point_coords"].items():
        print(f"{label:8s} {coord}")

    print("\n=== Recommended path ===")
    print(sp["path"])

    fig = brillouin_zone_3d(structure)
    html_name = path.with_suffix("").name + "_bz.html"
    fig.write_html(html_name)
    print(f"\nWrote interactive BZ plot to: {html_name}")

    json_name = path.with_suffix("").name + "_seekpath.json"
    with open(json_name, "w") as f:
        json.dump(sp, f, indent=2)
    print(f"Wrote SeeK-path data to: {json_name}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python Brilloin_zone_plot.py structure.cif_or_cell")

    main(sys.argv[1])
