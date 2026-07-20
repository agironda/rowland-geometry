"""
Crystal analyzer material data and crystallographic calculations.

This module provides lookup of supported analyzer materials and calculates the 
interplanar d-spacing of a plane family and the angles between two crystal 
plane families.

The most common hard X-ray analyzer materials are supported and hard-coded here
in a dictionary: silicon, germanium, quartz, sapphire, and lithium niobate.
"""

import numpy as np
from typing import Sequence

CRYSTALS = {

    # lattice constants from Hom, Kiszenick, Post J Appl Cryst 1975
    "si": { 
        "name": "Silicon",
        "formula": "Si",
        "aliases": ("silicon", "si"),
        "system": "cubic",
        "a": 5.430941, # angstrom
    },
    "ge": {
        "name": "Germanium",
        "formula": "Ge",
        "aliases": ("germanium", "ge"),
        "system": "cubic",
        "a": 5.65782, # angstrom
    },

    # lattice constants from Hasan MRS Bulletin 2017 Iss 06 Vol 42
    "quartz": {
        "name": "Quartz",
        "formula": "SiO2",
        "aliases": ("quartz", "sio", "sio2"),
        "system": "hexagonal",
        "a": 4.914, # angstrom
        "c": 5.405, # angstrom
    },
    "sapphire": {
        "name": "Sapphire",
        "formula": "Al2O3",
        "aliases": ("sapphire", "alo", "al2o3"),
        "system": "hexagonal",
        "a":  4.754, # angstrom
        "c": 12.982, # angstrom
    },
    "linbo3": {
        "name": "Lithium Niobate",
        "formula": "LiNbO3",
        "aliases": ("lithium niobate", "linbo3", "linbo"),
        "system": "hexagonal",
        "a":  5.148, # angstrom
        "c": 13.863, # angstrom
    },
}


def get_crystal(material: str) -> dict:
    """
    Returns data for a supported crystal analyzer material.

    Parameters
    ----------
    material : str
        Name or alias of the crystal analyzer material.

    Returns
    -------
    dict
        Crystal data containing the material name, formula, aliases, crystal
        system, and lattice parameters in angstroms.

    Examples
    --------
    >>> get_crystal("Si")["name"]
    'Silicon'
    >>> get_crystal("lithium niobate")["a"]
    5.148
    """
    key = material.strip().lower()

    for crystal in CRYSTALS.values():
        valid_keys = [
            crystal["name"].strip().lower(),
            crystal["formula"].strip().lower(),
            *[alias.strip().lower() for alias in crystal["aliases"]]
        ]

        if key in valid_keys:
            return crystal.copy()
        
    supported = ", ".join(crystal["name"] for crystal in CRYSTALS.values())

    raise ValueError(
        f"Unsupported crystal material or unrecognized alias. "
        f"Supported materials: {supported}"
    )


def d_spacing(
        material: str,
        reflection: Sequence[int]
        ) -> float:
    """
    Calculate the interplanar distance of a plane family in a given crystal 
    structure.

    Parameters
    ----------
    material : str
        Name or alias of the crystal analyzer material.
    reflection : Sequence[int]
        Miller indices of the reflection as [h, k, l]. For hexagonal crystal 
        systems, either [h, k, l] or [h, k, i, l] is supported.

    Returns
    -------
    float
        Interplanar distance (d-spacing) in angstroms.
    
    Notes
    -----
    The d-spacing relies on the lattice parameters of the analyzer material.
    Presently, the following crystals are supported: silicon, germanium, 
    quartz, sapphire, and lithium niobate. Aliases are acceptable for 
    specifying 'material', such as si, ge, linbo3, linbo, etc. 

    Examples
    --------
    >>> d_spacing("germanium", [5, 5, 1])
    """
    crystal = get_crystal(material)
    hkl = np.asarray(reflection)

    if not np.all(np.isfinite(hkl)):
        raise ValueError("Reflection indices must be finite")
    
    if not np.all(np.equal(hkl, np.round(hkl))):
        raise ValueError("Reflection indices must be integers")

    hkl = hkl.astype(int)

    if not np.any(hkl):
        raise ValueError("Reflection indices cannot all be zero")

    if crystal["system"] == "cubic":
        if hkl.shape != (3,):
            raise ValueError(
                "Reflection must be a 3 element index (h, k, l)"
            )
        
        h, k, l = hkl
        a = crystal["a"]

        d = a / np.sqrt(h*h + k*k + l*l)

    elif crystal["system"] == "hexagonal":
        if hkl.shape == (3,):
            h, k, l = hkl

        elif hkl.shape == (4,):
            h, k, i, l = hkl
            
            if i != -(h + k):
                raise ValueError(
                    "For 4-index notation, i must equal -(h + k)"
                    )

        else:
            raise ValueError(
                "Reflection must be a 3 or 4 element index (h, k, (i), l)"
            )
        
        a = crystal["a"]
        c = crystal["c"]

        denom = ((4/3)*(h*h + k*k + h*k)) + a*a*l*l/c/c
        d = a / np.sqrt(denom)

    else:
        raise ValueError(
            f"Unsupported crystal system {crystal['system']!r}"
        )

    return float(d)


def plane_angle(
        material: str,
        plane_1: Sequence[int],
        plane_2: Sequence[int]
        ) -> float:
    """
    Calculate the angle between two crystal plane families.

    Parameters
    ----------
    material : str
        Name or alias of the crystal analyzer material.
    plane_1 : Sequence[int]
        Miller indices of the reflection as [h, k, l]. For hexagonal crystal 
        systems, either [h, k, l] or [h, k, i, l] is supported.
    plane_2 : Sequence[int]
        Miller indices of the second plane family. Uses the same conventions 
        as 'plane_1'.

    Returns
    -------
    float
        Angle between the planes in degrees from 0 to 90.
    
    Examples
    --------
    >>> plane_angle("silicon", [5, 5, 1], [1, 0, 0])
    """
    crystal = get_crystal(material)

    vector_1 = _plane_normal(crystal, plane_1)
    vector_2 = _plane_normal(crystal, plane_2)

    dot = float(np.dot(vector_1, vector_2))
    norm = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)

    cos_angle = np.clip(abs(dot) / norm, 0.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def _plane_normal(
        crystal: dict,
        indices: Sequence[int],
        ) -> np.ndarray:
    """
    Return a Cartesian vector normal to the crystal plane.
    """
    hkl = np.asarray(indices)

    if not np.all(np.isfinite(hkl)):
        raise ValueError("Plane indices must be finite")

    if not np.all(np.equal(hkl, np.round(hkl))):
        raise ValueError("Plane indices must be integers")

    hkl = hkl.astype(int)

    if not np.any(hkl):
        raise ValueError("Plane indices cannot all be zero")

    if crystal["system"] == "cubic":
        if hkl.shape != (3,):
            raise ValueError(
                "Cubic planes must use a 3-element index (h, k, l)"
            )

        return hkl.astype(float)

    elif crystal["system"] == "hexagonal":
        if hkl.shape == (3,):
            h, k, l = hkl

        elif hkl.shape == (4,):
            h, k, i, l = hkl

            if i != -(h + k):
                raise ValueError(
                    "For 4-index notation, i must equal -(h + k)"
                )

        else:
            raise ValueError(
                "Hexagonal planes must use (h, k, l) or (h, k, i, l)"
            )

        a = crystal["a"]
        c = crystal["c"]

        return np.array([
            h / a,
            (h + 2*k) / (np.sqrt(3.0) * a),
            l / c,
        ])

    else:
        raise ValueError(
            f"Unsupported crystal system {crystal['system']!r}"
        )
