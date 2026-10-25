"""Tests for the public API of :mod:`rowland_geometry.crystals`.

The suite checks three complementary contracts:
- agreement with independent Gog, Seely, Gironda and Orlat literature values;
- analytical geometry and invariance properties;
- crystal lookup and invalid-input behavior.

Bragg angle and analyzer configuration tests belong in test_analyzer.py.
"""

import math
from itertools import permutations

import numpy as np
import pytest

from rowland_geometry.crystals import d_spacing, get_crystal, plane_angle
from tests.reference_data.gog_analyzer_atlas import GOG_TABLES
from tests.reference_data.orlat_2026_asymmetry import ORLAT_TABLES
from tests.reference_data.seely_2019_quartz import SEELY_CUT, SEELY_QUARTZ
from tests.reference_data.gironda_2024_si551 import GIRONDA_2024_ASYMMETRIC, GIRONDA_MATERIAL, GIRONDA_CUT


HC_EV_ANGSTROM = 12398.41904

# The atlas includes Bragg angles and incident energies; neither is used
# to test d_spacing(). Keep only the published backscattering energies.
GOG_E0_CASES = [
    pytest.param(material, hkl, e0, id=f"{edge}-{material}-{hkl}")
    for edge, _incident_energy, table in GOG_TABLES
    for material, hkl, e0, _bragg_angle in table
]

# Likewise, plane_angle() uses only the reflection, cut and alpha columns.
ORLAT_ALPHA_CASES = [
    pytest.param(material, hkl, cut, alpha, id=f"{edge}-{material}-{hkl}-{cut}")
    for edge, _incident_energy, table in ORLAT_TABLES
    for material, hkl, cut, alpha, _bragg_angle, _theta_m in table
]

SEELY_D_CASES = [
    pytest.param(hkl, two_d_nm, id=str(hkl))
    for hkl, two_d_nm, _alpha, _bragg_angle in SEELY_QUARTZ
]
SEELY_ALPHA_CASES = [
    pytest.param(hkl, alpha, id=str(hkl))
    for hkl, _two_d_nm, alpha, _bragg_angle in SEELY_QUARTZ
]

GIRONDA_ALPHA_CASES = [
    pytest.param(
        reflection,
        alpha,
        id=emission,
    )
    for emission, _energy, reflection, _bragg, alpha, _theta_m
    in GIRONDA_2024_ASYMMETRIC
]

# ----------------------------------------------------------------------------
# Material lookup: get_crystal()
# ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "material, expected_name",
    [
        ("Si", "Silicon"),
        (" SILICON ", "Silicon"),
        ("gE", "Germanium"),
        ("germanium", "Germanium"),
        ("Quartz", "Quartz"),
        ("sio2", "Quartz"),
        ("SAPPH", "Sapphire"),
        ("corundum", "Sapphire"),
        ("Al2O3", "Sapphire"),
        ("LiNbO3", "Lithium Niobate"),
        (" lithium niobate ", "Lithium Niobate"),
        ("linbo", "Lithium Niobate"),
    ],
)
def test_get_crystal_aliases(material, expected_name):
    assert get_crystal(material)["name"] == expected_name


@pytest.mark.parametrize(
    "material, expected_formula, crystal_system, lattice_setting",
    [
        ("Si", "Si", "cubic", "cubic"),
        ("Ge", "Ge", "cubic", "cubic"),
        ("Quartz", "SiO2", "trigonal", "hexagonal"),
        ("Sapphire", "Al2O3", "trigonal", "hexagonal"),
        ("LiNbO3", "LiNbO3", "trigonal", "hexagonal"),
    ],
)
def test_get_crystal_metadata(
    material, expected_formula, crystal_system, lattice_setting
):
    crystal = get_crystal(material)

    assert crystal["formula"] == expected_formula
    assert crystal["crystal_system"] == crystal_system
    assert crystal["lattice_setting"] == lattice_setting
    assert crystal["a"] > 0

    if lattice_setting == "hexagonal":
        assert crystal["c"] > 0


def test_get_crystal_returns_a_copy():
    first = get_crystal("si")
    original_a = first["a"]
    first["a"] = -1

    assert get_crystal("si")["a"] == original_a


@pytest.mark.parametrize("material", ["diamond", "", "not a crystal"])
def test_get_crystal_rejects_unknown_material(material):
    with pytest.raises(ValueError, match="Unsupported crystal material"):
        get_crystal(material)


# ----------------------------------------------------------------------------
# d_spacing(): independent literature benchmarks
# ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "material, hkl, expected_e0",
    GOG_E0_CASES,
)
def test_d_spacing_gog(material, hkl, expected_e0):
    """Compare calculated backscattering energy with published atlas values."""
    d = d_spacing(material, hkl)
    calculated_e0 = HC_EV_ANGSTROM / (2.0 * d)

    # Atlas E0 is tabulated to the nearest eV. Small residual differences may
    # also depend on the reference lattice parameters used by the atlas.
    assert calculated_e0 == pytest.approx(expected_e0, abs=1.0)


@pytest.mark.parametrize("hkl, expected_two_d_nm", SEELY_D_CASES)
def test_d_spacing_seely(hkl, expected_two_d_nm):
    """Seely reports 2d in nm; d_spacing returns d in angstroms."""
    calculated_two_d_nm = 2.0 * d_spacing("Quartz", hkl) / 10.0
    assert calculated_two_d_nm == pytest.approx(expected_two_d_nm, abs=0.0001)


# ----------------------------------------------------------------------------
# d_spacing(): analytical values and crystallographic invariants
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("material", ["Si", "Ge"])
def test_d_spacing_cubic_special_cases(material):
    """Cubic (100) and (111) have exact analytic d-spacing relationships."""
    a = get_crystal(material)["a"]

    assert d_spacing(material, (1, 0, 0)) == pytest.approx(a)
    assert d_spacing(material, (1, 1, 1)) == pytest.approx(a / math.sqrt(3))


@pytest.mark.parametrize("material", ["Quartz", "Sapphire", "LiNbO3"])
def test_d_spacing_hexagonal_special_cases(material):
    """Independent limiting cases for hexagonal reciprocal lattice geometry."""
    crystal = get_crystal(material)
    a, c = crystal["a"], crystal["c"]

    assert d_spacing(material, (0, 0, 1)) == pytest.approx(c)
    assert d_spacing(material, (1, 0, 0)) == pytest.approx(math.sqrt(3) * a / 2)
    assert d_spacing(material, (1, 1, 0)) == pytest.approx(a / 2)


@pytest.mark.parametrize(
    "material, hkl",
    [
        ("Ge", (1, 3, 11)),          # Se K atlas
        ("Si", (2, 4, 10)),          # Se K atlas
        ("Sapphire", (4, 4, 15)),    # Se K atlas
        ("Quartz", (3, 3, 8)),       # Se K atlas
        ("LiNbO3", (2, 4, -23)),     # Se K atlas
    ],
)
def test_d_spacing_sign_invariance(material, hkl):
    """The planes (hkl) and (-h,-k,-l) have the same interplanar distance."""
    reversed_hkl = tuple(-index for index in hkl)
    assert d_spacing(material, reversed_hkl) == pytest.approx(
        d_spacing(material, hkl)
    )


@pytest.mark.parametrize(
    "material, hkl",
    [
        ("Si", (2, 4, 10)),       # Se K atlas
        ("Ge", (1, 3, 11)),       # Se K atlas
    ],
)
def test_d_spacing_cubic_permutation_invariance(material, hkl):
    """In a cubic lattice, permutations of h, k, l have the same d-spacing."""
    expected = d_spacing(material, hkl)

    for permuted_hkl in set(permutations(hkl)):
        assert d_spacing(material, permuted_hkl) == pytest.approx(expected)


@pytest.mark.parametrize(
    "material, first, second",
    [
        ("Quartz", (1, -1, -5), (0, 1, -5)),  # Cr K atlas
        ("Sapphire", (1, 3, -5), (3, -4, -5)),  # Cr K atlas
        ("LiNbO3", (0, -4, -2), (4, 0, -2)),  # Cr K atlas
    ],
)
def test_d_spacing_atlas_same_spacing_pairs(material, first, second):
    """Selected distinct reflections with the same d-spacing in Gog's atlas."""
    assert d_spacing(material, first) == pytest.approx(d_spacing(material, second))


@pytest.mark.parametrize(
    "material, hkl",
    [
        ("Quartz", (5, 5, 0)),        # Se K atlas: basal-plane example
        ("Quartz", (3, 3, 8)),        # Se K atlas: mixed-index example
        ("Sapphire", (4, 4, 15)),     # Se K atlas
        ("LiNbO3", (2, 4, -23)),      # Se K atlas
    ],
)
def test_d_spacing_hexagonal_three_vs_four_indices(material, hkl):
    h, k, l = hkl
    hkil = (h, k, -(h + k), l)

    assert d_spacing(material, hkl) == pytest.approx(d_spacing(material, hkil))


@pytest.mark.parametrize(
    "material, hkl",
    [
        ("Si", (5, 3, 1)),
        ("Ge", (3, 3, 3)),
        ("Quartz", (1, 0, 2)),
        ("Sapphire", (1, 3, -5)),
        ("LiNbO3", (1, -2, 12)),
    ],
)
def test_d_spacing_index_scaling(material, hkl):
    scaled_hkl = tuple(2 * index for index in hkl)

    assert d_spacing(material, scaled_hkl) == pytest.approx(
        d_spacing(material, hkl) / 2
    )


@pytest.mark.parametrize(
    "indices",
    [
        (2, 4, 10),
        [2, 4, 10],
        np.array([2, 4, 10]),
        np.array([2.0, 4.0, 10.0]),
    ],
)
def test_d_spacing_accepts_supported_sequences(indices):
    assert d_spacing("Si", indices) == pytest.approx(d_spacing("Si", (2, 4, 10)))


# ----------------------------------------------------------------------------
# plane_angle(): independent literature benchmarks
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("hkl, expected_alpha", SEELY_ALPHA_CASES)
def test_plane_angle_seely(hkl, expected_alpha):
    """Literature angle between a quartz reflection and the (101) cut."""
    calculated = plane_angle("Quartz", hkl, SEELY_CUT)
    assert calculated == pytest.approx(expected_alpha, abs=0.02)


@pytest.mark.parametrize(
    "material, reflection, cut, expected_alpha",
    [
        ("Si", (3, 3, -1), (1, 1, 0), 13.26),
        ("Ge", (2, 2, 0), (1, 1, 1), 35.26),
    ],
)
def test_plane_angle_literature(
    material, reflection, cut, expected_alpha
):
    calculated = plane_angle(material, reflection, cut)

    assert calculated == pytest.approx(expected_alpha, abs=0.02)


@pytest.mark.parametrize(
    "material, reflection, cut, expected_alpha",
    ORLAT_ALPHA_CASES,
)
def test_plane_angle_orlat(material, reflection, cut, expected_alpha):
    calculated = plane_angle(material, reflection, cut)

    assert calculated == pytest.approx(expected_alpha, abs=0.06)


@pytest.mark.parametrize(
    "reflection, expected_alpha",
    GIRONDA_ALPHA_CASES,
)
def test_plane_angle_gironda(reflection, expected_alpha):
    """Literature angles relative to the Si(551) surface cut."""
    calculated = plane_angle(
        GIRONDA_MATERIAL,
        reflection,
        GIRONDA_CUT,
    )

    assert calculated == pytest.approx(expected_alpha, abs=0.01)

# ----------------------------------------------------------------------------
# plane_angle(): geometrical identities and invariants
# ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "material, hkl",
    [
        ("Si", (5, 1, 1)),
        ("Ge", (1, 3, 5)),
        ("Quartz", (5, 1, 1)),
        ("Sapphire", (1, 3, -5)),
        ("LiNbO3", (0, -4, -2)),
    ],
)
def test_plane_angle_parallel_and_antiparallel(material, hkl):
    opposite = tuple(-index for index in hkl)

    assert plane_angle(material, hkl, hkl) == pytest.approx(0.0, abs=1e-6)
    assert plane_angle(material, hkl, opposite) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "material",
    ["Si", "Ge", "Quartz", "Sapphire", "LiNbO3"],
)
def test_plane_angle_orthogonal_axes(material):
    # (100) and (001) are perpendicular for these crystal systems.
    assert plane_angle(material, (1, 0, 0), (0, 0, 1)) == pytest.approx(
        90.0, abs=1e-8
    )


def test_plane_angle_cubic_45_degrees():
    assert plane_angle("Si", (1, 0, 0), (1, 1, 0)) == pytest.approx(45.0)


@pytest.mark.parametrize(
    "material",
    ["Quartz", "Sapphire", "LiNbO3"],
)
def test_plane_angle_hexagonal_60_degrees(material):
    calculated = plane_angle(
        material,
        (1, 0, 0),
        (0, 1, 0),
    )

    assert calculated == pytest.approx(60.0, abs=1e-8)


@pytest.mark.parametrize(
    "material, plane_1, plane_2",
    [
        ("Si", (1, 1, 1), (5, 1, 1)),
        ("Quartz", (3, 3, 8), (5, 5, 0)),
        ("Sapphire", (1, 3, -5), (4, 4, 15)),
        ("LiNbO3", (2, 4, -23), (0, -4, -2)),
    ],
)
def test_plane_angle_symmetric_and_bounded(material, plane_1, plane_2):
    forward = plane_angle(material, plane_1, plane_2)
    reverse = plane_angle(material, plane_2, plane_1)

    assert forward == pytest.approx(reverse)
    assert 0 <= forward <= 90


@pytest.mark.parametrize("material", ["Quartz", "Sapphire", "LiNbO3"])
def test_plane_angle_hexagonal_three_vs_four_indices(material):
    first = (1, 2, -5)
    second = (4, 0, 1)
    first_four = (1, 2, -3, -5)
    second_four = (4, 0, -4, 1)

    assert plane_angle(material, first, second) == pytest.approx(
        plane_angle(material, first_four, second_four)
    )


# ----------------------------------------------------------------------------
# Invalid input: public functions
# ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "material, indices, message",
    [
        ("Si", (0, 0, 0), "cannot all be zero"),
        ("Quartz", (0, 0, 0, 0), "cannot all be zero"),
        ("Si", (1, 2), "3-element"),
        ("Ge", (1, 2, 3, -3), "3-element"),
        ("Quartz", (1, 2), "must use"),
        ("Sapphire", (1, 2, 3, 4, 5), "must use"),
        ("LiNbO3", (1, 0, 0, 2), "i must equal"),
        ("Si", (1, 0.5, 0), "must be integers"),
        ("Quartz", (1, 0, 0.1, 0), "must be integers"),
        ("Ge", (1, float("nan"), 0), "must be finite"),
        ("Sapphire", (1, float("inf"), 0), "must be finite"),
    ],
)
def test_d_spacing_rejects_invalid_indices(material, indices, message):
    with pytest.raises(ValueError, match=message):
        d_spacing(material, indices)


@pytest.mark.parametrize(
    "indices",
    [
        (1+2j, 0, 0),
        ("1", 0, 0),
        ("a", 1, 0),
        (True, False, False),
    ],
)
def test_d_spacing_rejects_nonreal_indices(indices):
    with pytest.raises(ValueError, match="real numbers"):
        d_spacing("Si", indices)


@pytest.mark.parametrize("bad_operand", ["first", "second"])
def test_plane_angle_validates_both_planes(bad_operand):
    good, bad = (1, 0, 0), (0, 0, 0)
    first, second = (bad, good) if bad_operand == "first" else (good, bad)

    with pytest.raises(ValueError, match="cannot all be zero"):
        plane_angle("Si", first, second)

