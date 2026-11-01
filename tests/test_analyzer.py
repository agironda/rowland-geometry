import numpy as np
import pytest

from rowland_geometry.analyzer import Analyzer
from tests.reference_data.gog_analyzer_atlas import GOG_TABLES
from tests.reference_data.gironda_2024_si551 import GIRONDA_2024_ASYMMETRIC, GIRONDA_MATERIAL, GIRONDA_CUT

GOG_ENERGY_CASES = [
    pytest.param(material, hkl, energy, bragg, id=f"{edge}-{material}-{hkl}")
    for edge, energy, table in GOG_TABLES
    for material, hkl, _e0, bragg in table
]

# Literature values for reflections and their alphas of Si[551]
GIRONDA_ALPHA_CASES = [
    pytest.param(reflection, alpha)
    for _emission, _energy, reflection, _bragg, alpha, _theta_m
    in GIRONDA_2024_ASYMMETRIC
]

# Literature values for reflections, bragg, and energy of Si[551]
GIRONDA_BRAGG_CASES = [
    pytest.param(energy, reflection, bragg)
    for _emission, energy, reflection, bragg, _alpha, _theta_m
    in GIRONDA_2024_ASYMMETRIC
]

# Does going from bragg -> energy -> bragg give the same result
@pytest.mark.parametrize("bragg", [85, 70, 55])
def test_round_trip(bragg):
    sbca = Analyzer("sapphire", (4, -1, 5))
    calculated = sbca.to_bragg(sbca.to_energy(bragg))
    assert calculated == pytest.approx(bragg, abs = 0.01)

# Check the Bragg -> energy conversion from Gironda Si data
# Bigger tolerance on here. Different lattice parameter in literature (?)
@pytest.mark.parametrize("energy, reflection, bragg", GIRONDA_BRAGG_CASES)
def test_bragg_to_energy(energy, reflection, bragg):
    sbca = Analyzer(GIRONDA_MATERIAL, reflection)
    calculated = sbca.to_energy(bragg)
    assert calculated == pytest.approx(energy, abs=5)

# Check the energy -> bragg conversion from Gironda Si data
# Bigger tolerance on here. Different lattice parameter in literature (?)
@pytest.mark.parametrize("energy, reflection, bragg", GIRONDA_BRAGG_CASES)
def test_energy_to_bragg(energy, reflection, bragg):
    sbca = Analyzer(GIRONDA_MATERIAL, reflection)
    calculated = sbca.to_bragg(energy)
    assert calculated == pytest.approx(bragg, abs=0.1)

# Check the alpha calculation from Gironda Si data
# Tight tolerance here because cubic structure, differences in my and literature's
# lattice parameters won't change the calculated alpha, assuming the math's right.
@pytest.mark.parametrize("reflection, alpha", GIRONDA_ALPHA_CASES)
def test_calculated_asymmetry(reflection, alpha):
    sbca = Analyzer(GIRONDA_MATERIAL, reflection, cut=GIRONDA_CUT)
    calculated = sbca.asymmetry
    assert calculated == pytest.approx(alpha, abs=0.01)

# Check the energy -> Bragg conversion from all Gog Analyzer Atlas data
@pytest.mark.parametrize("material, reflection, energy, bragg", GOG_ENERGY_CASES)
def test_energy_to_bragg_gog(material, reflection, energy, bragg):
    sbca = Analyzer(material, reflection)
    calculated = sbca.to_bragg(energy)
    assert calculated == pytest.approx(bragg, abs=0.01)

# Check that e0 corresponds to the energy at 90 deg bragg calculated by the class
@pytest.mark.parametrize(
    "material, reflection",
    [
        ("Si", (5, 3, 1)),
        ("Ge", (3, 3, 3)),
        ("Quartz", (1, 0, 2)),
        ("Sapphire", (4, -1, 5)),
        ("LiNbO3", (3, 1, 0)),
    ],
)
def test_backscattering_limit(material, reflection):
    sbca = Analyzer(material, reflection)
    assert sbca.to_energy(90) == pytest.approx(sbca.e0)
    assert sbca.to_bragg(sbca.e0) == pytest.approx(90.0)

# Check that the miscut (which can be signed) correctly sums with crystal
# asymmetry to give the riight overall asymmetry.
@pytest.mark.parametrize(
    "miscut, expected_alpha",
    [
        (0.0, 45.0),
        (2.0, 47.0),
        (-2.0, 43.0),
    ],
)
def test_miscut_asymmetry(miscut, expected_alpha):
    sbca = Analyzer(
        "Si",
        (1, 0, 0),
        cut=(1, 1, 0),
        miscut=miscut,
    )
    assert sbca.crystal_asymmetry == pytest.approx(45.0)
    assert sbca.miscut == pytest.approx(miscut)
    assert sbca.asymmetry == pytest.approx(expected_alpha)

# Check that the cut defaults to reflection if not specified and asymmetry is 0
def test_default_cut():
    sbca = Analyzer("Si", (5, 3, 1))
    np.testing.assert_array_equal(sbca.cut, sbca.reflection)
    assert sbca.crystal_asymmetry == pytest.approx(0.0)
    assert sbca.asymmetry == pytest.approx(0.0)

# Check that bragg must be between 0 and 90
@pytest.mark.parametrize("bragg", [0, -10, 90.1, 180, np.nan, np.inf, -np.inf])
def test_invalid_bragg(bragg):
    sbca = Analyzer("Si", (5, 3, 1))
    with pytest.raises(ValueError, match="Bragg angle"):
        sbca.to_energy(bragg)

# Check that energy must be greater than e0
@pytest.mark.parametrize("energy", [-100, 0, np.nan, np.inf, -np.inf])
def test_invalid_energy(energy):
    sbca = Analyzer("Si", (5, 3, 1))
    with pytest.raises(ValueError, match="Energy"):
        sbca.to_bragg(energy)

def test_energy_below_backscattering():
    sbca = Analyzer("Si", (5, 3, 1))
    with pytest.raises(ValueError, match="Energy"):
        sbca.to_bragg(sbca.e0 - 1.0)

@pytest.mark.parametrize(
    "kwargs",
    [
        {"bending_radius": 0},
        {"bending_radius": -500},
        {"bending_radius": np.inf},
        {"optic_diameter": 0},
        {"optic_diameter": -10},
        {"optic_diameter": np.nan},
        {"miscut": 90},
        {"miscut": -90},
        {"miscut": np.inf},
        {"miscut": np.nan},
    ],
)
def test_construction(kwargs):
    with pytest.raises(ValueError):
        Analyzer("Si", (5, 3, 1), **kwargs)

def test_analyzer_as_dict():
    sbca = Analyzer(
        "Si", (5, 3, 1),
        cut=(5, 5, 1),
        bending_radius=500,
        optic_diameter=100,
        miscut=1.0,
    )
    data = sbca.as_dict()
    assert data["material"] == "Silicon"
    assert data["reflection"] == [5, 3, 1]
    assert data["cut"] == [5, 5, 1]
    assert data["bending_radius"] == 500
    assert data["optic_diameter"] == 100
    assert data["miscut"] == 1.0
    assert data["asymmetry"] == (data["crystal_asymmetry"] + data["miscut"])


def test_analyzer_indices_read_only():
    sbca = Analyzer("Si", (5, 3, 1))
    with pytest.raises(ValueError):
        sbca.reflection[0] = 1
    with pytest.raises(ValueError):
        sbca.cut[0] = 1