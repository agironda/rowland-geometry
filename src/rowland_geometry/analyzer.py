"""
X-ray spherically bent crystal analyzer represented as a class Analyzer.

This module describes the Analyzer object, which holds all of the relevant 
parameters of a spherically bent crystal analyzer employed in hard x-ray
spectrometers, including the wafer material, reflection for photon analysis,
asymmetry and miscut, and the bending radius.

The analyzer object can be used to convert Bragg angle to analyzed photon 
energy and vice versa.
"""

from collections.abc import Sequence

import numpy as np

from .crystals import _validate_indices, d_spacing, get_crystal, plane_angle

# fundamental constants
HC = 12398.41904    # eV-angstroms


class Analyzer:
    """
    Object representation of a spherically bent crystal X-ray optic.

    Parameters
    ----------
    material : str
        Name, formula, or alias of a supported crystal material.
    reflection : Sequence[int]
        Miller indices of the diffracting plane.
    bending_radius : float
        Radius of curvature of the analyzer surface in millimetres.
    cut : Sequence[int], optional
        Miller indices of the nominally surface coincident plane family. 
        Defaults to ``reflection``.
    miscut : float, default 0.0
        Miscut angle between surface profile and ``cut`` plane family in degrees.
    optic_diameter : float, default 100.0
        Optic diameter in millimetres.

    Attributes
    ----------
    material : str
        Analyzer wafer material name.
    reflection : numpy.ndarray
        Three-index Miller representation of the diffracting plane family.
    cut : numpy.ndarray
        Three-index Miller representation of the surface cut plane family.
    bending_radius : float
        Radius of curvature of the analyzer surface in millimetres.
    optic_diameter : float
        Optic diameter in millimetres.
    d_hkl : float
        Interplanar spacing of the diffracting plane family in angstroms.
    miscut : float
        Miscut angle between surface profile and ``cut`` plane family in degrees.
    crystal_asymmetry : float
        Angle in degrees between the ``reflection`` and ``cut`` plane families.
    asymmetry : float
        Total asymmetry angle in degrees, the sum of ``crystal_asymmetry`` and ``miscut``.
    e0 : float
        Analyzed photon energy at backscattering (Bragg angle = 90) of the 
        specified reflection in eV.

    Methods
    -------
    to_energy()
        Computes the analyzed photon energy in eV for a given Bragg angle in 
        degrees for the specified plane family ``reflection``.

    to_bragg()
        Computes the Bragg angle in degrees for a given analyzed photon energy
        in eV for the specified plane family ``reflection``.

    as_dict()
        Returns the analyzer parameters and derived quantities as a dictionary.

    Notes
    -----
    All optic dimensions are in [mm], interplanar spacings are in [angstrom], 
    photon energies are in [eV] and all angles are in [deg].
    
    The public properties of the class are read only. A new Analyzer object 
    must be instantiated to change the parameters such as the reflection or 
    surface cut.

    There are no error checks if a specified reflection for photon analysis is
    allowed or forbidden based on the crystal system's selection rules.

    The miscut and crystal_asymmetry quantities describe different contributions to the
    total asymmetry angle. The miscut is the signed angle between the analyzer's
    surface and the nominal surface-cut plane family. For an SBCA, this angle is 
    nominally zero and the cut plane family is parallel to the analyzer surface.

    The crystal_asymmetry quantity is the crystallographic angle between the diffracting
    plane family and the cut plane family. It is independent of wafer miscut.
    The total asymmetry angle is therefore asymmetry = crystal_asymmetry + miscut.

    Examples
    --------
    >>> sbca = Analyzer("silicon", [5, 5, 1], 500)
    >>> sbca.to_bragg(8830)    # input in eV, output in degree
    67.39525...
    >>> sbca.to_energy(75)     # input in degree, output in eV
    8439.224...
    """
    
    def __init__(
        self,
        material: str,
        reflection: Sequence[int],
        bending_radius: float,
        *,
        cut: Sequence[int] | None = None,
        miscut: float = 0.0,
        optic_diameter: float = 100.0,
        ):
        
        bending_radius, optic_diameter, miscut = _validate_analyzer_scalars(
            bending_radius, 
            optic_diameter, 
            miscut,
        )
        
        self._bending_radius = bending_radius
        self._optic_diameter = optic_diameter
        self._miscut = miscut

        xtal = get_crystal(material)

        self._material = xtal["name"]
        self._reflection = _validate_indices(xtal, reflection)

        if cut is None: 
            self._cut = self._reflection.copy()
        else: 
            self._cut = _validate_indices(xtal, cut)

        self._reflection.setflags(write=False)
        self._cut.setflags(write=False)
        
        self._d_hkl = d_spacing(
            self._material, 
            self._reflection,
        )
        
        self._crystal_asymmetry = plane_angle(
            self._material, 
            self._reflection, 
            self._cut,
        )

        self._e0 = float(HC / (2.0 * self._d_hkl))  # eV

    @property
    def bending_radius(self) -> float:
        """Radius of curvature of the bent crystal analyzer surface in [mm]."""
        return self._bending_radius

    @property
    def optic_diameter(self) -> float:
        """Diameter of the optic in [mm]."""
        return self._optic_diameter

    @property
    def material(self) -> str:
        """Wafer material of the crystal analyzer."""
        return self._material

    @property
    def reflection(self) -> np.ndarray:
        """Indices of the diffracting reflection for photon analysis."""
        return self._reflection

    @property
    def cut(self) -> np.ndarray:
        """Indices of the plane family nominally coplanar with the wafer surface."""
        return self._cut

    @property
    def miscut(self) -> float:
        """Miscut angle (deg) between the wafer surface and the ``cut`` of the optic."""
        return self._miscut

    @property
    def crystal_asymmetry(self) -> float:
        """Crystallographic angle between the ``reflection`` and ``cut`` planes in degrees."""
        return self._crystal_asymmetry

    @property
    def asymmetry(self) -> float:
        """Total asymmetry angle alpha (deg), sum of ``miscut`` and ``crystal_asymmetry``."""
        return self._crystal_asymmetry + self._miscut

    @property
    def d_hkl(self) -> float:
        """Interplanar spacing of the analyzer's reflection in [angstrom]."""
        return self._d_hkl

    @property
    def e0(self) -> float:
        """Backscatter energy [eV] for the analyzer's reflection at 90 deg Bragg."""
        return self._e0

    def to_energy(self, angle_deg: float) -> float:
        """Compute the energy (eV) for a given Bragg angle (degrees)."""
        angle_deg = float(angle_deg)
        if not (0.0 < angle_deg <= 90.0) or not np.isfinite(angle_deg):
            raise ValueError("Bragg angle must be finite and between 0 and 90 deg")
        return float(self.e0 / np.sin(np.radians(angle_deg)))

    def to_bragg(self, energy: float) -> float:
        """Compute the Bragg angle (deg) for a given energy (eV)."""
        energy = float(energy)
        if energy < self.e0 or not np.isfinite(energy):
            raise ValueError("Energy must be finite and greater than e0")
        return float(np.degrees(np.asin(self.e0 / energy)))
    
    def as_dict(self) -> dict[str, object]:
        return {
            "material": self.material,
            "reflection": self.reflection.tolist(),
            "cut": self.cut.tolist(),
            "bending_radius": self.bending_radius,
            "optic_diameter": self.optic_diameter,
            "d_hkl": self.d_hkl,
            "e0": self.e0,
            "crystal_asymmetry": self.crystal_asymmetry,
            "miscut": self.miscut,
            "asymmetry": self.asymmetry,
            }


def _validate_analyzer_scalars(
        bending_radius: float,
        optic_diameter: float,
        miscut: float,
        ) -> tuple[float, float, float]:
    """
    Validate and normalize scalar inputs to the Analyzer class.
    Raises specific errors for incorrect formats.
    """
    bending_radius = float(bending_radius)
    optic_diameter = float(optic_diameter)
    miscut = float(miscut)

    if not np.isfinite(bending_radius) or bending_radius <= 0:
        raise ValueError(
            "bending_radius must be finite and positive"
        )

    if not np.isfinite(optic_diameter) or optic_diameter <= 0:
        raise ValueError(
            "optic_diameter must be finite and positive"
        )

    if not np.isfinite(miscut) or abs(miscut) >= 90:
        raise ValueError("miscut must be -90 < miscut < 90 degree and finite")

    return bending_radius, optic_diameter, miscut