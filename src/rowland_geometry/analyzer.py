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
from typing import Optional

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
        Defaults to `reflection`.
    miscut : float, default 0.0
        Miscut angle between surface profile and `cut` plane family in degrees.
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
        Miscut angle between surface profile and `cut` plane family in degrees.
    alpha_xtal : float
        Angle in degrees between the reflection and cut plane families.
    alpha : float
        Total asymmetry angle in degrees, the sum of alpha_xtal and miscut.
    E0 : float
        Analyzed photon energy at backscattering (Bragg angle = 90) of the 
        specified reflection in eV.

    Methods
    -------
    energy()
        Computes the analyzed photon energy in eV for a given Bragg angle in 
        degrees for the specified plane family `reflection`.

    bragg()
        Computes the Bragg angle in degrees for a given analyzed photon energy
        in eV for the specified plane family `reflection`.

    as_dict()
        Returns the analyzer parameters and derived quantities as a dictionary.

    Notes
    -----
    All optic dimensions are in [mm], interplanar spacings are in [angstrom], 
    photon energies are in [eV] and all angles are in [deg].
    
    The public properties of the class are read only. A new Analyzer object 
    must be instantiated to change the parameters such qs the reflection or 
    surface cut.

    The miscut and alpha_xtal quantities describe different contributions to the
    total asymmetry angle. The miscut is the signed angle between the analyzer's
    surface and the nominal surface-cut plane family. For an SBCA, this angle is 
    nominally zero and the cut plane family is parallel to the analyzer surface.

    The alpha_xtal quantity is the crystallographic angle between the diffracting
    plane family and the cut plane family. It is independent of wafer miscut.
    The total asymmetry angle is therefore alpha = alpha_xtal + miscut.
    """
    
    def __init__(
        self,
        material: str,
        reflection: Sequence[int],
        bending_radius: float,
        *,
        cut: Optional[Sequence[int]] = None,
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
        
        self._alpha_xtal = plane_angle(
            self._material, 
            self._reflection, 
            self._cut,
        )

        self._E0 = float(HC / (2.0 * self._d_hkl))  # eV

    @property
    def bending_radius(self) -> float:
        return self._bending_radius

    @property
    def optic_diameter(self) -> float:
        return self._optic_diameter

    @property
    def material(self) -> str:
        return self._material

    @property
    def reflection(self) -> np.ndarray:
        return self._reflection

    @property
    def cut(self) -> np.ndarray:
        return self._cut

    @property
    def miscut(self) -> float:
        """Miscut angle (deg) between the nominal surface and G0."""
        return self._miscut

    @property
    def alpha_xtal(self) -> float:
        """Angle (deg) from planes only (no miscut)."""
        return self._alpha_xtal

    @property
    def alpha(self) -> float:
        """Total asymmetry angle α (deg) including miscut."""
        return self._alpha_xtal + self._miscut

    @property
    def alpha_rad(self) -> float:
        return np.radians(self.alpha)

    @property
    def d_hkl(self) -> float:
        return self._d_hkl

    @property
    def E0(self) -> float:
        """Backscatter energy (eV) for this reflection (θ_B = 90°)."""
        return self._E0

    def energy(self, angle_deg: float) -> float:
        """Compute the energy (eV) for a given Bragg angle (degrees)."""
        if not (0.0 < angle_deg <= 90.0):
            raise ValueError("Bragg angle must be between 0 and 90 deg")
        return self.E0 / np.sin(np.radians(angle_deg))

    def bragg(self, energy: float) -> float:
        """Compute the Bragg angle (deg) for a given energy (eV)."""
        if energy < self.E0:
            raise ValueError("Energy below E0 for the given reflection")
        return np.degrees(np.asin(self.E0 / energy))
    
    def as_dict(self) -> dict:
        return {
            "material": self.material,
            "reflection": self.reflection.tolist(),
            "cut": self.cut.tolist(),
            "bending_radius": self.bending_radius,
            "optic_diameter": self.optic_diameter,
            "d_hkl": self.d_hkl,
            "E0": self.E0,
            "crystal_alpha": self.alpha_xtal,
            "miscut": self.miscut,
            "alpha": self.alpha,
            }


def _validate_analyzer_scalars(
        bending_radius: float,
        optic_diameter: float,
        miscut: float,
        ) -> tuple[float, float, float]:
    """
    Validate and normalize scalar inputs to the Analyzer class
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