"""
This module describes the RowlandSpectrometer class, which is a representation
of a Rowland geometry X-ray spectrometer with spherically bent focussing optics.

Thus, this class combines the Analyzer, RowlandCircle, and ReferenceFrame 
objects into the RowlandSpectrometer to present the geometric relations, analyzed 
energies of the spectrometer, and the coordinates of the optic elemenets in a 
reference frame relevant to physical hardware. 
"""

from __future__ import annotations

import numpy as np

from .analyzer import Analyzer
from .rowland_circle import RowlandCircle
from .frames import ReferenceFrame, apply_reference_frame


class RowlandSpectrometer:
    """
    Object representation of Rowland geometry focusing spectrometer in a frame 
    of reference corresponding to a physical laboratory set-up. Thus, this
    object class tracks the coordinates of optical elements and foci as a 
    function of Bragg angle or energy in an (optionally) specified reference 
    frame. The object is intentionally mutable to allow changing the analyzed
    energy or Bragg angle and reading out the updated geometry.
    
    Parameters
    ----------
    analyzer : Analyzer
        Spherically bent crystal optic that defines the Rowland circle.
    bragg : float, optional
        Bragg angle in ``degrees``. Exactly one of ``bragg`` or ``energy`` must
        be provided.
    energy : float, optional
        Analyzed photon energy in ``eV``. Exactly one of ``bragg`` or ``energy``
        must be provided.
    asymmetry_direction : {+1, -1}, default +1
        Specifies the orientation of the installed analyzer
    frame : ReferenceFrame, optional
        Specifies the orientation and the element fixed at the origin of the 
        spectrometer. If None, a default frame is used (origin at analyzer, +y
        toward Rowland circle's center).
        
    Attributes
    ----------
    analyzer : Analyzer
        Analyzer associated with the spectrometer.
    bragg : float
        Current Bragg angle in degrees.
    energy : float
        Current analyzed photon energy in eV.
    diameter : float
        Rowland-circle diameter in millimeters.
    asymmetry : float
        Total asymmetry angle in degrees, derived from analyzer.
    asymmetry_direction : int
        Sign (+1 or -1) of the asymmetry angle of the installed analyzer.
    geometry : RowlandCircle
        Independent copy of the current Rowland-circle geometry.
    frame : ReferenceFrame
        Reference frame of spectrometer used for coordinate reporting.
    extent_meridional : float
        Meridional focal extent in millimeters.
    extent_sagittal : float
        Sagittal focal extent in millimeters.

    Notes
    -----
    Changing ``bragg`` or ``energy`` rebuilds the internal Rowland-circle
    geometry. Coordinates reported in the configured ``ReferenceFrame`` therefore
    reflect the updated geometry.

    Replacing ``analyzer`` (i.e., swapping optics or hkl-hopping) keeps the 
    current Bragg angle fixed and recalculates the analyzed photon energy and
    Rowland geometry for the new optic/reflection.

    The internal ``RowlandCircle`` is not exposed directly. The ``geometry``
    property returns an independent copy so that its derived geometry cannot
    be modified externally.

    All angles are in degrees, photon energies are in eV, and lengths
    and coordinates are in millimeters.

    Examples
    --------
    >>> a = Analyzer("Si", (8, 8, 0), bending_radius=500.0)
    >>> spec = RowlandSpectrometer(a, bragg=85.0)
    >>> spec.energy
    9923.1...
    """

    def __init__(
        self,
        analyzer: Analyzer,
        *,
        bragg: float | None = None,
        energy: float | None = None,
        asymmetry_direction: int = +1,
        frame: ReferenceFrame | None = None,
    ):
        if not isinstance(analyzer, Analyzer):
            raise TypeError("analyzer must be an Analyzer")

        if (bragg is None) == (energy is None):
            raise ValueError("Provide exactly one of bragg or energy")

        self._analyzer = analyzer
        self._asymmetry_direction = _validate_asymmetry_direction(asymmetry_direction)

        if bragg is not None:
            bragg_now = float(bragg)
            energy_now = self.analyzer.to_energy(bragg_now)
        else:
            energy_now = float(energy)
            bragg_now = self.analyzer.to_bragg(energy_now)

        self._geometry = self._build_geometry(
            self._analyzer, 
            bragg_now, 
            self.asymmetry_direction
        )
        
        self._energy = float(energy_now)
        self.frame = ReferenceFrame() if frame is None else frame

    def _build_geometry(
            self, 
            analyzer: Analyzer, 
            bragg: float, 
            asymmetry_direction: int
        ) -> RowlandCircle:
        """Build Rowland geometry from the proposed spectrometer state."""
        return RowlandCircle(
            diameter=analyzer.bending_radius,
            bragg=bragg,
            asymmetry=analyzer.asymmetry * asymmetry_direction,
        )

    @property
    def geometry(self) -> RowlandCircle:
        """Copy of the current Rowland-circle geometry."""
        return self._geometry.copy()
    
    @property
    def analyzer(self) -> Analyzer:
        """Spherically bent optic of the spectrometer"""
        return self._analyzer

    @property
    def diameter(self) -> float:
        """Rowland diameter in mm, defined by Analyzer bending radius."""
        return self._geometry.diameter

    @property
    def asymmetry(self) -> float:
        """Asymmetry angle in deg, defined by Analyzer cut, miscut, and reflection"""
        return self._geometry.asymmetry

    @property
    def frame(self) -> ReferenceFrame:
        """Reference frame used for coordinate reporting."""
        return self._frame

    @property
    def extent_meridional(self) -> float:
        """Meridional focal extent in millimeters."""
        foci = self._geometry.foci_extents(self.analyzer.optic_diameter)
        return foci["extent_meridional"]

    @property
    def extent_sagittal(self) -> float:
        """Sagittal focal extent in millimeters."""
        foci = self._geometry.foci_extents(self.analyzer.optic_diameter)
        return foci["extent_sagittal"]
    
    @property
    def bragg(self) -> float:
        """Bragg angle in degrees."""
        return self._geometry.bragg

    @property
    def energy(self) -> float:
        """Analyzed photon energy (eV)."""
        return self._energy

    @property
    def asymmetry_direction(self) -> int:
        """Signed orientation of the asymmetry angle."""
        return self._asymmetry_direction

    @analyzer.setter
    def analyzer(self, analyzer: Analyzer) -> None:
        """
        Changes the optic (or the reflection) and regenerates the Rowland
        geometry at the current ``bragg`` angle. -- this really should regenerate at the same bragg+alpha, not the same bragg
        """
        if not isinstance(analyzer, Analyzer):
            raise TypeError("analyzer must be an Analyzer")

        bragg_now = self.bragg

        energy_new = float(analyzer.to_energy(bragg_now))

        geometry_new = self._build_geometry(
            analyzer,
            bragg_now,
            self.asymmetry_direction
        )

        self._analyzer = analyzer
        self._geometry = geometry_new
        self._energy = energy_new

    @asymmetry_direction.setter
    def asymmetry_direction(self, value: int) -> None:
        sign_new = _validate_asymmetry_direction(value)

        geometry_new = self._build_geometry(
            self.analyzer,
            self.bragg,
            sign_new,
        )

        self._asymmetry_direction = sign_new
        self._geometry = geometry_new

    @frame.setter
    def frame(self, value: ReferenceFrame) -> None:
        if not isinstance(value, ReferenceFrame):
            raise TypeError("frame must be a ReferenceFrame")
        self._frame = value

    @bragg.setter
    def bragg(self, value: float) -> None:
        bragg_new = float(value)
        energy_new = float(self.analyzer.to_energy(bragg_new))

        geometry_new = self._build_geometry(
            self.analyzer,
            bragg_new,
            self.asymmetry_direction
        )

        self._geometry = geometry_new
        self._energy = energy_new

    @energy.setter
    def energy(self, value: float) -> None:
        energy_new = float(value)
        bragg_new = float(self.analyzer.to_bragg(energy_new))

        geometry_new = self._build_geometry(
            self.analyzer,
            bragg_new,
            self.asymmetry_direction
        )

        self._geometry = geometry_new
        self._energy = energy_new

    def coords_canonical_frame(self) -> dict[str, np.ndarray]:
        """Return coordinates in the canonical Rowland-circle reference frame."""
        r = self._geometry
        return {
            "source": r.pos_source,
            "analyzer": r.pos_analyzer,
            "meridional": r.pos_meridional,
            "sagittal": r.pos_sagittal,
            "center": r.pos_center,
        }

    def coords_reference_frame(self) -> dict[str, np.ndarray]:
        """
        Coordinates in the configured reference frame.

        Returns
        -------
        dict
            Mapping name -> ``np.ndarray([x, y, z])`` (copies).
        """
        return apply_reference_frame(self.coords_canonical_frame(), self.frame)


    def as_dict(self, *, include_canonical: bool = False) -> dict:
        """
        Return a summary of the spectrometer state.

        Parameters
        ----------
        include_canonical
            If True, include canonical-frame coordinates in addition to reference-frame
            coordinates.

        Returns
        -------
        dict
            JSON-serializable dictionary describing analyzer, geometry, reference frame,
            and coordinates.
        """
        
        ref_coords = self.coords_reference_frame()

        data = {
            "energy": float(self.energy),
            "bragg": float(self.bragg),
            "asymmetry": float(self.asymmetry),
            "analyzer": self.analyzer.as_dict(),
            "rowland": {
                "diameter": float(self._geometry.diameter),
                "chord_rho": float(self._geometry.chord_rho),
                "chord_fm": float(self._geometry.chord_fm),
                "chord_fs": float(self._geometry.chord_fs),
            },
            "focus": {
                "optic_diameter": float(self.analyzer.optic_diameter),
                "extent_meridional": float(self.extent_meridional),
                "extent_sagittal": float(self.extent_sagittal),
            },
            "reference_frame": {
                "origin": self.frame.origin,
                "axis_element": self.frame.axis_element,
                "axis": self.frame.axis,
                "rotation": self.frame.rotation,
                "flip_x": self.frame.flip_x,
                "flip_y": self.frame.flip_y,
            },
            "coords_reference_frame": {name: coord.tolist() for name, coord in ref_coords.items()},
        }

        if include_canonical:
            canon = self.coords_canonical_frame()
            data["coords_canonical_frame"] = {name: coord.tolist() for name, coord in canon.items()}

        return data

    def snapshot(self) -> dict:
        """
        Return a flat snapshot suitable for CSV/row-oriented export.

        Returns
        -------
        dict
            Flat dictionary with scalar fields (floats/strings), including:
            ``energy``, ``bragg``, ``asymmetry``, ``theta_m``, Rowland chord values, and
            per-element coordinates as ``{name}.x``, ``{name}.y``, ``{name}.z``.

        Notes
        -----
        Intended for scanning and export. Values are cast to plain Python float types
        (not NumPy scalars).
        """
        def _f(x) -> float:
            return float(x)

        r = self._geometry

        row = {
            "energy": _f(self.energy),
            "bragg": _f(self.bragg),
            "asymmetry": _f(self.asymmetry),
            "theta_m": _f(self.bragg + self.asymmetry),
            "rowland.diameter": _f(r.diameter),
            "rowland.chord_rho": _f(r.chord_rho),
            "rowland.chord_fm": _f(r.chord_fm),
            "rowland.chord_fs": _f(r.chord_fs),
            "focus.extent_meridional": _f(self.extent_meridional),
            "focus.extent_sagittal": _f(self.extent_sagittal),
        }

        coords = self.coords_reference_frame()
        for name in ("source", "analyzer", "meridional", "sagittal", "center"):
            v = coords[name]
            row[f"{name}.x"] = _f(v[0])
            row[f"{name}.y"] = _f(v[1])
            row[f"{name}.z"] = _f(v[2])

        return row

def _validate_asymmetry_direction(value: int) -> int:
    if not isinstance(value, (int, np.integer)):
        raise TypeError("asymmetry_direction must be an integer.")

    if value not in (-1, 1):
        raise ValueError("asymmetry_direction must be either +1 or -1.")

    return value
    