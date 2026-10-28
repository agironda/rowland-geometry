"""
This module describes the DispersiveSpectrometer class. It is an object representation
of a Rowland geometry X-ray spectrometer with an off-circle points source. The source is 
displaced along the "rho" chord. 

Thus, this class combines the Analyzer, DispersiveRefocusing, and ReferenceFrame 
objects to present the geometric relations, energy bandwidth , and the coordinates 
of the optic elemenets in a reference frame relevant to physical hardware. 
"""

from __future__ import annotations

import numpy as np

from .analyzer import Analyzer
from .dispersive_refocusing import DispersiveRefocusing
from .frames import ReferenceFrame, apply_reference_frame

_DEFAULT_RESPONSE_BINS = 100

class DispersiveSpectrometer:

    def __init__(
            self,
            analyzer: Analyzer,
            displacement: float,
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
        self._displacement = float(displacement)

        if bragg is not None:
            bragg_now = float(bragg)
            energy_now = self.analyzer.to_energy(bragg_now)
        else:
            energy_now = float(energy)
            bragg_now = self.analyzer.to_bragg(energy_now)

        self._geometry = self._build_geometry(
            self._analyzer, 
            bragg_now, 
            self._displacement, 
            self._asymmetry_direction
        )

        self._energy = float(energy_now)
        self.frame = ReferenceFrame() if frame is None else frame

    def _build_geometry(
            self, 
            analyzer: Analyzer, 
            bragg: float, 
            displacement: float,
            asymmetry_direction: int
        ) -> DispersiveRefocusing:
        """Build dispersive geometry from the proposed spectrometer state."""
        return DispersiveRefocusing(
            diameter=analyzer.bending_radius,
            bragg=bragg,
            asymmetry=analyzer.asymmetry * asymmetry_direction,
            displacement=displacement,
            optic_diameter=analyzer.optic_diameter
        )

    @property
    def geometry(self) -> DispersiveRefocusing:
        """Current dispersive refocusing geometry."""
        return self._geometry.copy()

    @property
    def diameter(self) -> float:
        """Rowland diameter in mm, defined by Analyzer bending radius."""
        return self._geometry.diameter
    
    @property
    def displacement(self) -> float:
        """Source displacement along rho in millimeters."""
        return self._displacement
     
    @property
    def analyzer(self) -> Analyzer:
        """Spherically bent optic of the spectrometer"""
        return self._analyzer

    @property
    def asymmetry(self) -> float:
        """Asymmetry angle in degrees."""
        return self._geometry.asymmetry

    @property
    def asymmetry_direction(self) -> int:
        """Orientation of analyzer asymmetry, either +1 or -1."""
        return self._asymmetry_direction
    
    @property
    def bragg(self) -> float:
        """Bragg angle in degrees."""
        return self._geometry.bragg

    @property
    def energy(self) -> float:
        """Photon energy at the analyzer center (eV)."""
        return self._energy

    @property
    def frame(self) -> ReferenceFrame:
        """Reference frame used for coordinate reporting."""
        return self._frame

    @property
    def profile_analyzer_energy(self) -> np.ndarray:
        """Analyzed energies along the finite analyzer surface."""
        return np.array([
            self.analyzer.to_energy(bragg)
            for bragg in self._geometry.profile_analyzer_bragg
        ])

    @property
    def energy_width(self) -> float:
        """Energy range analyzed by the analyzer."""
        return float(np.ptp(self.profile_analyzer_energy))

    @frame.setter
    def frame(self, value: ReferenceFrame) -> None:
        if not isinstance(value, ReferenceFrame):
            raise TypeError("frame must be a ReferenceFrame")
        self._frame = value

    @displacement.setter
    def displacement(self, value: float) -> None:
        displacement_new = float(value)

        geometry_new = self._build_geometry(
            self.analyzer,
            self.bragg,
            displacement_new,
            self.asymmetry_direction
        )

        self._displacement = displacement_new
        self._geometry = geometry_new

    @bragg.setter
    def bragg(self, value: float) -> None:
        bragg_new = float(value)

        energy_new = float(self.analyzer.to_energy(bragg_new))

        geometry_new = self._build_geometry(
            self.analyzer,
            bragg_new,
            self.displacement,
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
            self.displacement,
            self.asymmetry_direction
        )

        self._geometry = geometry_new
        self._energy = energy_new

    @analyzer.setter
    def analyzer(self, analyzer: Analyzer) -> None:
        if not isinstance(analyzer, Analyzer):
            raise TypeError("analyzer must be an Analyzer")

        bragg_now = self.bragg

        energy_new = float(analyzer.to_energy(bragg_now))

        geometry_new = self._build_geometry(
            analyzer,
            bragg_now,
            self.displacement,
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
            self.displacement,
            sign_new,
        )

        self._asymmetry_direction = sign_new
        self._geometry = geometry_new

    def energy_response(
        self,
        bin_width: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the area-weighted, peak-normalized geometric energy response."""

        if bin_width is None:
            bin_width = self.energy_width / _DEFAULT_RESPONSE_BINS
        else:
            bin_width = float(bin_width)

        if not np.isfinite(bin_width) or bin_width <= 0:
            raise ValueError("bin_width must be finite and greater than 0.")

        energy = self.profile_analyzer_energy

        x = self._geometry.profile_analyzer_x
        strip_height = self._geometry.profile_analyzer_weight

        dx = x[1] - x[0]
        area_weight = strip_height * dx

        e_min = np.min(energy)
        e_max = np.max(energy)

        edges = np.arange(
            e_min,
            e_max + bin_width,
            bin_width,
        )

        response, edges = np.histogram(
            energy,
            bins=edges,
            weights=area_weight,
        )

        energy_axis = 0.5 * (edges[:-1] + edges[1:])

        if np.max(response) > 0:
            response = response / np.max(response)

        return energy_axis, response

    def analyzer_energy_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """Return analyzer surface position and analyzed energy."""
        return (
            self._geometry.profile_analyzer_x.copy(),
            self.profile_analyzer_energy,
        )

    def coords_canonical_frame(self) -> dict[str, np.ndarray]:
        """Return dispersive geometry coordinates in the canonical reference frame."""
        r = self._geometry
        return {
            "source": r.pos_source,
            "analyzer center": r.pos_analyzer_center,
            "analyzer edge": r.pos_analyzer_edges[0],
            "analyzer edge": r.pos_analyzer_edges[1],
            "meridional": r.pos_meridional,
            "sagittal": r.pos_sagittal,
            "center": r.pos_center,
        }

    def coords_reference_frame(self) -> dict[str, np.ndarray]:
        """ Return coordinates in the configured reference frame."""
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
            "bragg": float(self.bragg),
            "asymmetry": float(self.asymmetry),
            "analyzer": self.analyzer.as_dict(),
            "energy_center": float(self.energy),
            "energy_width": self.energy_width(),
            "energy_range": (np.min(self.profile_analyzer_energy),np.max(self.profile_analyzer_energy)),
            "dispersive_geometry": {
                "diameter": float(self._geometry.diameter),
                "chord_rho": float(self._geometry.chord_rho),
                "chord_fm": float(self._geometry.chord_fm),
                "chord_fs": float(self._geometry.chord_fs),
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
    
def _validate_asymmetry_direction(value: int) -> int:
    if not isinstance(value, (int, np.integer)):
        raise TypeError("asymmetry_direction must be an integer.")

    if value not in (-1, 1):
        raise ValueError("asymmetry_direction must be either +1 or -1.")

    return value
    
