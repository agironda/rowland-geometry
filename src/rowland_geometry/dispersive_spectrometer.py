from __future__ import annotations

import numpy as np

from .analyzer import Analyzer
from .dispersive_refocusing import DispersiveRefocusing
from .frames import ReferenceFrame, apply_reference_frame

class DispersiveSpectrometer:

    def __init__(
            self,
            analyzer: Analyzer,
            displacement: float,
            *,
            bragg: float | None = None,
            energy: float | None = None,
            frame: ReferenceFrame | None = None,
    ):
        if not isinstance(analyzer, Analyzer):
            raise TypeError("analyzer must be an Analyzer")

        if (bragg is None) == (energy is None):
            raise ValueError("Provide exactly one of bragg or energy")

        self._analyzer = analyzer
        self._displacement = float(displacement)

        if bragg is not None:
            bragg_now = float(bragg)
            energy_now = self.analyzer.to_energy(bragg_now)
        else:
            energy_now = float(energy)
            bragg_now = self.analyzer.to_bragg(energy_now)

        self._geometry = self._build_geometry(bragg_now)
        self._energy = float(energy_now)
        self.frame = ReferenceFrame() if frame is None else frame

    @property
    def analyzer(self) -> Analyzer:
        """Spherically bent optic of the spectrometer"""
        return self._analyzer

    @property
    def displacement(self) -> float:
        """Source displacement along rho in millimeters."""
        return self._displacement

    @property
    def geometry(self) -> DispersiveRefocusing:
        """Current dispersive refocusing geometry."""
        return self._geometry.copy()
    
    def _build_geometry(self, bragg: float) -> DispersiveRefocusing:
        """Build Rowland geometry from the current analyzer and Bragg angle."""
        return DispersiveRefocusing(
            diameter=self.analyzer.bending_radius,
            bragg=bragg,
            asymmetry=self.analyzer.asymmetry,
            displacement = self.displacement,
            optic_diameter= self.analyzer.optic_diameter
        )