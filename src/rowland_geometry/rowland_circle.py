"""
This module describes the RowlandCircle class, which holds all of the geometric
relations and coordinates of a source, analyzer, and foci in so-called Rowland
circle focusing geometry.

The geometric relations of the Rowland circle elements are driven by the 
diameter of the Rowland circle (equal to the bending radius of the spherical
optic), the Bragg angle, and alpha, the asymmetry angle.

The Rowland circle is central in the design of energy-scanning X-ray 
spectrometers. Thus the implementation of the class here is mutable - the same
Rowland circle object can have the diameter, Bragg angle, or asymmetry angle
changed which triggers a recalculation of the chords and coordinates relating
the source, analyzer, and foci.

The Rowland geometry here is only concerned with the relative relation of the 
circle's elements and is thus in an absolute, canonical reference frame. The 
use of energy to drive the Rowland circle in a specific reference frame are 
instead described in spectrometer.py module and the Spectrometer class.
"""
import numpy as np

class RowlandCircle:
    """
    Object representation of Rowwland circle focusing geometry and arrangement
    of its elements about the circle perimeter: source, optic, and foci.

    Parameters
    ----------
    diameter : str
        Name, formula, or alias of a supported crystal material.
    bragg : Sequence[int]
        Miller indices of the diffracting plane.
    alpha : float
        Radius of curvature of the analyzer surface in millimetres.

    Attributes
    ----------

    Methods
    -------

    Notes
    -----

    Examples
    --------
    """

    def __init__(
        self,
        diameter: float,
        bragg: float,
        *,
        alpha: float = 0.0,
    ):
                    
        self._diameter = _validate_length(diameter)
        self._bragg = _validate_bragg(bragg)
        self._alpha = _validate_alpha(alpha)
        
        self.calculate_geometry()
        
    def calculate_geometry(self) -> None:
        """Compute chord lengths and coordinates from current parameters."""
        self._calculate_chords()
        self._calculate_coords()

    def as_dict(self, *, optic_diameter: float | None = None) -> dict:
        """Return a serializable summary of the geometry."""
        data= {
            "diameter": self.diameter,
            "bragg": self.bragg,
            "alpha": self.alpha,
            "chord_rho": self.chord_rho,
            "chord_fm": self.chord_fm,
            "chord_fs": self.chord_fs,
            "pos_source": self.pos_source.tolist(),
            "pos_analyzer": self.pos_analyzer.tolist(),
            "pos_meridional": self.pos_meridional.tolist(),
            "pos_sagittal": self.pos_sagittal.tolist(),
            "pos_center": self.pos_center.tolist(),
        }
        if optic_diameter is not None:
            data["focus"] = self.focus_extents(optic_diameter)
        return data

    # --- core parameters as properties ---

    @property
    def diameter(self) -> float:
        return self._diameter

    @diameter.setter
    def diameter(self, value: float) -> None:
        self._diameter = _validate_length(value)
        self.calculate_geometry()

    @property
    def radius(self) -> float:
        """Rowland radius = diameter / 2."""
        return self._diameter / 2.0

    @property
    def bragg(self) -> float:
        return self._bragg

    @bragg.setter
    def bragg(self, value: float) -> None:
        self._bragg = _validate_bragg(value)
        self.calculate_geometry()

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = _validate_alpha(value)
        self.calculate_geometry()

    def extent_meridional(self, optic_diameter: float) -> float:
        d = _validate_length(optic_diameter)

        fs = float(self.chord_fs)
        if not np.isfinite(fs) or fs == 0.0:
            raise ValueError("Cannot compute extent: chord_fs is zero or non-finite")

        # manuscript form: 2(fs - fm) * tan(d / 2fs)
        extent = 2.0 * (self.chord_fs - self.chord_fm) * np.tan(d / (2*fs))
        return float(abs(extent))

    def extent_sagittal(self, optic_diameter: float) -> float:
        d = _validate_length(optic_diameter)

        rho = float(self.chord_rho)
        if not np.isfinite(rho) or rho == 0.0:
            raise ValueError("Cannot compute extent: chord_rho is zero or non-finite")

        gamma = 2.0 * np.arctan(d / (2.0 * rho))
        extent = 2.0 * (self.chord_fs - self.chord_fm) * np.tan(gamma / 2.0)
        return float(abs(extent))

    def focus_extents(self, optic_diameter: float) -> dict:
        return {
            "optic_diameter": float(optic_diameter),
            "extent_meridional": self.extent_meridional(optic_diameter),
            "extent_sagittal": self.extent_sagittal(optic_diameter),
        }
    
    # -------- geometry calculations --------
    
    def _calculate_chords(self) -> None:
        """Compute and cache geometric distances."""
        self.chord_rho = self._calc_chord_rho()
        self.chord_fm = self._calc_chord_fm()
        self.chord_fs = self._calc_chord_fs()

    def _calculate_coords(self) -> None:
        """Compute and cache coordinates for source, analyzer, and astigmatic foci."""
        self.pos_source = self._calc_coord_source()
        self.pos_analyzer = self._calc_coord_analyzer()
        self.pos_meridional = self._calc_coord_meridional()
        self.pos_sagittal = self._calc_coord_sagittal()
        self.pos_center = self._calc_coord_center()
        
    def _calc_chord_rho(self) -> float:
        """Compute source-analyzer chord length."""
        # ρ = D * sin(θB + α)
        angle = np.radians(self.bragg + self.alpha)
        return self.diameter * np.sin(angle)
        
    def _calc_chord_fm(self) -> float:
        """Compute analyzer–meridional-focus (on-circle) chord length."""
        # fm = D * sin(θB − α)
        angle = np.radians(self.bragg - self.alpha)
        return self.diameter * np.sin(angle)

    def _calc_chord_fs(self) -> float:
        """Compute analyzer–sagittal-focus (off-circle) chord length."""
        # fs = - D * sin^2(θB + α) / [sin(θB − α) * cos(2(θB + α))]
        angle_pos = np.radians(self.bragg + self.alpha)
        angle_neg = np.radians(self.bragg - self.alpha)
        numerator = np.sin(angle_pos) ** 2
        s = np.sin(angle_neg)
        c = np.cos(2 * angle_pos)
        
        # tolerance in trig-space; tune if you want earlier/later flagging
        eps = 1e-6

        if abs(s) < eps:
            delta_deg = float(self.bragg - self.alpha)
            raise ValueError(
                "Singular Rowland geometry computing chord_fs: sin(bragg - alpha) ≈ 0.\n"
                f"  bragg={self.bragg:.6g} deg, alpha={self.alpha:.6g} deg "
                f"(bragg-alpha={delta_deg:.6g} deg)\n"
                "This corresponds to bragg ≈ alpha, where fm = D*sin(bragg-alpha) → 0 and "
                "the fs expression divides by ~0.\n"
                "Move away by changing bragg or alpha so |bragg-alpha| is not near 0."
            )

        # 2*(bragg+alpha) ≈ 90°  <=>  bragg+alpha ≈ 45°
        if abs(c) < eps:
            sum_deg = float(self.bragg + self.alpha)
            raise ValueError(
                "Singular Rowland geometry computing chord_fs: cos(2*(bragg + alpha)) ≈ 0.\n"
                f"  bragg={self.bragg:.6g} deg, alpha={self.alpha:.6g} deg "
                f"(bragg+alpha={sum_deg:.6g} deg)\n"
                "This corresponds to bragg + alpha ≈ 45°, where the analytic sagittal-focus "
                "distance diverges (sagittal focus → ∞).\n"
                "Move away by changing bragg or alpha so (bragg+alpha) is not near 45°."
            )
            
        return -self.diameter * numerator / (s * c)
    
    def _calc_coord_analyzer(self) -> np.ndarray:
        """Compute analyzer coordinates (canonical frame)."""
        return np.array([0.0, 0.0, 0.0])
    
    def _calc_coord_source(self) -> np.ndarray:
        """Compute source coordinates (canonical frame)."""
        angle = np.radians(self.bragg + self.alpha)
        x = self.chord_rho * np.cos(angle)
        y = self.chord_rho * np.sin(angle)
        return np.array([-x, y, 0.0])

    def _calc_coord_meridional(self) -> np.ndarray:
        """Compute meridional focus coordinates (canonical frame)."""
        angle = np.radians(self.bragg - self.alpha)
        x = self.chord_fm * np.cos(angle)
        y = self.chord_fm * np.sin(angle)
        return np.array([x, y, 0.0])

    def _calc_coord_sagittal(self) -> np.ndarray:
        """Compute sagittal focus coordinates (canonical frame)."""
        angle = np.radians(self.bragg - self.alpha)
        x = self.chord_fs * np.cos(angle)
        y = self.chord_fs * np.sin(angle)
        return np.array([x, y, 0.0])
    
    def _calc_coord_center(self) -> np.ndarray:
        """Compute Rowland circle center coordinates (canonical frame)."""
        return np.array([0.0, self.radius, 0.0])

    def copy(self) -> "RowlandCircle":
        """Return a copy of this RowlandCircle WITHOUT recomputing geometry."""
        new = RowlandCircle.__new__(RowlandCircle)

        # core params
        new._diameter = float(self._diameter)
        new._bragg = float(self._bragg)
        new._alpha = float(self._alpha)

        # cached chords
        new.chord_rho = float(self.chord_rho)
        new.chord_fm = float(self.chord_fm)
        new.chord_fs = float(self.chord_fs)

        # cached coords (copy arrays)
        new.pos_source = np.array(self.pos_source, dtype=float, copy=True)
        new.pos_analyzer = np.array(self.pos_analyzer, dtype=float, copy=True)
        new.pos_meridional = np.array(self.pos_meridional, dtype=float, copy=True)
        new.pos_sagittal = np.array(self.pos_sagittal, dtype=float, copy=True)
        new.pos_center = np.array(self.pos_center, dtype=float, copy=True)

        return new

    @classmethod
    def from_analyzer(
        cls,
        analyzer: "Analyzer",
        *,
        bragg: float | None = None,
        energy: float | None = None,
    ) -> "RowlandCircle":
        """
        Construct a RowlandCircle from an Analyzer and either a Bragg angle or energy.
        """
        if analyzer.bending_radius is None:
            raise ValueError("Analyzer has no bending_radius defined.")
            
        # exactly one of bragg / energy
        if (bragg is None) == (energy is None):
            raise ValueError("Specify exactly one of 'bragg' or 'energy'")

        # compute bragg if only energy given
        if energy is not None:
            bragg_val = analyzer.bragg(energy)
        else:
            bragg_val = float(bragg)

        return cls(
            diameter=analyzer.bending_radius,
            bragg=bragg_val,
            alpha=analyzer.alpha,
        )

def _validate_length(length: float) -> float:
    """
    Validate and normalize scalar inputs to the Rowland Circle class.
    Raises specific errors for incorrect formats.
    """
    length = float(length)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Rowland circle lengths must be finite and > 0.")
    return length

def _validate_bragg(bragg: float) -> float:
    """
    Validate and normalize scalar inputs to the Rowland Circle class.
    Raises specific errors for incorrect formats. Must be in degrees.
    """
    bragg = float(bragg)
    if not np.isfinite(bragg) or not 0 < bragg < 90:
        raise ValueError("Bragg angle must be finite and satisfy 0 < Bragg < 90.")
    return bragg

def _validate_alpha(alpha: float) -> float:
    """
    Validate and normalize scalar inputs to the Rowland Circle class.
    Raises specific errors for incorrect formats. Must be in degrees.
    """
    alpha = float(alpha)
    if not np.isfinite(alpha) or abs(alpha) >= 90:
        raise ValueError("alpha angle must be finite and satisfy -90 < alpha < 90.")
    return alpha
