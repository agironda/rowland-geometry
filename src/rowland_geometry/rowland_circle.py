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
circle's elements and is thus in a defined canonical reference frame. The 
use of energy to drive the Rowland circle in a specific reference frame are 
instead handled in the spectrometer.py module and the Spectrometer class.
"""
import numpy as np


class RowlandCircle:
    """
    Object representation of Rowland circle focusing geometry and arrangement
    of its elements about the circle perimeter: source, optic, and foci.

    Parameters
    ----------
    diameter : float
        Diameter of the Rowland circle in millimeters. Equivalent to the 
        bending radius of the spherical analyzer.
    bragg : float
        Bragg angle in degrees. 
    alpha : float, default 0.0
        Asymmetry angle in degrees. Optional input, defaults to 0.

    Attributes
    ----------
    diameter : float
        Rowland-circle diameter in millimeters.
    bragg : float
        Bragg angle in degrees.
    alpha : float
        Asymmetry angle in degrees.
    chord_rho : float
        Source-to-analyzer distance.
    chord_fm : float
        Analyzer-to-meridional-focus distance.
    chord_fs : float
        Analyzer-to-sagittal-focus distance.
    pos_source : numpy.ndarray
        Source coordinates in the canonical reference frame.
    pos_analyzer : numpy.ndarray
        Analyzer coordinates in the canonical reference frame.
    pos_meridional : numpy.ndarray
        Meridional focus coordinates in the canonical reference frame.
    pos_sagittal : numpy.ndarray
        Sagittal focus coordinates in the canonical reference frame.
    pos_center : numpy.ndarray
        Rowland circle center coordinates in the canonical reference frame.

    Methods
    -------
    foci_extents(optic_diameter)
        Calculates the focal sizes of the meridional and sagittal foci. 
    
    as_dict()
        Returns a summary of the Rowland circle geometry.
    
    copy()
        Makes a new copy of the Rowland circle.

    Notes
    -----
    The Rowland-circle diameter is equal to the bending radius of the spherical
    analyzer.

    Changing ``diameter``, ``bragg``, or ``alpha`` automatically recalculates
    all chord lengths and coordinates. The class is otherwise immutable, i.e.
    geometry changes cannot be driven by changing the chord_rho for example.
    Coordinate properties return copies so that the stored geometry cannot be
    modified externally.

    The coordinates of the Rowland circle elements are defined in a canonical
    reference frame. This is defined with the analyzer at the origin ``(0, 0)`` 
    and the Rowland circle center located at ``(0, R)``, where R is the radius of 
    the Rowland circle. The source is always located to the left of the foci. 
    For coordinates in a different reference frame, see the Spectrometer class 
    and module.

    Examples
    --------

    Create a Rowland circle and inspect a derived chord length:

    >>> circle = RowlandCircle(diameter=500, bragg=70, alpha=10)
    >>> circle.chord_fm
    433.013...

    Changing the Bragg angle recalculates the geometry:

    >>> circle.bragg = 85
    >>> circle.chord_fm
    482.962...

    Calculate the focal extents for a 100 mm optic:

    >>> circle.foci_extents(optic_diameter=100)
    {'optic_diameter': 100.0,
     'extent_meridional': 7.435...,
     'extent_sagittal': 7.763...}

    Return the geometry as a dictionary:

    >>> circle.as_dict()
    {'diameter': 500.0,
     'bragg': 85.0,
     'alpha': 10.0,
     'chord_rho': 498.097...,
     'chord_fm': 482.962...,
     'chord_fs': 521.630...,
     'pos_source': [43.412..., 496.201..., 0.0],
     'pos_analyzer': [0.0, 0.0, 0.0],
     'pos_meridional': [125.0, 466.506..., 0.0],
     'pos_sagittal': [135.007..., 503.856..., 0.0],
     'pos_center': [0.0, 250.0, 0.0]}
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
        
        self._recalculate_geometry()

    def foci_extents(self, optic_diameter: float) -> dict:
        """
        Calculate the meridional and sagittal focal extents.

        Parameters
        ----------
        optic_diameter : float
            Diameter of the analyzer in millimeters.

        Returns
        -------
        dict
            Optic diameter and the meridional and sagittal focal extents.
        """
        od = _validate_length(optic_diameter)
        return {
            "optic_diameter": od,
            "extent_meridional": self._calc_extent_meridional(od),
            "extent_sagittal": self._calc_extent_sagittal(od),
        }
    
    def as_dict(self) -> dict:
        """
        Return a serializable summary of the current geometry.

        Returns
        -------
        dict
            Circle parameters, chord lengths, and canonical coordinates.
        """
        data = {
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
        return data

    def copy(self) -> "RowlandCircle":
        """
        Return a copy of the RowlandCircle object.

        Returns
        -------
        RowlandCircle
        """
        return type(self)(
            diameter=self.diameter,
            bragg=self.bragg,
            alpha=self.alpha,
        )

    @property
    def diameter(self) -> float:
        """Rowland circle diameter in millimeters"""
        return self._diameter

    @property
    def bragg(self) -> float:
        """Bragg angle in degrees."""
        return self._bragg

    @property
    def alpha(self) -> float:
        """Asymmetry angle in degrees."""
        return self._alpha

    @diameter.setter
    def diameter(self, value: float) -> None:
        self._diameter = _validate_length(value)
        self._recalculate_geometry()

    @bragg.setter
    def bragg(self, value: float) -> None:
        self._bragg = _validate_bragg(value)
        self._recalculate_geometry()

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = _validate_alpha(value)
        self._recalculate_geometry()

    @property
    def chord_rho(self) -> float:
        """Length from source to analyzer."""
        return self._chord_rho

    @property
    def chord_fm(self) -> float:
        """Length from analyzer to meridional focus (on-circle)."""
        return self._chord_fm

    @property
    def chord_fs(self) -> float:
        """Length from analyzer to sagittal focus (off-circle)."""
        return self._chord_fs

    @property
    def pos_source(self) -> np.ndarray:
        """Coordinates of the source in the canonical reference frame."""
        return self._pos_source.copy()

    @property
    def pos_analyzer(self) -> np.ndarray:
        """Coordinates of the analyzer in the canonical reference frame."""
        return self._pos_analyzer.copy()

    @property
    def pos_meridional(self) -> np.ndarray:
        """Coordinates of the meridional focus in the canonical reference frame."""
        return self._pos_meridional.copy()

    @property
    def pos_sagittal(self) -> np.ndarray:
        """Coordinates of the sagittal focus in the canonical reference frame."""
        return self._pos_sagittal.copy()

    @property
    def pos_center(self) -> np.ndarray:
        """Coordinates of the Rowland circle center in the canonical reference frame."""
        return self._pos_center.copy()

    def _recalculate_geometry(self) -> None:
        """Compute chord lengths and coordinates from current parameters."""
        chord_rho = self._calc_chord_rho()
        chord_fm = self._calc_chord_fm()
        chord_fs = self._calc_chord_fs()

        pos_source = self._calc_coord_source(chord_rho)
        pos_analyzer = self._calc_coord_analyzer()
        pos_meridional = self._calc_coord_meridional(chord_fm)
        pos_sagittal = self._calc_coord_sagittal(chord_fs)
        pos_center = self._calc_coord_center()

        self._chord_rho = chord_rho
        self._chord_fm = chord_fm
        self._chord_fs = chord_fs

        self._pos_source = pos_source
        self._pos_analyzer = pos_analyzer
        self._pos_meridional = pos_meridional
        self._pos_sagittal = pos_sagittal
        self._pos_center = pos_center
        
    def _calc_chord_rho(self) -> float:
        """Compute source-analyzer chord length."""
        # rho = D * sin(bragg + alpha)
        angle = np.radians(self.bragg + self.alpha)
        return float(self.diameter * np.sin(angle))
        
    def _calc_chord_fm(self) -> float:
        """Compute analyzer–meridional-focus (on-circle) chord length."""
        # fm = D * sin(bragg − alpha)
        angle = np.radians(self.bragg - self.alpha)
        return float(self.diameter * np.sin(angle))

    def _calc_chord_fs(self) -> float:
        """Compute analyzer–sagittal-focus (off-circle) chord length."""
        if np.isclose(self.bragg, self.alpha):
            raise ValueError(
                "Sagittal focus is undefined when bragg equals alpha."
            )

        if np.isclose(self.bragg + self.alpha, 45.0):
            raise ValueError(
                "Sagittal focus diverges when bragg + alpha equals 45 degrees."
            )

        # fs = - D * sin^2(bragg + alpha) / [sin(bragg − alpha) * cos(2(bragg + alpha))]
        angle_pos = np.radians(self.bragg + self.alpha)
        angle_neg = np.radians(self.bragg - self.alpha)
        numerator = np.sin(angle_pos) ** 2
        s = np.sin(angle_neg)
        c = np.cos(2 * angle_pos)
        return float(-self.diameter * numerator / (s * c))
    
    def _calc_coord_source(self, chord_rho: float) -> np.ndarray:
        """Compute source coordinates (canonical frame)."""
        angle = np.radians(self.bragg + self.alpha)
        x = chord_rho * np.cos(angle)
        y = chord_rho * np.sin(angle)
        return np.array([-x, y, 0.0])

    def _calc_coord_analyzer(self) -> np.ndarray:
        """Compute analyzer coordinates (canonical frame)."""
        return np.array([0.0, 0.0, 0.0])
    
    def _calc_coord_meridional(self, chord_fm: float) -> np.ndarray:
        """Compute meridional focus coordinates (canonical frame)."""
        angle = np.radians(self.bragg - self.alpha)
        x = chord_fm * np.cos(angle)
        y = chord_fm * np.sin(angle)
        return np.array([x, y, 0.0])

    def _calc_coord_sagittal(self, chord_fs: float) -> np.ndarray:
        """Compute sagittal focus coordinates (canonical frame)."""
        angle = np.radians(self.bragg - self.alpha)
        x = chord_fs * np.cos(angle)
        y = chord_fs * np.sin(angle)
        return np.array([x, y, 0.0])
    
    def _calc_coord_center(self) -> np.ndarray:
        """Compute Rowland circle center coordinates (canonical frame)."""
        return np.array([0.0, self.diameter/2.0, 0.0])

    def _calc_extent_meridional(self, optic_diameter: float) -> float:
        """Return the meridional focal size, perpendicular to the Rowland plane"""
        # manuscript form: 2(fs - fm) * tan(d / 2fs)
        tangent = np.tan(optic_diameter / (2*self.chord_fs))
        extent = (2.0 * (self.chord_fs - self.chord_fm) * tangent)
        return float(abs(extent))

    def _calc_extent_sagittal(self, optic_diameter: float) -> float:
        """Return the sagittal focal size, parallel to the Rowland plane"""
        gamma = 2.0 * np.arctan(optic_diameter / (2.0 * self.chord_rho))
        extent = 2.0 * (self.chord_fs - self.chord_fm) * np.tan(gamma / 2.0)
        return float(abs(extent))

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