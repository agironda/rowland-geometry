import numpy as np
from .rowland_circle import RowlandCircle
from .lens import meridional_q, sagittal_q

_PROFILE_POINTS = 1001

class DispersiveRefocusing:

    def __init__(
        self,
        diameter: float,
        bragg: float,
        displacement: float,
        optic_diameter: float = 100.0,
        *,
        asymmetry: float = 0.0,
    ):
        self._on_circle_geometry = RowlandCircle(diameter, bragg, asymmetry=asymmetry)
        self._displacement = _validate_displacement(displacement, self._on_circle_geometry.chord_rho)
        self._optic_diameter = _validate_optic_diameter(optic_diameter, diameter)
        self._profile_analyzer_x = self._calc_profile_analyzer_x()

        self._recalculate_geometry()

    @property
    def on_circle_geometry(self) -> RowlandCircle:
        """Unperturbed on-circle Rowland Circle geometry."""
        return self._on_circle_geometry.copy()

    @property
    def diameter(self) -> float:
        """Rowland circle diameter in millimeters"""
        return self._on_circle_geometry.diameter

    @property
    def bragg(self) -> float:
        """Bragg angle in degrees."""
        return self._on_circle_geometry.bragg

    @property
    def asymmetry(self) -> float:
        """Asymmetry angle in degrees."""
        return self._on_circle_geometry.asymmetry

    @property
    def displacement(self) -> float:
        """Displacement of source along rho in millimeters"""
        return self._displacement

    @property
    def optic_diameter(self) -> float:
        """Aperture size of the analyzer in millimeters"""
        return self._optic_diameter

    @diameter.setter
    def diameter(self, value: float) -> None:
        geometry = self.on_circle_geometry
        geometry.diameter = value
        _validate_optic_diameter(self.optic_diameter, geometry.diameter)
        _validate_displacement(self.displacement, geometry.chord_rho)
        self._on_circle_geometry.diameter = value
        self._recalculate_geometry()

    @bragg.setter
    def bragg(self, value: float) -> None:
        geometry = self.on_circle_geometry
        geometry.bragg = value
        _validate_optic_diameter(self.optic_diameter, geometry.diameter)
        _validate_displacement(self.displacement, geometry.chord_rho)
        self._on_circle_geometry.bragg = value
        self._recalculate_geometry()

    @asymmetry.setter
    def asymmetry(self, value: float) -> None:
        geometry = self.on_circle_geometry
        geometry.asymmetry = value
        _validate_optic_diameter(self.optic_diameter, geometry.diameter)
        _validate_displacement(self.displacement, geometry.chord_rho)
        self._on_circle_geometry.asymmetry = value
        self._recalculate_geometry()

    @displacement.setter
    def displacement(self, value: float) -> None:
        self._displacement = _validate_displacement(value, self._on_circle_geometry.chord_rho)
        self._recalculate_geometry()

    @optic_diameter.setter
    def optic_diameter(self, value: float) -> None:
        self._optic_diameter = _validate_optic_diameter(value, self.diameter)
        self._recalculate_geometry()

    def _recalculate_geometry(self):

        chord_rho = self._calc_chord_rho()
        chord_fm = self._calc_chord_fm(chord_rho)
        chord_fs = self._calc_chord_fs(chord_rho)

        pos_source = self._calc_coord_source(chord_rho)
        pos_meridional = self._calc_coord_meridional(chord_fm)
        pos_sagittal = self._calc_coord_sagittal(chord_fs)
        pos_center = self._calc_coord_center()

        self._chord_rho = chord_rho
        self._chord_fm = chord_fm
        self._chord_fs = chord_fs

        self._pos_source = pos_source
        self._pos_meridional = pos_meridional
        self._pos_sagittal = pos_sagittal
        self._pos_center = pos_center

        return

    def _calc_chord_rho(self) -> float:
        """Compute source-analyzer chord length."""
        on_circle = self._on_circle_geometry
        rho = self.displacement + on_circle.chord_rho
        return rho

    def _calc_chord_fm(self, chord_rho) -> float:
        """Compute source-analyzer chord length."""
        t1 = self.bragg + self.asymmetry
        t2 = self.bragg - self.asymmetry
        r = self.diameter
        fm = meridional_q(chord_rho, t1, t2, r)
        return fm

    def _calc_chord_fs(self, chord_rho) -> float:
        """Compute source-analyzer chord length."""
        t1 = self.bragg + self.asymmetry
        t2 = self.bragg - self.asymmetry
        r = self.diameter
        fs = sagittal_q(chord_rho, t1, t2, r)
        return fs

    def _calc_coord_source(self, chord_rho) -> np.ndarray:
        on_circle = self._on_circle_geometry
        rho_hat = on_circle.pos_source / on_circle.chord_rho
        return chord_rho * rho_hat

    def _calc_coord_meridional(self, chord_fm) -> np.ndarray:
        on_circle = self._on_circle_geometry
        fm_hat = on_circle.pos_meridional / on_circle.chord_fm
        return chord_fm * fm_hat

    def _calc_coord_sagittal(self, chord_fs) -> np.ndarray:
        on_circle = self._on_circle_geometry
        fs_hat = on_circle.pos_sagittal / on_circle.chord_fs
        return chord_fs * fs_hat

    def _calc_coord_center(self) -> np.ndarray:
        on_circle = self._on_circle_geometry
        return on_circle.pos_center

    def _calc_profile_analyzer_x(self) -> np.ndarray:
        return np.linspace(
            -self._optic_diameter / 2.0, 
            self._optic_diameter / 2.0, 
            _PROFILE_POINTS
            )



    def _calculate_analyzer_pos(self, x: float) -> np.ndarray:
        """Return the analyzer-surface coordinates at x."""
        x_pos = x
        y_pos = _analyzer_profile(x_pos, self._on_circle_geometry.diameter)
        return np.array([x_pos, y_pos, 0.0])

    def _calculate_mechanical_angle(self, x: float) -> float:
        """Return the local mechanical angle at analyzer position x."""
        sbca_xy = self._calculate_analyzer_pos(x)
        tangent = _analyzer_tangent(x, self._on_circle_geometry.diameter)
        ray = self._pos_source - sbca_xy
        ray_hat = ray / np.linalg.norm(ray)
        dot = np.clip(np.dot(ray_hat, tangent), -1.0, 1.0)
        return float(np.degrees(np.arccos(dot)))

    def _calculate_analyzer_profile(self) -> np.ndarray:
        return np.array([
            self._calculate_analyzer_pos(x)
            for x in self._analyzer_x_profile
        ])

    def _calculate_mechanical_angle_profile(self) -> np.ndarray:
        return np.array([
            self._calculate_mechanical_angle(x)
            for x in self._analyzer_x_profile
        ])

def _analyzer_profile(x: float, bending_radius: float) -> float:
    """Return the y-coordinate of the Johann analyzer surface at x."""
    R = bending_radius

    if abs(x) > R:
        raise ValueError("x coordinate lies beyond the curvature circle.")

    return float(R - np.sqrt(R**2 - x**2))

def _analyzer_tangent(x: float, bending_radius: float) -> np.ndarray:
    """Return the unit tangent to the Johann analyzer surface at x."""
    R = bending_radius
    y = _analyzer_profile(x, R)

    return np.array([(y - R) / R, -x / R, 0.0])

def _validate_displacement(displacement: float, chord_rho: float) -> float:
    """
    Validate a source displacement along the rho direction.
    """
    displacement = float(displacement)

    if not np.isfinite(displacement):
        raise ValueError("Rho displacement must be finite.")

    if displacement == 0:
        raise ValueError("Rho displacement cannot be 0.")

    if displacement <= -chord_rho:
        raise ValueError(
            "Negative rho displacement must have a magnitude smaller "
            "than the on-circle rho chord displacement."
        )

    return displacement

def _validate_optic_diameter(
    optic_diameter: float,
    bending_radius: float,
) -> float:
    """
    Validate the physical analyzer diameter. Ignores surface bending, i.e. flat aperture.
    """

    optic_diameter = float(optic_diameter)

    if not np.isfinite(optic_diameter) or optic_diameter <= 0:
        raise ValueError("Optic diameter must be finite and greater than 0.")

    if optic_diameter > 2 * bending_radius:
        raise ValueError(
            "Optic diameter cannot exceed twice the bending radius."
        )

    return optic_diameter



