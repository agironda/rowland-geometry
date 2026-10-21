import numpy as np
from .rowland_circle import RowlandCircle

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

        self._pos_source = self._calculate_displaced_source()

        self._analyzer_x_profile = np.linspace(
            -self._optic_diameter / 2.0, 
            self._optic_diameter / 2.0, 
            _PROFILE_POINTS
            )

        self._analyzer_pos_profile = self._calculate_analyzer_profile()
        self._analyzer_ang_profile = self._calculate_mechanical_angle_profile()

        self._analyzer_bragg_profile = self._analyzer_ang_profile - self._on_circle_geometry.asymmetry     

    def _calculate_displaced_source(self) -> np.ndarray:
        """Return source coordinates after displacement along rho."""

        rho = self._on_circle_geometry.chord_rho
        rho_hat = self._on_circle_geometry.pos_source / rho
        rho_star = rho + self._displacement

        return rho_star * rho_hat

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
