"""
Lens equations for bent crystal analyzers with asymmetric diffraction.

Meridional / tangential plane:

    sin²(theta1) / p + sin²(theta2) / q
        = [sin(theta1) + |sin(theta2)|] / Rm

Sagittal plane:

    1 / p + 1 / q
        = [sin(theta1) + |sin(theta2)|] / Rs

where

    p      source-to-crystal distance
    q      crystal-to-focus distance
    theta1 incident angle relative to the crystal surface
    theta2 exit angle relative to the crystal surface
    Rm     meridional bending radius of curvature
    Rs     sagittal bending radius of curvature

For the asymmetric Rowland geometry used in this package:

    theta1 = bragg + asymmetry
    theta2 = bragg - asymmetry
"""

import numpy as np

def meridional_p(q: float, theta1: float, theta2: float, radius: float) -> float:
    
    q, theta1, theta2, radius = _validate_inputs(q, theta1, theta2, radius)

    t1 = np.radians(theta1)
    t2 = np.radians(theta2)

    fterm = (np.sin(t1) + abs(np.sin(t2))) / radius
    qterm = np.sin(t2)**2 / q

    denominator = fterm - qterm
    
    if np.isclose(denominator, 0.0): 
        return np.inf
    
    p = np.sin(t1)**2 / denominator
    
    return float(p)

def meridional_q(p: float, theta1: float, theta2: float, radius: float) -> float:

    p, theta1, theta2, radius = _validate_inputs(p, theta1, theta2, radius)

    t1 = np.radians(theta1)
    t2 = np.radians(theta2)

    fterm = (np.sin(t1) + abs(np.sin(t2))) / radius
    pterm = np.sin(t1)**2 / p

    denominator = fterm - pterm

    if np.isclose(denominator, 0.0): 
        return np.inf
    
    q = np.sin(t2)**2 / denominator
    
    return float(q)

def sagittal_p(q: float, theta1: float, theta2: float, radius: float) -> float:

    q, theta1, theta2, radius = _validate_inputs(q, theta1, theta2, radius)

    t1 = np.radians(theta1)
    t2 = np.radians(theta2)

    fterm = (np.sin(t1) + abs(np.sin(t2))) / radius
    qterm = 1 / q

    denominator = fterm - qterm

    if np.isclose(denominator, 0.0): 
        return np.inf

    p = 1 / denominator

    return float(p)

def sagittal_q(p: float, theta1: float, theta2: float, radius: float) -> float:

    p, theta1, theta2, radius = _validate_inputs(p, theta1, theta2, radius)

    t1 = np.radians(theta1)
    t2 = np.radians(theta2)

    fterm = (np.sin(t1) + abs(np.sin(t2))) / radius
    pterm = 1 / p

    denominator = fterm - pterm

    if np.isclose(denominator, 0.0): 
        return np.inf

    q = 1 / denominator

    return float(q)

def _validate_inputs(
        distance: float,
        theta1: float,
        theta2: float,
        radius: float,
        ) -> tuple[float, float, float, float]:
    """Validate and normalize inputs for the bent-crystal lens equations."""

    distance = float(distance)
    theta1 = float(theta1)
    theta2 = float(theta2)
    radius = float(radius)

    if np.isnan(distance):
        raise ValueError("Distance must not be NaN.")

    if distance == 0.0:
        raise ValueError("Distance must be nonzero.")

    if not np.isfinite(theta1) or not np.isfinite(theta2):
        raise ValueError("theta1 and theta2 must be finite.")

    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Radius must be finite and greater than 0.")

    return distance, theta1, theta2, radius