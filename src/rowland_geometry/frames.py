"""
Reference frame module for Rowland circle spectrometer geometry.

This module defines the ``ReferenceFrame`` class and tools for transforming
coordinates from the canonical Rowland-circle reference frame into a 
laboratory reference frame matching a physical spectrometer.

A reference frame is defined by an origin, an element used to establish the
positive x- or y-axis direction, an optional in-plane rotation, and optional
axis reflections.

Positive ``rotation`` values rotate the transformed coordinates clockwise
about the selected origin. Negative values rotate counterclockwise. The 
``rotation`` is applied before any axis reflections.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np


Coords = dict[str, np.ndarray]


@dataclass(frozen=True)
class ReferenceFrame:
    """
    Definition of a coordinate reference frame.

    Parameters
    ----------
    origin : str, default "analyzer"
        Spectrometer element placed at the origin.
    axis_element : str, default "center"
        Spectrometer element used to define the reference-axis direction.
    axis : {"x", "y"}, default "y"
        Axis with which the vector from ``origin`` to ``axis_element`` is
        initially aligned.
    rotation : float, default 0.0
        Additional in-plane rotation in degrees about ``origin``. Positive
        values rotate clockwise and negative values rotate counterclockwise.
    flip_x : bool, default False
        If True, reflect coordinates across the y-axis.
    flip_y : bool, default False
        If True, reflect coordinates across the x-axis.

    Notes
    -----
    ``ReferenceFrame`` objects are immutable after instantiation.

    The ``origin`` and ``axis_element`` names are validated against the supplied
    coordinate mapping when the frame is applied.

    Positive ``rotation`` values rotate the transformed coordinates clockwise
    about the selected origin. Negative values rotate counterclockwise. The 
    ``rotation`` is applied before any axis reflections.

    Examples
    --------
    Define a frame centered on the source with the source-to-analyzer direction
    aligned with the positive x-axis:

    >>> frame = ReferenceFrame(
    ...     origin="source",
    ...     axis_element="analyzer",
    ...     axis="x",
    ... )

    Add a 20 degree clockwise rotation:

    >>> frame = ReferenceFrame(
    ...     origin="source",
    ...     axis_element="analyzer",
    ...     axis="x",
    ...     rotation=20,
    ... )
    """
    origin: str = "analyzer"
    axis_element: str = "center"
    axis: str = "y"
    rotation: float = 0.0
    flip_x: bool = False
    flip_y: bool = False

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")

        if self.origin == self.axis_element:
            raise ValueError("origin and axis_element must be different")

        rotation = float(self.rotation)
        if not np.isfinite(rotation):
            raise ValueError("rotation must be finite")

        object.__setattr__(self, "rotation", rotation)


def _copy_coords(coords: Mapping[str, np.ndarray]) -> Coords:
    return {k: np.array(v, dtype=float, copy=True) for k, v in coords.items()}


def apply_reference_frame(
        coords_in: Mapping[str, np.ndarray], 
        frame: ReferenceFrame
        ) -> Coords:
    """
    Transform canonical coordinates into a specified reference frame.

    Parameters
    ----------
    coords_in : Mapping[str, numpy.ndarray]
        Mapping of element names to three-dimensional coordinates.
    frame : ReferenceFrame
        Reference-frame definition to apply.

    Returns
    -------
    dict
        Transformed coordinates keyed by element name.

    Notes
    -----
    The transformation is applied in the following order:

    1. Translate coordinates so that ``frame.origin`` is at the origin.
    2. Rotate so that the vector from ``origin`` to ``axis_element`` aligns
       with the requested positive x- or y-axis.
    3. Apply the additional clockwise rotation specified by
       ``frame.rotation``.
    4. Apply optional x- and y-axis reflections.
    """
    coords = _copy_coords(coords_in)

    if frame.origin not in coords:
        raise ValueError(f"Unknown origin '{frame.origin}'")
    if frame.axis_element not in coords:
        raise ValueError(f"Unknown axis_element '{frame.axis_element}'")

    # 1) translate
    origin_vec = coords[frame.origin].copy()
    for k in coords:
        coords[k] = coords[k] - origin_vec

    # 2 & 3) rotate in XY plane
    v = coords[frame.axis_element].copy()
    v[2] = 0.0
    norm = float(np.linalg.norm(v[:2]))
    if norm == 0:
        raise ValueError("origin and axis_element coincide in canonical frame")

    theta_current = float(np.arctan2(v[1], v[0]))
    theta_target = 0.0 if frame.axis == "x" else float(np.pi / 2.0)
    dtheta = theta_target - theta_current
    dtheta -= np.radians(frame.rotation)

    c = float(np.cos(dtheta))
    s = float(np.sin(dtheta))
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)

    for k in coords:
        coords[k] = R @ coords[k]

    # 4) flips
    sx = -1.0 if frame.flip_x else 1.0
    sy = -1.0 if frame.flip_y else 1.0
    M = np.array([[sx, 0.0, 0.0],
                  [0.0, sy, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)

    for k in coords:
        coords[k] = M @ coords[k]

    return coords
