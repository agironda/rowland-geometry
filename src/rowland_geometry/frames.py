"""
Reference frame module for Rowland circle spectrometer geometry.

This module defines the ``ReferenceFrame`` class and tools for transforming
coordinates from the canonical Rowland-circle reference frame into a 
laboratory reference frame matching a physical spectrometer.

A reference frame is defined by an origin, an element used to establish the
positive x- or y-axis direction, optional in-plane and out-of-plane rotations, 
and optional axis reflections.

Out-of-plane rotations are limited to axes from the origin to one of the 
spectrometer elements. This allows individual Rowland circles to be arranged
about a common axis connecting two spectrometer elements, typically connecting
the source to the meridional or sagittal foci, which results in a common refocus
point for multiple Rowland circles (i.e. a convenient place to put a detector).

Positive ``in_plane_rotation`` values rotate the transformed coordinates clockwise
about the selected origin. Negative values rotate counterclockwise. The 
``in_plane_rotation`` is applied before any axis reflections.
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
    in_plane_rotation : float, default 0.0
        Additional in-plane rotation in degrees about ``origin``. Positive
        values rotate clockwise and negative values rotate counterclockwise.
    out_of_plane_rotation : float, default 0.0
        Rotation in degrees about the axis connecting ``origin`` to the
        element specified by ``out_of_plane_axis``. Positive rotations
        follow the right-hand rule.
    out_of_plane_axis: str, default "meridional"
        Spectrometer element defining the out-of-plane rotation axis.
        The axis points from ``origin`` toward this element.
    flip_x : bool, default False
        If True, reflect coordinates across the y-axis.
    flip_y : bool, default False
        If True, reflect coordinates across the x-axis.

    Notes
    -----
    ``ReferenceFrame`` objects are immutable after instantiation.

    Order of transformations:
    1.  Translate so spectrometer element specified as ``origin`` is at (0,0,0).
    2.  Align (i.e., in-plane rotate) so the spectrometer element ``axis_element``
        lies along the x or y ``axis``.
    3.  Apply an additional scalar in-plane rotation if specified.
    4.  Apply axis reflections / mirroring ``flip_x`` and ``flip_y``
    5.  Apply out-of-plane rotation about the specified axis from ``origin``.

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
    ...     in_plane_rotation=20,
    ... )
    """
    origin: str = "analyzer"
    axis_element: str = "center"
    axis: str = "y"
    in_plane_rotation: float = 0.0
    out_of_plane_rotation: float = 0.0
    out_of_plane_axis: str = "meridional"
    flip_x: bool = False
    flip_y: bool = False

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")

        if self.origin == self.axis_element:
            raise ValueError("origin and axis_element must be different")

        in_plane_rotation = float(self.in_plane_rotation)
        out_of_plane_rotation = float(self.out_of_plane_rotation)

        if not np.isfinite(in_plane_rotation):
            raise ValueError("In plane rotation must be finite")

        if not np.isfinite(out_of_plane_rotation):
            raise ValueError("Out of plane rotation must be finite")

        if self.origin == self.out_of_plane_axis and out_of_plane_rotation != 0.0:
            raise ValueError("origin and out_of_plane_axis must be different")

        object.__setattr__(self, "out_of_plane_rotation", out_of_plane_rotation)
        object.__setattr__(self, "in_plane_rotation", in_plane_rotation)


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

    1.  Translate coordinates so that ``frame.origin`` is at the origin.
    2.  Rotate so that the vector from ``origin`` to ``axis_element`` aligns
        with the requested positive x- or y-axis.
    3.  Apply the additional in-plane clockwise rotation about ``origin`` as
        specified by ``frame.in_plane_rotation``.
    4.  Apply any x- and y-axis reflections.
    5.  Apply out-of-plane rotation about axis from ``origin`` to the element
        specified by ``frame.out_of_plane_axis``.
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
    dtheta -= np.radians(frame.in_plane_rotation)

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

    # 5) Out-of-plane rotation
    if frame.out_of_plane_rotation != 0.0:
        if frame.out_of_plane_axis not in coords:
            raise ValueError(
                f"Unknown out_of_plane_axis '{frame.out_of_plane_axis}'"
            )

        u = coords[frame.out_of_plane_axis].copy()
        norm = float(np.linalg.norm(u))

        if norm == 0.0:
            raise ValueError("Out-of-plane rotation axis has zero length")

        u /= norm

        theta = np.radians(frame.out_of_plane_rotation)
        c = float(np.cos(theta))
        s = float(np.sin(theta))

        ux, uy, uz = u

        K = np.array([
            [0.0, -uz, uy],
            [uz, 0.0, -ux],
            [-uy, ux, 0.0],
        ])

        R = c * np.eye(3) + (1.0 - c) * np.outer(u, u) + s * K

        for k in coords:
            coords[k] = R @ coords[k]

    return coords
