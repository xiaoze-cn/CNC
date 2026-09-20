"""Rigid transformations between camera and turntable coordinates."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _unit_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("axis must be a finite non-zero vector")
    return axis / norm


def rotation_matrix(axis: Iterable[float], angle_degrees: float) -> np.ndarray:
    """Return a 3x3 right-handed Rodrigues rotation matrix."""

    axis = _unit_axis(np.asarray(tuple(axis), dtype=np.float64))
    theta = np.deg2rad(float(angle_degrees))
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3) * np.cos(theta)
        + (1 - np.cos(theta)) * np.outer(axis, axis)
        + np.sin(theta) * skew
    )


def transform_about_axis(
    points: np.ndarray,
    *,
    origin: Iterable[float],
    axis: Iterable[float],
    angle_degrees: float,
) -> np.ndarray:
    """Rotate points around a world-space axis while preserving translation."""

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    origin = np.asarray(tuple(origin), dtype=np.float64).reshape(3)
    if not np.isfinite(origin).all():
        raise ValueError("origin must be finite")
    rotation = rotation_matrix(axis, angle_degrees)
    finite = np.isfinite(points).all(axis=1)
    result = points.copy()
    result[finite] = (
        (points[finite].astype(np.float64) - origin) @ rotation.T + origin
    ).astype(np.float32)
    return result


def to_turntable_coordinates(
    points: np.ndarray,
    *,
    origin: Iterable[float],
    axis: Iterable[float],
) -> np.ndarray:
    """Convert camera XYZ to a centered, right-handed, Z-up turntable frame."""

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    origin = np.asarray(tuple(origin), dtype=np.float64).reshape(3)
    z_axis = _unit_axis(np.asarray(tuple(axis), dtype=np.float64))
    camera_x = np.array([1.0, 0.0, 0.0])
    x_axis = camera_x - np.dot(camera_x, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        camera_x = np.array([0.0, 1.0, 0.0])
        x_axis = camera_x - np.dot(camera_x, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    basis = np.column_stack((x_axis, y_axis, z_axis))
    return ((points.astype(np.float64) - origin) @ basis).astype(np.float32)
