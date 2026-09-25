"""Recover the turntable plane, rotation axis, and center from marker tracks."""

from __future__ import annotations

from typing import Any

import numpy as np


def fit_plane(
    frame_points: list[np.ndarray], prior_axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    points = np.concatenate(frame_points, axis=0)
    center = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    axis = vectors[-1]
    axis /= np.linalg.norm(axis)
    if float(np.dot(axis, prior_axis)) < 0:
        axis = -axis
    camera_x = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    x_axis = camera_x - np.dot(camera_x, axis) * axis
    if np.linalg.norm(x_axis) < 1e-6:
        camera_x = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        x_axis = camera_x - np.dot(camera_x, axis) * axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(axis, x_axis)
    residuals = (points - center) @ axis
    return center, axis, x_axis, y_axis, float(np.sqrt(np.mean(residuals**2)))


def project_plane(
    points: np.ndarray,
    center: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> np.ndarray:
    relative = points - center
    return np.column_stack((relative @ x_axis, relative @ y_axis))


def _metric(values: list[float], operation: str) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if operation == "mean":
        return float(np.mean(array))
    if operation == "max_abs":
        return float(np.max(np.abs(array)))
    if operation == "median_abs":
        return float(np.median(np.abs(array)))
    if operation == "median":
        return float(np.median(array))
    return float(np.max(array))


def solve_axis(
    frames: list[dict[str, Any]],
    *,
    reference_index: int,
    plane_center: np.ndarray,
    axis: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    plane_rms: float,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Solve one run's axis calibration and attach per-frame 3D transforms."""

    successful = [frame for frame in frames if frame["status"] == "ok"]
    moving = [frame for frame in successful if frame["index"] != reference_index]
    if len(moving) < 3:
        raise ValueError(
            f"current-run calibration matched only {len(moving)} moving frames; need at least 3"
        )
    equations = np.concatenate(
        [np.eye(2) - np.asarray(frame["plane_rotation"]) for frame in moving], axis=0
    )
    rhs = np.concatenate(
        [np.asarray(frame["plane_translation_mm"]) for frame in moving]
    )
    center_2d, _, _, _ = np.linalg.lstsq(equations, rhs, rcond=None)
    origin = plane_center + center_2d[0] * x_axis + center_2d[1] * y_axis
    center_residuals = equations @ center_2d - rhs
    basis = np.column_stack((x_axis, y_axis, axis))
    for frame in successful:
        rotation_2d = np.asarray(frame["plane_rotation"], dtype=np.float64)
        local_rotation = np.eye(3)
        local_rotation[:2, :2] = rotation_2d
        rotation_3d = basis @ local_rotation @ basis.T
        transform = np.eye(4)
        transform[:3, :3] = rotation_3d
        transform[:3, 3] = origin - rotation_3d @ origin
        frame["frame_transform"] = transform.tolist()

    pixel_angle_frames = [
        frame for frame in moving if frame["pixel_angle"].get("status") == "ok"
    ]
    pixel_differences = [
        (
            (
                frame["pixel_angle"]["measured_angle_degrees"]
                - frame["measured_angle_degrees"]
                + 180.0
            )
            % 360.0
        )
        - 180.0
        for frame in pixel_angle_frames
    ]
    summary = {
        "frames": len(frames),
        "successful_frames": len(successful),
        "median_matches": _metric([frame["matches"] for frame in successful], "median"),
        "angle_mean": _metric(
            [frame["angle_error_degrees"] for frame in moving], "mean"
        ),
        "angle_max": _metric(
            [frame["angle_error_degrees"] for frame in moving], "max_abs"
        ),
        "fit_median": _metric(
            [frame["fit_rms_mm"] for frame in moving], "median"
        ),
        "fit_max": _metric([frame["fit_rms_mm"] for frame in moving], "max"),
        "pixel_median": _metric(
            [
                frame["pixel_homography"]["rms_px"]
                for frame in moving
                if frame["pixel_homography"]["rms_px"] is not None
            ],
            "median",
        ),
        "pixel_max": _metric(
            [
                frame["pixel_homography"]["rms_px"]
                for frame in moving
                if frame["pixel_homography"]["rms_px"] is not None
            ],
            "max",
        ),
        "pixel_frames": len(pixel_angle_frames) + 1,
        "pixel_angle_mean": _metric(
            [frame["pixel_angle"]["angle_error_degrees"] for frame in pixel_angle_frames],
            "mean",
        ),
        "pixel_angle_max": _metric(
            [frame["pixel_angle"]["angle_error_degrees"] for frame in pixel_angle_frames],
            "max_abs",
        ),
        "cross_angle_median": _metric(
            pixel_differences, "median_abs"
        ),
        "cross_angle_max": _metric(pixel_differences, "max_abs"),
    }
    frame_rms = [float(frame["fit_rms_mm"]) for frame in moving]
    calibration = {
        "axis": axis.tolist(),
        "origin_mm": origin.tolist(),
        "frame_rms_mm": frame_rms,
        "match_rms": frame_rms,
        "axis_residual_mm": float(np.sqrt(np.mean(center_residuals**2))),
        "plane_normal": axis.tolist(),
        "plane_offset_mm": -float(np.dot(axis, plane_center)),
        "plane_rms_mm": plane_rms,
        "axis_alignment_degrees": 0.0,
        "source": "current_run_random_fiducials",
        "source_angles_degrees": [
            float(frame["commanded_angle_degrees"]) for frame in frames
        ],
        "measured_angles_degrees": [
            float(frame["measured_angle_degrees"])
            for frame in frames
            if frame["status"] == "ok"
        ],
        "matched_frames": len(successful),
        "median_matches": summary["median_matches"],
        "tracking_summary": summary,
    }
    return calibration, summary, len(successful)
