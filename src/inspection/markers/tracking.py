"""Track turntable markers across captured frames."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from camera.acquisition import CaptureLayout

from .calibration import fit_plane, project_plane, solve_axis


@dataclass(frozen=True, slots=True)
class CalibrationQualityLimits:
    """Conservative run-level gates derived from recorded calibration residuals."""

    minimum_frames: int = 6
    minimum_median_matches: float = 6.0
    min_image_ratio: float = 0.80
    max_gap: float = 75.0
    plane_rms: float = 0.35
    axis_residual: float = 0.075
    fit_rms: float = 0.30
    angle_error: float = 0.25
    image_rms: float = 0.30
    pixel_rms: float = 1.50
    pixel_angle: float = 0.35
    cross_angle: float = 0.35


def _max_gap(angles: list[float]) -> float | None:
    if len(angles) < 2:
        return None
    normalized = np.unique(np.mod(np.asarray(angles, dtype=np.float64), 360.0))
    if len(normalized) < 2:
        return 360.0
    gaps = np.diff(np.r_[normalized, normalized[0] + 360.0])
    return float(np.max(gaps))


def evaluate_quality(
    tracking: dict[str, Any],
    limits: CalibrationQualityLimits | None = None,
) -> dict[str, Any]:
    """Evaluate whether a solved marker run is safe to use for reconstruction."""

    limits = limits or CalibrationQualityLimits()
    summary = tracking.get("summary", {})
    calibration = tracking.get("calibration", {})
    image_plane = tracking.get("image_plane", {})
    frames = int(summary.get("frames", 0) or 0)
    successful = int(summary.get("successful_frames", 0) or 0)
    image_markers = int(image_plane.get("markers", 0) or 0)
    image_inliers = int(image_plane.get("inliers", 0) or 0)
    image_inlier_ratio = (
        float(image_inliers / image_markers) if image_markers > 0 else None
    )
    maximum_view_gap = _max_gap(
        [float(value) for value in calibration.get("source_angles_degrees", [])]
    )
    measurements = {
        "frames": frames,
        "successful_frames": successful,
        "median_matches": summary.get("median_matches"),
        "image_ratio": image_inlier_ratio,
        "max_gap": maximum_view_gap,
        "plane_rms_mm": calibration.get("plane_rms_mm"),
        "axis_residual_mm": calibration.get("axis_residual_mm"),
        "fit_max": summary.get("fit_max"),
        "angle_max": summary.get("angle_max"),
        "image_rms": image_plane.get("rms_mm"),
        "pixel_max": summary.get("pixel_max"),
        "pixel_angle_max": summary.get("pixel_angle_max"),
        "cross_angle_max": summary.get("cross_angle_max"),
        "pixel_frames": summary.get("pixel_frames"),
    }
    checks: dict[str, dict[str, Any]] = {}

    def minimum_check(name: str, value: Any, minimum: float) -> None:
        passed = value is not None and np.isfinite(value) and float(value) >= minimum
        checks[name] = {
            "status": "pass" if passed else "fail",
            "value": value,
            "minimum": minimum,
        }

    def maximum_check(name: str, value: Any, maximum: float) -> None:
        passed = value is not None and np.isfinite(value) and float(value) <= maximum
        checks[name] = {
            "status": "pass" if passed else "fail",
            "value": value,
            "maximum": maximum,
        }

    minimum_check("frames", frames, limits.minimum_frames)
    checks["all_tracked"] = {
        "status": "pass" if frames > 0 and successful == frames else "fail",
        "value": successful,
        "expected": frames,
    }
    checks["all_angles"] = {
        "status": (
            "pass"
            if frames > 0
            and measurements["pixel_frames"] == frames
            else "fail"
        ),
        "value": measurements["pixel_frames"],
        "expected": frames,
    }
    minimum_check(
        "median_matches",
        measurements["median_matches"],
        limits.minimum_median_matches,
    )
    minimum_check(
        "image_ratio",
        image_inlier_ratio,
        limits.min_image_ratio,
    )
    maximum_check(
        "max_gap",
        maximum_view_gap,
        limits.max_gap,
    )
    maximum_check(
        "plane_rms_mm", measurements["plane_rms_mm"], limits.plane_rms
    )
    maximum_check(
        "axis_residual_mm",
        measurements["axis_residual_mm"],
        limits.axis_residual,
    )
    maximum_check(
        "fit_max",
        measurements["fit_max"],
        limits.fit_rms,
    )
    maximum_check(
        "angle_max",
        measurements["angle_max"],
        limits.angle_error,
    )
    maximum_check(
        "image_rms",
        measurements["image_rms"],
        limits.image_rms,
    )
    maximum_check(
        "pixel_max",
        measurements["pixel_max"],
        limits.pixel_rms,
    )
    maximum_check(
        "pixel_angle_max",
        measurements["pixel_angle_max"],
        limits.pixel_angle,
    )
    maximum_check(
        "cross_angle_max",
        measurements["cross_angle_max"],
        limits.cross_angle,
    )
    failures = [name for name, check in checks.items() if check["status"] != "pass"]
    return {
        "status": "pass" if not failures else "fail",
        "basis": "conservative limits derived from recorded run residuals; not a measurement uncertainty budget",
        "limits": asdict(limits),
        "measurements": measurements,
        "checks": checks,
        "failures": failures,
    }

def _load_metadata(capture_dir: Path) -> dict[str, Any]:
    layout = CaptureLayout(capture_dir)
    path = layout.resolve("capture.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _nearest_pairs(
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    maximum_distance_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = reference[:, None, :] - observed[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    reference_choices = np.argmin(distances, axis=1)
    observed_choices = np.argmin(distances, axis=0)
    reference_indices: list[int] = []
    observed_indices: list[int] = []
    selected_distances: list[float] = []
    for reference_index, observed_index in enumerate(reference_choices):
        distance = float(distances[reference_index, observed_index])
        if (
            observed_choices[observed_index] == reference_index
            and distance <= maximum_distance_mm
        ):
            reference_indices.append(reference_index)
            observed_indices.append(int(observed_index))
            selected_distances.append(distance)
    return (
        np.asarray(reference_indices, dtype=np.int64),
        np.asarray(observed_indices, dtype=np.int64),
        np.asarray(selected_distances, dtype=np.float64),
    )


def _signed_angle(rotation: np.ndarray, axis: np.ndarray) -> float:
    skew = np.asarray(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ),
        dtype=np.float64,
    )
    sine = 0.5 * float(np.dot(axis, skew))
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.rad2deg(np.arctan2(sine, cosine)))


def _homography_metrics(
    reference_pixels: np.ndarray,
    observed_pixels: np.ndarray,
) -> dict[str, float | int | None]:
    if len(reference_pixels) < 4:
        return {"inliers": 0, "inlier_ratio": 0.0, "rms_px": None, "p95_px": None}
    homography, mask = cv2.findHomography(
        observed_pixels.astype(np.float64),
        reference_pixels.astype(np.float64),
        cv2.RANSAC,
        1.5,
        maxIters=5000,
        confidence=0.999,
    )
    if homography is None or mask is None:
        return {"inliers": 0, "inlier_ratio": 0.0, "rms_px": None, "p95_px": None}
    projected = cv2.perspectiveTransform(
        observed_pixels.reshape(1, -1, 2).astype(np.float64), homography
    ).reshape(-1, 2)
    residuals = np.linalg.norm(projected - reference_pixels, axis=1)
    inliers = mask.reshape(-1).astype(bool)
    values = residuals[inliers]
    return {
        "inliers": int(inliers.sum()),
        "inlier_ratio": float(np.mean(inliers)),
        "rms_px": float(np.sqrt(np.mean(values**2))) if len(values) else None,
        "p95_px": float(np.percentile(values, 95)) if len(values) else None,
    }


def _plane_coordinates(
    points: np.ndarray, origin: np.ndarray, axis: np.ndarray
) -> np.ndarray:
    camera_x = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    x_axis = camera_x - np.dot(camera_x, axis) * axis
    if np.linalg.norm(x_axis) < 1e-6:
        camera_x = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        x_axis = camera_x - np.dot(camera_x, axis) * axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(axis, x_axis)
    relative = points - origin
    return np.column_stack((relative @ x_axis, relative @ y_axis))


def _rigid_2d(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1, :] *= -1
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    aligned = source @ rotation.T + translation
    rms = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rotation, translation, rms


def _track_pixels(
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    commanded_angle_degrees: float,
    maximum_distance_mm: float,
) -> dict[str, float | int]:
    match = _match_markers(
        reference,
        observed,
        commanded_angle_degrees=commanded_angle_degrees,
        maximum_distance_mm=maximum_distance_mm,
    )
    if match is None:
        return {"matches": 0, "status": "insufficient_matches"}
    reference_indices, _, rotation, _, rms, _ = match
    mapping_angle = float(np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0])))
    measured = (-mapping_angle) % 360.0
    commanded = commanded_angle_degrees % 360.0
    error = ((measured - commanded + 180.0) % 360.0) - 180.0
    return {
        "status": "ok",
        "matches": int(len(reference_indices)),
        "measured_angle_degrees": measured,
        "angle_error_degrees": error,
        "fit_rms_mm": rms,
    }


def _marker_arrays(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    markers = metadata.get("turntable", {}).get("markers", [])
    valid = [marker for marker in markers if marker.get("point_mm") is not None]
    points = np.asarray([marker["point_mm"] for marker in valid], dtype=np.float64)
    pixels = np.asarray(
        [[marker["x_px"], marker["y_px"]] for marker in valid], dtype=np.float64
    )
    return points.reshape(-1, 3), pixels.reshape(-1, 2)


def _match_markers(
    reference: np.ndarray,
    observed: np.ndarray,
    *,
    commanded_angle_degrees: float,
    maximum_distance_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    """Match partial random marker sets without a previously known rotation center."""

    angle = np.deg2rad(-commanded_angle_degrees)
    initial_rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))),
        dtype=np.float64,
    )
    rotated = observed @ initial_rotation.T
    translations = (reference[:, None, :] - rotated[None, :, :]).reshape(-1, 2)
    best: tuple[int, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for translation in translations:
        aligned = rotated + translation
        reference_indices, observed_indices, distances = _nearest_pairs(
            reference,
            aligned,
            maximum_distance_mm=maximum_distance_mm,
        )
        if not len(distances):
            continue
        rms = float(np.sqrt(np.mean(distances**2)))
        score = (len(reference_indices), -rms)
        if best is None or score > (best[0], -best[1]):
            best = (
                len(reference_indices),
                rms,
                translation.copy(),
                reference_indices,
                observed_indices,
            )
    if best is None or best[0] < 6:
        return None

    _, initial_rms, _, reference_indices, observed_indices = best
    rotation, translation, rms = _rigid_2d(
        observed[observed_indices], reference[reference_indices]
    )
    for _ in range(3):
        aligned = observed @ rotation.T + translation
        next_reference, next_observed, _ = _nearest_pairs(
            reference,
            aligned,
            maximum_distance_mm=maximum_distance_mm,
        )
        if len(next_reference) < 6:
            break
        reference_indices, observed_indices = next_reference, next_observed
        rotation, translation, rms = _rigid_2d(
            observed[observed_indices], reference[reference_indices]
        )
    return (
        reference_indices,
        observed_indices,
        rotation,
        translation,
        rms,
        initial_rms,
    )


def analyze_tracks(
    manifest_path: str | Path,
    calibration_path: str | Path,
    *,
    maximum_distance_mm: float = 2.0,
) -> dict[str, Any]:
    """Estimate this run's axis and center from its own random marker tracks."""

    if maximum_distance_mm <= 0:
        raise ValueError("maximum_distance_mm must be positive")
    manifest_path = Path(manifest_path)
    calibration_path = Path(calibration_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior = json.loads(calibration_path.read_text(encoding="utf-8"))
    prior_axis = np.asarray(prior["axis"], dtype=np.float64)
    prior_axis /= np.linalg.norm(prior_axis)
    records = manifest.get("frames", [])
    if len(records) < 2:
        raise ValueError("marker tracking needs at least two frames")

    frame_data: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    for record in records:
        capture_dir = manifest_path.parent / record["directory"]
        points, pixels = _marker_arrays(_load_metadata(capture_dir))
        if len(points) < 6:
            raise ValueError(f"frame {record['index']} has fewer than 6 markers")
        frame_data.append((record, points, pixels))

    plane_center, axis, x_axis, y_axis, plane_rms = fit_plane(
        [points for _, points, _ in frame_data], prior_axis
    )
    plane_points = [
        project_plane(points, plane_center, x_axis, y_axis)
        for _, points, _ in frame_data
    ]
    reference_record, _, reference_pixels = frame_data[0]
    reference_plane = plane_points[0]
    reference_angle = float(reference_record["degrees"])
    image_to_plane, plane_mask = cv2.findHomography(
        reference_pixels,
        reference_plane,
        cv2.RANSAC,
        0.5,
        maxIters=5000,
        confidence=0.999,
    )
    if image_to_plane is None or plane_mask is None:
        raise ValueError("unable to calibrate the image-to-turntable-plane homography")
    pixel_reference = cv2.perspectiveTransform(
        reference_pixels.reshape(1, -1, 2), image_to_plane
    ).reshape(-1, 2)
    plane_calibration_residuals = np.linalg.norm(
        pixel_reference - reference_plane, axis=1
    )
    plane_calibration_inliers = plane_mask.reshape(-1).astype(bool)
    if int(np.count_nonzero(plane_calibration_inliers)) < 4:
        raise ValueError(
            "image-to-turntable-plane homography has fewer than 4 inliers"
        )

    frames: list[dict[str, Any]] = []
    for (record, _, observed_pixels), observed_plane in zip(frame_data, plane_points):
        commanded = float(record["degrees"]) - reference_angle
        match = _match_markers(
            reference_plane,
            observed_plane,
            commanded_angle_degrees=commanded,
            maximum_distance_mm=maximum_distance_mm,
        )
        if match is None:
            frames.append(
                {
                    "index": int(record["index"]),
                    "commanded_angle_degrees": commanded % 360.0,
                    "status": "insufficient_matches",
                    "reference_markers": int(len(reference_plane)),
                    "observed_markers": int(len(observed_plane)),
                    "matches": 0,
                }
            )
            continue
        reference_indices, observed_indices, rotation, translation, rms, initial_rms = match
        mapping_angle = float(np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0])))
        measured = (-mapping_angle) % 360.0
        commanded_normalized = commanded % 360.0
        angle_error = ((measured - commanded_normalized + 180.0) % 360.0) - 180.0
        pixel_observed = cv2.perspectiveTransform(
            observed_pixels.reshape(1, -1, 2), image_to_plane
        ).reshape(-1, 2)
        frames.append(
            {
                "index": int(record["index"]),
                "commanded_angle_degrees": commanded_normalized,
                "measured_angle_degrees": measured,
                "angle_error_degrees": angle_error,
                "status": "ok",
                "reference_markers": int(len(reference_plane)),
                "observed_markers": int(len(observed_plane)),
                "matches": int(len(reference_indices)),
                "initial_rms": float(initial_rms),
                "fit_rms_mm": float(rms),
                "pixel_homography": _homography_metrics(
                    reference_pixels[reference_indices], observed_pixels[observed_indices]
                ),
                "pixel_angle": _track_pixels(
                    pixel_reference,
                    pixel_observed,
                    commanded_angle_degrees=commanded,
                    maximum_distance_mm=maximum_distance_mm,
                ),
                "plane_rotation": rotation.tolist(),
                "plane_translation_mm": translation.tolist(),
            }
        )

    calibration, summary, successful_count = solve_axis(
        frames,
        reference_index=int(reference_record["index"]),
        plane_center=plane_center,
        axis=axis,
        x_axis=x_axis,
        y_axis=y_axis,
        plane_rms=plane_rms,
    )
    result = {
        "status": "ok" if successful_count == len(frames) else "partial",
        "method": "current-run-plane-and-translation-voting",
        "manifest": str(manifest_path),
        "orientation_reference": str(calibration_path),
        "match_limit": float(maximum_distance_mm),
        "reference_frame": int(reference_record["index"]),
        "image_plane": {
            "inliers": int(plane_calibration_inliers.sum()),
            "markers": int(len(reference_pixels)),
            "rms_mm": float(np.sqrt(np.mean(plane_calibration_residuals[plane_calibration_inliers] ** 2))),
            "homography": image_to_plane.tolist(),
        },
        "summary": summary,
        "calibration": calibration,
        "frames": frames,
    }
    result["quality_gate"] = evaluate_quality(result)
    if result["status"] == "ok" and result["quality_gate"]["status"] != "pass":
        result["status"] = "quality_failed"
    return result


def estimate_calibration(
    manifest_path: str | Path,
    calibration_path: str | Path,
    *,
    maximum_distance_mm: float = 2.0,
) -> dict[str, Any]:
    """Estimate the current axis and center without reusing a prior center."""

    tracking = analyze_tracks(
        manifest_path,
        calibration_path,
        maximum_distance_mm=maximum_distance_mm,
    )
    diagnostic_path = Path(manifest_path).with_name("tracking.json")
    diagnostic_path.write_text(
        json.dumps(tracking, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if tracking.get("status") != "ok":
        failed = [
            int(frame["index"])
            for frame in tracking.get("frames", [])
            if frame.get("status") != "ok"
        ]
        summary = tracking.get("summary", {})
        quality_failures = tracking.get("quality_gate", {}).get("failures", [])
        if quality_failures:
            raise ValueError(
                "标记标定质量门禁失败：" + ", ".join(quality_failures)
            )
        raise ValueError(
            f"标记跟踪失败：完成 {summary.get('successful_frames', 0)}/"
            f"{summary.get('frames', 0)} 帧，失败帧 {failed}。"
        )
    calibration = dict(tracking["calibration"])
    calibration["quality_gate"] = tracking["quality_gate"]
    return calibration
