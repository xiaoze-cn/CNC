"""Visual marker detection and 3D circle fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class MarkerCandidate:
    x_px: float
    y_px: float
    radius_px: float
    contrast: float
    inner_brightness: float
    point_mm: tuple[float, float, float] | None = None
    component_label: int | None = None
    component_bbox_px: tuple[int, int, int, int] | None = None
    component_area_px: int | None = None
    center_method: str | None = None
    component_point_count: int | None = None
    circle_boundary_point_count: int | None = None
    circle_inlier_count: int | None = None
    circle_radius_mm: float | None = None
    circle_rms_mm: float | None = None

    def data(self) -> dict[str, object]:
        return asdict(self)


def load_mono8(path: str | Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError("marker image must be a single-channel grayscale image")
    return image


def detect_marker_components(
    image: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None = None,
    threshold: float = 20.0,
    blur_kernel: int = 5,
    min_area: int = 1200,
    max_area: int = 5000,
    min_width: int = 40,
    max_width: int = 130,
    min_height: int = 20,
    max_height: int = 70,
    min_aspect: float = 1.2,
    max_aspect: float = 3.5,
) -> list[MarkerCandidate]:
    """Find the wide, bright elliptical fiducials used on the turntable.

    The white disks are often elongated by perspective and textured by the
    projector, so Hough circles alone misses them. Connected components on a
    lightly blurred intensity threshold provide a complementary candidate set;
    this remains a candidate detector, not an automatic identity decision.
    """

    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError("image must be HxW Mono8")
    height, width = image.shape
    x0, y0, x1, y1 = (0, 0, width, height) if roi is None else tuple(int(v) for v in roi)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("roi must be x0,y0,x1,y1 inside image")
    if blur_kernel < 1 or blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be a positive odd integer")
    cropped = cv2.GaussianBlur(image[y0:y1, x0:x1], (blur_kernel, blur_kernel), 0)
    _, binary = cv2.threshold(cropped, float(threshold), 255, cv2.THRESH_BINARY)
    _, _, stats, centers = cv2.connectedComponentsWithStats(binary)
    yy, xx = np.ogrid[:cropped.shape[0], :cropped.shape[1]]
    candidates: list[MarkerCandidate] = []
    for label, (stats_row, center) in enumerate(zip(stats[1:], centers[1:]), start=1):
        x, y, comp_width, comp_height, area = (int(value) for value in stats_row)
        aspect = comp_width / max(comp_height, 1)
        if not (
            min_area <= area <= max_area
            and min_width <= comp_width <= max_width
            and min_height <= comp_height <= max_height
            and min_aspect <= aspect <= max_aspect
        ):
            continue
        cx, cy = float(center[0]), float(center[1])
        radius = 0.5 * max(comp_width, comp_height)
        distance = np.hypot(xx - cx, yy - cy)
        inner = cropped[distance <= radius * 0.45]
        ring = cropped[(distance >= radius * 1.15) & (distance <= radius * 1.8)]
        if len(inner) == 0 or len(ring) == 0:
            continue
        inner_brightness = float(np.mean(inner))
        contrast = inner_brightness - float(np.mean(ring))
        candidates.append(
            MarkerCandidate(
                x_px=cx + x0,
                y_px=cy + y0,
                radius_px=radius,
                contrast=contrast,
                inner_brightness=inner_brightness,
                component_label=label,
                component_bbox_px=(x + x0, y + y0, comp_width, comp_height),
                component_area_px=area,
            )
        )
    return sorted(candidates, key=lambda candidate: (candidate.y_px, candidate.x_px))


def attach_pointcloud_coordinates(
    candidates: Iterable[MarkerCandidate],
    points: np.ndarray,
    *,
    image_shape: tuple[int, int],
    sample_radius_scale: float = 0.35,
) -> list[MarkerCandidate]:
    """Attach robust median XYZ coordinates to image candidates."""

    height, width = image_shape
    points = np.asarray(points, dtype=np.float32).reshape(height, width, 3)
    result: list[MarkerCandidate] = []
    for candidate in candidates:
        radius = max(2.0, candidate.radius_px * sample_radius_scale)
        x0 = max(0, int(candidate.x_px - radius))
        x1 = min(width, int(candidate.x_px + radius + 1))
        y0 = max(0, int(candidate.y_px - radius))
        y1 = min(height, int(candidate.y_px + radius + 1))
        grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
        mask = (grid_x - candidate.x_px) ** 2 + (grid_y - candidate.y_px) ** 2 <= radius**2
        local = points[y0:y1, x0:x1][mask]
        finite = local[np.isfinite(local).all(axis=1)]
        point = tuple(float(value) for value in np.median(finite, axis=0)) if len(finite) else None
        result.append(
            replace(candidate, point_mm=point, center_method="image-window-median")
        )
    return result


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("normal must be a finite non-zero vector")
    normal /= length
    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(np.abs(normal)))] = 1.0
    u = np.cross(normal, reference)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return normal, u, v


def _circle_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    a, b, c = np.asarray(points, dtype=np.float64).reshape(3, 2)
    matrix = 2.0 * np.asarray([b - a, c - a])
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-9:
        return None
    rhs = np.asarray([b @ b - a @ a, c @ c - a @ a])
    center = np.linalg.solve(matrix, rhs)
    return center, float(np.linalg.norm(a - center))


def _least_squares_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    anchor = np.mean(points, axis=0)
    local = points - anchor
    matrix = np.column_stack((2.0 * local, np.ones(len(local))))
    rhs = np.sum(local**2, axis=1)
    solution, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    radius_squared = float(solution[2] + solution[0] ** 2 + solution[1] ** 2)
    if radius_squared <= 0.0:
        raise ValueError("circle fit produced a non-positive radius")
    return solution[:2] + anchor, float(np.sqrt(radius_squared))


def _robust_circle_fit(
    points: np.ndarray,
    *,
    residual_threshold_mm: float = 0.20,
    iterations: int = 160,
) -> tuple[np.ndarray, float, np.ndarray, float] | None:
    """Fit a circle to boundary points with deterministic three-point RANSAC."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 12 or not np.isfinite(points).all():
        return None
    rng = np.random.default_rng(0)
    best_inliers: np.ndarray | None = None
    best_rms = np.inf
    for sample in rng.choice(len(points), size=(iterations, 3), replace=True):
        if len(set(int(index) for index in sample)) < 3:
            continue
        circle = _circle_from_three(points[sample])
        if circle is None:
            continue
        center, radius = circle
        if not (0.25 <= radius <= 25.0):
            continue
        residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        inliers = residuals <= residual_threshold_mm
        count = int(np.count_nonzero(inliers))
        if count < 12:
            continue
        rms = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
        if best_inliers is None or count > np.count_nonzero(best_inliers) or (
            count == np.count_nonzero(best_inliers) and rms < best_rms
        ):
            best_inliers = inliers
            best_rms = rms
    if best_inliers is None:
        return None

    center, radius = _least_squares_circle(points[best_inliers])
    for _ in range(3):
        residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        refined = residuals <= residual_threshold_mm
        if np.count_nonzero(refined) < 12 or np.array_equal(refined, best_inliers):
            break
        best_inliers = refined
        center, radius = _least_squares_circle(points[best_inliers])
    residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    rms = float(np.sqrt(np.mean(residuals[best_inliers] ** 2)))
    return center, radius, best_inliers, rms


def attach_marker_component_coordinates(
    candidates: Iterable[MarkerCandidate],
    image: np.ndarray,
    points: np.ndarray,
    *,
    normal: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    threshold: float = 20.0,
    blur_kernel: int = 5,
    residual_threshold_mm: float = 0.20,
    max_circle_rms_mm: float = 0.18,
    min_inlier_ratio: float = 0.55,
) -> list[MarkerCandidate]:
    """Recover marker centers by fitting their full 3D component boundaries.

    Connected components are segmented in image space, but their circle is fit
    after projecting boundary XYZ samples into the physical turntable plane.
    The legacy center-window median is retained only as a per-marker fallback.
    """

    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError("image must be HxW Mono8")
    height, width = image.shape
    organized = np.asarray(points, dtype=np.float64).reshape(height, width, 3)
    x0, y0, x1, y1 = (0, 0, width, height) if roi is None else tuple(int(v) for v in roi)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("roi must be x0,y0,x1,y1 inside image")
    if blur_kernel < 1 or blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be a positive odd integer")
    cropped = cv2.GaussianBlur(image[y0:y1, x0:x1], (blur_kernel, blur_kernel), 0)
    _, binary = cv2.threshold(cropped, float(threshold), 255, cv2.THRESH_BINARY)
    _, labels, _, _ = cv2.connectedComponentsWithStats(binary)
    normal, u, v = _plane_basis(normal)

    candidate_list = list(candidates)
    fallback = attach_pointcloud_coordinates(
        candidate_list,
        organized,
        image_shape=image.shape,
    )
    result: list[MarkerCandidate] = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    for candidate, fallback_candidate in zip(candidate_list, fallback):
        label = candidate.component_label
        center_x = int(round(candidate.x_px)) - x0
        center_y = int(round(candidate.y_px)) - y0
        label_is_valid = (
            label is not None
            and 0 < label <= int(labels.max())
            and 0 <= center_x < labels.shape[1]
            and 0 <= center_y < labels.shape[0]
            and labels[center_y, center_x] == label
        )
        if not label_is_valid:
            result.append(fallback_candidate)
            continue
        component = labels == label
        eroded = cv2.erode(component.astype(np.uint8), kernel, iterations=1).astype(bool)
        boundary = component & ~eroded
        ys, xs = np.nonzero(component)
        boundary_ys, boundary_xs = np.nonzero(boundary)
        component_xyz = organized[ys + y0, xs + x0]
        boundary_xyz = organized[boundary_ys + y0, boundary_xs + x0]
        component_xyz = component_xyz[np.isfinite(component_xyz).all(axis=1)]
        boundary_xyz = boundary_xyz[np.isfinite(boundary_xyz).all(axis=1)]
        if len(component_xyz) < 24 or len(boundary_xyz) < 12:
            result.append(
                replace(
                    fallback_candidate,
                    component_point_count=len(component_xyz),
                    circle_boundary_point_count=len(boundary_xyz),
                )
            )
            continue

        plane_offsets = component_xyz @ normal
        plane_offset = float(np.median(plane_offsets))
        deviations = np.abs(boundary_xyz @ normal - plane_offset)
        median_deviation = float(np.median(deviations))
        axial_limit = max(0.35, 4.0 * 1.4826 * median_deviation)
        boundary_xyz = boundary_xyz[deviations <= axial_limit]
        if len(boundary_xyz) < 12:
            result.append(
                replace(
                    fallback_candidate,
                    component_point_count=len(component_xyz),
                    circle_boundary_point_count=len(boundary_xyz),
                )
            )
            continue

        anchor = np.median(component_xyz, axis=0)
        boundary_2d = np.column_stack(
            ((boundary_xyz - anchor) @ u, (boundary_xyz - anchor) @ v)
        )
        fit = _robust_circle_fit(
            boundary_2d,
            residual_threshold_mm=residual_threshold_mm,
        )
        if fit is None:
            result.append(
                replace(
                    fallback_candidate,
                    component_point_count=len(component_xyz),
                    circle_boundary_point_count=len(boundary_xyz),
                )
            )
            continue
        center_2d, radius, inliers, rms = fit
        inlier_count = int(np.count_nonzero(inliers))
        accepted = (
            inlier_count >= 12
            and inlier_count / len(boundary_xyz) >= min_inlier_ratio
            and 0.5 <= radius <= 20.0
            and rms <= max_circle_rms_mm
        )
        if not accepted:
            result.append(
                replace(
                    fallback_candidate,
                    component_point_count=len(component_xyz),
                    circle_boundary_point_count=len(boundary_xyz),
                    circle_inlier_count=inlier_count,
                    circle_radius_mm=float(radius),
                    circle_rms_mm=rms,
                )
            )
            continue
        center_3d = (
            anchor
            + center_2d[0] * u
            + center_2d[1] * v
            + (plane_offset - float(anchor @ normal)) * normal
        )
        result.append(
            replace(
                candidate,
                point_mm=tuple(float(value) for value in center_3d),
                center_method="component-boundary-circle-fit",
                component_point_count=len(component_xyz),
                circle_boundary_point_count=len(boundary_xyz),
                circle_inlier_count=inlier_count,
                circle_radius_mm=float(radius),
                circle_rms_mm=rms,
            )
        )
    return result
