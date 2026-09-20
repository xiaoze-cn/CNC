"""Turntable subject extraction and capture-side observation persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from camera.acquisition import Camera, CaptureLayout, CaptureStoragePolicy
from inspection.geometry.pointcloud import clean_points, write_ascii_ply
from inspection.geometry.transforms import rotation_matrix, transform_about_axis
from inspection.markers.detection import (
    attach_marker_component_coordinates,
    detect_marker_components,
    load_mono8,
)


# 标定种子只用于提供初始转轴估计
# 实际采集中标记平面在临时坐标中可能有几毫米厚度
# 原因可能是透视 点图偏差或轻微转轴漂移
# 这些限制用于稳健选择候选点而不是测量容差
_MARKER_PLANE_GAP_MM = 1.50
_MIN_MARKER_RADIUS_MM = 10.0
_MIN_MARKERS = 6


@dataclass(frozen=True, slots=True)
class TurntableLocation:
    axis: tuple[float, float, float]
    center_mm: tuple[float, float, float]
    marker_plane_offset_mm: float
    marker_plane_rms_mm: float
    marker_count: int
    marker_radius_min_mm: float
    marker_radius_max_mm: float
    subject_radius_mm: float
    marker_circle_fit_count: int = 0
    marker_circle_radius_median_mm: float | None = None
    marker_circle_rms_median_mm: float | None = None
    marker_circle_rms_max_mm: float | None = None


@dataclass(frozen=True, slots=True)
class FrameObservation:
    points: np.ndarray
    frame_index: int
    angle_degrees: float | None = None
    source_indices: np.ndarray | None = None
    confidence: np.ndarray | None = None
    image: np.ndarray | None = None
    incidence_cosine: np.ndarray | None = None
    normals: np.ndarray | None = None
    camera_origin: np.ndarray | None = None
    alignment_rotation: np.ndarray | None = None
    rotation_origin: np.ndarray | None = None
    depth_mm: np.ndarray | None = None
    intrinsics: tuple[float, float, float, float] | None = None


def normal_incidence_cosine(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if len(points) != len(normals):
        raise ValueError("points and normals must have the same length")
    point_norm = np.linalg.norm(points, axis=1)
    normal_norm = np.linalg.norm(normals, axis=1)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(normals).all(axis=1)
        & (point_norm > 0)
        & (normal_norm > 0)
    )
    cosine = np.full(len(points), np.nan, dtype=np.float64)
    cosine[valid] = np.abs(
        np.einsum("ij,ij->i", points[valid], normals[valid])
        / (point_norm[valid] * normal_norm[valid])
    )
    return np.clip(cosine, 0.0, 1.0)


def locate_turntable(
    image: np.ndarray,
    points: np.ndarray,
    calibration: dict[str, Any],
    *,
    marker_threshold: float = 10.0,
) -> tuple[TurntableLocation, list[dict[str, object]]]:
    """Locate the stage plane from its markers and calibrated rotation axis."""

    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape
    roi = (0, int(height * 0.24), int(width * 0.82), int(height * 0.78))
    axis = np.asarray(calibration["axis"], dtype=np.float64).reshape(3)
    axis /= np.linalg.norm(axis)
    origin = np.asarray(calibration["origin_mm"], dtype=np.float64).reshape(3)
    candidates = detect_marker_components(image, roi=roi, threshold=marker_threshold)
    candidates = attach_marker_component_coordinates(
        candidates,
        image,
        points,
        normal=axis,
        roi=roi,
        threshold=marker_threshold,
    )
    candidates = [
        candidate
        for candidate in candidates
        if candidate.point_mm is not None
        and candidate.center_method == "component-boundary-circle-fit"
    ]
    if len(candidates) < _MIN_MARKERS:
        raise ValueError(
            f"转台定位圆拟合不足：只找到 {len(candidates)} 个合格三维圆，至少需要 {_MIN_MARKERS} 个"
        )

    marker_points = np.asarray(
        [candidate.point_mm for candidate in candidates], dtype=np.float64
    )
    offsets = (marker_points - origin) @ axis
    # 沿临时转轴排序后选择最大的连续点簇
    # 这样可以排除远离转台平面的明显误检
    # 同时容忍当前点图中可见的两到三毫米范围扩散
    order = np.argsort(offsets)
    sorted_offsets = offsets[order]
    split_points = np.flatnonzero(np.diff(sorted_offsets) > _MARKER_PLANE_GAP_MM)
    starts = np.r_[0, split_points + 1]
    ends = np.r_[split_points + 1, len(sorted_offsets)]
    cluster_sizes = ends - starts
    largest = int(np.argmax(cluster_sizes))
    cluster_indices = order[starts[largest] : ends[largest]]
    inliers = np.zeros(len(offsets), dtype=bool)
    inliers[cluster_indices] = True
    marker_points = marker_points[inliers]
    selected = [candidate for candidate, keep in zip(candidates, inliers) if keep]
    if len(marker_points) < _MIN_MARKERS:
        raise ValueError(
            f"转台平面内定位点不足：只剩 {len(marker_points)} 个，"
            f"至少需要 {_MIN_MARKERS} 个"
        )

    plane_offset = float(np.median((marker_points - origin) @ axis))
    plane_residuals = (marker_points - origin) @ axis - plane_offset
    center = origin + plane_offset * axis
    relative = marker_points - center
    axial = relative @ axis
    radii = np.linalg.norm(relative - np.outer(axial, axis), axis=1)
    # 少量伪检组件可能落在转轴附近
    # 旧的二十毫米截断也会删除最新采集中的真实内环标记
    # 保留这些标记并只拒绝实际位于转轴上的点
    radial_inliers = radii >= _MIN_MARKER_RADIUS_MM
    if int(np.count_nonzero(radial_inliers)) < _MIN_MARKERS:
        raise ValueError(
            "定位点的转台半径分布无效："
            f"有效半径点 {int(np.count_nonzero(radial_inliers))} 个，"
            f"至少需要 {_MIN_MARKERS} 个"
        )
    radii = radii[radial_inliers]
    radius_min = float(np.min(radii))
    radius_max = float(np.percentile(radii, 95))
    fitted = [
        candidate
        for candidate in selected
        if candidate.circle_radius_mm is not None and candidate.circle_rms_mm is not None
    ]
    circle_radii = np.asarray(
        [candidate.circle_radius_mm for candidate in fitted], dtype=np.float64
    )
    circle_rms = np.asarray(
        [candidate.circle_rms_mm for candidate in fitted], dtype=np.float64
    )
    location = TurntableLocation(
        axis=tuple(float(value) for value in axis),
        center_mm=tuple(float(value) for value in center),
        marker_plane_offset_mm=plane_offset,
        marker_plane_rms_mm=float(np.sqrt(np.mean(plane_residuals**2))),
        marker_count=len(marker_points),
        marker_radius_min_mm=radius_min,
        marker_radius_max_mm=radius_max,
        subject_radius_mm=float(radius_max * 0.75),
        marker_circle_fit_count=len(fitted),
        marker_circle_radius_median_mm=(
            float(np.median(circle_radii)) if len(circle_radii) else None
        ),
        marker_circle_rms_median_mm=(
            float(np.median(circle_rms)) if len(circle_rms) else None
        ),
        marker_circle_rms_max_mm=float(np.max(circle_rms)) if len(circle_rms) else None,
    )
    return location, [candidate.data() for candidate in selected]


def turntable_subject_mask(
    points: np.ndarray,
    location: TurntableLocation,
    *,
    min_height_mm: float = 1.0,
    max_height_mm: float = 80.0,
) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(cloud).all(axis=1)
    valid = cloud[finite]
    axis = np.asarray(location.axis, dtype=np.float64)
    center = np.asarray(location.center_mm, dtype=np.float64)
    relative = valid.astype(np.float64) - center
    height = relative @ axis
    radius = np.linalg.norm(relative - np.outer(height, axis), axis=1)
    keep_valid = (
        (height >= min_height_mm)
        & (height <= max_height_mm)
        & (radius <= location.subject_radius_mm)
    )
    keep = np.zeros(len(cloud), dtype=bool)
    keep[np.flatnonzero(finite)] = keep_valid
    return keep


def _z_window_mask(
    points: np.ndarray,
    *,
    z_min_mm: float | None,
    z_max_mm: float | None,
) -> np.ndarray:
    """Mark points in the camera-coordinate Z window after localization."""

    if (z_min_mm is None) != (z_max_mm is None):
        raise ValueError("z_min_mm and z_max_mm must be provided together")
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    mask = np.isfinite(cloud).all(axis=1)
    if z_min_mm is None:
        return mask
    if not np.isfinite([z_min_mm, z_max_mm]).all() or z_min_mm >= z_max_mm:
        raise ValueError("z_min_mm and z_max_mm must be finite and ordered")
    mask &= cloud[:, 2] >= z_min_mm
    mask &= cloud[:, 2] <= z_max_mm
    return mask


def _write_subject_indices(directory: Path, mask: np.ndarray) -> Path:
    layout = CaptureLayout(directory)
    path = layout.write_path("processed", "source_indices.npy")
    np.save(path, np.flatnonzero(mask).astype(np.int64))
    metadata_path = layout.resolve("capture.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.setdefault("files", {})["source_indices_npy"] = layout.relative(path)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return path


def _write_valid_z_indices(directory: Path, mask: np.ndarray) -> Path:
    layout = CaptureLayout(directory)
    path = layout.write_path("processed", "valid.npy")
    np.save(path, np.flatnonzero(mask).astype(np.int64))
    metadata_path = layout.resolve("capture.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.setdefault("files", {})["valid_indices_npy"] = layout.relative(path)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return path


def process_turntable_capture(
    directory: str | Path,
    calibration: str | Path,
    *,
    min_height_mm: float = 1.0,
    max_height_mm: float = 80.0,
    write_source_indices: bool | None = None,
    write_ply: bool = False,
    z_min_mm: float | None = None,
    z_max_mm: float | None = None,
) -> Path:
    """Recreate the subject cloud from persisted source observations."""

    directory = Path(directory)
    layout = CaptureLayout(directory)
    points_path = layout.resolve("points.npy")
    if not points_path.is_file():
        raise FileNotFoundError(f"采集目录缺少 source/points.npy: {points_path}")
    organized = np.asarray(np.load(points_path), dtype=np.float64) * 1000.0
    image = load_mono8(layout.resolve("image.png"))
    calibration_payload = json.loads(Path(calibration).read_text(encoding="utf-8"))
    location, markers = locate_turntable(image, organized, calibration_payload)
    subject_mask = turntable_subject_mask(
        organized,
        location,
        min_height_mm=min_height_mm,
        max_height_mm=max_height_mm,
    )
    z_mask = _z_window_mask(
        organized,
        z_min_mm=z_min_mm,
        z_max_mm=z_max_mm,
    )
    subject_mask &= z_mask
    subject = organized.reshape(-1, 3)[subject_mask]
    if len(subject) < 1000:
        raise ValueError(f"主体点云过少：{len(subject)} 个点")

    layout.ensure()
    output_cloud = layout.write_path("processed", "cloud.npy")
    output_ply = layout.write_path("processed", "cloud.ply")
    np.save(output_cloud, subject)
    if write_ply:
        write_ascii_ply(output_ply, subject)
    metadata_path = layout.resolve("capture.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"采集目录缺少 metadata/capture.json: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["files"]["points_npy"] = layout.relative(points_path)
    metadata["files"]["cloud_npy"] = layout.relative(output_cloud)
    metadata["files"]["cloud_ply"] = layout.relative(output_ply) if write_ply else None
    metadata["turntable"] = {**asdict(location), "markers": markers}
    metadata["subject"] = {
        "selection": "inside_marker_ring_and_above_turntable_plane",
        "min_height_mm": min_height_mm,
        "max_height_mm": max_height_mm,
        "points": int(len(subject)),
    }
    metadata["z_window"] = {
        "selection": "post_localization_camera_coordinates",
        "min_mm": z_min_mm,
        "max_mm": z_max_mm,
        "valid_points": int(z_mask.sum()),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if write_source_indices is None:
        write_source_indices = any(
            layout.resolve(name).is_file()
            for name in ("depth.npy", "confidence.npy", "normals.npy", "image.npy")
        )
    if write_source_indices:
        _write_subject_indices(directory, subject_mask)
    if z_min_mm is not None:
        _write_valid_z_indices(directory, z_mask)
    return output_ply if write_ply else output_cloud


def save_turntable_capture(
    camera: Camera,
    frame: Any,
    directory: str | Path,
    calibration: str | Path,
    *,
    min_height_mm: float = 1.0,
    max_height_mm: float = 80.0,
    z_min_mm: float | None = None,
    z_max_mm: float | None = None,
    storage_policy: CaptureStoragePolicy | None = None,
) -> Path:
    """Extract and persist one copied turntable observation."""

    if frame.image is None:
        raise ValueError("转台定位需要本帧 Mono8 图像")
    calibration_payload = json.loads(Path(calibration).read_text(encoding="utf-8"))
    location, markers = locate_turntable(frame.image, frame.points, calibration_payload)
    subject_mask = turntable_subject_mask(
        frame.points,
        location,
        min_height_mm=min_height_mm,
        max_height_mm=max_height_mm,
    )
    z_mask = _z_window_mask(
        frame.points,
        z_min_mm=z_min_mm,
        z_max_mm=z_max_mm,
    )
    subject_mask &= z_mask
    subject = frame.points[subject_mask]
    if len(subject) < 1000:
        raise ValueError(f"主体点云过少：{len(subject)} 个点")
    output = camera.save(
        frame,
        directory,
        cloud_points=subject,
        extra_metadata={
            "turntable": {**asdict(location), "markers": markers},
            "subject": {
                "selection": "inside_marker_ring_and_above_turntable_plane",
                "min_height_mm": min_height_mm,
                "max_height_mm": max_height_mm,
                "points": int(len(subject)),
            },
            "z_window": {
                "selection": "post_localization_camera_coordinates",
                "min_mm": z_min_mm,
                "max_mm": z_max_mm,
                "valid_points": int(z_mask.sum()),
            },
        },
        storage_policy=storage_policy,
    )
    if z_min_mm is not None:
        _write_valid_z_indices(Path(directory), z_mask)
    if storage_policy is not None and storage_policy.save_source_indices:
        _write_subject_indices(Path(directory), subject_mask)
    return output


def _fit_pinhole_intrinsics(
    organized_points: np.ndarray,
) -> tuple[float, float, float, float]:
    points = np.asarray(organized_points)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("organized point map must have shape HxWx3")
    _, width = points.shape[:2]
    flat = points.reshape(-1, 3)
    sample_indices = np.arange(0, len(flat), max(1, len(flat) // 20000))
    sample = flat[sample_indices]
    valid = np.isfinite(sample).all(axis=1) & (sample[:, 2] > 0)
    sample = sample[valid]
    sample_indices = sample_indices[valid]
    if len(sample) < 100:
        raise ValueError("not enough organized points to recover camera projection")
    rows, columns = np.divmod(sample_indices, width)
    x_model = np.column_stack((sample[:, 0] / sample[:, 2], np.ones(len(sample))))
    y_model = np.column_stack((sample[:, 1] / sample[:, 2], np.ones(len(sample))))
    fx, cx = np.linalg.lstsq(x_model, columns, rcond=None)[0]
    fy, cy = np.linalg.lstsq(y_model, rows, rcond=None)[0]
    if not np.isfinite((fx, fy, cx, cy)).all() or fx <= 0 or fy <= 0:
        raise ValueError("recovered camera projection is invalid")
    return float(fx), float(fy), float(cx), float(cy)


def load_frame_observations(
    manifest_path: str | Path,
    calibration_path: str | Path,
    *,
    angle_sign: float = -1.0,
) -> list[FrameObservation]:
    """Load aligned clouds and their source-level sensor evidence."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    observations: list[FrameObservation] = []
    intrinsics: tuple[float, float, float, float] | None = None
    for record in manifest.get("frames", []):
        capture_dir = manifest_path.parent / record["directory"]
        cloud_path = (manifest_path.parent / record["path"]).with_suffix(".npy")
        raw_points = np.asarray(np.load(cloud_path), dtype=np.float64).reshape(-1, 3)
        angle = float(record["measured_degrees"])
        transform_angle = angle_sign * angle
        alignment_rotation = rotation_matrix(calibration["axis"], transform_angle)
        transformed = transform_about_axis(
            raw_points,
            origin=calibration["origin_mm"],
            axis=calibration["axis"],
            angle_degrees=transform_angle,
        )
        finite_mask = np.isfinite(transformed).all(axis=1)
        aligned = transformed[finite_mask].astype(np.float32, copy=False)
        indices_path = capture_dir / "processed" / "source_indices.npy"
        indices: np.ndarray | None = None
        confidence: np.ndarray | None = None
        image: np.ndarray | None = None
        incidence_cosine: np.ndarray | None = None
        aligned_normals: np.ndarray | None = None
        depth_mm: np.ndarray | None = None
        camera_origin = transform_about_axis(
            np.zeros((1, 3), dtype=np.float32),
            origin=calibration["origin_mm"],
            axis=calibration["axis"],
            angle_degrees=transform_angle,
        )[0]
        if indices_path.is_file():
            candidate_indices = np.asarray(
                np.load(indices_path), dtype=np.int64
            ).reshape(-1)
            if len(candidate_indices) == len(raw_points):
                indices = candidate_indices[finite_mask]
                confidence_path = capture_dir / "source" / "confidence.npy"
                image_path = capture_dir / "source" / "image.npy"
                normals_path = capture_dir / "source" / "normals.npy"
                if confidence_path.is_file():
                    flat_confidence = np.asarray(np.load(confidence_path)).reshape(-1)
                    if len(flat_confidence) > int(indices.max(initial=-1)):
                        confidence = flat_confidence[indices]
                if image_path.is_file():
                    source_image = np.asarray(np.load(image_path))
                    if source_image.ndim >= 2 and len(source_image.reshape(-1)) > int(
                        indices.max(initial=-1)
                    ):
                        image = source_image
                if normals_path.is_file():
                    source_normals = np.asarray(np.load(normals_path))
                    if source_normals.ndim >= 2 and source_normals.shape[-1] == 3:
                        flat_normals = source_normals.reshape(-1, 3)
                        if len(flat_normals) > int(indices.max(initial=-1)):
                            selected_normals = flat_normals[indices]
                            incidence_cosine = normal_incidence_cosine(
                                raw_points[finite_mask], selected_normals
                            )
                            if len(aligned) == len(selected_normals):
                                aligned_normals = (
                                    np.asarray(selected_normals, dtype=np.float64)
                                    @ alignment_rotation.T
                                ).astype(np.float32)
                depth_path = capture_dir / "source" / "depth.npy"
                organized_path = capture_dir / "source" / "points.npy"
                if depth_path.is_file():
                    depth_mm = (
                        np.asarray(np.load(depth_path), dtype=np.float32) * 1000.0
                    )
                if intrinsics is None and organized_path.is_file():
                    intrinsics = _fit_pinhole_intrinsics(
                        np.load(organized_path, mmap_mode="r")
                    )
        observations.append(
            FrameObservation(
                points=aligned,
                frame_index=int(record["index"]),
                angle_degrees=float(record["degrees"]),
                source_indices=indices,
                confidence=confidence,
                image=image,
                incidence_cosine=incidence_cosine,
                normals=aligned_normals,
                camera_origin=camera_origin,
                alignment_rotation=alignment_rotation,
                rotation_origin=np.asarray(calibration["origin_mm"], dtype=np.float64),
                depth_mm=depth_mm,
                intrinsics=intrinsics,
            )
        )
    if not observations:
        raise ValueError(f"扫描清单中没有可处理的帧: {manifest_path}")
    return observations
