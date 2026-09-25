"""CAD surface visibility classification from recorded camera evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from inspection.geometry.transforms import rotation_matrix
from inspection.reconstruction.observations import fit_intrinsics


@dataclass(frozen=True, slots=True)
class CadVisibilityFrame:
    name: str
    step_to_camera: np.ndarray
    camera_origin_step: np.ndarray
    intrinsics: tuple[float, float, float, float]
    valid_depth_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class CadVisibilityResult:
    observed_mask: np.ndarray
    no_return: np.ndarray
    bad_return: np.ndarray
    occluded_mask: np.ndarray
    outside: np.ndarray
    insufficient_evidence_mask: np.ndarray
    report: dict[str, Any]


def _turntable_basis(axis: np.ndarray) -> np.ndarray:
    z_axis = np.asarray(axis, dtype=np.float64).reshape(3)
    z_axis /= np.linalg.norm(z_axis)
    camera_x = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = camera_x - np.dot(camera_x, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        camera_x = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = camera_x - np.dot(camera_x, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def _rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _step_transform(
    calibration: dict[str, Any],
    measured_angle_degrees: float,
    angle_sign: float,
    cloud_to_step: np.ndarray,
) -> np.ndarray:
    axis = np.asarray(calibration["axis"], dtype=np.float64)
    origin = np.asarray(calibration["origin_mm"], dtype=np.float64)
    alignment_rotation = rotation_matrix(
        axis, angle_sign * float(measured_angle_degrees)
    )
    camera_to_aligned = _rigid_transform(
        alignment_rotation,
        origin - alignment_rotation @ origin,
    )
    basis = _turntable_basis(axis)
    aligned_to_turntable = _rigid_transform(
        basis.T,
        -basis.T @ origin,
    )
    return (
        np.asarray(cloud_to_step, dtype=np.float64).reshape(4, 4)
        @ aligned_to_turntable
        @ camera_to_aligned
    )


def load_frames(
    placement_dir: str | Path,
    cloud_to_step: np.ndarray,
) -> tuple[list[CadVisibilityFrame], list[str]]:
    """Recover each recorded camera pose directly in STEP coordinates."""

    placement_dir = Path(placement_dir)
    manifest_path = placement_dir / "run.json"
    calibration_path = placement_dir / "calibration.json"
    report_path = placement_dir / "report.json"
    issues: list[str] = []
    if not manifest_path.is_file() or not calibration_path.is_file():
        return [], [f"{placement_dir}: run.json or calibration.json is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    angle_sign = -1.0
    if report_path.is_file():
        placement_report = json.loads(report_path.read_text(encoding="utf-8"))
        angle_sign = float(
            placement_report.get("processing", {})
            .get("config", {})
            .get("angle_sign", angle_sign)
        )
    intrinsics_by_shape: dict[
        tuple[int, int], tuple[float, float, float, float]
    ] = {}
    frames: list[CadVisibilityFrame] = []
    records = manifest.get("frames", [])
    if not records:
        return [], [f"{placement_dir}: run manifest contains no frames"]
    for record in records:
        frame_name = f"{placement_dir.name}/{int(record['index']):03d}"
        capture_dir = placement_dir / record["directory"]
        depth_path = capture_dir / "source" / "depth.npy"
        points_path = capture_dir / "source" / "points.npy"
        if not depth_path.is_file() or not points_path.is_file():
            issues.append(f"{frame_name}: source depth or organized points are missing")
            continue
        if record.get("measured_degrees") is None:
            issues.append(f"{frame_name}: measured angle is missing")
            continue
        try:
            organized_points = np.load(points_path, mmap_mode="r")
            depth = np.asarray(np.load(depth_path), dtype=np.float32) * 1000.0
            if depth.ndim != 2:
                raise ValueError("depth map is not two-dimensional")
            if organized_points.ndim != 3 or organized_points.shape[2] != 3:
                raise ValueError("organized point map must have shape HxWx3")
            grid_shape = tuple(int(value) for value in organized_points.shape[:2])
            if depth.shape != grid_shape:
                raise ValueError(
                    f"depth shape {depth.shape} does not match point map {grid_shape}"
                )
            if grid_shape not in intrinsics_by_shape:
                intrinsics_by_shape[grid_shape] = fit_intrinsics(
                    organized_points
                )
            intrinsics = intrinsics_by_shape[grid_shape]
            camera_to_step = _step_transform(
                calibration,
                float(record["measured_degrees"]),
                angle_sign,
                cloud_to_step,
            )
            frames.append(
                CadVisibilityFrame(
                    name=frame_name,
                    step_to_camera=np.linalg.inv(camera_to_step),
                    camera_origin_step=camera_to_step[:3, 3].copy(),
                    intrinsics=intrinsics,
                    valid_depth_mask=np.isfinite(depth) & (depth > 0),
                )
            )
        except (OSError, ValueError, np.linalg.LinAlgError) as exc:
            issues.append(f"{frame_name}: {exc}")
    return frames, issues


def _valid_depth(
    valid_depth: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    inside: np.ndarray,
) -> np.ndarray:
    height, width = valid_depth.shape
    result = np.zeros(len(u), dtype=bool)
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            sample_u = u + column_offset
            sample_v = v + row_offset
            valid = (
                inside
                & (sample_u >= 0)
                & (sample_u < width)
                & (sample_v >= 0)
                & (sample_v < height)
            )
            result[valid] |= valid_depth[sample_v[valid], sample_u[valid]]
    return result


def classify_visibility(
    mesh: o3d.geometry.TriangleMesh,
    sampled_points: np.ndarray,
    observed_mask: np.ndarray,
    frames: list[CadVisibilityFrame],
    *,
    surface_tolerance_mm: float,
    batch_points: int = 100_000,
    evidence_complete: bool = True,
) -> CadVisibilityResult:
    """Partition CAD samples into mutually exclusive observation states."""

    points = np.asarray(sampled_points, dtype=np.float64).reshape(-1, 3)
    observed = np.asarray(observed_mask, dtype=bool).reshape(-1)
    if len(points) != len(observed):
        raise ValueError("sampled_points and observed_mask must have equal length")
    if batch_points < 1:
        raise ValueError("batch_points must be positive")
    in_fov_any = np.zeros(len(points), dtype=bool)
    visible_any = np.zeros(len(points), dtype=bool)
    any_return = np.zeros(len(points), dtype=bool)
    if frames:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        ray_tolerance = max(0.20, 2.0 * float(surface_tolerance_mm))
        for frame in frames:
            rotation = frame.step_to_camera[:3, :3]
            translation = frame.step_to_camera[:3, 3]
            fx, fy, cx, cy = frame.intrinsics
            height, width = frame.valid_depth_mask.shape
            for start in range(0, len(points), batch_points):
                stop = min(len(points), start + batch_points)
                batch = points[start:stop]
                camera_points = (rotation @ batch.T).T + translation
                z = camera_points[:, 2]
                projectable = np.isfinite(camera_points).all(axis=1) & (z > 0)
                u_float = np.zeros(len(batch), dtype=np.float64)
                v_float = np.zeros(len(batch), dtype=np.float64)
                u_float[projectable] = fx * camera_points[projectable, 0] / z[
                    projectable
                ] + cx
                v_float[projectable] = fy * camera_points[projectable, 1] / z[
                    projectable
                ] + cy
                u = np.rint(u_float).astype(np.int64)
                v = np.rint(v_float).astype(np.int64)
                inside = (
                    projectable
                    & (u >= 0)
                    & (u < width)
                    & (v >= 0)
                    & (v < height)
                )
                global_slice = slice(start, stop)
                in_fov_any[global_slice] |= inside
                local_indices = np.flatnonzero(inside)
                if not len(local_indices):
                    continue
                vectors = batch[local_indices] - frame.camera_origin_step
                target_distance = np.linalg.norm(vectors, axis=1)
                finite_ray = np.isfinite(target_distance) & (target_distance > 0)
                if not np.any(finite_ray):
                    continue
                ray_indices = local_indices[finite_ray]
                directions = vectors[finite_ray] / target_distance[finite_ray, None]
                origins = np.repeat(
                    frame.camera_origin_step.reshape(1, 3), len(ray_indices), axis=0
                )
                rays = np.column_stack((origins, directions)).astype(np.float32)
                hit_distance = (
                    scene.cast_rays(o3d.core.Tensor(rays))["t_hit"]
                    .numpy()
                    .astype(np.float64)
                )
                line_of_sight = (
                    np.isfinite(hit_distance)
                    & (np.abs(hit_distance - target_distance[finite_ray]) <= ray_tolerance)
                )
                visible_local = np.zeros(len(batch), dtype=bool)
                visible_local[ray_indices] = line_of_sight
                valid_return = _valid_depth(
                    frame.valid_depth_mask, u, v, inside
                )
                visible_any[global_slice] |= visible_local
                any_return[global_slice] |= visible_local & valid_return

    unobserved = ~observed
    evidence_available = bool(frames) and evidence_complete
    visible_unobserved = unobserved & visible_any & evidence_available
    visible_unqualified_return = visible_unobserved & any_return
    visible_no_return = visible_unobserved & ~any_return
    occluded = unobserved & in_fov_any & ~visible_any & evidence_available
    out_of_view = unobserved & ~in_fov_any & evidence_available
    insufficient = (
        np.zeros(len(points), dtype=bool)
        if evidence_available
        else unobserved.copy()
    )
    masks = (
        observed,
        visible_no_return,
        visible_unqualified_return,
        occluded,
        out_of_view,
        insufficient,
    )
    assigned = np.sum(np.column_stack(masks), axis=1)
    if not np.all(assigned == 1):
        raise RuntimeError("CAD visibility classes must be mutually exclusive and exhaustive")
    counts = {
        "observed": int(np.count_nonzero(observed)),
        "no_return_candidate": int(np.count_nonzero(visible_no_return)),
        "visible_unqualified_return": int(
            np.count_nonzero(visible_unqualified_return)
        ),
        "occluded": int(np.count_nonzero(occluded)),
        "out_of_view": int(np.count_nonzero(out_of_view)),
        "insufficient_evidence": int(np.count_nonzero(insufficient)),
    }
    total = max(1, len(points))
    return CadVisibilityResult(
        observed_mask=observed,
        no_return=visible_no_return,
        bad_return=visible_unqualified_return,
        occluded_mask=occluded,
        outside=out_of_view,
        insufficient_evidence_mask=insufficient,
        report={
            "status": (
                "available"
                if evidence_available
                else "partial"
                if frames
                else "unavailable"
            ),
            "evidence_complete": evidence_available,
            "method": "per-frame camera projection plus CAD triangle ray casting",
            "frames": len(frames),
            "ray_tolerance": max(
                0.20, 2.0 * float(surface_tolerance_mm)
            ),
            "counts": counts,
            "ratios": {name: count / total for name, count in counts.items()},
            "missing_material_confirmed": False,
            "note": (
                "no_return_candidate is a capture follow-up candidate, not proof "
                "of missing material; visible_unqualified_return has sensor depth but no "
                "nearby point in the formal fused cloud"
            ),
        },
    )
