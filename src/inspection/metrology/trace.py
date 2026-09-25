"""Trace one recorded turntable frame back onto the registered STEP model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from inspection.reconstruction.observations import load_observations
from inspection.geometry.transforms import to_turntable

from .deviation import (
    DEFAULT_TOLERANCE_MM,
    _cloud_distances,
    _signed_distances,
    _classify_deviation,
    _distance_colors,
    _load_points,
    _direction_quality,
    _write_cloud,
)
from .model import load_mesh


@dataclass(frozen=True)
class TraceResult:
    report: dict[str, Any]
    output_dir: Path


_SIDE_ALIASES = {
    "a": ("placement_A",),
    "b": ("placement_B",),
}


def _resolve_side(root: Path, side: str) -> tuple[str, Path]:
    key = side.lower().strip()
    if key not in _SIDE_ALIASES:
        raise ValueError("side must be a or b")
    for name in _SIDE_ALIASES[key]:
        candidate = root / name
        if candidate.is_dir():
            return name, candidate
    names = " 或 ".join(str(root / name) for name in _SIDE_ALIASES[key])
    raise FileNotFoundError(f"找不到放置面采集目录: {names}")


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    candidate = root / path
    return candidate if candidate.is_file() else path


def _resolve_alignment(merge: dict[str, Any], side: str, legacy_name: str) -> np.ndarray:
    alignments = merge.get("alignments", {})
    for name in (_SIDE_ALIASES[side][0], legacy_name):
        entry = alignments.get(name)
        if entry and "step_transform" in entry:
            transform = np.asarray(entry["step_transform"], dtype=np.float64)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError(f"{name} 的 STEP 位姿矩阵无效")
            return transform
    raise KeyError(f"merge.json 中没有 {side} 侧的 STEP 位姿")


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _match_count(points: np.ndarray, path: Path, radius_mm: float) -> int:
    if not len(points) or not path.is_file():
        return 0
    target = _load_points(path)
    if not len(target):
        return 0
    tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    )
    radius_squared = float(radius_mm) ** 2
    matched = 0
    for point in np.asarray(points, dtype=np.float64):
        count, _, squared = tree.search_knn_vector_3d(point, 1)
        if count and squared[0] <= radius_squared:
            matched += 1
    return matched


def _frame_mask(
    points: np.ndarray,
    side_dir: Path,
    *,
    radius_mm: float,
) -> tuple[np.ndarray, bool]:
    """Keep frame samples represented by the formal reconstructed side cloud."""

    formal_path = side_dir / "cloud.npy"
    if not formal_path.is_file():
        formal_path = side_dir / "cloud.ply"
    if not formal_path.is_file():
        return np.ones(len(points), dtype=bool), False
    formal = _load_points(formal_path)
    if not len(formal):
        return np.zeros(len(points), dtype=bool), True
    source_geometry = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    )
    formal_geometry = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(formal.astype(np.float64, copy=False))
    )
    distances = np.asarray(
        source_geometry.compute_point_cloud_distance(formal_geometry),
        dtype=np.float64,
    )
    return distances <= float(radius_mm), True


def trace_frame(
    source: str | Path,
    *,
    side: str,
    frame_number: int,
    tolerance_mm: float | None = None,
) -> TraceResult:
    """Project one persisted capture frame into the final STEP coordinates."""

    root = Path(source)
    if not root.is_dir():
        raise NotADirectoryError(f"检测目录不存在: {root}")
    if frame_number < 1:
        raise ValueError("视角编号必须从 1 开始")

    side_key = side.lower().strip()
    legacy_name, side_dir = _resolve_side(root, side_key)
    state_path = root / "inspection.json"
    merge_path = root / "merge.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"缺少检测记录: {state_path}")
    if not merge_path.is_file():
        raise FileNotFoundError(f"缺少放置面位姿记录: {merge_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    step_path = _resolve_path(state.get("model", ""), root)
    if not step_path.is_file():
        raise FileNotFoundError(f"STEP 模型不存在: {step_path}")
    tolerance = float(
        state.get("tolerance_mm", DEFAULT_TOLERANCE_MM)
        if tolerance_mm is None
        else tolerance_mm
    )
    if tolerance <= 0:
        raise ValueError("tolerance_mm must be positive")

    manifest_path = side_dir / "run.json"
    calibration_path = side_dir / "calibration.json"
    if not manifest_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError(
            f"{side_dir} 缺少 run.json 或 calibration.json，无法复用逐帧位姿"
        )
    side_report_path = side_dir / "report.json"
    angle_sign = -1.0
    if side_report_path.is_file():
        side_report = json.loads(side_report_path.read_text(encoding="utf-8"))
        angle_sign = float(
            side_report.get("processing", {})
            .get("config", {})
            .get("angle_sign", angle_sign)
        )
    observations = load_observations(
        manifest_path,
        calibration_path,
        angle_sign=angle_sign,
    )
    if frame_number > len(observations):
        raise ValueError(
            f"{legacy_name} 只有 {len(observations)} 个视角，不能选择第 {frame_number} 个"
        )
    observation = observations[frame_number - 1]
    transform = _resolve_alignment(merge, side_key, legacy_name)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    turntable_points = to_turntable(
        observation.points,
        origin=calibration["origin_mm"],
        axis=calibration["axis"],
    )
    side_report = {}
    if side_report_path.is_file():
        side_report = json.loads(side_report_path.read_text(encoding="utf-8"))
    processing_config = side_report.get("processing", {}).get("config", {})
    formal_filter_radius = max(
        0.1,
        2.0 * float(processing_config.get("voxel_size_mm", 0.05)),
    )
    formal_mask, formal_filter_used = _frame_mask(
        turntable_points,
        side_dir,
        radius_mm=formal_filter_radius,
    )
    filtered_turntable_points = turntable_points[formal_mask]
    aligned_points = _apply_transform(filtered_turntable_points, transform)

    mesh = load_mesh(step_path, tolerance)
    distances = _cloud_distances(mesh, aligned_points)
    direction_quality = _direction_quality(mesh)
    signed_distances = (
        _signed_distances(mesh, aligned_points)
        if direction_quality["reliable"]
        else None
    )
    deviation = _classify_deviation(
        mesh,
        distances,
        tolerance,
        signed_distances=signed_distances,
        signed_distance_reliable=direction_quality["reliable"],
    )
    normal_mask = ~(deviation.problem_mask | deviation.recessed_mask)

    output_dir = root / "trace" / f"{_SIDE_ALIASES[side_key][0]}_{frame_number:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_cloud(
        output_dir / "cloud.ply",
        aligned_points[normal_mask],
        color=(0.62, 0.66, 0.72),
    )
    _write_cloud(
        output_dir / "problem.ply",
        aligned_points[deviation.problem_mask],
        colors=_distance_colors(
            distances[deviation.problem_mask], tolerance, kind="problem"
        ),
    )
    _write_cloud(
        output_dir / "recessed.ply",
        aligned_points[deviation.recessed_mask],
        colors=_distance_colors(
            distances[deviation.recessed_mask], tolerance, kind="recessed"
        ),
    )
    _write_cloud(
        output_dir / "coverage.ply",
        aligned_points[normal_mask],
        color=(0.0, 0.75, 1.0),
    )
    # 单个视角无法区分未观测的背面和 STEP 中真实缺失的区域
    # 使用空缺失点云保持渲染器接口一致
    _write_cloud(output_dir / "missing.ply", np.empty((0, 3), dtype=np.float64))
    _write_cloud(output_dir / "unobserved.ply", np.empty((0, 3), dtype=np.float64))
    o3d.io.write_triangle_mesh(str(output_dir / "mesh.ply"), mesh)
    np.save(output_dir / "cloud.npy", aligned_points.astype(np.float32))

    final_match_radius = max(0.15, tolerance)
    final_problem = root / "view" / "problem.ply"
    final_recessed = root / "view" / "recessed.ply"
    report = {
        "status": "ok",
        "processing_status": "ok",
        "status_scope": "processing_only",
        "conformance": {
            "status": "indeterminate",
            "reasons": [
                "a single frame cannot establish surface coverage",
                "feature-level acceptance rules are not configured",
                *(
                    []
                    if deviation.signed_distance_reliable
                    else ["STEP tessellation is not watertight; deviation direction is unavailable"]
                ),
            ],
        },
        "mode": "single-frame-trace",
        "source": str(root),
        "side": side_key,
        "side_directory": str(side_dir),
        "frame_number": int(frame_number),
        "frame_index": int(observation.frame_index),
        "commanded_angle_degrees": observation.angle_degrees,
        "step": str(step_path),
        "tolerance_mm": tolerance,
        "raw_points": int(len(turntable_points)),
        "formal_reconstruction_filter": {
            "used": formal_filter_used,
            "radius_mm": formal_filter_radius,
            "retained_point_count": int(len(filtered_turntable_points)),
            "removed_point_count": int(len(turntable_points) - len(filtered_turntable_points)),
            "note": "frame points are retained only when represented by the formal side cloud",
        },
        "point_cloud_count": int(len(aligned_points)),
        "mesh_direction_quality": direction_quality,
        "classification": {
            "within_tolerance_count": int(np.count_nonzero(normal_mask)),
            "mode": deviation.mode,
            "signed_distance_reliable": deviation.signed_distance_reliable,
            "unsigned_bad_count": int(np.count_nonzero(distances > tolerance)),
            "exterior_bad_count": (
                int(np.count_nonzero(deviation.exterior_mask))
                if deviation.signed_distance_reliable
                else None
            ),
            "recessed_bad_count": (
                int(np.count_nonzero(deviation.recessed_mask))
                if deviation.signed_distance_reliable
                else None
            ),
            "unclassified_bad_count": int(
                np.count_nonzero(deviation.unclassified_mask)
            ),
            "missing_not_evaluated": True,
        },
        "qualified_point_cloud": {
            "count": int(np.count_nonzero(normal_mask)),
            "ratio": float(np.mean(normal_mask)) if len(normal_mask) else 0.0,
            "color": "bright cyan blue",
            "note": "measured points within the STEP tolerance band for this view",
        },
        "final_bad_overlap": {
            "match_radius_mm": final_match_radius,
            "problem_match_count": _match_count(
                aligned_points[deviation.problem_mask],
                final_problem,
                final_match_radius,
            ),
            "exterior_match_count": (
                _match_count(
                    aligned_points[deviation.exterior_mask],
                    final_problem,
                    final_match_radius,
                )
                if deviation.signed_distance_reliable
                else None
            ),
            "recessed_match_count": _match_count(
                aligned_points[deviation.recessed_mask],
                final_recessed,
                final_match_radius,
            ),
            "note": "nearest-neighbor attribution to final bad clouds; approximate",
        },
        "alignment": {"cloud_to_step": transform.tolist()},
        "artifacts": {
            "cloud": str(output_dir / "cloud.ply"),
            "problem": str(output_dir / "problem.ply"),
            "recessed": str(output_dir / "recessed.ply"),
            "coverage": str(output_dir / "coverage.ply"),
            "missing": str(output_dir / "missing.ply"),
            "unobserved": str(output_dir / "unobserved.ply"),
            "mesh": str(output_dir / "mesh.ply"),
        },
        "note": (
            "This view classifies measured points only. STEP missing is not evaluated "
            "because a single camera angle naturally leaves occluded surface unobserved."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return TraceResult(report=report, output_dir=output_dir)
