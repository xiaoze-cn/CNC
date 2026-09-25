"""Point-cloud cleanup, reduction, and overlap measurements.

All coordinates are millimetres.  The functions are deliberately independent
of camera and motor SDKs so recorded scans can be reprocessed offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .transforms import rotate_axis



def clean_points(
    points: np.ndarray,
    *,
    z_min: float | None = None,
    z_max: float | None = None,
) -> np.ndarray:
    result = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    mask = np.isfinite(result).all(axis=1)
    if z_min is not None:
        mask &= result[:, 2] >= z_min
    if z_max is not None:
        mask &= result[:, 2] <= z_max
    return result[mask]


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
    *,
    representative: str = "medoid",
) -> np.ndarray:
    """Reduce density with a deterministic geometric representative.

    The default ``medoid`` chooses the actual point nearest the voxel centroid,
    so the result does not depend on frame or array ordering.
    """

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if representative not in {"medoid", "centroid"}:
        raise ValueError("representative must be medoid or centroid")
    points = clean_points(points)
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
    np.add.at(sums, inverse, points.astype(np.float64))
    counts = np.bincount(inverse, minlength=len(unique_keys)).astype(np.float64)
    centroids = sums / counts[:, None]
    if representative == "centroid":
        return centroids.astype(np.float32)

    distance_squared = np.sum((points.astype(np.float64) - centroids[inverse]) ** 2, axis=1)
    minimum = np.full(len(unique_keys), np.inf, dtype=np.float64)
    np.minimum.at(minimum, inverse, distance_squared)
    candidates = np.flatnonzero(
        np.isclose(distance_squared, minimum[inverse], rtol=1e-12, atol=1e-15)
    )
    # 使用坐标而不是来源和帧顺序解决完全相同的几何结果
    candidate_order = np.lexsort(
        (
            points[candidates, 2],
            points[candidates, 1],
            points[candidates, 0],
            inverse[candidates],
        )
    )
    ordered = candidates[candidate_order]
    _, first_candidate = np.unique(inverse[ordered], return_index=True)
    keep = ordered[first_candidate]
    return points[np.sort(keep)]


def write_ply(path: str | Path, points: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    finite = clean_points(points)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(finite)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(handle, finite, fmt="%.7g")
    return path


def _grid_distances(source: np.ndarray, target: np.ndarray, cell_size: float) -> np.ndarray:
    """Approximate source-to-target nearest distances using a uniform grid."""

    source = clean_points(source)
    target = clean_points(target)
    if not len(source) or not len(target):
        return np.empty(0, dtype=np.float64)
    cells: dict[tuple[int, int, int], list[int]] = {}
    keys = np.floor(target / cell_size).astype(np.int64)
    for index, key in enumerate(keys):
        cells.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(index)
    source_keys = np.floor(source / cell_size).astype(np.int64)
    distances: list[float] = []
    for point, key in zip(source, source_keys):
        candidates: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    candidates.extend(cells.get((int(key[0] + dx), int(key[1] + dy), int(key[2] + dz)), ()))
        if candidates:
            delta = target[np.asarray(candidates)] - point
            distances.append(float(np.sqrt(np.min(np.sum(delta * delta, axis=1)))))
    return np.asarray(distances, dtype=np.float64)


def overlap_metrics(
    source: np.ndarray,
    target: np.ndarray,
    *,
    voxel_size: float = 1.0,
    max_points: int = 10000,
) -> dict[str, float]:
    """Report approximate nearest-neighbour overlap error in millimetres."""

    if voxel_size <= 0 or max_points < 1:
        raise ValueError("voxel_size must be positive and max_points must be positive")
    source = voxel_downsample(clean_points(source), voxel_size)
    target = voxel_downsample(clean_points(target), voxel_size)
    if len(source) > max_points:
        source = source[np.linspace(0, len(source) - 1, max_points, dtype=np.int64)]
    distances = _grid_distances(source, target, voxel_size)
    if not len(distances):
        return {"overlap_points": 0.0, "overlap_ratio": 0.0}
    return {
        "overlap_points": float(len(distances)),
        "overlap_ratio": float(len(distances) / len(source)),
        "overlap_rms_mm": float(np.sqrt(np.mean(distances**2))),
        "overlap_p95_mm": float(np.percentile(distances, 95)),
    }


def turntable_overlap(
    manifest_path: str | Path,
    *,
    axis: Iterable[float],
    origin: Iterable[float],
    voxel_size: float = 1.0,
    angle_sign: float = -1.0,
    transformed_points: Sequence[np.ndarray] | None = None,
) -> dict[str, object]:
    """Compute adjacent-frame overlap metrics after a provisional axis transform."""

    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    if transformed_points is None:
        transformed: list[np.ndarray] = []
        for frame in payload.get("frames", []):
            points = np.load((root / frame["path"]).with_suffix(".npy"))
            transformed.append(
                rotate_axis(
                    points,
                    origin=origin,
                    axis=axis,
                    angle_degrees=angle_sign * float(frame["measured_degrees"]),
                )
            )
    else:
        transformed = [np.asarray(points) for points in transformed_points]
    pairs = []
    for index in range(len(transformed) - 1):
        pairs.append({"from_index": index, "to_index": index + 1, **overlap_metrics(transformed[index], transformed[index + 1], voxel_size=voxel_size)})
    return {"voxel_size_mm": voxel_size, "pairs": pairs}
