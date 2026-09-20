"""Visibility and spatial-connectivity evidence for reconstructed points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from inspection.geometry.pointcloud import clean_points, voxel_downsample


@dataclass(frozen=True, slots=True)
class SpatialComponentSelection:
    valid_points: np.ndarray
    uncertain_points: np.ndarray
    component_count: int
    retained_component_count: int


def classify_spatial_components(
    points: np.ndarray,
    *,
    radius_mm: float,
    min_component_ratio: float,
) -> SpatialComponentSelection:
    """Retain every spatial component large enough relative to the largest."""

    cloud = clean_points(points)
    empty = np.empty((0, 3), dtype=cloud.dtype if len(cloud) else np.float32)
    if len(cloud) < 4 or min_component_ratio <= 0:
        return SpatialComponentSelection(cloud, empty, 0, 0)

    import open3d as o3d

    geometry = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(cloud.astype(np.float64, copy=False))
    )
    labels = np.asarray(
        geometry.cluster_dbscan(eps=radius_mm, min_points=4, print_progress=False),
        dtype=np.int64,
    )
    component_labels = labels[labels >= 0]
    if not len(component_labels):
        return SpatialComponentSelection(empty, cloud, 0, 0)
    counts = np.bincount(component_labels)
    threshold = max(4, int(np.ceil(int(counts.max(initial=0)) * min_component_ratio)))
    retained_labels = counts >= threshold
    keep = (labels >= 0) & retained_labels[np.maximum(labels, 0)]
    return SpatialComponentSelection(
        cloud[keep],
        cloud[~keep],
        int(np.count_nonzero(counts)),
        int(np.count_nonzero(retained_labels)),
    )


def apply_spatial_component_evidence(
    valid_points: np.ndarray,
    uncertain_points: np.ndarray,
    config: Any,
) -> tuple[np.ndarray, np.ndarray, SpatialComponentSelection]:
    selection = classify_spatial_components(
        valid_points,
        radius_mm=config.component_radius_mm,
        min_component_ratio=config.min_component_ratio,
    )
    uncertain_parts = [uncertain_points, selection.uncertain_points]
    combined_uncertain = voxel_downsample(
        np.concatenate([part for part in uncertain_parts if len(part)], axis=0)
        if any(len(part) for part in uncertain_parts)
        else np.empty((0, 3)),
        config.voxel_size_mm,
        representative=config.voxel_representative,
    )
    return selection.valid_points, combined_uncertain, selection


def normalized_rows(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    norms = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(vectors).all(axis=1) & np.isfinite(norms) & (norms > 0)
    result = np.zeros_like(vectors)
    result[valid] = vectors[valid] / norms[valid, None]
    return result, valid


def neighbor_visibility(
    points: np.ndarray,
    observation: Any,
    *,
    occlusion_tolerance_mm: float,
    occlusion_min_neighbors: int = 1,
) -> tuple[np.ndarray, int]:
    """Test whether points are visible in a neighboring organized depth map."""

    count = len(points)
    if not 1 <= occlusion_min_neighbors <= 9:
        raise ValueError("occlusion_min_neighbors must be in [1, 9]")
    required = (
        observation.alignment_rotation,
        observation.rotation_origin,
        observation.depth_mm,
        observation.intrinsics,
    )
    if any(value is None for value in required):
        return np.ones(count, dtype=bool), 0
    rotation = np.asarray(observation.alignment_rotation, dtype=np.float64).reshape(3, 3)
    origin = np.asarray(observation.rotation_origin, dtype=np.float64).reshape(3)
    camera_points = (
        (np.asarray(points, dtype=np.float64).reshape(-1, 3) - origin) @ rotation
        + origin
    )
    fx, fy, cx, cy = observation.intrinsics
    z = camera_points[:, 2]
    projectable = np.isfinite(camera_points).all(axis=1) & (z > 0)
    u_float = np.zeros(count, dtype=np.float64)
    v_float = np.zeros(count, dtype=np.float64)
    u_float[projectable] = fx * camera_points[projectable, 0] / z[projectable] + cx
    v_float[projectable] = fy * camera_points[projectable, 1] / z[projectable] + cy
    u = np.rint(u_float).astype(np.int64)
    v = np.rint(v_float).astype(np.int64)
    depth = np.asarray(observation.depth_mm, dtype=np.float32)
    height, width = depth.shape
    inside = projectable & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    nearest_depth = np.full(count, np.inf, dtype=np.float64)
    closer_neighbors = np.zeros(count, dtype=np.int8)
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            sample_u = u + column_offset
            sample_v = v + row_offset
            sample_valid = (
                inside
                & (sample_u >= 0)
                & (sample_u < width)
                & (sample_v >= 0)
                & (sample_v < height)
            )
            values = np.full(count, np.inf, dtype=np.float64)
            values[sample_valid] = depth[sample_v[sample_valid], sample_u[sample_valid]]
            values[~np.isfinite(values) | (values <= 0)] = np.inf
            nearest_depth = np.minimum(nearest_depth, values)
            closer_neighbors += (values < z - occlusion_tolerance_mm).astype(np.int8)
    occluded = inside & (closer_neighbors >= occlusion_min_neighbors)
    return inside & ~occluded, int(np.count_nonzero(occluded))
