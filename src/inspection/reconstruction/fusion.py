"""Single-side point-cloud fusion and result persistence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from inspection.geometry.pointcloud import (
    clean_points,
    turntable_overlap_metrics,
    voxel_downsample,
    write_ascii_ply,
)
from inspection.geometry.transforms import to_turntable_coordinates

from .evidence import (
    apply_spatial_component_evidence,
    neighbor_visibility,
    normalized_rows,
)
from .observations import FrameObservation, load_frame_observations


CPU_WORKERS = max(2, min(4, os.cpu_count() or 2))


@dataclass(frozen=True, slots=True)
class PointCloudProcessingConfig:
    """All parameters that can affect a single-side point-cloud result."""

    voxel_size_mm: float = 0.05
    angle_sign: float = -1.0
    support_radius_mm: float = 0.15
    min_views: int = 3
    min_support_angle_degrees: float = 0.0
    max_support_angle_degrees: float | None = None
    min_normal_consistency_cosine: float | None = 0.60
    require_same_observation_side: bool = True
    use_occlusion_visibility: bool = True
    occlusion_tolerance_mm: float = 0.50
    occlusion_min_neighbors: int = 1
    overlap_voxel_size_mm: float = 1.0
    voxel_representative: str = "medoid"
    confidence_min: float | None = 0.40
    edge_uncertain_px: float = 1.0
    image_gradient_threshold: float = 32.0
    depth_gradient_threshold_mm: float = 0.50
    normal_change_cosine: float = 0.85
    confidence_drop_ratio: float = 0.75
    confidence_drop_abs: float = 0.10
    edge_policy: str = "hard"
    min_incidence_cosine: float | None = None
    component_radius_mm: float = 0.30
    min_component_ratio: float = 0.005

    @classmethod
    def production(
        cls,
        *,
        voxel_size_mm: float = 0.05,
        angle_sign: float = -1.0,
    ) -> "PointCloudProcessingConfig":
        """Return the released profile used by formal scans.

        Edge evidence is hard by default for conservative formal measurement.
        The soft policy remains available for offline investigation.
        """

        return cls(voxel_size_mm=voxel_size_mm, angle_sign=angle_sign)

    def validate(self) -> None:
        if self.voxel_size_mm <= 0:
            raise ValueError("voxel_size_mm must be positive")
        if self.angle_sign == 0:
            raise ValueError("angle_sign must be non-zero")
        if self.support_radius_mm <= 0:
            raise ValueError("support_radius_mm must be positive")
        if self.min_views < 1:
            raise ValueError("min_views must be at least 1")
        if not 0 <= self.min_support_angle_degrees <= 180:
            raise ValueError("min_support_angle_degrees must be in [0, 180]")
        if (
            self.max_support_angle_degrees is not None
            and not 0 < self.max_support_angle_degrees <= 180
        ):
            raise ValueError(
                "max_support_angle_degrees must be in (0, 180] when provided"
            )
        if (
            self.max_support_angle_degrees is not None
            and self.min_support_angle_degrees > self.max_support_angle_degrees
        ):
            raise ValueError(
                "min_support_angle_degrees cannot exceed max_support_angle_degrees"
            )
        if (
            self.min_normal_consistency_cosine is not None
            and not 0 <= self.min_normal_consistency_cosine <= 1
        ):
            raise ValueError("min_normal_consistency_cosine must be in [0, 1]")
        if self.occlusion_tolerance_mm < 0:
            raise ValueError("occlusion_tolerance_mm must be non-negative")
        if self.occlusion_min_neighbors < 1 or self.occlusion_min_neighbors > 9:
            raise ValueError("occlusion_min_neighbors must be in [1, 9]")
        if self.overlap_voxel_size_mm <= 0:
            raise ValueError("overlap_voxel_size_mm must be positive")
        if self.voxel_representative not in {"medoid", "centroid"}:
            raise ValueError("voxel_representative must be medoid or centroid")
        if self.confidence_min is not None and not np.isfinite(self.confidence_min):
            raise ValueError("confidence_min must be finite when provided")
        if self.edge_uncertain_px < 0:
            raise ValueError("edge_uncertain_px must be non-negative")
        if self.image_gradient_threshold < 0:
            raise ValueError("image_gradient_threshold must be non-negative")
        if self.depth_gradient_threshold_mm < 0:
            raise ValueError("depth_gradient_threshold_mm must be non-negative")
        if not 0 <= self.normal_change_cosine <= 1:
            raise ValueError("normal_change_cosine must be in [0, 1]")
        if not 0 < self.confidence_drop_ratio <= 1:
            raise ValueError("confidence_drop_ratio must be in (0, 1]")
        if self.confidence_drop_abs < 0:
            raise ValueError("confidence_drop_abs must be non-negative")
        if self.edge_policy not in {"soft", "hard", "off"}:
            raise ValueError("edge_policy must be soft, hard, or off")
        if self.min_incidence_cosine is not None and not 0 <= self.min_incidence_cosine <= 1:
            raise ValueError("min_incidence_cosine must be in [0, 1] when provided")
        if self.component_radius_mm <= 0:
            raise ValueError("component_radius_mm must be positive")
        if not 0 <= self.min_component_ratio < 1:
            raise ValueError("min_component_ratio must be in [0, 1)")

    def data(self) -> dict[str, Any]:
        return {
            "profile": "production" if self == self.production(
                voxel_size_mm=self.voxel_size_mm,
                angle_sign=self.angle_sign,
            ) else "custom",
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class PointCloudProcessingResult:
    points: np.ndarray
    report: dict[str, Any]
    uncertain_points: np.ndarray
    likely_noise_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )


@dataclass(frozen=True, slots=True)
class MultiviewEvidenceSelection:
    """Candidate partition produced before the final coordinate conversion."""

    valid_points: np.ndarray
    uncertain_points: np.ndarray
    candidate_points: int
    valid_candidates: int
    uncertain_candidates: int
    sensor_invalid_candidates: int = 0
    spatial_component_candidates: int = 0
    spatial_components: int = 0
    retained_spatial_components: int = 0
    confidence_invalid_candidates: int = 0
    edge_invalid_candidates: int = 0
    incidence_invalid_candidates: int = 0
    likely_noise_candidates: int = 0
    likely_noise_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    occluded_view_tests: int = 0
    normal_rejected_matches: int = 0
    opposite_side_rejected_matches: int = 0
    image_gradient_candidates: int = 0
    depth_gradient_candidates: int = 0
    normal_change_candidates: int = 0
    confidence_drop_candidates: int = 0
    soft_edge_supported_candidates: int = 0
    compute_device_name: str = "cpu"
    effective_max_support_angle_degrees: float | None = None


def _effective_max_support_angle(
    config: PointCloudProcessingConfig,
    frame_angles_degrees: Sequence[float | None],
) -> float:
    """Derive the smallest window that gives every frame enough support views."""

    configured = config.max_support_angle_degrees
    required_support = config.min_views - 1
    if configured is not None:
        return configured
    if required_support <= 0:
        return 180.0
    if any(angle is None for angle in frame_angles_degrees):
        raise ValueError("自动视角证据匹配要求每个非空帧都包含采集角度")
    if len(frame_angles_degrees) < config.min_views:
        raise ValueError(
            f"正式测量至少需要 {config.min_views} 个非空采集帧，"
            f"当前只有 {len(frame_angles_degrees)} 个"
        )

    angles = np.asarray(frame_angles_degrees, dtype=np.float64)
    required_angles: list[float] = []
    for index, angle in enumerate(angles):
        other_angles = np.delete(angles, index)
        separations = np.abs((other_angles - angle + 180.0) % 360.0 - 180.0)
        if len(separations) < required_support:
            return configured
        partitioned = np.partition(separations, required_support - 1)
        required_angles.append(float(partitioned[required_support - 1]))

    return max(required_angles)


def _observation_grid_shape(observation: FrameObservation) -> tuple[int, int] | None:
    """Return the source sensor grid shape used by point indices, when available."""

    for value in (observation.image, observation.depth_mm):
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim >= 2:
            height, width = array.shape[:2]
            if height > 0 and width > 0:
                return int(height), int(width)
    return None


def _scatter_source_values(
    values: np.ndarray,
    indices: np.ndarray,
    size: int,
) -> np.ndarray:
    """Place selected point values back on their organized source grid."""

    result = np.full(size, np.nan, dtype=np.float32)
    valid = (indices >= 0) & (indices < size)
    result[indices[valid]] = np.asarray(values, dtype=np.float32).reshape(-1)[valid]
    return result


def _neighbor_difference_edges(values: np.ndarray, threshold: float) -> np.ndarray:
    """Mark pixels whose 4-connected neighbor differs by more than a threshold."""

    grid = np.asarray(values, dtype=np.float32)
    edges = np.zeros(grid.shape, dtype=bool)
    if grid.ndim != 2:
        return edges
    finite = np.isfinite(grid)
    horizontal_valid = finite[:, :-1] & finite[:, 1:]
    horizontal = np.where(
        horizontal_valid,
        np.abs(grid[:, :-1] - grid[:, 1:]),
        0.0,
    )
    vertical_valid = finite[:-1, :] & finite[1:, :]
    vertical = np.where(
        vertical_valid,
        np.abs(grid[:-1, :] - grid[1:, :]),
        0.0,
    )
    edges[:, :-1] |= horizontal >= threshold
    edges[:, 1:] |= horizontal >= threshold
    edges[:-1, :] |= vertical >= threshold
    edges[1:, :] |= vertical >= threshold
    return edges


def _normal_change_edges(
    normals: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Mark pixels whose neighboring normals change beyond a cosine threshold."""

    grid = np.asarray(normals, dtype=np.float32)
    edges = np.zeros(grid.shape[:2], dtype=bool)
    if grid.ndim != 3 or grid.shape[-1] != 3:
        return edges
    length = np.linalg.norm(grid, axis=2)
    valid = np.isfinite(grid).all(axis=2) & (length > 1e-8)
    normalized = np.zeros_like(grid)
    normalized[valid] = grid[valid] / length[valid, None]
    for first, second, first_slice, second_slice in (
        (normalized[:, :-1], normalized[:, 1:], (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        (normalized[:-1, :], normalized[1:, :], (slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        pair_valid = valid[first_slice] & valid[second_slice]
        cosine = np.abs(np.einsum("ij,ij->i", first.reshape(-1, 3), second.reshape(-1, 3))).reshape(pair_valid.shape)
        changed = pair_valid & (cosine < threshold)
        edges[first_slice] |= changed
        edges[second_slice] |= changed
    return edges


def _confidence_drop_edges(confidence: np.ndarray, ratio: float, absolute: float) -> np.ndarray:
    """Mark confidence values that fall sharply below their local 4-neighbor level."""

    grid = np.asarray(confidence, dtype=np.float32)
    edges = np.zeros(grid.shape, dtype=bool)
    if grid.ndim != 2:
        return edges
    neighbors: list[np.ndarray] = []
    for source, target in (
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        shifted = np.full_like(grid, np.nan)
        shifted[target] = grid[source]
        neighbors.append(shifted)
    stack = np.stack(neighbors, axis=0)
    finite_count = np.isfinite(stack).sum(axis=0)
    local_median = np.full(grid.shape, np.nan, dtype=np.float32)
    valid = finite_count > 0
    local_median[valid] = np.nanmedian(stack[:, valid], axis=0)
    drop = local_median - grid
    return (
        np.isfinite(grid)
        & valid
        & (drop >= absolute)
        & (grid <= local_median * ratio)
    )


def _edge_evidence(
    observation: FrameObservation,
    point_count: int,
    config: PointCloudProcessingConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    """Combine image, depth, normal, and local-confidence edge evidence."""

    combined = np.zeros(point_count, dtype=bool)
    counts = {
        "image_gradient_candidates": 0,
        "depth_gradient_candidates": 0,
        "normal_change_candidates": 0,
        "confidence_drop_candidates": 0,
    }
    if observation.source_indices is None or len(observation.source_indices) != point_count:
        return combined, counts
    indices = np.asarray(observation.source_indices, dtype=np.int64).reshape(-1)
    grid_shape = _observation_grid_shape(observation)
    if grid_shape is None:
        return combined, counts
    grid_size = grid_shape[0] * grid_shape[1]
    source_valid = (indices >= 0) & (indices < grid_size)
    if not np.any(source_valid):
        return combined, counts
    safe_indices = np.clip(indices, 0, max(0, grid_size - 1))

    evidence: dict[str, np.ndarray] = {}
    if observation.image is not None:
        import cv2

        image = np.asarray(observation.image)
        if image.ndim == 3:
            if image.shape[2] >= 3:
                image = cv2.cvtColor(
                    np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
                )
            else:
                image = image[..., 0]
        if image.ndim == 2 and image.shape[:2] == grid_shape:
            image_float = np.nan_to_num(image.astype(np.float32), nan=0.0)
            gradient_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)
            gradient = np.hypot(gradient_x, gradient_y)
            gradient_edges = (
                gradient >= config.image_gradient_threshold
                if config.image_gradient_threshold > 0
                else np.zeros_like(gradient, dtype=bool)
            )
            canny = cv2.Canny(
                np.clip(image_float, 0, 255).astype(np.uint8), 40, 120
            )
            distance = cv2.distanceTransform(
                (canny == 0).astype(np.uint8), cv2.DIST_L2, 3
            )
            evidence["image_gradient_candidates"] = (
                (gradient_edges | (distance <= config.edge_uncertain_px))
                .reshape(-1)[safe_indices]
            )

    if observation.depth_mm is not None:
        depth = np.asarray(observation.depth_mm, dtype=np.float32)
        if depth.ndim == 2 and depth.shape[:2] == grid_shape:
            if config.depth_gradient_threshold_mm > 0:
                evidence["depth_gradient_candidates"] = _neighbor_difference_edges(
                    depth, config.depth_gradient_threshold_mm
                ).reshape(-1)[safe_indices]

    if observation.normals is not None:
        normals = np.asarray(observation.normals, dtype=np.float32).reshape(-1, 3)
        if len(normals) == point_count:
            normal_grid = np.full((grid_size, 3), np.nan, dtype=np.float32)
            normal_grid[indices[source_valid]] = normals[source_valid]
            evidence["normal_change_candidates"] = _normal_change_edges(
                normal_grid.reshape(*grid_shape, 3), config.normal_change_cosine
            ).reshape(-1)[safe_indices]

    if observation.confidence is not None:
        confidence = np.asarray(observation.confidence, dtype=np.float32).reshape(-1)
        if len(confidence) == point_count:
            confidence_grid = _scatter_source_values(confidence, indices, grid_size)
            evidence["confidence_drop_candidates"] = _confidence_drop_edges(
                confidence_grid.reshape(grid_shape),
                config.confidence_drop_ratio,
                config.confidence_drop_abs,
            ).reshape(-1)[safe_indices]

    for name, values in evidence.items():
        values = np.asarray(values, dtype=bool)
        values &= source_valid
        counts[name] = int(np.count_nonzero(values))
        combined |= values
    return combined, counts


@lru_cache(maxsize=1)
def _cuda_runtime_probe() -> bool:
    """Check a CUDA kernel in a child process so a broken driver cannot hang us."""

    probe = (
        "import numpy as np, open3d as o3d; "
        "d=o3d.core.Device('CUDA:0'); "
        "x=o3d.core.Tensor(np.zeros((1, 3), dtype=np.float32), device=d); "
        "(x + x).cpu()"
    )
    try:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(
            [sys.executable, "-c", probe],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _resolve_compute_device(*, prefer_gpu: bool = False) -> tuple[object, str]:
    """Keep the CUDA path available while defaulting formal processing to CPU."""

    import open3d as o3d

    if not prefer_gpu:
        return o3d.core.Device("CPU:0"), "cpu"

    try:
        cuda_available = bool(o3d.core.cuda.is_available())
        cuda_available &= o3d.core.cuda.device_count() > 0
    except Exception:
        cuda_available = False
    if cuda_available and _cuda_runtime_probe():
        return o3d.core.Device("CUDA:0"), "cuda"
    return o3d.core.Device("CPU:0"), "cpu"


def multiview_evidence_processor(
    frames: Sequence[np.ndarray],
    config: PointCloudProcessingConfig,
    *,
    observations: Sequence[FrameObservation] | None = None,
    prefer_gpu: bool = False,
) -> MultiviewEvidenceSelection:
    """Classify observations using cross-view evidence before reducing them.

    Points without enough independent-view support are retained as uncertain
    observations. They are excluded from the formal cloud but are written to a
    separate artifact for review instead of being silently discarded.
    """

    # 保留所有有限观测用于证据判断
    # 体素降采样只在支持分类后进行因此不会改变真实观测是否获得支持
    clouds: list[np.ndarray] = []
    cloud_observations: list[FrameObservation | None] = []
    frame_angles_degrees: list[float | None] = []
    sensor_masks: list[np.ndarray] = []
    hard_sensor_masks: list[np.ndarray] = []
    edge_masks: list[np.ndarray] = []
    sensor_invalid_candidates = 0
    confidence_invalid_candidates = 0
    edge_invalid_candidates = 0
    incidence_invalid_candidates = 0
    image_gradient_candidates = 0
    depth_gradient_candidates = 0
    normal_change_candidates = 0
    confidence_drop_candidates = 0
    for frame_index, frame in enumerate(frames):
        cloud = clean_points(frame)
        if len(cloud):
            clouds.append(cloud)
            observation = (
                observations[frame_index]
                if observations is not None and frame_index < len(observations)
                else None
            )
            cloud_observations.append(observation)
            frame_angles_degrees.append(
                observation.angle_degrees
                if observation is not None
                else None
            )
            mask = np.ones(len(cloud), dtype=bool)
            hard_mask = np.ones(len(cloud), dtype=bool)
            edge_mask = np.zeros(len(cloud), dtype=bool)
            if observation is not None:
                if config.confidence_min is not None and observation.confidence is not None:
                    confidence = np.asarray(observation.confidence).reshape(-1)
                    if len(confidence) == len(cloud):
                        confidence_valid = np.isfinite(confidence) & (
                            confidence >= config.confidence_min
                        )
                        confidence_invalid_candidates += int(
                            np.count_nonzero(~confidence_valid)
                        )
                        mask &= confidence_valid
                        hard_mask &= confidence_valid
                if config.edge_policy != "off" and config.edge_uncertain_px > 0:
                    edge_candidates, edge_counts = _edge_evidence(
                        observation, len(cloud), config
                    )
                    edge_valid = ~edge_candidates
                    edge_mask = edge_candidates
                    edge_invalid_candidates += int(np.count_nonzero(edge_candidates))
                    image_gradient_candidates += edge_counts["image_gradient_candidates"]
                    depth_gradient_candidates += edge_counts["depth_gradient_candidates"]
                    normal_change_candidates += edge_counts["normal_change_candidates"]
                    confidence_drop_candidates += edge_counts["confidence_drop_candidates"]
                    if config.edge_policy == "hard":
                        mask &= edge_valid
                if (
                    config.min_incidence_cosine is not None
                    and observation.incidence_cosine is not None
                ):
                    incidence = np.asarray(observation.incidence_cosine).reshape(-1)
                    if len(incidence) == len(cloud):
                        incidence_valid = np.isfinite(incidence) & (
                            incidence >= config.min_incidence_cosine
                        )
                        incidence_invalid_candidates += int(
                            np.count_nonzero(~incidence_valid)
                        )
                        mask &= incidence_valid
                        hard_mask &= incidence_valid
            sensor_masks.append(mask)
            hard_sensor_masks.append(hard_mask)
            edge_masks.append(edge_mask)
            sensor_invalid_candidates += int(np.count_nonzero(~mask))
    if not clouds:
        return MultiviewEvidenceSelection(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            0,
            0,
            0,
            sensor_invalid_candidates,
            confidence_invalid_candidates=confidence_invalid_candidates,
            edge_invalid_candidates=edge_invalid_candidates,
            incidence_invalid_candidates=incidence_invalid_candidates,
            image_gradient_candidates=image_gradient_candidates,
            depth_gradient_candidates=depth_gradient_candidates,
            normal_change_candidates=normal_change_candidates,
            confidence_drop_candidates=confidence_drop_candidates,
        )
    if config.min_views > len(clouds):
        raise ValueError("min_views must be between 1 and the number of non-empty frames")
    if config.min_views == 1 or len(clouds) == 1:
        formal_masks = (
            [hard & ~edge for hard, edge in zip(hard_sensor_masks, edge_masks)]
            if config.edge_policy == "soft"
            else sensor_masks
        )
        valid_parts = [cloud[mask] for cloud, mask in zip(clouds, formal_masks)]
        uncertain_parts = [cloud[~mask] for cloud, mask in zip(clouds, formal_masks)]
        all_points = voxel_downsample(
            np.concatenate(valid_parts, axis=0),
            config.voxel_size_mm,
            representative=config.voxel_representative,
        )
        uncertain_points = voxel_downsample(
            np.concatenate(uncertain_parts, axis=0)
            if any(len(part) for part in uncertain_parts)
            else np.empty((0, 3)),
            config.voxel_size_mm,
            representative=config.voxel_representative,
        )
        valid_points, uncertain_points, components = apply_spatial_component_evidence(
            all_points, uncertain_points, config
        )
        return MultiviewEvidenceSelection(
            valid_points,
            uncertain_points,
            int(sum(len(cloud) for cloud in clouds)),
            int(sum(len(part) for part in valid_parts)),
            int(sum(len(part) for part in uncertain_parts)),
            sensor_invalid_candidates,
            int(len(components.uncertain_points)),
            components.component_count,
            components.retained_component_count,
            confidence_invalid_candidates,
            edge_invalid_candidates,
            incidence_invalid_candidates,
            image_gradient_candidates=image_gradient_candidates,
            depth_gradient_candidates=depth_gradient_candidates,
            normal_change_candidates=normal_change_candidates,
            confidence_drop_candidates=confidence_drop_candidates,
        )

    import open3d as o3d
    compute_device, device_name = _resolve_compute_device(prefer_gpu=prefer_gpu)

    searches: list[object] = []
    for cloud in clouds:
        search = o3d.core.nns.NearestNeighborSearch(
            o3d.core.Tensor(
                cloud, dtype=o3d.core.Dtype.Float32, device=compute_device
            )
        )
        search.knn_index()
        searches.append(search)
    normalized_normals: list[tuple[np.ndarray, np.ndarray] | None] = []
    for cloud, observation in zip(clouds, cloud_observations):
        if observation is not None and observation.normals is not None:
            normals = np.asarray(observation.normals).reshape(-1, 3)
            normalized_normals.append(
                normalized_rows(normals) if len(normals) == len(cloud) else None
            )
        else:
            normalized_normals.append(None)
    required_support = config.min_views - 1
    if (
        required_support > 0
        and (
            config.min_support_angle_degrees > 0
            or config.max_support_angle_degrees is None
            or config.max_support_angle_degrees < 180
        )
    ) and any(
        angle is None for angle in frame_angles_degrees
    ):
        raise ValueError(
            "视角证据门槛要求每个非空帧都包含采集角度"
        )
    effective_max_support_angle = _effective_max_support_angle(
        config, frame_angles_degrees
    )
    valid_parts: list[np.ndarray] = []
    uncertain_parts: list[np.ndarray] = []
    likely_noise_screen_parts: list[np.ndarray] = []
    valid_candidates = 0
    uncertain_candidates = 0
    likely_noise_candidates = 0
    soft_edge_supported_candidates = 0
    occluded_view_tests = 0
    normal_rejected_matches = 0
    opposite_side_rejected_matches = 0
    executor = (
        ThreadPoolExecutor(max_workers=CPU_WORKERS)
        if device_name == "cpu"
        else None
    )

    def _run_search(
        item: tuple[int, object], query_tensor: object
    ) -> tuple[int, np.ndarray, np.ndarray]:
        other_index, search = item
        indices_tensor, squared_distance_tensor = search.knn_search(query_tensor, 1)
        return (
            other_index,
            indices_tensor.cpu().numpy().reshape(-1),
            squared_distance_tensor.cpu().numpy().reshape(-1),
        )

    for index, cloud in enumerate(clouds):
        support_count = np.zeros(len(cloud), dtype=np.int64)
        observable_count = np.zeros(len(cloud), dtype=np.int64)
        query = o3d.core.Tensor(
            cloud, dtype=o3d.core.Dtype.Float32, device=compute_device
        )
        eligible_searches: list[tuple[int, object]] = []
        for other_index, search in enumerate(searches):
            if other_index == index:
                continue
            if frame_angles_degrees[index] is None or frame_angles_degrees[other_index] is None:
                separation = 0.0
            else:
                separation = abs(
                    (
                        float(frame_angles_degrees[other_index])
                        - float(frame_angles_degrees[index])
                        + 180.0
                    )
                    % 360.0
                    - 180.0
                )
            if not (
                config.min_support_angle_degrees - 1e-9
                <= separation
                <= effective_max_support_angle + 1e-9
            ):
                continue
            eligible_searches.append((other_index, search))
        if executor is None:
            search_results = (_run_search(item, query) for item in eligible_searches)
        else:
            search_results = executor.map(
                lambda item: _run_search(item, query), eligible_searches
            )
        for other_index, neighbor_indices, squared_distances in search_results:
            other_observation = cloud_observations[other_index]
            if not config.use_occlusion_visibility or other_observation is None:
                observable = np.ones(len(cloud), dtype=bool)
            else:
                observable, occluded = neighbor_visibility(
                    cloud,
                    other_observation,
                    occlusion_tolerance_mm=config.occlusion_tolerance_mm,
                    occlusion_min_neighbors=config.occlusion_min_neighbors,
                )
                occluded_view_tests += occluded
            observable_count += observable
            target_sensor_valid = hard_sensor_masks[other_index][neighbor_indices]
            match = observable & target_sensor_valid & (
                squared_distances <= config.support_radius_mm**2
            )
            source_normal_data = normalized_normals[index]
            target_normal_data = normalized_normals[other_index]
            if (
                config.min_normal_consistency_cosine is not None
                and source_normal_data is not None
                and target_normal_data is not None
            ):
                source_normals, source_normal_valid = source_normal_data
                target_normals, target_normal_valid = target_normal_data
                matched_target_normals = target_normals[neighbor_indices]
                both_valid = source_normal_valid & target_normal_valid[neighbor_indices]
                normal_cosine = np.abs(
                    np.einsum("ij,ij->i", source_normals, matched_target_normals)
                )
                normal_consistent = ~both_valid | (
                    normal_cosine >= config.min_normal_consistency_cosine
                )
                normal_rejected_matches += int(
                    np.count_nonzero(match & ~normal_consistent)
                )
                match &= normal_consistent
            if (
                config.require_same_observation_side
                and source_normal_data is not None
                and cloud_observations[index] is not None
                and cloud_observations[index].camera_origin is not None
                and other_observation is not None
                and other_observation.camera_origin is not None
            ):
                source_normals, source_normal_valid = source_normal_data
                source_views, source_view_valid = normalized_rows(
                    np.asarray(cloud_observations[index].camera_origin) - cloud
                )
                target_views, target_view_valid = normalized_rows(
                    np.asarray(other_observation.camera_origin) - cloud
                )
                source_facing = np.einsum("ij,ij->i", source_normals, source_views)
                target_facing = np.einsum("ij,ij->i", source_normals, target_views)
                direction_valid = (
                    source_normal_valid & source_view_valid & target_view_valid
                )
                same_side = ~direction_valid | (source_facing * target_facing > 0)
                opposite_side_rejected_matches += int(
                    np.count_nonzero(match & ~same_side)
                )
                match &= same_side
            support_count += match
        supported = support_count >= required_support
        if config.edge_policy == "soft":
            edge_recovery = supported & edge_masks[index] & hard_sensor_masks[index]
            soft_edge_supported_candidates += int(np.count_nonzero(edge_recovery))
            valid = supported & hard_sensor_masks[index]
        else:
            valid = supported & sensor_masks[index]
        likely_noise_screen = (
            (support_count == 0)
            & (observable_count >= required_support)
            & ~sensor_masks[index]
        )
        valid_parts.append(clouds[index][valid])
        uncertain_parts.append(clouds[index][~valid & ~likely_noise_screen])
        likely_noise_screen_parts.append(clouds[index][likely_noise_screen])
        valid_candidates += int(np.count_nonzero(valid))
        uncertain_candidates += int(np.count_nonzero(~valid))
        likely_noise_candidates += int(np.count_nonzero(likely_noise_screen))

    if executor is not None:
        executor.shutdown(wait=True)

    valid_points = voxel_downsample(
        np.concatenate(valid_parts, axis=0) if valid_parts else np.empty((0, 3)),
        config.voxel_size_mm,
        representative=config.voxel_representative,
    )
    uncertain_points = voxel_downsample(
        np.concatenate(uncertain_parts, axis=0)
        if uncertain_parts and any(len(part) for part in uncertain_parts)
        else np.empty((0, 3)),
        config.voxel_size_mm,
        representative=config.voxel_representative,
    )
    likely_noise_screen_points = voxel_downsample(
        np.concatenate(likely_noise_screen_parts, axis=0)
        if any(len(part) for part in likely_noise_screen_parts)
        else np.empty((0, 3)),
        config.voxel_size_mm,
        representative=config.voxel_representative,
    )
    likely_noise_points = np.empty((0, 3), dtype=np.float32)
    if len(likely_noise_screen_points) and len(valid_points):
        screened_geometry = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                likely_noise_screen_points.astype(np.float64, copy=False)
            )
        )
        valid_geometry = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(valid_points.astype(np.float64, copy=False))
        )
        confirmed_distance = np.asarray(
            screened_geometry.compute_point_cloud_distance(valid_geometry),
            dtype=np.float64,
        )
        isolated = confirmed_distance > config.component_radius_mm
        likely_noise_points = likely_noise_screen_points[isolated]
        near_confirmed = likely_noise_screen_points[~isolated]
        uncertain_points = voxel_downsample(
            np.concatenate((uncertain_points, near_confirmed), axis=0),
            config.voxel_size_mm,
            representative=config.voxel_representative,
        )
    elif len(likely_noise_screen_points):
        uncertain_points = voxel_downsample(
            np.concatenate((uncertain_points, likely_noise_screen_points), axis=0),
            config.voxel_size_mm,
            representative=config.voxel_representative,
        )
    valid_points, uncertain_points, components = apply_spatial_component_evidence(
        valid_points, uncertain_points, config
    )
    return MultiviewEvidenceSelection(
        valid_points,
        uncertain_points,
        valid_candidates + uncertain_candidates,
        valid_candidates,
        uncertain_candidates,
        sensor_invalid_candidates,
        int(len(components.uncertain_points)),
        components.component_count,
        components.retained_component_count,
        confidence_invalid_candidates,
        edge_invalid_candidates,
        incidence_invalid_candidates,
        likely_noise_candidates,
        likely_noise_points,
        occluded_view_tests,
        normal_rejected_matches,
        opposite_side_rejected_matches,
        image_gradient_candidates,
        depth_gradient_candidates,
        normal_change_candidates,
        confidence_drop_candidates,
        soft_edge_supported_candidates,
        device_name,
        effective_max_support_angle,
    )


def process_point_cloud(
    manifest_path: str | Path,
    calibration_path: str | Path,
    config: PointCloudProcessingConfig | None = None,
    *,
    prefer_gpu: bool = False,
) -> PointCloudProcessingResult:
    """Process one recorded side with the released multiview evidence pipeline."""

    manifest_path = Path(manifest_path)
    calibration_path = Path(calibration_path)
    config = config or PointCloudProcessingConfig.production()
    config.validate()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    observations = load_frame_observations(
        manifest_path,
        calibration_path,
        angle_sign=config.angle_sign,
    )
    loaded_seconds = time.perf_counter() - started
    frames = [observation.points for observation in observations]
    evidence_started = time.perf_counter()
    selection = multiview_evidence_processor(
        frames, config, observations=observations, prefer_gpu=prefer_gpu
    )
    effective_max_support_angle = (
        selection.effective_max_support_angle_degrees
        if selection.effective_max_support_angle_degrees is not None
        else (
            config.max_support_angle_degrees
            if config.max_support_angle_degrees is not None
            else 180.0
        )
    )
    evidence_seconds = time.perf_counter() - evidence_started
    processed = clean_points(selection.valid_points)
    uncertain = clean_points(selection.uncertain_points)
    likely_noise = clean_points(selection.likely_noise_points)
    if not len(processed):
        if len(uncertain):
            raise ValueError(
                "点云没有满足正式测量证据门槛的点；不确定点已保留，请调整质量门槛或补拍"
            )
        raise ValueError("点云处理结果为空")
    points = to_turntable_coordinates(
        processed,
        origin=calibration["origin_mm"],
        axis=calibration["axis"],
    ).astype(np.float32, copy=False)
    uncertain_points = to_turntable_coordinates(
        uncertain,
        origin=calibration["origin_mm"],
        axis=calibration["axis"],
    ).astype(np.float32, copy=False)
    likely_noise_points = to_turntable_coordinates(
        likely_noise,
        origin=calibration["origin_mm"],
        axis=calibration["axis"],
    ).astype(np.float32, copy=False)
    coordinate_seconds = time.perf_counter() - evidence_started - evidence_seconds
    metrics = turntable_overlap_metrics(
        manifest_path,
        axis=calibration["axis"],
        origin=calibration["origin_mm"],
        voxel_size=config.overlap_voxel_size_mm,
        angle_sign=config.angle_sign,
        transformed_points=frames,
    )
    metrics["coordinate_system"] = {
        "name": "turntable",
        "unit": "mm",
        "origin": "rotation_axis_at_calibrated_marker_plane",
        "x": "camera_image_right",
        "y": "away_from_camera",
        "z": "above_turntable",
    }
    metrics["processing"] = {
        "processor": "multiview_evidence",
        "config": config.data(),
        "input_frames": len(frames),
        "input_points": int(sum(len(frame) for frame in frames)),
        "output_points": int(len(points)),
        "candidate_points": selection.candidate_points,
        "valid_candidates": selection.valid_candidates,
        "uncertain_candidates": selection.uncertain_candidates,
        "sensor_invalid_candidates": selection.sensor_invalid_candidates,
        "confidence_invalid_candidates": selection.confidence_invalid_candidates,
        "edge_invalid_candidates": selection.edge_invalid_candidates,
        "image_gradient_candidates": selection.image_gradient_candidates,
        "depth_gradient_candidates": selection.depth_gradient_candidates,
        "normal_change_candidates": selection.normal_change_candidates,
        "confidence_drop_candidates": selection.confidence_drop_candidates,
        "soft_edge_supported_candidates": selection.soft_edge_supported_candidates,
        "compute_device": selection.compute_device_name,
        "support_angle_degrees": {
            "mode": (
                "automatic"
                if config.max_support_angle_degrees is None
                else "fixed"
            ),
            "configured_maximum": config.max_support_angle_degrees,
            "effective_maximum": effective_max_support_angle,
        },
        "timing_seconds": {
            "load_and_transform": round(loaded_seconds, 3),
            "evidence_matching": round(evidence_seconds, 3),
            "coordinate_and_metrics": round(coordinate_seconds, 3),
            "total": round(time.perf_counter() - started, 3),
        },
        "incidence_invalid_candidates": selection.incidence_invalid_candidates,
        "spatial_component_candidates": selection.spatial_component_candidates,
        "spatial_components": selection.spatial_components,
        "retained_spatial_components": selection.retained_spatial_components,
        "uncertain_output_points": int(len(uncertain_points)),
        "likely_noise_screen_candidates": selection.likely_noise_candidates,
        "likely_noise_output_points": int(len(likely_noise_points)),
        "occluded_view_tests": selection.occluded_view_tests,
        "normal_rejected_matches": selection.normal_rejected_matches,
        "opposite_side_rejected_matches": selection.opposite_side_rejected_matches,
        "deletion_policy": (
            "invalid numeric data is removed; a point is classified as likely noise only "
            "when it has zero adjacent-view support, fails sensor evidence, and is farther "
            "than component_radius_mm from confirmed geometry; all other unsupported "
            "observations remain uncertain; edge evidence is soft when edge_policy=soft "
            "and a supported edge point can remain in the formal cloud"
        ),
    }
    metrics["evidence"] = {
        "method": "multiview_support_distance",
        "valid_definition": (
            f"support_count >= {config.min_views - 1} other views separated by at least "
            f"{config.min_support_angle_degrees:g} and at most "
            f"{effective_max_support_angle:g} degrees"
        ),
        "uncertain_definition": (
            f"support_count < {config.min_views - 1} eligible other views"
        ),
        "likely_noise_definition": (
            "zero eligible-view support AND failed sensor evidence AND spatially isolated "
            f"more than {config.component_radius_mm:g} mm from confirmed geometry"
        ),
        "support_constraints": {
            "normal_consistency_cosine_min": config.min_normal_consistency_cosine,
            "same_observation_side_required": config.require_same_observation_side,
            "occlusion_visibility_enabled": config.use_occlusion_visibility,
            "occlusion_tolerance_mm": config.occlusion_tolerance_mm,
            "occlusion_min_neighbors": config.occlusion_min_neighbors,
            "visibility_rule": (
                "project into the neighbor organized depth map; views with a surface "
                "closer than the candidate by more than the tolerance are occluded"
            ),
        },
        "edge_policy": config.edge_policy,
        "minimum_support_angle_degrees": config.min_support_angle_degrees,
        "configured_maximum_support_angle_degrees": config.max_support_angle_degrees,
        "maximum_support_angle_degrees": effective_max_support_angle,
        "support_angle_mode": (
            "automatic" if config.max_support_angle_degrees is None else "fixed"
        ),
        "sensor_thresholds": {
            "confidence_min": config.confidence_min,
            "edge_uncertain_px": config.edge_uncertain_px,
            "image_gradient_threshold": config.image_gradient_threshold,
            "depth_gradient_threshold_mm": config.depth_gradient_threshold_mm,
            "normal_change_cosine": config.normal_change_cosine,
            "confidence_drop_ratio": config.confidence_drop_ratio,
            "confidence_drop_abs": config.confidence_drop_abs,
            "min_incidence_cosine": config.min_incidence_cosine,
        },
        "spatial_component_thresholds": {
            "radius_mm": config.component_radius_mm,
            "min_component_ratio": config.min_component_ratio,
            "rule": "retain every component at least ratio * largest_component; do not keep only the largest",
        },
        "source_index_frames": int(
            sum(observation.source_indices is not None for observation in observations)
        ),
        "confidence_frames": int(
            sum(observation.confidence is not None for observation in observations)
        ),
        "image_frames": int(
            sum(observation.image is not None for observation in observations)
        ),
        "normal_frames": int(
            sum(observation.incidence_cosine is not None for observation in observations)
        ),
        "aligned_normal_frames": int(
            sum(observation.normals is not None for observation in observations)
        ),
        "visibility_frames": int(
            sum(
                observation.depth_mm is not None
                and observation.intrinsics is not None
                for observation in observations
            )
        ),
    }
    # 保留已有报告字段以兼容下游读取程序
    metrics["denoise"] = {
        "method": "multiview_consensus",
        "support_radius_mm": config.support_radius_mm,
        "min_views": config.min_views,
        "min_support_angle_degrees": config.min_support_angle_degrees,
        "configured_max_support_angle_degrees": config.max_support_angle_degrees,
        "max_support_angle_degrees": effective_max_support_angle,
        "output_points": int(len(points)),
    }
    return PointCloudProcessingResult(
        points=points,
        report=metrics,
        uncertain_points=uncertain_points,
        likely_noise_points=likely_noise_points,
    )


def write_processing_result(
    result: PointCloudProcessingResult,
    cloud_path: str | Path,
    *,
    report_path: str | Path | None = None,
) -> Path:
    """Persist a processing result without touching source capture artifacts."""

    cloud_path = Path(cloud_path)
    if cloud_path.suffix.lower() != ".ply":
        raise ValueError("点云输出路径必须使用 .ply 后缀")
    cloud_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cloud_path.with_suffix(".npy"), result.points)
    write_ascii_ply(cloud_path, result.points)
    uncertain_path = cloud_path.with_name(f"{cloud_path.stem}-uncertain{cloud_path.suffix}")
    np.save(uncertain_path.with_suffix(".npy"), result.uncertain_points)
    write_ascii_ply(uncertain_path, result.uncertain_points)
    likely_noise_path = cloud_path.with_name(
        f"{cloud_path.stem}-likely-noise{cloud_path.suffix}"
    )
    np.save(likely_noise_path.with_suffix(".npy"), result.likely_noise_points)
    write_ascii_ply(likely_noise_path, result.likely_noise_points)
    result.report.setdefault("artifacts", {}).update(
        {
            "cloud": str(cloud_path),
            "cloud_npy": str(cloud_path.with_suffix(".npy")),
            "uncertain": str(uncertain_path),
            "uncertain_npy": str(uncertain_path.with_suffix(".npy")),
            "likely_noise": str(likely_noise_path),
            "likely_noise_npy": str(likely_noise_path.with_suffix(".npy")),
        }
    )
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return cloud_path
