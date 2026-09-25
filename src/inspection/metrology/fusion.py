"""Local fusion for independently aligned placement scans.

STEP geometry is used only to associate observations with a local surface and
to define its normal direction.  Distance to the nominal STEP surface is not a
quality score: preferring the closest observation would erase real defects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import open3d as o3d

@dataclass(frozen=True, slots=True)
class LocalPlacementFusionConfig:
    cell_size: float
    voxel_size: float
    agreement_mm: float
    selection_mode: str = "nominal"
    normal_quantization: float = 0.25
    replacement_score_ratio: float = 0.75
    projection_batch_points: int = 250_000

    @classmethod
    def for_metrology(
        cls,
        *,
        tolerance_mm: float,
        voxel_size: float,
        selection_mode: str = "nominal",
    ) -> "LocalPlacementFusionConfig":
        return cls(
            cell_size=max(
                3.0 * voxel_size,
                1.5 * tolerance_mm,
            ),
            voxel_size=voxel_size,
            agreement_mm=max(tolerance_mm, 2.0 * voxel_size),
            selection_mode=selection_mode,
        )

    def validate(self) -> None:
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if self.agreement_mm <= 0:
            raise ValueError("agreement_mm must be positive")
        if self.selection_mode not in {"nominal", "consensus"}:
            raise ValueError("selection_mode must be nominal or consensus")
        if self.normal_quantization <= 0:
            raise ValueError("normal_quantization must be positive")
        if not 0 < self.replacement_score_ratio <= 1:
            raise ValueError("replacement_score_ratio must be in (0, 1]")
        if self.projection_batch_points < 1:
            raise ValueError("projection_batch_points must be positive")


@dataclass(frozen=True, slots=True)
class PlacementFusionLayer:
    name: str
    points: np.ndarray
    point_quality: np.ndarray | None = None
    registration_uncertainty_mm: float | None = None
    trusted_quality: bool = False


@dataclass(frozen=True, slots=True)
class PlacementFusionResult:
    points: np.ndarray
    source_placement: np.ndarray
    surface_cell: np.ndarray
    scatter_score: np.ndarray
    placement_support: np.ndarray
    conflict: np.ndarray
    supplemental: np.ndarray
    conflict_points: np.ndarray
    conflict_source_placement: np.ndarray
    conflict_surface_cell: np.ndarray
    report: dict[str, Any]


@dataclass(slots=True)
class _LayerCells:
    keys: np.ndarray
    point_cell: np.ndarray
    mean_residual: np.ndarray
    scatter_score: np.ndarray
    point_count: np.ndarray


def _surface_projection(
    scene: o3d.t.geometry.RaycastingScene,
    points: np.ndarray,
    config: LocalPlacementFusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    surface_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray] = []
    for start in range(0, len(points), config.projection_batch_points):
        query = o3d.core.Tensor(
            points[start : start + config.projection_batch_points],
            dtype=o3d.core.Dtype.Float32,
        )
        closest = scene.compute_closest_points(query)
        surface_parts.append(closest["points"].numpy().astype(np.float64))
        normal_parts.append(
            closest["primitive_normals"].numpy().astype(np.float64)
        )
    if not surface_parts:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty
    surface = np.concatenate(surface_parts, axis=0)
    normals = np.concatenate(normal_parts, axis=0)
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(normals).all(axis=1) & (lengths > 0)
    normals[valid] /= lengths[valid, None]
    normals[~valid] = 0.0
    return surface, normals


def _layer_cells(
    scene: o3d.t.geometry.RaycastingScene,
    layer: PlacementFusionLayer,
    config: LocalPlacementFusionConfig,
) -> tuple[np.ndarray, _LayerCells]:
    raw_points = np.asarray(layer.points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(raw_points).all(axis=1)
    points = raw_points[finite]
    if not len(points):
        raise ValueError(f"placement {layer.name!r} has no finite points")
    point_quality: np.ndarray | None = None
    if layer.point_quality is not None:
        point_quality = np.asarray(layer.point_quality, dtype=np.float64).reshape(-1)
        if len(point_quality) != len(raw_points):
            raise ValueError(
                f"placement {layer.name!r} point_quality must match its point count"
            )
        point_quality = point_quality[finite]

    surface, normals = _surface_projection(scene, points, config)
    residual = np.einsum("ij,ij->i", points - surface, normals)
    spatial_key = np.floor(surface / config.cell_size).astype(np.int64)
    normal_key = np.rint(normals / config.normal_quantization).astype(np.int64)
    point_keys = np.concatenate((spatial_key, normal_key), axis=1)
    keys, point_cell = np.unique(point_keys, axis=0, return_inverse=True)
    count = np.bincount(point_cell, minlength=len(keys)).astype(np.int64)
    residual_sum = np.bincount(point_cell, weights=residual, minlength=len(keys))
    residual_square_sum = np.bincount(
        point_cell,
        weights=residual * residual,
        minlength=len(keys),
    )
    mean = residual_sum / count
    variance = np.maximum(0.0, residual_square_sum / count - mean * mean)
    scatter = np.sqrt(variance)

    # The sampling term prevents a one-point cell from looking perfectly stable.
    capped_count = np.minimum(count, 16)
    score = scatter + config.voxel_size / np.sqrt(capped_count)
    if point_quality is not None:
        quality = np.where(np.isfinite(point_quality), point_quality, 0.0)
        quality = np.clip(quality, 0.0, 1.0)
        quality_sum = np.bincount(
            point_cell,
            weights=quality,
            minlength=len(keys),
        )
        mean_quality = quality_sum / count
        score += (1.0 - mean_quality) * config.agreement_mm
    if layer.registration_uncertainty_mm is not None:
        uncertainty = float(layer.registration_uncertainty_mm)
        if not np.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError("registration_uncertainty_mm must be non-negative")
        score += uncertainty

    return points.astype(np.float32), _LayerCells(
        keys=keys,
        point_cell=point_cell.astype(np.int64, copy=False),
        mean_residual=mean,
        scatter_score=score,
        point_count=count,
    )


def _cluster_observations(
    residuals: np.ndarray,
    agreement_mm: float,
) -> list[np.ndarray]:
    order = np.argsort(residuals, kind="stable")
    clusters: list[list[int]] = []
    cluster_min = 0.0
    for observation_index in order:
        value = float(residuals[observation_index])
        if not clusters or value - cluster_min > agreement_mm:
            clusters.append([int(observation_index)])
            cluster_min = value
        else:
            clusters[-1].append(int(observation_index))
    return [np.asarray(cluster, dtype=np.int64) for cluster in clusters]


def _medoid_indices(points: np.ndarray, voxel_size_mm: float) -> np.ndarray:
    """Return actual-point voxel representatives while preserving attributes."""

    if not len(points):
        return np.empty(0, dtype=np.int64)
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    keys = np.floor(values / voxel_size_mm).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
    np.add.at(sums, inverse, values)
    counts = np.bincount(inverse, minlength=len(unique_keys))
    centroids = sums / counts[:, None]
    distance_squared = np.sum((values - centroids[inverse]) ** 2, axis=1)
    minimum = np.full(len(unique_keys), np.inf, dtype=np.float64)
    np.minimum.at(minimum, inverse, distance_squared)
    candidates = np.flatnonzero(
        np.isclose(distance_squared, minimum[inverse], rtol=1e-12, atol=1e-15)
    )
    order = np.lexsort(
        (
            values[candidates, 2],
            values[candidates, 1],
            values[candidates, 0],
            inverse[candidates],
        )
    )
    ordered = candidates[order]
    _, first = np.unique(inverse[ordered], return_index=True)
    return np.sort(ordered[first]).astype(np.int64, copy=False)


def fuse_layers(
    mesh: o3d.geometry.TriangleMesh,
    layers: Sequence[PlacementFusionLayer],
    config: LocalPlacementFusionConfig,
) -> PlacementFusionResult:
    """Select the locally strongest placement without averaging conflicts."""

    config.validate()
    if not layers:
        raise ValueError("at least one placement layer is required")
    names = [layer.name for layer in layers]
    if len(names) != len(set(names)):
        raise ValueError("placement layer names must be unique")

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    layer_points: list[np.ndarray] = []
    layer_cells: list[_LayerCells] = []
    for layer in layers:
        points, cells = _layer_cells(scene, layer, config)
        layer_points.append(points)
        layer_cells.append(cells)

    all_cell_keys = np.concatenate([cells.keys for cells in layer_cells], axis=0)
    global_keys, observation_global_cell = np.unique(
        all_cell_keys,
        axis=0,
        return_inverse=True,
    )
    offsets = np.cumsum([0, *[len(cells.keys) for cells in layer_cells]])
    layer_global_cells = [
        observation_global_cell[offsets[index] : offsets[index + 1]]
        for index in range(len(layers))
    ]

    observation_cell = np.concatenate(layer_global_cells)
    observation_placement = np.concatenate(
        [
            np.full(len(cells.keys), index, dtype=np.int32)
            for index, cells in enumerate(layer_cells)
        ]
    )
    observation_residual = np.concatenate(
        [cells.mean_residual for cells in layer_cells]
    )
    observation_score = np.concatenate(
        [cells.scatter_score for cells in layer_cells]
    )
    observation_count = np.concatenate(
        [cells.point_count for cells in layer_cells]
    )
    order = np.argsort(observation_cell, kind="stable")
    sorted_cells = observation_cell[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_cells)) + 1]
    ends = np.r_[starts[1:], len(order)]

    cell_count = len(global_keys)
    accepted = np.zeros((cell_count, len(layers)), dtype=bool)
    cell_layer_score = np.full(
        (cell_count, len(layers)),
        np.nan,
        dtype=np.float32,
    )
    support = np.zeros(cell_count, dtype=np.int16)
    conflict = np.zeros(cell_count, dtype=bool)
    unresolved_conflict = np.zeros(cell_count, dtype=bool)
    supplemental = np.zeros(cell_count, dtype=bool)
    ambiguous_conflicts = 0
    consensus_overrides = 0
    quality_replacements = 0
    nominal_selections = 0
    nominal_singletons = 0

    for start, end in zip(starts, ends):
        group = order[start:end]
        global_cell = int(observation_cell[group[0]])
        placements = observation_placement[group]
        residuals = observation_residual[group]
        scores = observation_score[group]
        cell_layer_score[global_cell, placements] = scores
        clusters = _cluster_observations(residuals, config.agreement_mm)
        conflict[global_cell] = len(clusters) > 1

        baseline_local = int(np.argmin(placements))
        baseline_cluster_index = next(
            index
            for index, cluster in enumerate(clusters)
            if baseline_local in cluster
        )
        cluster_support = np.asarray([len(cluster) for cluster in clusters])
        maximum_support = int(cluster_support.max())
        maximum_clusters = np.flatnonzero(cluster_support == maximum_support)
        winner_cluster_index = baseline_cluster_index
        if config.selection_mode == "nominal" and len(clusters) > 1:
            cluster_distance = np.asarray(
                [
                    abs(float(np.median(residuals[cluster])))
                    for cluster in clusters
                ],
                dtype=np.float64,
            )
            winner_cluster_index = int(np.argmin(cluster_distance))
            nominal_selections += 1
            if len(clusters[winner_cluster_index]) == 1:
                nominal_singletons += 1
        elif (
            len(maximum_clusters) == 1
            and maximum_support >= 2
            and maximum_support > int(cluster_support[baseline_cluster_index])
        ):
            winner_cluster_index = int(maximum_clusters[0])
            consensus_overrides += 1
        elif len(clusters) > 1 and len(maximum_clusters) > 1:
            baseline_best_score = float(
                np.min(scores[clusters[baseline_cluster_index]])
            )
            eligible_quality_clusters = [
                int(cluster_index)
                for cluster_index in maximum_clusters
                if cluster_index != baseline_cluster_index
                and any(
                    layers[int(placements[local_index])].trusted_quality
                    for local_index in clusters[int(cluster_index)]
                )
            ]
            if eligible_quality_clusters:
                candidate_scores = np.asarray(
                    [
                        np.min(scores[clusters[cluster_index]])
                        for cluster_index in eligible_quality_clusters
                    ]
                )
                best_candidate = int(np.argmin(candidate_scores))
                if (
                    candidate_scores[best_candidate]
                    < baseline_best_score * config.replacement_score_ratio
                ):
                    winner_cluster_index = eligible_quality_clusters[best_candidate]
                    quality_replacements += 1
                else:
                    ambiguous_conflicts += 1
                    unresolved_conflict[global_cell] = True
            else:
                ambiguous_conflicts += 1
                unresolved_conflict[global_cell] = True

        winner_cluster = clusters[winner_cluster_index]
        baseline_placement = int(placements[baseline_local])
        accepted[global_cell, placements[winner_cluster]] = True
        support[global_cell] = len(winner_cluster)
        supplemental[global_cell] = baseline_placement > 0

    formal_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    cell_parts: list[np.ndarray] = []
    conflict_parts: list[np.ndarray] = []
    conflict_source_parts: list[np.ndarray] = []
    conflict_cell_parts: list[np.ndarray] = []
    selected_counts: dict[str, int] = {}
    for placement_index, (layer, points, cells, cell_ids) in enumerate(
        zip(layers, layer_points, layer_cells, layer_global_cells)
    ):
        point_global_cell = cell_ids[cells.point_cell]
        selected = (
            accepted[point_global_cell, placement_index]
            & ~unresolved_conflict[point_global_cell]
        )
        formal_parts.append(points[selected])
        source_parts.append(
            np.full(np.count_nonzero(selected), placement_index, dtype=np.int16)
        )
        cell_parts.append(point_global_cell[selected])
        nonselected = ~selected
        conflict_rejected = nonselected & conflict[point_global_cell]
        conflict_parts.append(points[conflict_rejected])
        conflict_source_parts.append(
            np.full(
                np.count_nonzero(conflict_rejected),
                placement_index,
                dtype=np.int16,
            )
        )
        conflict_cell_parts.append(point_global_cell[conflict_rejected])
        selected_counts[layer.name] = int(np.count_nonzero(selected))

    formal = np.concatenate(formal_parts, axis=0)
    formal_input_points = len(formal)
    source = np.concatenate(source_parts)
    surface_cell = np.concatenate(cell_parts)
    keep = _medoid_indices(formal, config.voxel_size)
    formal = formal[keep]
    source = source[keep]
    surface_cell = surface_cell[keep]
    output_score = cell_layer_score[surface_cell, source]
    output_support = support[surface_cell]
    output_conflict = conflict[surface_cell]
    output_supplemental = supplemental[surface_cell]

    conflict_points = np.concatenate(conflict_parts, axis=0)
    conflict_source = np.concatenate(conflict_source_parts)
    conflict_surface_cell = np.concatenate(conflict_cell_parts)
    conflict_keep = _medoid_indices(
        conflict_points,
        config.voxel_size,
    )
    conflict_points = conflict_points[conflict_keep]
    conflict_source = conflict_source[conflict_keep]
    conflict_surface_cell = conflict_surface_cell[conflict_keep]
    output_by_placement = {
        layer.name: int(np.count_nonzero(source == index))
        for index, layer in enumerate(layers)
    }
    observed_placements = np.bincount(
        observation_cell,
        minlength=cell_count,
    )
    nominal_mode = config.selection_mode == "nominal"
    report: dict[str, Any] = {
        "method": "cad-surface-local-selective-fusion",
        "policy": (
            "in each conflicting surface cell select the agreement "
            "cluster whose median signed normal residual is closest to the STEP "
            "surface; never average across conflicting clusters"
            if nominal_mode
            else
            "accept unique coverage and all observations in the winning agreement "
            "cluster; require an independent-placement support majority to override "
            "the incumbent cluster; quarantine unresolved ties from the formal cloud; "
            "never average across conflicting clusters"
        ),
        "cad_usage": (
            "nominal mode uses absolute normal distance to STEP to select among "
            "conflicting observation clusters; this can suppress real deviations"
            if nominal_mode
            else
            "STEP supplies surface association and normal direction only; absolute "
            "distance to nominal geometry is not used as an observation quality score"
        ),
        "config": asdict(config),
        "placements": names,
        "input_points": {
            layer.name: int(len(points))
            for layer, points in zip(layers, layer_points)
        },
        "selected_points": selected_counts,
        "output_points": int(len(formal)),
        "output_counts": output_by_placement,
        "surface_cells": cell_count,
        "single_placement_cells": int(np.count_nonzero(observed_placements == 1)),
        "overlap_cells": int(np.count_nonzero(observed_placements > 1)),
        "conflict_cells": int(np.count_nonzero(conflict)),
        "unresolved_conflict_cells": int(np.count_nonzero(unresolved_conflict)),
        "ambiguous_conflict_cells": int(ambiguous_conflicts),
        "consensus_override_cells": int(consensus_overrides),
        "quality_replacement_cells": int(quality_replacements),
        "nominal_selection_cells": int(nominal_selections),
        "nominal_singleton_cells": int(nominal_singletons),
        "quality_override_placements": [
            layer.name for layer in layers if layer.trusted_quality
        ],
        "supplemental_cells": int(np.count_nonzero(supplemental)),
        "conflict_output_points": int(len(conflict_points)),
        "voxel_removed": int(
            formal_input_points - len(formal)
        ),
        "formal_conflicts": int(np.count_nonzero(output_conflict)),
        "formal_supplemental_points": int(np.count_nonzero(output_supplemental)),
        "quality_note": (
            "scatter_score is a relative selection score based on within-cell "
            "normal scatter, capped sampling support, optional normalized sensor "
            "quality, and optional independently estimated registration uncertainty; "
            "it is not a calibrated measurement uncertainty and cannot override a "
            "conflict unless its placement explicitly enables trusted quality override"
        ),
    }
    return PlacementFusionResult(
        points=formal.astype(np.float32, copy=False),
        source_placement=source,
        surface_cell=surface_cell.astype(np.int64, copy=False),
        scatter_score=output_score,
        placement_support=output_support,
        conflict=output_conflict,
        supplemental=output_supplemental,
        conflict_points=conflict_points,
        conflict_source_placement=conflict_source,
        conflict_surface_cell=conflict_surface_cell.astype(np.int64, copy=False),
        report=report,
    )
