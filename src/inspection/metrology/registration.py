"""Register STEP meshes and measured point clouds."""

from __future__ import annotations

from itertools import permutations, product
from typing import Any

import numpy as np
import open3d as o3d


def _transformation_separation(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float]:
    first = np.asarray(first, dtype=np.float64).reshape(4, 4)
    second = np.asarray(second, dtype=np.float64).reshape(4, 4)
    relative_rotation = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    rotation_degrees = float(np.rad2deg(np.arccos(cosine)))
    translation_mm = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    return rotation_degrees, translation_mm


def _assess_ambiguity(
    candidates: list[dict[str, Any]],
    *,
    relative_score_margin: float = 0.05,
    score_margin: float = 0.02,
    distinct_rotation_degrees: float = 2.0,
    distinct_translation_mm: float = 0.50,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("registration ambiguity needs at least one candidate")
    ordered = sorted(candidates, key=lambda item: float(item["score_mm"]))
    best = ordered[0]
    best_score = float(best["score_mm"])
    score_limit = best_score + max(
        score_margin,
        relative_score_margin * max(best_score, 1e-12),
    )
    alternatives: list[dict[str, Any]] = []
    summarized: list[dict[str, Any]] = []
    for candidate in ordered:
        rotation, translation = _transformation_separation(
            best["transformation"], candidate["transformation"]
        )
        summary = {
            "score_mm": float(candidate["score_mm"]),
            "fitness": float(candidate["fitness"]),
            "inlier_rmse_mm": float(candidate["inlier_rmse_mm"]),
            "rotation_delta": rotation,
            "translation_delta": translation,
            "transformation": np.asarray(candidate["transformation"]).tolist(),
        }
        summarized.append(summary)
        if (
            float(candidate["score_mm"]) <= score_limit
            and (
                rotation > distinct_rotation_degrees
                or translation > distinct_translation_mm
            )
        ):
            alternatives.append(summary)
    return {
        "status": "ambiguous" if alternatives else "unique",
        "best_score_mm": best_score,
        "score_limit": float(score_limit),
        "relative_score_margin": relative_score_margin,
        "score_margin": score_margin,
        "distinct_rotation_degrees": distinct_rotation_degrees,
        "distinct_translation_mm": distinct_translation_mm,
        "alternatives": alternatives,
        "candidates": summarized,
    }


def _axis_rotations() -> list[np.ndarray]:
    rotations: list[np.ndarray] = []
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0:
                rotations.append(matrix)
    return rotations


def register(
    mesh: o3d.geometry.TriangleMesh,
    points: np.ndarray,
    tolerance_mm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find the best coarse-to-fine rigid registration."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 10:
        raise ValueError("STEP registration requires at least 10 finite cloud points")
    if len(mesh.vertices) < 3 or len(mesh.triangles) < 1:
        raise ValueError("STEP registration requires a non-empty triangle mesh")
    o3d.utility.random.seed(42)
    mesh_points = np.asarray(
        mesh.sample_points_uniformly(number_of_points=80000).points
    )
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh_points))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    voxel = max(tolerance_mm / 3.0, 0.25)
    source_coarse = source.voxel_down_sample(voxel)
    target_coarse = target.voxel_down_sample(voxel)
    normal_search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=max(4.0 * voxel, 1.0), max_nn=40
    )
    source_coarse.estimate_normals(normal_search)
    target_coarse.estimate_normals(normal_search)
    source_center = mesh_points.mean(axis=0)
    target_center = points.mean(axis=0)
    coarse_distance = max(4.0 * tolerance_mm, 4.0)
    fine_distance = max(2.0 * tolerance_mm, 2.0)
    rotations = _axis_rotations()
    coarse_candidates: list[dict[str, Any]] = []
    for rotation in rotations:
        initial = np.eye(4)
        initial[:3, :3] = rotation
        initial[:3, 3] = target_center - rotation @ source_center
        coarse = o3d.pipelines.registration.registration_icp(
            source_coarse,
            target_coarse,
            coarse_distance,
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        result = o3d.pipelines.registration.registration_icp(
            source_coarse,
            target_coarse,
            fine_distance,
            coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
        )
        score = float(result.inlier_rmse + (1.0 - result.fitness) * fine_distance)
        coarse_candidates.append(
            {
                "score_mm": score,
                "transformation": result.transformation,
                "fitness": float(result.fitness),
                "inlier_rmse_mm": float(result.inlier_rmse),
            }
        )

    source_fine = source.voxel_down_sample(max(voxel / 2.0, 0.12))
    target_fine = target.voxel_down_sample(max(voxel / 2.0, 0.12))
    fine_search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=max(2.0 * voxel, 0.6), max_nn=50
    )
    source_fine.estimate_normals(fine_search)
    target_fine.estimate_normals(fine_search)
    refinement_seeds: list[dict[str, Any]] = []
    for candidate in sorted(
        coarse_candidates, key=lambda item: float(item["score_mm"])
    ):
        if all(
            (
                lambda separation: separation[0] > 1.0
                or separation[1] > 0.25
            )(
                _transformation_separation(
                    seed["transformation"], candidate["transformation"]
                )
            )
            for seed in refinement_seeds
        ):
            refinement_seeds.append(candidate)
        if len(refinement_seeds) >= 6:
            break
    refined_candidates: list[dict[str, Any]] = []
    final_distance = max(tolerance_mm, 0.75)
    for seed in refinement_seeds:
        transform = np.asarray(seed["transformation"], dtype=np.float64)
        refinements: list[dict[str, float]] = []
        for distance in (fine_distance, final_distance):
            refined = o3d.pipelines.registration.registration_icp(
                source_fine,
                target_fine,
                distance,
                transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
            )
            transform = refined.transformation
            refinements.append(
                {
                    "distance_mm": float(distance),
                    "fitness": float(refined.fitness),
                    "inlier_rmse_mm": float(refined.inlier_rmse),
                }
            )
        final = refinements[-1]
        refined_candidates.append(
            {
                "score_mm": float(
                    final["inlier_rmse_mm"]
                    + (1.0 - final["fitness"]) * final_distance
                ),
                "transformation": transform,
                "fitness": final["fitness"],
                "inlier_rmse_mm": final["inlier_rmse_mm"],
                "refinements": refinements,
            }
        )
    refined_candidates.sort(key=lambda item: float(item["score_mm"]))
    best = refined_candidates[0]
    ambiguity = _assess_ambiguity(refined_candidates)
    return np.asarray(best["transformation"]), {
        "method": "multistart-coarse-to-fine-icp",
        "starts": len(rotations),
        "refined_distinct_candidates": len(refined_candidates),
        "voxel_mm": float(voxel),
        "fitness": best["fitness"],
        "inlier_rmse_mm": best["inlier_rmse_mm"],
        "refinements": best["refinements"],
        "ambiguity": ambiguity,
    }


def register_clouds(
    reference_points: np.ndarray,
    points: np.ndarray,
    tolerance_mm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """将一个放置面的点云配准到公共参考点云"""

    reference_points = np.asarray(reference_points, dtype=np.float64).reshape(-1, 3)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    reference_points = reference_points[np.isfinite(reference_points).all(axis=1)]
    points = points[np.isfinite(points).all(axis=1)]
    if len(reference_points) < 10 or len(points) < 10:
        raise ValueError("ICP 配准需要足够的有效点")

    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reference_points))
    voxel = max(tolerance_mm / 3.0, 0.25)
    source_coarse = source.voxel_down_sample(voxel)
    target_coarse = target.voxel_down_sample(voxel)
    normal_search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=max(4.0 * voxel, 1.0), max_nn=40
    )
    source_coarse.estimate_normals(normal_search)
    target_coarse.estimate_normals(normal_search)
    source_center = np.asarray(source_coarse.points).mean(axis=0)
    target_center = np.asarray(target_coarse.points).mean(axis=0)
    coarse_distance = max(4.0 * tolerance_mm, 4.0)
    fine_distance = max(2.0 * tolerance_mm, 2.0)
    best: tuple[float, np.ndarray, dict[str, Any]] | None = None
    for rotation in _axis_rotations():
        initial = np.eye(4)
        initial[:3, :3] = rotation
        initial[:3, 3] = target_center - rotation @ source_center
        coarse = o3d.pipelines.registration.registration_icp(
            source_coarse,
            target_coarse,
            coarse_distance,
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        result = o3d.pipelines.registration.registration_icp(
            source_coarse,
            target_coarse,
            fine_distance,
            coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
        )
        score = float(result.inlier_rmse + (1.0 - result.fitness) * fine_distance)
        if best is None or score < best[0]:
            best = (
                score,
                result.transformation,
                {
                    "fitness": float(result.fitness),
                    "inlier_rmse_mm": float(result.inlier_rmse),
                },
            )
    if best is None:
        raise RuntimeError("cloud registration did not produce a candidate pose")

    source_fine = source.voxel_down_sample(max(voxel / 2.0, 0.12))
    target_fine = target.voxel_down_sample(max(voxel / 2.0, 0.12))
    fine_search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=max(2.0 * voxel, 0.6), max_nn=50
    )
    source_fine.estimate_normals(fine_search)
    target_fine.estimate_normals(fine_search)
    transform = best[1]
    refinements: list[dict[str, float]] = []
    for distance in (fine_distance, max(tolerance_mm, 0.75)):
        refined = o3d.pipelines.registration.registration_icp(
            source_fine,
            target_fine,
            distance,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
        )
        transform = refined.transformation
        refinements.append(
            {
                "distance_mm": float(distance),
                "fitness": float(refined.fitness),
                "inlier_rmse_mm": float(refined.inlier_rmse),
            }
        )
    return transform, {
        "method": "multistart-coarse-to-fine-cloud-icp",
        "starts": len(_axis_rotations()),
        "voxel_mm": float(voxel),
        "fitness": refinements[-1]["fitness"],
        "inlier_rmse_mm": refinements[-1]["inlier_rmse_mm"],
        "refinements": refinements,
    }
