"""Register STEP meshes and measured point clouds."""

from __future__ import annotations

from itertools import permutations, product
from typing import Any

import numpy as np
import open3d as o3d


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
    best: tuple[float, np.ndarray, dict[str, Any]] | None = None
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
        if best is None or score < best[0]:
            best = (
                score,
                result.transformation,
                {
                    "fitness": float(result.fitness),
                    "inlier_rmse_mm": float(result.inlier_rmse),
                },
            )
    assert best is not None

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
        "method": "multistart-coarse-to-fine-icp",
        "starts": len(rotations),
        "voxel_mm": float(voxel),
        "fitness": refinements[-1]["fitness"],
        "inlier_rmse_mm": refinements[-1]["inlier_rmse_mm"],
        "refinements": refinements,
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
    assert best is not None

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
