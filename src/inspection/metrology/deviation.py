"""STEP-to-cloud deviation measurement and result generation."""

from __future__ import annotations

import json
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from inspection.viewer import DISPLAY_COLORS, DISPLAY_STYLE
from .model import load_mesh
from .fusion import (
    LocalPlacementFusionConfig,
    PlacementFusionLayer,
    fuse_layers,
)
from .registration import register
from .visibility import (
    CadVisibilityFrame,
    classify_visibility,
    load_frames,
)


DEFAULT_TOLERANCE_MM = 0.1
MIN_STEP_REGISTRATION_FITNESS = 0.2
MIN_REGISTRATION_RETAINED_RATIO = 0.8


@dataclass
class ComparisonResult:
    report: dict[str, Any]
    output_dir: Path


@dataclass(frozen=True, slots=True)
class _DeviationClassification:
    problem_mask: np.ndarray
    recessed_mask: np.ndarray
    exterior_mask: np.ndarray
    unclassified_mask: np.ndarray
    signed_distance_reliable: bool
    mode: str


def _stats(
    values: np.ndarray,
    threshold: float,
    *,
    bad_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    values = values[finite]
    if not len(values):
        return {"count": 0, "p50_mm": None, "p90_mm": None, "p95_mm": None,
                "p99_mm": None, "max_mm": None, "within_tolerance_ratio": 0.0,
                "bad_count": 0, "bad_ratio": 0.0}
    if bad_mask is None:
        bad = values > threshold
    else:
        bad_array = np.asarray(bad_mask, dtype=bool).reshape(-1)
        if len(bad_array) != len(finite):
            raise ValueError("bad_mask must contain one value per distance")
        bad = bad_array[finite]
    return {
        "count": int(len(values)),
        "p50_mm": float(np.percentile(values, 50)),
        "p90_mm": float(np.percentile(values, 90)),
        "p95_mm": float(np.percentile(values, 95)),
        "p99_mm": float(np.percentile(values, 99)),
        "max_mm": float(np.max(values)),
        "within_tolerance_ratio": float(np.mean(~bad)),
        "bad_count": int(np.count_nonzero(bad)),
        "bad_ratio": float(np.mean(bad)),
    }


def _quality_gate(
    distances: np.ndarray,
    tolerance_mm: float,
) -> dict[str, Any]:
    """Validate a placement against the fixed STEP reference after registration."""

    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    p50 = float(np.percentile(distances, 50))
    p90 = float(np.percentile(distances, 90))
    p50_limit = max(2.5 * tolerance_mm, 0.25)
    p90_limit = max(7.5 * tolerance_mm, 0.75)
    passed = p50 <= p50_limit and p90 <= p90_limit
    return {
        "status": "pass" if passed else "fail",
        "p50_mm": p50,
        "p90_mm": p90,
        "maximum_p50_mm": float(p50_limit),
        "maximum_p90_mm": float(p90_limit),
    }


def _outlier_limit(tolerance_mm: float) -> float:
    """Return the distance beyond which points cannot be credible part defects."""

    return max(50.0 * tolerance_mm, 5.0)


def _load_points(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        points = np.load(path)
    else:
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def _measure_path(path: Path) -> Path:
    candidates = (
        path.with_name(f"{path.stem}-trusted.npy"),
        path.with_name(f"{path.stem}-trusted.ply"),
    )
    return next((item for item in candidates if item.is_file()), path)


def _mesh_distances(mesh_points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    cloud_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))
    tree = o3d.geometry.KDTreeFlann(cloud_pcd)
    distances = np.empty(len(mesh_points), dtype=np.float64)
    for start in range(0, len(mesh_points), 5000):
        for index, point in enumerate(mesh_points[start:start + 5000], start):
            count, _, squared = tree.search_knn_vector_3d(point, 1)
            distances[index] = np.sqrt(squared[0]) if count else np.inf
    return distances


def _cloud_distances(mesh: o3d.geometry.TriangleMesh, points: np.ndarray) -> np.ndarray:
    """Return unsigned point-to-surface distances.

    The distance is intentionally unsigned: a measured point recessed into a
    STEP solid (for example, the bottom of a real pit absent from the STEP)
    still has a non-zero distance to the nominal surface and is therefore
    eligible for the red problem-point output when it exceeds tolerance.
    """

    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    distances: list[np.ndarray] = []
    for start in range(0, len(points), 100000):
        query = o3d.core.Tensor(points[start:start + 100000], dtype=o3d.core.Dtype.Float32)
        distances.append(scene.compute_distance(query).numpy().astype(np.float64))
    return np.concatenate(distances) if distances else np.empty(0, dtype=np.float64)


def _signed_distances(
    mesh: o3d.geometry.TriangleMesh,
    points: np.ndarray,
) -> np.ndarray:
    """Return signed distances; positive values are on the mesh exterior.

    The STEP tessellation can contain open shells, so the sign is an
    inspection-side classification rather than a replacement for geometric
    metrology. It is used here to prevent recessed points from being rendered
    as exterior over-thickness defects.
    """

    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    distances: list[np.ndarray] = []
    for start in range(0, len(points), 100000):
        query = o3d.core.Tensor(points[start:start + 100000], dtype=o3d.core.Dtype.Float32)
        distances.append(scene.compute_signed_distance(query).numpy().astype(np.float64))
    return np.concatenate(distances) if distances else np.empty(0, dtype=np.float64)


def _classify_deviation(
    mesh: o3d.geometry.TriangleMesh,
    unsigned_distances: np.ndarray,
    tolerance_mm: float,
    *,
    signed_distances: np.ndarray | None = None,
    signed_distance_reliable: bool | None = None,
) -> _DeviationClassification:
    """Classify direction only when the tessellated STEP is watertight."""

    unsigned = np.asarray(unsigned_distances, dtype=np.float64).reshape(-1)
    over_tolerance = np.isfinite(unsigned) & (unsigned > tolerance_mm)
    signed_reliable = (
        _direction_quality(mesh)["reliable"]
        if signed_distance_reliable is None
        else bool(signed_distance_reliable)
    )
    if not signed_reliable:
        empty = np.zeros(len(unsigned), dtype=bool)
        return _DeviationClassification(
            problem_mask=over_tolerance,
            recessed_mask=empty,
            exterior_mask=empty,
            unclassified_mask=over_tolerance,
            signed_distance_reliable=False,
            mode="unsigned_only",
        )

    if signed_distances is None:
        raise ValueError("signed_distances are required for a watertight mesh")
    signed = np.asarray(signed_distances, dtype=np.float64).reshape(-1)
    if len(signed) != len(unsigned):
        raise ValueError("signed and unsigned distance arrays must have equal length")
    finite_signed = np.isfinite(signed)
    exterior = over_tolerance & finite_signed & (signed > 0.0)
    recessed = over_tolerance & finite_signed & (signed < 0.0)
    unclassified = over_tolerance & ~(exterior | recessed)
    return _DeviationClassification(
        problem_mask=exterior | unclassified,
        recessed_mask=recessed,
        exterior_mask=exterior,
        unclassified_mask=unclassified,
        signed_distance_reliable=True,
        mode="signed_watertight_mesh",
    )


def _direction_quality(mesh: o3d.geometry.TriangleMesh) -> dict[str, bool]:
    """Return the topology checks required before using distance direction."""

    quality = {
        "watertight": bool(mesh.is_watertight()),
        "edge_manifold": bool(mesh.is_edge_manifold(allow_boundary_edges=False)),
        "vertex_manifold": bool(mesh.is_vertex_manifold()),
        "orientable": bool(mesh.is_orientable()),
        "self_intersecting": bool(mesh.is_self_intersecting()),
    }
    quality["reliable"] = bool(
        quality["watertight"]
        and quality["edge_manifold"]
        and quality["vertex_manifold"]
        and quality["orientable"]
        and not quality["self_intersecting"]
    )
    return quality


def _write_cloud(
    path: Path,
    points: np.ndarray,
    color: tuple[float, float, float] | None = None,
    *,
    colors: np.ndarray | None = None,
) -> None:
    if not len(points):
        path.write_text(
            "ply\nformat ascii 1.0\nelement vertex 0\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n",
            encoding="ascii",
        )
        return
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if len(colors) != len(points):
            raise ValueError("colors must contain one RGB value per point")
        cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    elif color is not None:
        cloud.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def _distance_colors(
    distances: np.ndarray,
    tolerance_mm: float,
    *,
    kind: str,
) -> np.ndarray:
    """Map distance excess to a light-to-dark problem color.

    This is a visual severity scale, not a calibrated probability of defect.
    The upper end is clipped at the 95th percentile so isolated extreme values
    do not wash out the rest of the map.
    """

    if kind not in {"problem", "recessed", "missing"}:
        raise ValueError("kind must be problem, recessed, or missing")
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    if not len(distances):
        return np.empty((0, 3), dtype=np.float64)
    excess = np.maximum(distances - float(tolerance_mm), 0.0)
    high = float(np.percentile(excess, 95))
    if high <= 1e-12:
        level = np.ones(len(excess), dtype=np.float64)
    else:
        level = np.clip(excess / high, 0.0, 1.0)
    if kind == "problem":
        light = np.asarray([1.0, 0.72, 0.38])
        dark = np.asarray(DISPLAY_COLORS["problem"])
    elif kind == "recessed":
        light = np.asarray([0.72, 0.50, 1.0])
        dark = np.asarray(DISPLAY_COLORS["recessed"])
    else:
        light = np.asarray([0.50, 0.75, 1.0])
        dark = np.asarray(DISPLAY_COLORS["unobserved"])
    return light[None, :] * (1.0 - level[:, None]) + dark[None, :] * level[:, None]


def show_comparison(
    output_dir: Path,
    *,
    show_blue: bool = False,
    blue_cell: float = 0.75,
) -> None:
    """Show deviations over a matte STEP surface with GPU depth occlusion."""

    if blue_cell <= 0:
        raise ValueError("blue_cell must be positive")
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    output_dir = Path(output_dir)
    trace_mode = (output_dir / "coverage.ply").is_file()
    report_path = output_dir / "report.json"
    if report_path.is_file():
        try:
            trace_mode = trace_mode or (
                json.loads(report_path.read_text(encoding="utf-8")).get("mode")
                == "single-frame-trace"
            )
        except (OSError, json.JSONDecodeError):
            trace_mode = False
    conflict_path = output_dir / "cloud-placement-conflicts.ply"
    if not conflict_path.is_file() and output_dir.name == "view":
        conflict_path = output_dir.parent / "cloud-placement-conflicts.ply"
    paths = {
        "mesh": output_dir / "mesh.ply",
        "conflict": conflict_path,
        "coverage": output_dir / "coverage.ply",
        "problem": output_dir / "problem.ply",
        "recessed": output_dir / "recessed.ply",
        "missing": (
            output_dir / "unobserved.ply"
            if (output_dir / "unobserved.ply").is_file()
            else output_dir / "missing.ply"
        ),
    }
    optional = {"recessed", "coverage", "conflict"}
    if not show_blue:
        optional.add("missing")
    missing_files = [
        str(path)
        for name, path in paths.items()
        if name not in optional and not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"对比窗口缺少文件: {', '.join(missing_files)}")

    def read_cloud(path: Path) -> o3d.geometry.PointCloud:
        if not path.is_file():
            return o3d.geometry.PointCloud()
        header = path.read_bytes()[:4096]
        if b"element vertex 0" in header:
            return o3d.geometry.PointCloud()
        return o3d.io.read_point_cloud(str(path))

    mesh = o3d.io.read_triangle_mesh(str(paths["mesh"]))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(DISPLAY_COLORS["mesh"])
    problem = read_cloud(paths["problem"])
    if not problem.is_empty():
        problem.paint_uniform_color(DISPLAY_COLORS["problem"])
    coverage = (
        read_cloud(paths["coverage"])
        if paths["coverage"].is_file()
        else o3d.geometry.PointCloud()
    )
    if not coverage.is_empty():
        coverage.paint_uniform_color(DISPLAY_COLORS["unobserved"])
    recessed = (
        read_cloud(paths["recessed"])
        if paths["recessed"].is_file()
        else o3d.geometry.PointCloud()
    )
    if not recessed.is_empty():
        recessed.paint_uniform_color(DISPLAY_COLORS["recessed"])
    missing = (
        read_cloud(paths["missing"])
        if show_blue
        else o3d.geometry.PointCloud()
    )
    if not missing.is_empty():
        missing.paint_uniform_color(DISPLAY_COLORS["unobserved"])
    conflict = read_cloud(paths["conflict"])
    if not conflict.is_empty():
        conflict.paint_uniform_color(DISPLAY_COLORS["conflict"])

    def vtk_points(values: np.ndarray) -> vtk.vtkPoints:
        result = vtk.vtkPoints()
        result.SetData(
            numpy_to_vtk(np.asarray(values, dtype=np.float32), deep=True)
        )
        return result

    mesh_data = vtk.vtkPolyData()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    mesh_data.SetPoints(vtk_points(vertices))
    normal_data = numpy_to_vtk(normals, deep=True)
    normal_data.SetNumberOfComponents(3)
    normal_data.SetName("Normals")
    mesh_data.GetPointData().SetNormals(normal_data)
    triangle_cells = np.column_stack(
        (np.full(len(triangles), 3, dtype=np.int64), triangles)
    )
    mesh_cells = vtk.vtkCellArray()
    mesh_cells.SetCells(
        len(triangles),
        numpy_to_vtkIdTypeArray(triangle_cells.reshape(-1), deep=True),
    )
    mesh_data.SetPolys(mesh_cells)
    mesh_rgb = np.round(np.asarray(DISPLAY_COLORS["mesh"]) * 255.0).astype(np.uint8)
    mesh_colors = np.tile(mesh_rgb, (len(triangles), 1))
    mesh_colors = np.column_stack(
        (mesh_colors, np.full(len(triangles), 255, dtype=np.uint8))
    )
    mesh_color_data = numpy_to_vtk(
        mesh_colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    mesh_color_data.SetNumberOfComponents(4)
    mesh_color_data.SetName("SurfaceOpacity")
    mesh_data.GetCellData().SetScalars(mesh_color_data)
    mesh_mapper = vtk.vtkPolyDataMapper()
    mesh_mapper.SetInputData(mesh_data)
    mesh_mapper.SetColorModeToDirectScalars()
    mesh_mapper.SetScalarModeToUseCellData()
    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mesh_mapper)
    mesh_actor.GetProperty().SetOpacity(0.30 if trace_mode else 1.0)
    mesh_actor.GetProperty().SetInterpolationToPhong()
    mesh_actor.GetProperty().SetAmbient(0.28)
    mesh_actor.GetProperty().SetDiffuse(0.72)
    mesh_actor.GetProperty().SetSpecular(0.08)
    mesh_actor.GetProperty().SetSpecularPower(20.0)
    # 完整对比使用不透明深度外壳隐藏背面点
    # 单帧追溯故意使用半透明外壳以便显示筛选后的内部缺陷并进行归因
    if not trace_mode:
        mesh_actor.ForceOpaqueOn()

    def cloud_polydata(cloud: o3d.geometry.PointCloud) -> vtk.vtkPolyData | None:
        points = np.asarray(cloud.points, dtype=np.float32)
        if not len(points):
            return None
        point_data = vtk.vtkPolyData()
        point_data.SetPoints(vtk_points(points))
        vertex_cells = np.column_stack(
            (
                np.ones(len(points), dtype=np.int64),
                np.arange(len(points), dtype=np.int64),
            )
        )
        vertices_vtk = vtk.vtkCellArray()
        vertices_vtk.SetCells(
            len(points),
            numpy_to_vtkIdTypeArray(vertex_cells.reshape(-1), deep=True),
        )
        point_data.SetVerts(vertices_vtk)
        colors = np.asarray(cloud.colors, dtype=np.float64)
        if len(colors) != len(points):
            colors = np.full((len(points), 3), 0.9, dtype=np.float64)
        colors_vtk = numpy_to_vtk(
            np.round(np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8),
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )
        colors_vtk.SetNumberOfComponents(3)
        colors_vtk.SetName("Colors")
        point_data.GetPointData().SetScalars(colors_vtk)
        return point_data

    mesh_center = vertices.mean(axis=0)

    def display_cloud(
        cloud: o3d.geometry.PointCloud,
        limit: int,
        *,
        radial_lift_mm: float = 0.0,
        voxel_size_mm: float = 0.0,
    ) -> o3d.geometry.PointCloud:
        """Reduce only the interactive display copy; source PLY files stay full-resolution."""

        reduced = (
            cloud.voxel_down_sample(voxel_size_mm)
            if voxel_size_mm > 0 and not cloud.is_empty()
            else cloud
        )
        count = len(reduced.points)
        if count <= limit:
            pass
        else:
            # 将显示采样分布到完整的源点顺序
            # 扫描顺序通常呈条带状直接每隔固定数量取点会留下可见条带
            # 即使保存的点云是完整的也会出现这个问题
            indices = np.linspace(0, count - 1, num=limit, dtype=np.int64)
            sampled = o3d.geometry.PointCloud()
            sampled.points = o3d.utility.Vector3dVector(
                np.asarray(reduced.points)[indices]
            )
            if reduced.has_colors():
                sampled.colors = o3d.utility.Vector3dVector(
                    np.asarray(reduced.colors)[indices]
                )
            reduced = sampled
        if radial_lift_mm:
            points = np.asarray(reduced.points).copy()
            directions = points - mesh_center
            lengths = np.linalg.norm(directions, axis=1)
            valid = lengths > 1e-9
            points[valid] += (
                directions[valid] / lengths[valid, None]
            ) * float(radial_lift_mm)
            lifted = o3d.geometry.PointCloud()
            lifted.points = o3d.utility.Vector3dVector(points)
            if reduced.has_colors():
                lifted.colors = o3d.utility.Vector3dVector(
                    np.asarray(reduced.colors).copy()
                )
            reduced = lifted
        return reduced

    def point_actor(point_data: vtk.vtkPolyData) -> vtk.vtkActor:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(point_data)
        mapper.SetColorModeToDirectScalars()
        mapper.SetScalarModeToUsePointData()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(DISPLAY_STYLE["point_size"])
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        return actor

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*DISPLAY_STYLE["background"])
    renderer.SetUseDepthPeeling(trace_mode)
    if trace_mode:
        renderer.SetMaximumNumberOfPeels(4)
        renderer.SetOcclusionRatio(0.1)
    renderer.AddActor(mesh_actor)

    window = vtk.vtkRenderWindow()
    window.SetWindowName("STEP Comparison")
    window.SetSize(DISPLAY_STYLE["width"], DISPLAY_STYLE["height"])
    window.SetAlphaBitPlanes(1)
    window.SetMultiSamples(0)
    window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    bounds = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bounds.get_center(), dtype=np.float64)
    extent = max(float(np.max(bounds.get_extent())), 1.0)
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*center)
    camera.SetPosition(center[0], center[1] - 3.0 * extent, center[2])
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.SetViewAngle(30.0)
    renderer.ResetCameraClippingRange()

    # 保持交互响应
    # 这些限制只影响显示不会影响保存的偏差点云和报告统计
    source_clouds = [
        # 覆盖采样点已经位于 STEP 表面
        # 不要使用缺陷点的通用中心径向抬高
        # 在凹槽和斜面上抬高会把采样点推到相邻表面
        cloud_polydata(display_cloud(coverage, 120_000)),
        cloud_polydata(display_cloud(problem, 35_000, radial_lift_mm=DISPLAY_STYLE["cloud_lift"])),
        cloud_polydata(display_cloud(recessed, 100_000, radial_lift_mm=DISPLAY_STYLE["cloud_lift"])),
        cloud_polydata(
            display_cloud(
                missing,
                65_000,
                radial_lift_mm=DISPLAY_STYLE["cloud_lift"],
                voxel_size_mm=blue_cell,
            )
        )
        if show_blue
        else None,
        cloud_polydata(display_cloud(conflict, 35_000, radial_lift_mm=DISPLAY_STYLE["cloud_lift"])),
    ]
    for source in source_clouds:
        if source is not None:
            renderer.AddActor(point_actor(source))
    # 完整场景组装后再初始化并只渲染一次
    # 之前的两阶段渲染在交互开始前无谓地支付了两次透明排序开销
    interactor.Initialize()
    window.Render()
    interactor.Start()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    cloud = report["cloud_to_mesh"]
    surface = report["surface_coverage"]
    mesh = report["mesh"]
    alignment = report["alignment"]
    conformance = report["conformance"]
    directional = cloud["directional_classification"]
    rows = [
        ("流程状态", report["processing_status"]),
        ("符合性结论", conformance["status"]),
        ("配准方法", alignment.get("method", alignment["requested"])),
        ("配准 Fitness", f"{alignment['fitness']:.4f}" if alignment.get("fitness") is not None else "N/A"),
        ("配准 RMSE", f"{alignment['inlier_rmse_mm']:.3f} mm" if alignment.get("inlier_rmse_mm") is not None else "N/A"),
        ("点云数量", f"{report['point_cloud_count']:,}"),
        ("模型三角形", f"{mesh['triangles']:,}"),
        ("实测点 P95 距离", f"{cloud['p95_mm']:.3f} mm"),
        ("实测点容差内", f"{cloud['within_tolerance_ratio'] * 100:.2f}%"),
        ("实测超差点", f"{cloud['bad_count']:,}"),
        ("方向分类", directional["mode"]),
        (
            "外凸超差点",
            f"{directional['exterior_bad_count']:,}"
            if directional["exterior_bad_count"] is not None
            else "N/A",
        ),
        (
            "内凹超差点",
            f"{directional['recessed_bad_count']:,}"
            if directional["recessed_bad_count"] is not None
            else "N/A",
        ),
        ("估算表面观测率", f"{surface['estimated_observed_ratio'] * 100:.2f}%"),
        ("估算未观测比例", f"{surface['estimated_unobserved_ratio'] * 100:.2f}%"),
    ]
    local_fusion = report.get("placement_merge", {}).get("local_fusion")
    visibility = report.get("cad_visibility", {})
    visibility_counts = visibility.get("counts", {})
    if visibility:
        rows.extend(
            [
                ("CAD 可见性", visibility.get("status", "unavailable")),
                ("可见性证据帧", visibility.get("frames", 0)),
                (
                    "可见无回波候选",
                    f"{visibility_counts.get('no_return_candidate', 0):,}",
                ),
                (
                    "可见但未入正式点云",
                    f"{visibility_counts.get('visible_unqualified_return', 0):,}",
                ),
                ("CAD 自遮挡", f"{visibility_counts.get('occluded', 0):,}"),
                ("视场外", f"{visibility_counts.get('out_of_view', 0):,}"),
                (
                    "证据不足",
                    f"{visibility_counts.get('insufficient_evidence', 0):,}",
                ),
            ]
        )
    if local_fusion:
        placement_contribution = ", ".join(
            f"{name}: {count:,}"
            for name, count in local_fusion["output_counts"].items()
        )
        rows.extend(
            [
                ("局部融合面片", f"{local_fusion['surface_cells']:,}"),
                ("多摆放重叠面片", f"{local_fusion['overlap_cells']:,}"),
                ("摆放冲突面片", f"{local_fusion['conflict_cells']:,}"),
                ("未决冲突面片", f"{local_fusion['unresolved_conflict_cells']:,}"),
                ("补充覆盖面片", f"{local_fusion['supplemental_cells']:,}"),
                ("隔离冲突点", f"{local_fusion['conflict_output_points']:,}"),
                ("正式点来源", placement_contribution),
            ]
        )
    table = "".join(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>" for key, value in rows)
    conflict_link = (
        "<a href='../cloud-placement-conflicts.ply'>placement conflicts</a>"
        if local_fusion
        else ""
    )
    body = f"""<!doctype html>
<meta charset='utf-8'><title>STEP / Point Cloud Comparison</title>
<style>body{{font:15px system-ui,sans-serif;max-width:900px;margin:36px auto;color:#20252b}}h1{{font-size:24px}}table{{border-collapse:collapse;min-width:520px}}th,td{{border:1px solid #d5dbe1;padding:9px 12px;text-align:left}}th{{background:#f2f5f7;width:220px}}a{{margin-right:18px}}</style>
<h1>STEP / Point Cloud Comparison</h1>
<p><b>STEP:</b> {html.escape(str(report['step']))}<br><b>Point cloud:</b> {html.escape(str(report['point_cloud']))}<br><b>Tolerance:</b> {report['tolerance_mm']:.3f} mm</p>
<table>{table}</table>
    <p><a href='problem.ply'>problem.ply</a><a href='recessed.ply'>recessed.ply</a><a href='unobserved.ply'>unobserved.ply</a><a href='visible_no_return.ply'>visible-no-return</a><a href='visible_unqualified_return.ply'>visible-unqualified-return</a><a href='occluded.ply'>occluded</a><a href='out_of_view.ply'>out-of-view</a><a href='mesh.ply'>mesh.ply</a>{conflict_link}<a href='report.json'>report.json</a></p>
<p><small>流程成功不代表工件合格。未观测区域不是已确认的材料缺失；在 STEP 网格不封闭时，problem.ply 只表示无符号距离超差，不能区分外凸和内凹。</small></p>
"""
    path.write_text(body, encoding="utf-8")


def compare_step(
    step_path: Path,
    cloud_path: Path,
    output_dir: Path,
    *,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
    alignment: str = "best-fit",
    mesh_samples: int = 300000,
    show_window: bool = False,
    visibility_frames: list[CadVisibilityFrame] | None = None,
    visibility_issues: list[str] | None = None,
) -> ComparisonResult:
    step_path = Path(step_path)
    cloud_path = Path(cloud_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points = _load_points(cloud_path)
    if not len(points):
        raise ValueError(
            f"点云有效点不足：{cloud_path} 没有有限点"
        )
    mesh = load_mesh(step_path, tolerance_mm)
    if alignment == "best-fit":
        if len(points) < 10:
            raise ValueError(
                f"STEP 配准至少需要 10 个有限点，当前只有 {len(points)} 个"
            )
        model_to_cloud, registration = register(mesh, points, tolerance_mm)
        transform = np.linalg.inv(model_to_cloud)
        aligned_points = (transform[:3, :3] @ points.T).T + transform[:3, 3]
        registration_diagnostic = output_dir / "registration.json"
        registration_diagnostic.write_text(
            json.dumps(
                {
                    "source": str(cloud_path),
                    "cloud_transform": transform.tolist(),
                    **registration,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if float(registration["fitness"]) < MIN_STEP_REGISTRATION_FITNESS:
            raise ValueError(
                "STEP 配准失败 "
                f"fitness={registration['fitness']:.3f}，"
                f"低于 {MIN_STEP_REGISTRATION_FITNESS:.3f}"
            )
        if registration.get("ambiguity", {}).get("status") == "ambiguous":
            raise ValueError(
                "STEP 配准存在近似同分的不同姿态；"
                "需要 CAD 基准或夹具基准消除对称歧义"
            )
    elif alignment == "identity":
        transform = np.eye(4)
        aligned_points = points
        registration = {"method": "identity", "fitness": None, "inlier_rmse_mm": None}
    else:
        raise ValueError(f"不支持的 alignment: {alignment}")

    cloud_distances = _cloud_distances(mesh, aligned_points)
    registration_gate = (
        _quality_gate(cloud_distances, tolerance_mm)
        if alignment == "best-fit"
        else None
    )
    if registration_gate is not None and registration_gate["status"] != "pass":
        diagnostic = json.loads(registration_diagnostic.read_text(encoding="utf-8"))
        diagnostic["quality_gate"] = registration_gate
        registration_diagnostic.write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise ValueError(
            "STEP 配准质量不合格 "
            f"p50={registration_gate['p50_mm']:.3f} mm "
            f"p90={registration_gate['p90_mm']:.3f} mm"
        )
    direction_quality = _direction_quality(mesh)
    signed_reliable = direction_quality["reliable"]
    cloud_signed_distances = (
        _signed_distances(mesh, aligned_points)
        if signed_reliable
        else None
    )
    classification = _classify_deviation(
        mesh,
        cloud_distances,
        tolerance_mm,
        signed_distances=cloud_signed_distances,
        signed_distance_reliable=signed_reliable,
    )
    sampled = np.asarray(mesh.sample_points_uniformly(number_of_points=max(1000, mesh_samples)).points)
    mesh_distances = _mesh_distances(sampled, aligned_points)
    unobserved_mask = mesh_distances > tolerance_mm
    visibility = classify_visibility(
        mesh,
        sampled,
        ~unobserved_mask,
        visibility_frames or [],
        surface_tolerance_mm=tolerance_mm,
        evidence_complete=bool(visibility_frames) and not visibility_issues,
    )
    bad_points = aligned_points[classification.problem_mask]
    recessed_points = aligned_points[classification.recessed_mask]
    unobserved_points = sampled[unobserved_mask]
    _write_cloud(
        output_dir / "problem.ply",
        bad_points,
        colors=_distance_colors(
            cloud_distances[classification.problem_mask], tolerance_mm, kind="problem"
        ),
    )
    _write_cloud(
        output_dir / "recessed.ply",
        recessed_points,
        colors=_distance_colors(
            cloud_distances[classification.recessed_mask], tolerance_mm, kind="recessed"
        ),
    )
    _write_cloud(
        output_dir / "unobserved.ply",
        unobserved_points,
        colors=_distance_colors(
            mesh_distances[unobserved_mask], tolerance_mm, kind="missing"
        ),
    )
    # Keep the historical filename for viewers and downstream tools. Its
    # contents are observational coverage gaps, not confirmed missing material.
    _write_cloud(
        output_dir / "missing.ply",
        unobserved_points,
        colors=_distance_colors(
            mesh_distances[unobserved_mask], tolerance_mm, kind="missing"
        ),
    )
    visibility_artifacts = {
        "visible_no_return": (
            visibility.no_return,
            (0.95, 0.15, 0.75),
        ),
        "visible_unqualified_return": (
            visibility.bad_return,
            (0.95, 0.75, 0.05),
        ),
        "occluded": (visibility.occluded_mask, (0.15, 0.35, 0.95)),
        "out_of_view": (visibility.outside, (0.45, 0.48, 0.52)),
        "insufficient_evidence": (
            visibility.insufficient_evidence_mask,
            (0.10, 0.75, 0.80),
        ),
    }
    for artifact_name, (artifact_mask, artifact_color) in visibility_artifacts.items():
        _write_cloud(
            output_dir / f"{artifact_name}.ply",
            sampled[artifact_mask],
            artifact_color,
        )
    np.save(output_dir / "cloud.npy", aligned_points)
    o3d.io.write_triangle_mesh(str(output_dir / "mesh.ply"), mesh)

    report = {
        "status": "ok",
        "processing_status": "ok",
        "status_scope": "processing_only",
        "step": str(step_path),
        "point_cloud": str(cloud_path),
        "output_dir": str(output_dir),
        "point_unit": "mm",
        "alignment": {
            "requested": alignment,
            "cloud_transform": transform.tolist(),
            "quality_gate": registration_gate,
            **registration,
        },
        "tolerance_mm": float(tolerance_mm),
        "mesh": {
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
            **direction_quality,
        },
        "point_cloud_count": int(len(aligned_points)),
        "cloud_to_mesh": {
            **_stats(cloud_distances, tolerance_mm),
            "unsigned_bad_count": int(np.count_nonzero(cloud_distances > tolerance_mm)),
            "directional_classification": {
                "mode": classification.mode,
                "reliable": classification.signed_distance_reliable,
                "exterior_bad_count": (
                    int(np.count_nonzero(classification.exterior_mask))
                    if classification.signed_distance_reliable
                    else None
                ),
                "recessed_bad_count": (
                    int(np.count_nonzero(classification.recessed_mask))
                    if classification.signed_distance_reliable
                    else None
                ),
                "unclassified_bad_count": int(
                    np.count_nonzero(classification.unclassified_mask)
                ),
            },
            # Legacy fields remain machine-readable but are null when direction
            # cannot be justified by a watertight tessellation.
            "exterior_bad_count": (
                int(np.count_nonzero(classification.exterior_mask))
                if classification.signed_distance_reliable
                else None
            ),
            "exterior_bad_ratio": (
                float(np.mean(classification.exterior_mask))
                if classification.signed_distance_reliable
                else None
            ),
            "recessed_bad_count": (
                int(np.count_nonzero(classification.recessed_mask))
                if classification.signed_distance_reliable
                else None
            ),
            "recessed_bad_ratio": (
                float(np.mean(classification.recessed_mask))
                if classification.signed_distance_reliable
                else None
            ),
        },
        "mesh_to_cloud": {**_stats(mesh_distances, tolerance_mm), "sample_count": int(len(sampled)), "coverage_ratio": float(np.mean(mesh_distances <= tolerance_mm))},
        "artifacts": {
            "problem": str(output_dir / "problem.ply"),
            "recessed": str(output_dir / "recessed.ply"),
            "unobserved": str(output_dir / "unobserved.ply"),
            "missing_legacy_alias": str(output_dir / "missing.ply"),
            **{
                name: str(output_dir / f"{name}.ply")
                for name in visibility_artifacts
            },
            "cloud": str(output_dir / "cloud.npy"),
            "mesh": str(output_dir / "mesh.ply"),
            "htmlreport": str(output_dir / "report.html"),
        },
        "surface_coverage": {
            "estimated_observed_ratio": float(np.mean(mesh_distances <= tolerance_mm)),
            "estimated_unobserved_ratio": float(np.mean(mesh_distances > tolerance_mm)),
            "estimated_covered_ratio": float(np.mean(mesh_distances <= tolerance_mm)),
            "estimated_missing_ratio": float(np.mean(mesh_distances > tolerance_mm)),
            "classification": (
                "cad_visibility_partition"
                if visibility.report["status"] == "available"
                else "observational_coverage_only"
            ),
            "visibility_analysis": visibility.report["status"],
            "missing_material_evaluated": False,
            "method": "uniform surface samples; ratios approximate observed and unobserved area ratios",
            "note": (
                "unobserved samples may be occluded, outside the reachable camera views, "
                "or rejected by sensor evidence; they are not confirmed missing material"
            ),
        },
        "cad_visibility": {
            **visibility.report,
            "load_issues": list(visibility_issues or []),
            "mesh_limitations": [
                name
                for name, limited in (
                    ("mesh_not_watertight", not direction_quality["watertight"]),
                    ("mesh_self_intersecting", direction_quality["self_intersecting"]),
                    ("mesh_not_orientable", not direction_quality["orientable"]),
                )
                if limited
            ],
        },
        "conformance": {
            "status": "indeterminate",
            "reasons": [
                *(
                    ["surface visibility evidence is unavailable"]
                    if visibility.report["status"] != "available"
                    else [
                        "visible no-return samples are capture follow-up candidates, not confirmed missing material"
                    ]
                ),
                "feature-level acceptance rules are not configured",
                *(
                    []
                    if classification.signed_distance_reliable
                    else ["STEP tessellation is not watertight; deviation direction is unavailable"]
                ),
            ],
            "note": "processing success and registration plausibility do not establish workpiece conformance",
        },
        "distance_note": (
            "Absolute nearest-surface distance determines point-level tolerance excess. "
            + (
                "Positive signed excess is stored in problem.ply and negative signed excess in recessed.ply."
                if classification.signed_distance_reliable
                else "The mesh is not watertight, so all unsigned excess is stored in problem.ply without directional meaning."
            )
        ),
        "visualization": {
            "color_encoding": (
                "signed_distance_class_and_absolute_distance_excess"
                if classification.signed_distance_reliable
                else "unsigned_distance_excess"
            ),
            "problem": (
                "light orange to vermilion; darker means larger exterior distance excess"
                if classification.signed_distance_reliable
                else "light orange to vermilion; darker means larger unsigned distance excess"
            ),
            "recessed": (
                "bright lime green; darker means larger recessed distance excess"
                if classification.signed_distance_reliable
                else "empty because directional classification is unavailable"
            ),
            "unobserved": "electric blue; darker means larger model-to-cloud coverage gap",
            "scale_upper_clip": "95th percentile of displayed excess distances",
            "confidence_note": "Color depth expresses distance severity, not calibrated measurement confidence.",
            "problem_side": (
                "positive signed distance and unclassified signed-zero excess"
                if classification.signed_distance_reliable
                else "all unsigned point-to-surface distance excess"
            ),
            "recessed_side": (
                "negative signed distance and absolute distance over tolerance"
                if classification.signed_distance_reliable
                else "not evaluated"
            ),
            "signed_distance_reliable": classification.signed_distance_reliable,
            "signed_distance_note": (
                "STEP mesh is not watertight; exterior/interior classification is disabled"
                if not classification.signed_distance_reliable
                else "STEP mesh is watertight; positive signed distance denotes exterior points"
            ),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(output_dir / "report.html", report)
    if show_window:
        show_comparison(output_dir)
    return ComparisonResult(report=report, output_dir=output_dir)


def merge_scans(
    step_path: str | Path,
    placement_clouds: list[tuple[str, str | Path]],
    output_dir: str | Path,
    *,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
    voxel_size: float = 0.05,
    mesh_samples: int = 300000,
    fusion_mode: str = "nominal",
) -> ComparisonResult:
    """将多个放置面配准合并并与 STEP 对比"""

    if tolerance_mm <= 0:
        raise ValueError("tolerance_mm must be positive")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if fusion_mode not in {"nominal", "consensus"}:
        raise ValueError("fusion_mode must be nominal or consensus")
    step_path = Path(step_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not placement_clouds:
        raise ValueError("至少需要一个放置面点云")

    mesh = load_mesh(step_path, tolerance_mm)
    aligned_clouds: list[np.ndarray] = []
    alignments: dict[str, dict[str, Any]] = {}
    rejected_artifacts: dict[str, dict[str, str]] = {}
    visibility_frames: list[CadVisibilityFrame] = []
    visibility_issues: list[str] = []
    outlier_limit_mm = _outlier_limit(tolerance_mm)
    for name, raw_cloud_path in placement_clouds:
        cloud_path = Path(raw_cloud_path)
        base_points = _load_points(cloud_path)
        if len(base_points) < 10:
            raise ValueError(
                f"放置面 {name} 有效点不足：只有 {len(base_points)} 个有限点，至少需要 10 个"
            )
        measure_path = _measure_path(cloud_path)
        measure_points = _load_points(measure_path)
        if len(measure_points) < 10:
            measure_path = cloud_path
            measure_points = base_points
        measure_mode = (
            "local_consensus"
            if measure_path != cloud_path
            else "input_cloud"
        )
        model_to_cloud, registration = register(mesh, base_points, tolerance_mm)
        transform = np.linalg.inv(model_to_cloud)
        aligned_base = (
            (transform[:3, :3] @ base_points.T).T + transform[:3, 3]
        )
        measure_all = (
            (transform[:3, :3] @ measure_points.T).T + transform[:3, 3]
        )
        registration_diagnostic = output_dir / f"{name}-registration.json"
        registration_diagnostic.write_text(
            json.dumps(
                {
                    "placement": name,
                    "source": str(cloud_path),
                    "step_transform": transform.tolist(),
                    **registration,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if float(registration["fitness"]) < MIN_STEP_REGISTRATION_FITNESS:
            raise ValueError(
                f"{name} STEP 配准失败 fitness={registration['fitness']:.3f}"
            )
        if registration.get("ambiguity", {}).get("status") == "ambiguous":
            raise ValueError(
                f"{name} STEP 配准存在近似同分的不同姿态；"
                "需要 CAD 基准或夹具基准消除对称歧义"
            )

        step_distances = _cloud_distances(mesh, aligned_base)
        quality_gate = _quality_gate(step_distances, tolerance_mm)
        if quality_gate["status"] != "pass":
            raise ValueError(
                f"{name} STEP 配准质量不合格 "
                f"p50={quality_gate['p50_mm']:.3f} mm "
                f"p90={quality_gate['p90_mm']:.3f} mm"
            )

        measure_distances = _cloud_distances(mesh, measure_all)
        retained_mask = measure_distances <= outlier_limit_mm
        aligned = measure_all[retained_mask]
        rejected = measure_all[~retained_mask]
        retained_ratio = float(len(aligned) / len(measure_all))
        if retained_ratio < MIN_REGISTRATION_RETAINED_RATIO:
            raise ValueError(
                f"{name} STEP 配准后仅保留 {retained_ratio:.1%} 点，"
                f"低于最低要求 {MIN_REGISTRATION_RETAINED_RATIO:.1%}"
            )

        aligned_clouds.append(aligned)
        np.save(output_dir / f"{name}.npy", aligned.astype(np.float32))
        _write_cloud(output_dir / f"{name}.ply", aligned, (0.65, 0.68, 0.72))
        rejected_npy = output_dir / f"{name}-registration-outliers.npy"
        rejected_ply = output_dir / f"{name}-registration-outliers.ply"
        np.save(rejected_npy, rejected.astype(np.float32))
        _write_cloud(rejected_ply, rejected, (0.84, 0.18, 0.0))
        rejected_artifacts[name] = {
            "npy": str(rejected_npy),
            "ply": str(rejected_ply),
        }
        alignments[name] = {
            "registration_source": str(cloud_path),
            "measurement_source": str(measure_path),
            "measurement_mode": measure_mode,
            "input_points": int(len(base_points)),
            "measurement_points": int(len(measure_points)),
            "retained_points": int(len(aligned)),
            "registration_outlier_points": int(len(rejected)),
            "outlier_limit": float(outlier_limit_mm),
            "retained_ratio": retained_ratio,
            "minimum_fitness": MIN_STEP_REGISTRATION_FITNESS,
            "step_transform": transform.tolist(),
            "reference": "step",
            "diagnostic": str(registration_diagnostic),
            "prefilter_distance": _stats(
                step_distances, tolerance_mm
            ),
            "measurement_distance": _stats(
                measure_distances, tolerance_mm
            ),
            "quality_gate": quality_gate,
            **registration,
        }
        placement_frames, placement_issues = load_frames(
            cloud_path.parent,
            transform,
        )
        visibility_frames.extend(placement_frames)
        visibility_issues.extend(placement_issues)

    fusion = fuse_layers(
        mesh,
        [
            PlacementFusionLayer(name=name, points=points)
            for (name, _), points in zip(placement_clouds, aligned_clouds)
        ],
        LocalPlacementFusionConfig.for_metrology(
            tolerance_mm=tolerance_mm,
            voxel_size=voxel_size,
            selection_mode=fusion_mode,
        ),
    )
    combined = fusion.points
    combined_path = output_dir / "cloud.npy"
    np.save(combined_path, combined)
    _write_cloud(output_dir / "cloud.ply", combined, (0.72, 0.75, 0.78))
    conflict_npy = output_dir / "cloud-placement-conflicts.npy"
    conflict_ply = output_dir / "cloud-placement-conflicts.ply"
    provenance_path = output_dir / "cloud-provenance.npz"
    np.save(conflict_npy, fusion.conflict_points)
    _write_cloud(conflict_ply, fusion.conflict_points, (0.90, 0.32, 0.05))
    np.savez(
        provenance_path,
        source_placement=fusion.source_placement,
        surface_cell=fusion.surface_cell,
        scatter_score=fusion.scatter_score,
        placement_support=fusion.placement_support,
        conflict=fusion.conflict,
        supplemental=fusion.supplemental,
        conflict_source_placement=fusion.conflict_source_placement,
        conflict_surface_cell=fusion.conflict_surface_cell,
        placement_names=np.asarray([name for name, _ in placement_clouds]),
    )
    if len(combined) < 10:
        raise ValueError(
            "多摆放局部融合没有产生足够的正式点；"
            "观测可能全部处于未决冲突，请检查冲突点云和配准"
        )

    merge_report = {
        "status": "ok",
        "processing_status": "ok",
        "status_scope": "processing_and_registration_only",
        "method": (
            "base-registration-and-nominal-priority-fusion"
            if fusion_mode == "nominal"
            else "base-registration-and-consensus-measurement-fusion"
        ),
        "fusion_mode": fusion_mode,
        "cad_bias": fusion_mode == "nominal",
        "measurement_policy": (
            "register each placement with its broad support cloud; use its local "
            "consensus cloud for measurement; when placements conflict select the "
            "cluster closest to STEP"
            if fusion_mode == "nominal"
            else
            "register each placement with its broad support cloud; use its local "
            "three-view consensus cloud for measurement and cross-placement fusion"
        ),
        "step": str(step_path),
        "point_unit": "mm",
        "tolerance_mm": float(tolerance_mm),
        "voxel_size_mm": float(voxel_size),
        "placements": [name for name, _ in placement_clouds],
        "alignments": alignments,
        "local_fusion": fusion.report,
        "quality_gate": {
            "status": "pass",
            "scope": "registration_plausibility_only",
            "establishes_workpiece_conformance": False,
            "minimum_fitness": MIN_STEP_REGISTRATION_FITNESS,
            "minimum_retained_ratio": MIN_REGISTRATION_RETAINED_RATIO,
            "outlier_limit": float(outlier_limit_mm),
            "failed_placements": [],
        },
        "combined_points": int(len(combined)),
        "cad_visibility_evidence": {
            "frames": len(visibility_frames),
            "load_issues": visibility_issues,
        },
        "artifacts": {
            "cloud": str(output_dir / "cloud.ply"),
            "provenance": str(provenance_path),
            "placement_conflicts": {
                "npy": str(conflict_npy),
                "ply": str(conflict_ply),
            },
            "placements": {
                name: str(output_dir / f"{name}.ply")
                for name, _ in placement_clouds
            },
            "registration_outliers": rejected_artifacts,
        },
    }
    (output_dir / "merge.json").write_text(
        json.dumps(merge_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = compare_step(
        step_path,
        combined_path,
        output_dir / "view",
        tolerance_mm=tolerance_mm,
        alignment="identity",
        mesh_samples=mesh_samples,
        visibility_frames=visibility_frames,
        visibility_issues=visibility_issues,
    )
    combined_p50 = float(result.report["cloud_to_mesh"]["p50_mm"])
    combined_p90 = float(result.report["cloud_to_mesh"]["p90_mm"])
    p50_limit = max(2.5 * tolerance_mm, 0.25)
    p90_limit = max(7.5 * tolerance_mm, 0.75)
    combined_gate_passed = combined_p50 <= p50_limit and combined_p90 <= p90_limit
    merge_report["quality_gate"]["combined_step_distance"] = {
        "status": "pass" if combined_gate_passed else "fail",
        "p50_mm": combined_p50,
        "p90_mm": combined_p90,
        "maximum_p50_mm": float(p50_limit),
        "maximum_p90_mm": float(p90_limit),
    }
    merge_report["quality_gate"]["status"] = (
        "pass" if combined_gate_passed else "fail"
    )
    if not combined_gate_passed:
        merge_report["status"] = "error"
        merge_report["processing_status"] = "error"
        result.report["status"] = "error"
        result.report["processing_status"] = "error"
    result.report["placement_merge"] = merge_report
    result.report["quality_gate"] = merge_report["quality_gate"]
    merge_report["conformance"] = result.report["conformance"]
    (output_dir / "merge.json").write_text(
        json.dumps(merge_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (result.output_dir / "report.json").write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(result.output_dir / "report.html", result.report)
    if not combined_gate_passed:
        raise ValueError(
            "合并点云 STEP 配准质量不合格 "
            f"p50={combined_p50:.3f} mm p90={combined_p90:.3f} mm"
        )
    return result
