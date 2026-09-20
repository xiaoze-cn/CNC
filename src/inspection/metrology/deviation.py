"""STEP-to-cloud deviation measurement and result generation."""

from __future__ import annotations

import json
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from .model import load_step_mesh
from .registration import register, register_clouds


DEFAULT_TOLERANCE_MM = 0.1
MIN_STEP_REGISTRATION_FITNESS = 0.2


@dataclass
class ComparisonResult:
    report: dict[str, Any]
    output_dir: Path


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


def _load_points(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        points = np.load(path)
    else:
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def _mesh_to_cloud_distances(mesh_points: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    cloud_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))
    tree = o3d.geometry.KDTreeFlann(cloud_pcd)
    distances = np.empty(len(mesh_points), dtype=np.float64)
    for start in range(0, len(mesh_points), 5000):
        for index, point in enumerate(mesh_points[start:start + 5000], start):
            count, _, squared = tree.search_knn_vector_3d(point, 1)
            distances[index] = np.sqrt(squared[0]) if count else np.inf
    return distances


def _cloud_to_mesh_distances(mesh: o3d.geometry.TriangleMesh, points: np.ndarray) -> np.ndarray:
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


def _cloud_to_mesh_signed_distances(
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
        dark = np.asarray([0.84, 0.18, 0.0])
    elif kind == "recessed":
        light = np.asarray([0.62, 1.0, 0.12])
        dark = np.asarray([0.08, 0.82, 0.0])
    else:
        light = np.asarray([0.16, 0.72, 1.0])
        dark = np.asarray([0.0, 0.16, 1.0])
    return light[None, :] * (1.0 - level[:, None]) + dark[None, :] * level[:, None]


def show_comparison(output_dir: Path) -> None:
    """Show deviations over a matte STEP surface with GPU depth occlusion."""

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
    paths = {
        "mesh": output_dir / "mesh.ply",
        "cloud": output_dir / "cloud.ply",
        "coverage": output_dir / "coverage.ply",
        "problem": output_dir / "problem.ply",
        "recessed": output_dir / "recessed.ply",
        "missing": output_dir / "missing.ply",
    }
    missing_files = [
        str(path)
        for name, path in paths.items()
        if name not in {"recessed", "cloud", "coverage"} and not path.is_file()
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
    mesh.paint_uniform_color((0.58, 0.62, 0.66))
    problem = read_cloud(paths["problem"])
    if not problem.has_colors():
        problem.paint_uniform_color((0.84, 0.18, 0.0))
    cloud = (
        read_cloud(paths["cloud"])
        if paths["cloud"].is_file()
        else o3d.geometry.PointCloud()
    )
    if not cloud.is_empty() and not cloud.has_colors():
        cloud.paint_uniform_color((0.62, 0.66, 0.72))
    coverage = (
        read_cloud(paths["coverage"])
        if paths["coverage"].is_file()
        else o3d.geometry.PointCloud()
    )
    if not coverage.is_empty() and not coverage.has_colors():
        coverage.paint_uniform_color((0.0, 0.75, 1.0))
    recessed = (
        read_cloud(paths["recessed"])
        if paths["recessed"].is_file()
        else o3d.geometry.PointCloud()
    )
    if not recessed.is_empty() and not recessed.has_colors():
        recessed.paint_uniform_color((0.08, 0.82, 0.0))
    missing = read_cloud(paths["missing"])
    if not missing.has_colors():
        missing.paint_uniform_color((0.0, 0.16, 1.0))

    def vtk_points(values: np.ndarray) -> vtk.vtkPoints:
        result = vtk.vtkPoints()
        result.SetData(
            numpy_to_vtk(np.asarray(values, dtype=np.float32), deep=True)
        )
        return result

    mesh_data = vtk.vtkPolyData()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    mesh_data.SetPoints(vtk_points(vertices))
    triangle_cells = np.column_stack(
        (np.full(len(triangles), 3, dtype=np.int64), triangles)
    )
    mesh_cells = vtk.vtkCellArray()
    mesh_cells.SetCells(
        len(triangles),
        numpy_to_vtkIdTypeArray(triangle_cells.reshape(-1), deep=True),
    )
    mesh_data.SetPolys(mesh_cells)
    triangle_centers = vertices[triangles].mean(axis=1)
    center = vertices.mean(axis=0)
    radial_distance = np.linalg.norm(triangle_centers - center, axis=1)
    low, high = np.percentile(radial_distance, (5.0, 95.0))
    if high - low <= 1e-6:
        radial_level = np.full(len(radial_distance), 0.5, dtype=np.float32)
    else:
        radial_level = np.clip(
            (radial_distance - low) / (high - low), 0.0, 1.0
        ).astype(np.float32)
    # 使用从内到外的固定色调渐变
    # 外壳故意设为不透明以进行深度测试避免隐藏的背面点穿透显示
    tone = np.round(125.0 + 55.0 * radial_level).astype(np.uint8)
    mesh_colors = np.column_stack(
        (
            tone,
            np.minimum(tone.astype(np.uint16) + 8, 255).astype(np.uint8),
            np.minimum(tone.astype(np.uint16) + 18, 255).astype(np.uint8),
            np.full(len(triangles), 255, dtype=np.uint8),
        )
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
    mesh_actor.GetProperty().SetAmbient(1.0)
    mesh_actor.GetProperty().SetDiffuse(0.0)
    mesh_actor.GetProperty().SetSpecular(0.0)
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
    ) -> o3d.geometry.PointCloud:
        """Reduce only the interactive display copy; source PLY files stay full-resolution."""

        count = len(cloud.points)
        if count <= limit:
            reduced = cloud
        else:
            # 将显示采样分布到完整的源点顺序
            # 扫描顺序通常呈条带状直接每隔固定数量取点会留下可见条带
            # 即使保存的点云是完整的也会出现这个问题
            indices = np.linspace(0, count - 1, num=limit, dtype=np.int64)
            reduced = o3d.geometry.PointCloud()
            reduced.points = o3d.utility.Vector3dVector(np.asarray(cloud.points)[indices])
            if cloud.has_colors():
                reduced.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[indices])
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
        actor.GetProperty().SetPointSize(4.0)
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        return actor

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.08, 0.09, 0.10)
    renderer.SetUseDepthPeeling(trace_mode)
    if trace_mode:
        renderer.SetMaximumNumberOfPeels(4)
        renderer.SetOcclusionRatio(0.1)
    renderer.AddActor(mesh_actor)

    window = vtk.vtkRenderWindow()
    window.SetWindowName("STEP Comparison")
    window.SetSize(1280, 800)
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
        cloud_polydata(display_cloud(cloud, 120_000, radial_lift_mm=0.35))
        if not trace_mode
        else None,
        # 覆盖采样点已经位于 STEP 表面
        # 不要使用缺陷点的通用中心径向抬高
        # 在凹槽和斜面上抬高会把采样点推到相邻表面
        cloud_polydata(display_cloud(coverage, 120_000)),
        cloud_polydata(display_cloud(problem, 35_000, radial_lift_mm=0.35)),
        cloud_polydata(display_cloud(recessed, 100_000, radial_lift_mm=0.35)),
        cloud_polydata(display_cloud(missing, 65_000, radial_lift_mm=0.35)),
    ]
    for source in source_clouds:
        if source is not None:
            renderer.AddActor(point_actor(source))
    # 完整场景组装后再初始化并只渲染一次
    # 之前的两阶段渲染在交互开始前无谓地支付了两次透明排序开销
    interactor.Initialize()
    window.Render()
    interactor.Start()


def _write_html_report(path: Path, report: dict[str, Any]) -> None:
    cloud = report["cloud_to_mesh"]
    surface = report["surface_coverage"]
    mesh = report["mesh"]
    alignment = report["alignment"]
    rows = [
        ("配准方法", alignment.get("method", alignment["requested"])),
        ("配准 Fitness", f"{alignment['fitness']:.4f}" if alignment.get("fitness") is not None else "N/A"),
        ("配准 RMSE", f"{alignment['inlier_rmse_mm']:.3f} mm" if alignment.get("inlier_rmse_mm") is not None else "N/A"),
        ("点云数量", f"{report['point_cloud_count']:,}"),
        ("模型三角形", f"{mesh['triangles']:,}"),
        ("实测点 P95 距离", f"{cloud['p95_mm']:.3f} mm"),
        ("实测点容差内", f"{cloud['within_tolerance_ratio'] * 100:.2f}%"),
        ("实测超差点", f"{cloud['bad_count']:,}"),
        ("外凸超差点", f"{cloud.get('exterior_bad_count', 0):,}"),
        ("内凹超差点", f"{cloud.get('recessed_bad_count', 0):,}"),
        ("估算表面覆盖率", f"{surface['estimated_covered_ratio'] * 100:.2f}%"),
        ("估算缺失比例", f"{surface['estimated_missing_ratio'] * 100:.2f}%"),
    ]
    table = "".join(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>" for key, value in rows)
    body = f"""<!doctype html>
<meta charset='utf-8'><title>STEP / Point Cloud Comparison</title>
<style>body{{font:15px system-ui,sans-serif;max-width:900px;margin:36px auto;color:#20252b}}h1{{font-size:24px}}table{{border-collapse:collapse;min-width:520px}}th,td{{border:1px solid #d5dbe1;padding:9px 12px;text-align:left}}th{{background:#f2f5f7;width:220px}}a{{margin-right:18px}}</style>
<h1>STEP / Point Cloud Comparison</h1>
<p><b>STEP:</b> {html.escape(str(report['step']))}<br><b>Point cloud:</b> {html.escape(str(report['point_cloud']))}<br><b>Tolerance:</b> {report['tolerance_mm']:.3f} mm</p>
<table>{table}</table>
    <p><a href='problem.ply'>problem.ply</a><a href='recessed.ply'>recessed.ply</a><a href='missing.ply'>missing.ply</a><a href='mesh.ply'>mesh.ply</a><a href='report.json'>report.json</a></p>
<p><small>Distances are unsigned nearest-surface distances. Color depth expresses distance excess over tolerance, not calibrated measurement confidence. Surface coverage is estimated from uniform model samples.</small></p>
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
) -> ComparisonResult:
    step_path = Path(step_path)
    cloud_path = Path(cloud_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points = _load_points(cloud_path)
    mesh = load_step_mesh(step_path, tolerance_mm)
    if alignment == "best-fit":
        model_to_cloud, registration = register(mesh, points, tolerance_mm)
        transform = np.linalg.inv(model_to_cloud)
        aligned_points = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    elif alignment == "identity":
        transform = np.eye(4)
        aligned_points = points
        registration = {"method": "identity", "fitness": None, "inlier_rmse_mm": None}
    else:
        raise ValueError(f"不支持的 alignment: {alignment}")

    cloud_distances = _cloud_to_mesh_distances(mesh, aligned_points)
    cloud_signed_distances = _cloud_to_mesh_signed_distances(mesh, aligned_points)
    sampled = np.asarray(mesh.sample_points_uniformly(number_of_points=max(1000, mesh_samples)).points)
    mesh_distances = _mesh_to_cloud_distances(sampled, aligned_points)
    # 将绝对距离超差分为外部材料和凹陷
    bad_mask = (cloud_distances > tolerance_mm) & (cloud_signed_distances > 0.0)
    recessed_mask = (cloud_distances > tolerance_mm) & (cloud_signed_distances < 0.0)
    missing_mask = mesh_distances > tolerance_mm
    bad_points = aligned_points[bad_mask]
    recessed_points = aligned_points[recessed_mask]
    missing_points = sampled[missing_mask]
    _write_cloud(
        output_dir / "problem.ply",
        bad_points,
        colors=_distance_colors(
            cloud_distances[bad_mask], tolerance_mm, kind="problem"
        ),
    )
    _write_cloud(
        output_dir / "recessed.ply",
        recessed_points,
        colors=_distance_colors(
            cloud_distances[recessed_mask], tolerance_mm, kind="recessed"
        ),
    )
    _write_cloud(
        output_dir / "missing.ply",
        missing_points,
        colors=_distance_colors(
            mesh_distances[missing_mask], tolerance_mm, kind="missing"
        ),
    )
    np.save(output_dir / "cloud.npy", aligned_points)
    o3d.io.write_triangle_mesh(str(output_dir / "mesh.ply"), mesh)

    report = {
        "status": "ok",
        "step": str(step_path),
        "point_cloud": str(cloud_path),
        "output_dir": str(output_dir),
        "point_unit": "mm",
        "alignment": {"requested": alignment, "transform_cloud_to_model": transform.tolist(), **registration},
        "tolerance_mm": float(tolerance_mm),
        "mesh": {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles), "watertight": bool(mesh.is_watertight())},
        "point_cloud_count": int(len(aligned_points)),
        "cloud_to_mesh": {
            **_stats(cloud_distances, tolerance_mm),
            "exterior_bad_count": int(np.count_nonzero(bad_mask)),
            "exterior_bad_ratio": float(np.mean(bad_mask)),
            "recessed_bad_count": int(np.count_nonzero(recessed_mask)),
            "recessed_bad_ratio": float(np.mean(recessed_mask)),
        },
        "mesh_to_cloud": {**_stats(mesh_distances, tolerance_mm), "sample_count": int(len(sampled)), "coverage_ratio": float(np.mean(mesh_distances <= tolerance_mm))},
        "artifacts": {"problem": str(output_dir / "problem.ply"), "recessed": str(output_dir / "recessed.ply"), "missing": str(output_dir / "missing.ply"), "cloud": str(output_dir / "cloud.npy"), "mesh": str(output_dir / "mesh.ply"), "htmlreport": str(output_dir / "report.html")},
        "surface_coverage": {
            "estimated_covered_ratio": float(np.mean(mesh_distances <= tolerance_mm)),
            "estimated_missing_ratio": float(np.mean(mesh_distances > tolerance_mm)),
            "method": "uniform surface samples; ratios approximate area ratios",
        },
        "distance_note": (
            "Absolute nearest-surface distance determines tolerance failure. "
            "Positive signed failures are stored in problem.ply; negative "
            "signed failures are stored in recessed.ply."
        ),
        "visualization": {
            "color_encoding": "signed_distance_class_and_absolute_distance_excess",
            "problem": "light orange to vermilion; darker means larger exterior distance excess",
            "recessed": "bright lime green; darker means larger recessed distance excess",
            "missing": "electric blue; darker means larger mesh-to-cloud distance excess",
            "scale_upper_clip": "95th percentile of displayed excess distances",
            "confidence_note": "Color depth expresses distance severity, not calibrated measurement confidence.",
            "problem_side": "only points with positive signed distance (outside the STEP surface) are displayed",
            "recessed_side": "points with negative signed distance and absolute distance over tolerance are displayed",
            "signed_distance_reliable": bool(mesh.is_watertight()),
            "signed_distance_note": (
                "STEP mesh is not watertight; exterior/interior sign is an approximate display classification"
                if not mesh.is_watertight()
                else "STEP mesh is watertight; positive signed distance denotes exterior points"
            ),
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html_report(output_dir / "report.html", report)
    if show_window:
        show_comparison(output_dir)
    return ComparisonResult(report=report, output_dir=output_dir)


def merge_placement_scans(
    step_path: str | Path,
    placement_clouds: list[tuple[str, str | Path]],
    output_dir: str | Path,
    *,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
    voxel_size: float = 0.05,
    mesh_samples: int = 300000,
) -> ComparisonResult:
    """将多个放置面配准合并并与 STEP 对比"""

    if tolerance_mm <= 0:
        raise ValueError("tolerance_mm must be positive")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    step_path = Path(step_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not placement_clouds:
        raise ValueError("至少需要一个放置面点云")

    mesh = load_step_mesh(step_path, tolerance_mm)
    aligned_clouds: list[np.ndarray] = []
    alignments: dict[str, dict[str, Any]] = {}
    reference: np.ndarray | None = None
    for index, (name, raw_cloud_path) in enumerate(placement_clouds):
        cloud_path = Path(raw_cloud_path)
        points = _load_points(cloud_path)
        if index == 0:
            model_to_cloud, registration = register(mesh, points, tolerance_mm)
            if float(registration["fitness"]) < MIN_STEP_REGISTRATION_FITNESS:
                raise ValueError(
                    f"{name} STEP 配准失败 fitness={registration['fitness']:.3f}"
                )
            transform = np.linalg.inv(model_to_cloud)
            aligned = (transform[:3, :3] @ points.T).T + transform[:3, 3]
            alignment_key = "transform_cloud_to_step"
        else:
            if reference is None:
                raise RuntimeError("缺少多面 ICP 公共参考点云")
            transform, registration = register_clouds(reference, points, tolerance_mm)
            aligned = (transform[:3, :3] @ points.T).T + transform[:3, 3]
            alignment_key = "transform_cloud_to_step"
        if float(registration["fitness"]) < MIN_STEP_REGISTRATION_FITNESS:
            raise ValueError(
                f"{name} ICP 配准失败 fitness={registration['fitness']:.3f}"
            )
        aligned_clouds.append(aligned)
        np.save(output_dir / f"{name}.npy", aligned.astype(np.float32))
        _write_cloud(output_dir / f"{name}.ply", aligned, (0.65, 0.68, 0.72))
        alignments[name] = {
            "source": str(cloud_path),
            "input_points": int(len(points)),
            "minimum_fitness": MIN_STEP_REGISTRATION_FITNESS,
            alignment_key: transform.tolist(),
            "reference": "step" if index == 0 else "merged_placements",
            **registration,
        }
        combined_reference = np.concatenate(aligned_clouds, axis=0)
        reference = np.asarray(
            o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(combined_reference)
            ).voxel_down_sample(voxel_size).points,
            dtype=np.float64,
        )

    combined = np.concatenate(aligned_clouds, axis=0)
    geometry = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(combined))
    combined = np.asarray(geometry.voxel_down_sample(voxel_size).points, dtype=np.float32)
    combined_path = output_dir / "cloud.npy"
    np.save(combined_path, combined)
    _write_cloud(output_dir / "cloud.ply", combined, (0.72, 0.75, 0.78))

    merge_report = {
        "status": "ok",
        "method": "first-step-registration-then-placement-icp",
        "step": str(step_path),
        "point_unit": "mm",
        "tolerance_mm": float(tolerance_mm),
        "voxel_size_mm": float(voxel_size),
        "placements": [name for name, _ in placement_clouds],
        "alignments": alignments,
        "combined_points": int(len(combined)),
        "artifacts": {
            "cloud": str(output_dir / "cloud.ply"),
            "placements": {
                name: str(output_dir / f"{name}.ply")
                for name, _ in placement_clouds
            },
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
    )
    result.report["placement_merge"] = merge_report
    (result.output_dir / "report.json").write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_html_report(result.output_dir / "report.html", result.report)
    return result
