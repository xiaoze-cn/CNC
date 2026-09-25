"""Human-facing preview for inspection results, images, and point clouds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
POINT_CLOUD_SUFFIXES = {".npy", ".ply"}
DISPLAY_COLORS = {
    "mesh": (0.36, 0.38, 0.42),
    "cloud": (0.03, 0.18, 0.07),
    "trusted": (0.03, 0.18, 0.07),
    "candidate": (0.90, 0.73, 0.24),
    "conflict": (0.85, 0.28, 0.37),
    "unobserved": (0.18, 0.50, 0.93),
    "problem": (0.98, 0.45, 0.09),
    "recessed": (0.49, 0.23, 0.93),
}
DISPLAY_STYLE = {
    "background": (0.08, 0.09, 0.10),
    "width": 1280,
    "height": 800,
    "point_size": 4.0,
    "cloud_lift": 0.12,
}


def _to_oklab(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float64)
    linear = np.where(
        colors <= 0.04045,
        colors / 12.92,
        ((colors + 0.055) / 1.055) ** 2.4,
    )
    lms = linear @ np.asarray(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    ).T
    lms = np.cbrt(np.clip(lms, 0.0, None))
    return lms @ np.asarray(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    ).T


def _to_srgb(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.float64)
    lms = colors @ np.asarray(
        [
            [1.0, 0.3963377774, 0.2158037573],
            [1.0, -0.1055613458, -0.0638541728],
            [1.0, -0.0894841775, -1.2914855480],
        ]
    ).T
    lms = lms**3
    linear = lms @ np.asarray(
        [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010],
        ]
    ).T
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.maximum(linear, 0.0) ** (1.0 / 2.4) - 0.055,
    )


def _latest_result() -> Path:
    search_groups = (
        list(Path("data/inspect").glob("*/cloud.ply")),
        list(Path("data").glob("**/processed/cloud.ply")),
        list(Path("data").glob("**/*.ply")),
        list(Path("data").glob("**/*.png")),
    )
    for candidates in search_groups:
        existing = [path for path in candidates if path.exists()]
        if existing:
            return max(existing, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError("data 中还没有可以预览的图像或点云")


def _show_image(path: Path) -> None:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    print(f"[show] 图像：{path} 尺寸：{image.shape} 类型：{image.dtype}")
    cv2.imshow(f"Inspection preview - {path.name}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _height_colors(
    points: np.ndarray,
    *,
    axis: int,
    reverse: bool = False,
) -> np.ndarray:
    height = points[:, axis]
    low, high = np.percentile(height, (2, 98))
    if high - low <= 1e-6:
        level = np.full(len(height), 0.5, dtype=np.float64)
    else:
        level = np.clip((height - low) / (high - low), 0.0, 1.0)
    if reverse:
        level = 1.0 - level

    stops = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    palette = np.asarray(
        [
            [0.03, 0.18, 0.07],
            [0.04, 0.42, 0.10],
            [0.20, 0.65, 0.12],
            [0.62, 0.76, 0.15],
            [0.95, 0.75, 0.22],
        ]
    )
    # 在 OKLab 中插值使相同高度变化对应更均匀的视觉颜色阶梯
    palette_oklab = _to_oklab(palette)
    interpolated = np.column_stack(
        [np.interp(level, stops, palette_oklab[:, channel]) for channel in range(3)]
    )
    return np.clip(_to_srgb(interpolated), 0.0, 1.0)


def _load_cloud(path: Path, max_points: int, voxel_size: float):
    import open3d as o3d

    if path.suffix.lower() == ".npy":
        points = np.asarray(np.load(path), dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if max_points > 0 and len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
            points = points[indices]
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    geometry = o3d.io.read_point_cloud(str(path))
    if geometry.is_empty():
        raise ValueError(f"点云为空或格式不受支持: {path}")
    if voxel_size > 0:
        geometry = geometry.voxel_down_sample(voxel_size)
    if max_points > 0 and len(geometry.points) > max_points:
        geometry = geometry.uniform_down_sample(
            max(1, len(geometry.points) // max_points)
        )
    return geometry


def _show_cloud(
    path: Path,
    *,
    max_points: int,
    point_size: float,
    voxel_size: float,
    color_mode: str = "height",
    show_blue: bool = False,
    show_evidence: bool = False,
    blue_cell: float = 0.75,
    title: str | None = None,
    z_up: bool = False,
) -> None:
    import open3d as o3d

    if blue_cell <= 0:
        raise ValueError("blue_cell must be positive")
    geometry = _load_cloud(path, max_points, voxel_size)
    evidence_labels: list[str] = []
    if show_evidence:
        specs = (
            ("trusted", DISPLAY_COLORS["trusted"], "可信"),
            ("candidate", DISPLAY_COLORS["candidate"], "候选"),
            ("conflict", DISPLAY_COLORS["conflict"], "冲突"),
            ("single", DISPLAY_COLORS["cloud"], "单次"),
        )
        combined = o3d.geometry.PointCloud()
        limit = max_points // len(specs) if max_points > 0 else 0
        for suffix, color, label in specs:
            item_path = path.with_name(
                f"{path.stem}-{suffix}{path.suffix}"
            )
            if not item_path.is_file():
                continue
            item = _load_cloud(item_path, limit, voxel_size)
            if item.is_empty():
                continue
            item.paint_uniform_color(color)
            combined += item
            evidence_labels.append(
                f"{label}={len(item.points)}"
            )
        if combined.is_empty():
            raise FileNotFoundError(f"没有找到点云证据分类文件: {path.parent}")
        geometry = combined

    points = np.asarray(geometry.points)
    if evidence_labels:
        pass
    elif color_mode == "height":
        # STEP 坐标使用 Y 轴表示从窄顶到底部的方向
        color_axis = 1
        reverse_colors = True
        geometry.colors = o3d.utility.Vector3dVector(
            _height_colors(points, axis=color_axis, reverse=reverse_colors)
        )
    elif color_mode == "yellow":
        geometry.paint_uniform_color((0.86, 0.82, 0.46))
    elif color_mode == "green":
        geometry.paint_uniform_color(DISPLAY_COLORS["cloud"])
    elif color_mode == "gray":
        geometry.paint_uniform_color((0.58, 0.62, 0.66))
    else:
        raise ValueError(f"不支持的点云配色模式: {color_mode}")

    overlays: list[tuple[str, object]] = []
    comparison_dir = path.parent / "view"
    if (
        not evidence_labels
        and path.name in {"cloud.ply", "cloud.npy"}
        and comparison_dir.is_dir()
    ):
        overlay_specs = [
            ("problem.ply", DISPLAY_COLORS["problem"]),
            ("recessed.ply", DISPLAY_COLORS["recessed"]),
        ]
        if show_blue:
            coverage_name = (
                "unobserved.ply"
                if (comparison_dir / "unobserved.ply").is_file()
                else "missing.ply"
            )
            overlay_specs.append((coverage_name, DISPLAY_COLORS["unobserved"]))
        for name, color in overlay_specs:
            overlay_path = comparison_dir / name
            if not overlay_path.is_file():
                continue
            overlay = o3d.io.read_point_cloud(str(overlay_path))
            if overlay.is_empty():
                continue
            overlay_voxel = (
                max(voxel_size, blue_cell)
                if name in {"unobserved.ply", "missing.ply"}
                else voxel_size
            )
            if overlay_voxel > 0:
                overlay = overlay.voxel_down_sample(overlay_voxel)
            if max_points > 0 and len(overlay.points) > max_points:
                overlay = overlay.uniform_down_sample(
                    max(1, len(overlay.points) // max_points)
                )
            # 当分类点与实测点云重合时仍保持可见避免深度测试遮挡
            overlay_points = np.asarray(overlay.points)
            center = np.asarray(geometry.get_center(), dtype=np.float64)
            direction = overlay_points - center
            lengths = np.linalg.norm(direction, axis=1)
            valid = lengths > 1e-9
            shifted = overlay_points.copy()
            # 仅在显示时抬高分类点避免近距离缩放时与实测点云发生深度冲突
            shifted[valid] += (
                direction[valid] / lengths[valid, None]
            ) * DISPLAY_STYLE["cloud_lift"]
            overlay.points = o3d.utility.Vector3dVector(shifted)
            overlay.paint_uniform_color(color)
            overlays.append((name, overlay))
        conflict_path = path.parent / "cloud-placement-conflicts.ply"
        if conflict_path.is_file():
            conflict = o3d.io.read_point_cloud(str(conflict_path))
            if not conflict.is_empty():
                if voxel_size > 0:
                    conflict = conflict.voxel_down_sample(voxel_size)
                if max_points > 0 and len(conflict.points) > max_points:
                    conflict = conflict.uniform_down_sample(
                        max(1, len(conflict.points) // max_points)
                    )
                conflict.paint_uniform_color(DISPLAY_COLORS["conflict"])
                overlays.append((conflict_path.name, conflict))

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    print(
        f"[show] 点云：{path} 点数：{len(points)} 最小值：{minimum.tolist()} "
        f"最大值：{maximum.tolist()} 配色：{color_mode} "
        f"证据：{evidence_labels} 叠加：{[name for name, _ in overlays]}"
    )

    viewer = o3d.visualization.Visualizer()
    window_title = title or f"Point cloud - {path.name}"
    if not viewer.create_window(
        window_name=window_title,
        width=DISPLAY_STYLE["width"],
        height=DISPLAY_STYLE["height"],
    ):
        raise RuntimeError("Open3D 预览窗口创建失败")
    viewer.add_geometry(geometry)
    for _, overlay in overlays:
        viewer.add_geometry(overlay)
    options = viewer.get_render_option()
    options.background_color = np.asarray(DISPLAY_STYLE["background"])
    options.point_size = min(point_size, 6.0) if overlays else point_size
    view = viewer.get_view_control()
    view.set_lookat(geometry.get_axis_aligned_bounding_box().get_center())
    if z_up:
        view.set_up([0.0, 0.0, 1.0])
        view.set_front([0.0, 1.0, 0.0])
        view.set_zoom(0.72)
    else:
        view.set_up([0.0, -1.0, 0.0])
        view.set_front([0.0, 0.0, -1.0])
        view.set_zoom(0.72)
    viewer.run()
    viewer.destroy_window()


def _show_run(
    run_dir: Path,
    *,
    max_points: int,
    point_size: float,
    voxel_size: float,
    color_mode: str,
    show_blue: bool,
    show_evidence: bool,
    blue_cell: float,
) -> None:
    cloud = run_dir / "cloud.ply"
    if not cloud.is_file():
        raise FileNotFoundError(f"运行目录中没有完整点云: {cloud}")
    _show_cloud(
        cloud,
        max_points=max_points,
        point_size=point_size,
        voxel_size=voxel_size,
        color_mode=color_mode,
        show_blue=show_blue,
        show_evidence=show_evidence,
        blue_cell=blue_cell,
        title=f"Complete point cloud - {run_dir.name}",
        z_up=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="show", description="预览最新或指定的图像、主体点云")
    parser.add_argument("path", nargs="?", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument(
        "--point-size",
        type=float,
        default=DISPLAY_STYLE["point_size"],
    )
    parser.add_argument("--voxel-size", type=float, default=0.0, help="点云预览降采样尺寸，单位 mm")
    parser.add_argument(
        "--color-mode",
        choices=("green", "yellow", "gray", "height"),
        default="green",
        help="点云配色，默认统一深绿；height 为纵向渐变；yellow 为纯浅黄；gray 为中性灰",
    )
    parser.add_argument(
        "--blue",
        dest="show_blue",
        action="store_true",
        help="叠加显示蓝色 STEP 未观测区域（不等同于材料缺失）",
    )
    parser.add_argument("--blue-cell", type=float, default=0.75)
    parser.add_argument(
        "--evidence",
        dest="show_evidence",
        action="store_true",
        help="按可信、候选、冲突、单次四色显示局部多视角证据",
    )
    args = parser.parse_args(argv)
    path = args.path or _latest_result()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        _show_run(
            path,
            max_points=args.max_points,
            point_size=args.point_size,
            voxel_size=args.voxel_size,
            color_mode=args.color_mode,
            show_blue=args.show_blue,
            show_evidence=args.show_evidence,
            blue_cell=args.blue_cell,
        )
    elif path.suffix.lower() in IMAGE_SUFFIXES:
        _show_image(path)
    elif path.suffix.lower() in POINT_CLOUD_SUFFIXES:
        complete_run_cloud = path.name in {"cloud.ply", "cloud.npy"} and any(
            (path.parent / manifest).is_file()
            for manifest in ("inspection.json", "run.json")
        )
        _show_cloud(
            path,
            max_points=args.max_points,
            point_size=args.point_size,
            voxel_size=args.voxel_size,
            color_mode=args.color_mode,
            show_blue=args.show_blue,
            show_evidence=args.show_evidence,
            blue_cell=args.blue_cell,
            title=f"Complete point cloud - {path.parent.name}" if complete_run_cloud else None,
            z_up=complete_run_cloud,
        )
    else:
        raise ValueError(f"只支持图像、PLY/NPY 点云或包含完整点云的 run 目录: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
