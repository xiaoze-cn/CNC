"""Command-line entry point for the inspection workflow."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from camera import Camera, CameraConfig, CaptureStoragePolicy
from inspection.metrology.deviation import (
    DEFAULT_TOLERANCE_MM,
    show_comparison,
)
from inspection.metrology.trace import trace_frame
from inspection.operations import (
    ScanConfig,
    inspect_workpiece,
    merge_placements,
)
from rotator import detect_rotator
from inspection.viewer import main as show_viewer
from inspection.storage import compact_root


DEFAULT_MODEL = Path("data/model/镜头架.STEP")
DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parent / "inspection" / "markers" / "turntable.json"
)


def _frame_count(value: str) -> int:
    frames = int(value)
    if not 1 <= frames <= 720:
        raise argparse.ArgumentTypeError("每面帧数必须在 1 到 720 之间")
    return frames


def _default_output(command: str) -> Path:
    return Path("data") / command / f"{datetime.now():%Y%m%d%H%M%S}"


def _format_elapsed(seconds: float) -> str:
    total_centiseconds = max(0, round(seconds * 100))
    minutes, remainder = divmod(total_centiseconds, 60 * 100)
    if minutes == 0:
        return f"{remainder / 100:.2f}秒"
    return f"{minutes}分{remainder / 100:.2f}秒"


def _latest_calibration() -> Path:
    if DEFAULT_CALIBRATION.is_file():
        return DEFAULT_CALIBRATION
    raise FileNotFoundError("未找到默认转台标定文件，请使用 --calibration 指定")


def _latest_comparison() -> Path:
    candidates = [path.parent for path in Path("data").glob("**/view/report.json")]
    if not candidates:
        raise FileNotFoundError("data 中还没有 STEP 对比结果")
    return max(candidates, key=lambda path: (path / "report.json").stat().st_mtime)


def _latest_inspection() -> Path:
    candidates = [
        path.parent
        for path in Path("data/inspect").glob("*/inspection.json")
        if (path.parent / "placement_A").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError("data/inspect 中还没有检测结果")
    return max(candidates, key=lambda path: (path / "inspection.json").stat().st_mtime)


def _frame_number(value: str) -> int:
    normalized = value[1:] if value.startswith("-") else value
    frame = int(normalized)
    if frame < 1:
        raise argparse.ArgumentTypeError("视角编号必须从 1 开始")
    return frame


def _expand_args(argv: list[str] | None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] != "show":
        return values
    expanded: list[str] = []
    for index, value in enumerate(values):
        if index:
            try:
                cell = float(value[2:]) if value.startswith("--") else -1.0
            except ValueError:
                cell = -1.0
            if cell > 0:
                expanded.extend(("--blue", "--blue-cell", str(cell)))
                continue
        expanded.append(value)
    return expanded


def _add_options(
    parser: argparse.ArgumentParser,
    *,
    include_output: bool = False,
    include_gpu: bool = False,
    include_tolerance: bool = False,
) -> None:
    """添加采集和完整检测共用参数"""

    if include_output:
        parser.add_argument("--output", type=Path, default=None, help="输出目录")
    parser.add_argument("--calibration", type=Path, default=None, help="转台标定文件")
    parser.add_argument("--port", default=None, help="转台串口，默认自动探测")
    parser.add_argument(
        "--frames",
        dest="frames",
        type=_frame_count,
        default=ScanConfig().frames,
        metavar="N",
        help="每个放置面整圈采集帧数，默认 18；步进角自动计算为 360/N",
    )
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument(
        "--storage-profile",
        choices=("compact", "metrology", "complete"),
        default="metrology",
        help=(
            "采集文件保留级别，默认 metrology；metrology 保留计量融合需要的"
            "传感器证据，complete 还保留诊断用冗余文件"
        ),
    )
    storage.add_argument(
        "--complete",
        dest="storage_profile",
        action="store_const",
        const="complete",
        help="保留完整传感器证据数组",
    )
    if include_gpu:
        parser.add_argument("--gpu", dest="prefer_gpu", action="store_true", help="使用 GPU 加速点云匹配")
    if include_tolerance:
        parser.add_argument(
            "--lim",
            dest="tolerance",
            metavar="MM",
            type=float,
            default=DEFAULT_TOLERANCE_MM,
            help=f"超差阈值，默认 {DEFAULT_TOLERANCE_MM:g} mm",
        )


def _show(
    path: Path | None,
    *,
    color_mode: str = "green",
    show_blue: bool = False,
    show_evidence: bool = False,
    blue_cell: float = 0.75,
) -> int:
    if blue_cell <= 0:
        raise ValueError("blue_cell must be positive")
    viewer_args = ["--color-mode", color_mode]
    if show_blue:
        viewer_args.append("--blue")
        viewer_args.extend(("--blue-cell", str(blue_cell)))
    if show_evidence:
        viewer_args.append("--evidence")
    if path is None:
        try:
            show_comparison(
                _latest_comparison(),
                show_blue=show_blue,
                blue_cell=blue_cell,
            )
            return 0
        except FileNotFoundError:
            return show_viewer(viewer_args)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file() and path.name == "report.json":
        show_comparison(path.parent, show_blue=show_blue, blue_cell=blue_cell)
        return 0
    if path.is_dir():
        comparison = path / "view"
        if (comparison / "report.json").is_file():
            show_comparison(
                comparison,
                show_blue=show_blue,
                blue_cell=blue_cell,
            )
            return 0
        if (path / "report.json").is_file() and (path / "mesh.ply").is_file():
            show_comparison(path, show_blue=show_blue, blue_cell=blue_cell)
            return 0
    return show_viewer([str(path), *viewer_args])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inspection")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="交互执行多放置面检测")
    inspect.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL, help="STEP 模型")
    _add_options(inspect, include_output=True, include_gpu=True, include_tolerance=True)
    inspect.add_argument(
        "--consensus",
        action="store_true",
        help="多放置面冲突改用一致性优先，默认使用标称优先",
    )

    trace = commands.add_parser(
        "trace",
        help="将指定放置面的单个采集视角贴回 STEP 并查看坏点贡献",
    )
    trace_side = trace.add_mutually_exclusive_group(required=True)
    trace_side.add_argument("-a", dest="side", action="store_const", const="a", help="选择放置面 A")
    trace_side.add_argument("-b", dest="side", action="store_const", const="b", help="选择放置面 B")
    trace.add_argument(
        "source_or_frame",
        metavar="SOURCE_OR_FRAME",
        help="检测目录；也可省略目录直接传视角编号",
    )
    trace.add_argument(
        "frame_value",
        nargs="?",
        metavar="-FRAME",
        help="视角编号，从 1 开始，例如 -1",
    )
    trace.add_argument(
        "--lim",
        dest="tolerance",
        metavar="MM",
        type=float,
        default=None,
        help="覆盖检测记录中的超差阈值",
    )
    merge = commands.add_parser(
        "merge",
        help="重建所有放置面 合并点云并生成 STEP 对比结果",
    )
    merge.add_argument("source", type=Path, help="包含 placement_A 等放置面目录的检测目录")
    merge.add_argument("model", nargs="?", type=Path, default=None, help="STEP 模型；默认从检测记录读取")
    merge.add_argument("--gpu", dest="prefer_gpu", action="store_true", help="使用 GPU 加速单面点云证据匹配")
    merge.add_argument(
        "--lim",
        dest="tolerance",
        metavar="MM",
        type=float,
        default=DEFAULT_TOLERANCE_MM,
        help=f"超差阈值，默认 {DEFAULT_TOLERANCE_MM:g} mm",
    )
    merge.add_argument(
        "--consensus",
        action="store_true",
        help="多放置面冲突改用一致性优先，默认使用标称优先",
    )

    show = commands.add_parser("show", help="查看最近一次或指定结果")
    show.add_argument("path", nargs="?", type=Path, default=None, help="结果目录、报告、点云或图像")
    show.add_argument(
        "--color-mode",
        choices=("green", "yellow", "gray", "height"),
        default="green",
        help="单独点云预览配色，默认统一深绿；height 为纵向渐变；yellow 为纯浅黄；gray 为中性灰",
    )
    show.add_argument(
        "--blue",
        dest="show_blue",
        action="store_true",
        help="叠加蓝色 STEP 未观测区域",
    )
    show.add_argument(
        "--blue-cell",
        type=float,
        default=0.75,
        help=argparse.SUPPRESS,
    )
    show.add_argument(
        "--evidence",
        dest="show_evidence",
        action="store_true",
        help="按可信、候选、冲突、单次四色显示局部多视角证据",
    )

    commands.add_parser("doctor", help="检查相机和转台是否在线")
    compact = commands.add_parser("compact", help="删除可重算的采集中间文件并压缩源点图")
    compact.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("data/inspect"),
        help="采集根目录，默认 data/inspect",
    )
    compact.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计将释放的空间，不修改文件",
    )
    return parser


def _wait_first() -> str:
    try:
        return input("放置面 A 完成后继续：")
    except EOFError as exc:
        raise RuntimeError("当前终端不能读取放置面命令") from exc


def _wait_placement() -> str:
    try:
        return input("重新放置工件后继续：")
    except EOFError as exc:
        raise RuntimeError("当前终端不能读取放置面命令") from exc


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(_expand_args(argv))
    if args.command == "inspect":
        if not args.model.is_file():
            raise FileNotFoundError(f"STEP 模型文件不存在: {args.model}")
        output = args.output or _default_output("inspect")
        cloud = inspect_workpiece(
            output,
            ScanConfig(
                port=args.port,
                frames=args.frames,
            ),
            args.calibration or _latest_calibration(),
            args.model,
            wait_for_placement=_wait_placement,
            wait_first=_wait_first,
            camera_config=CameraConfig(),
            tolerance_mm=args.tolerance,
            prefer_gpu=args.prefer_gpu,
            fusion_mode="consensus" if args.consensus else "nominal",
            storage_policy=(
                CaptureStoragePolicy.complete()
                if args.storage_profile == "complete"
                else CaptureStoragePolicy.metrology()
                if args.storage_profile == "metrology"
                else CaptureStoragePolicy.compact()
            ),
        )
        print(f"[inspect] 结果文件：{output / 'view' / 'report.json'}")
        show_comparison(output / "view")
        return 0
    if args.command == "merge":
        source = args.source
        if not source.is_dir():
            raise NotADirectoryError(f"检测目录不存在: {source}")
        started = time.perf_counter()
        print("[merge] 重建合并和 STEP 对比开始", flush=True)
        try:
            mode = "consensus" if args.consensus else "nominal"
            print(f"[merge] 融合策略：{mode}", flush=True)
            result = merge_placements(
                source,
                args.model,
                tolerance_mm=args.tolerance,
                prefer_gpu=args.prefer_gpu,
                fusion_mode=mode,
            )
        except BaseException:
            elapsed = time.perf_counter() - started
            print(f"[merge] 重建合并和 STEP 对比失败 用时 {_format_elapsed(elapsed)}", flush=True)
            raise
        elapsed = time.perf_counter() - started
        print(f"[merge] 重建合并和 STEP 对比完成 用时 {_format_elapsed(elapsed)}", flush=True)
        print(f"[merge] 结果文件：{result.output_dir / 'report.json'}")
        show_comparison(result.output_dir)
        return 0
    if args.command == "trace":
        if args.frame_value is None:
            source = _latest_inspection()
            frame_value = args.source_or_frame
        else:
            source = Path(args.source_or_frame)
            frame_value = args.frame_value
        result = trace_frame(
            source,
            side=args.side,
            frame_number=_frame_number(frame_value),
            tolerance_mm=args.tolerance,
        )
        print(f"[trace] 结果目录：{result.output_dir}")
        show_comparison(result.output_dir)
        return 0
    if args.command == "show":
        return _show(
            args.path,
            color_mode=args.color_mode,
            show_blue=args.show_blue,
            show_evidence=args.show_evidence,
            blue_cell=args.blue_cell,
        )
    if args.command == "doctor":
        devices = Camera.list_devices()
        rotator = detect_rotator()
        print(
            json.dumps(
                {
                    "camera": [asdict(device) for device in devices],
                    "rotator": rotator.data(),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0
    if args.command == "compact":
        summary = compact_root(args.root, dry_run=args.dry_run)
        action = "预计释放" if args.dry_run else "已释放"
        print(
            json.dumps(
                {
                    "root": str(args.root),
                    "captures": summary.captures,
                    "converted_points": summary.converted_points,
                    "deleted_files": summary.deleted_files,
                    "converted_reclaimed_gb": round(
                        summary.converted_reclaimed_bytes / (1024**3), 3
                    ),
                    "reclaimed_gb": round(
                        (summary.reclaimed_bytes + summary.converted_reclaimed_bytes)
                        / (1024**3),
                        3,
                    ),
                    "message": (
                        f"{action}约 "
                        f"{(summary.reclaimed_bytes + summary.converted_reclaimed_bytes) / (1024**3):.3f} GiB"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
