"""交互式多放置面检测流程"""

from __future__ import annotations

import json
import string
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from camera import CameraConfig, CaptureStoragePolicy
from inspection.metrology.deviation import DEFAULT_TOLERANCE_MM

from .config import ScanConfig
from .stages import acquire_placement, build_placement, merge_placements


_T = TypeVar("_T")


class InspectionCancelled(RuntimeError):
    """表示操作员主动取消检测"""


def _parse_placement_command(value: str) -> tuple[str, int | None]:
    """解析放置面命令以及可选的采集次数"""

    parts = value.strip().upper().split()
    if parts in (["-F"], ["-Q"]):
        return parts[0], None
    if not parts or parts[0] != "-C" or len(parts) > 2:
        raise ValueError("无效操作")
    if len(parts) == 1:
        return "-C", None

    frame_option = parts[1]
    if not frame_option.startswith("-") or len(frame_option) == 1:
        raise ValueError("采集次数无效")
    try:
        frames = int(frame_option[1:])
    except ValueError as exc:
        raise ValueError("采集次数无效") from exc
    if not 1 <= frames <= 720:
        raise ValueError("采集次数必须在 1 到 720 之间")
    return "-C", frames


def _format_elapsed(seconds: float) -> str:
    """将秒数格式化为分钟和秒"""

    total_centiseconds = max(0, round(seconds * 100))
    minutes, remainder = divmod(total_centiseconds, 60 * 100)
    if minutes == 0:
        return f"{remainder / 100:.2f}秒"
    return f"{minutes}分{remainder / 100:.2f}秒"


def _run_timed(label: str, operation: Callable[[], _T]) -> _T:
    """运行一个阶段并输出耗时"""

    started = time.perf_counter()
    print(f"[inspect] {label}开始", flush=True)
    try:
        result = operation()
    except BaseException:
        elapsed = time.perf_counter() - started
        print(f"[inspect] {label}失败 用时 {_format_elapsed(elapsed)}", flush=True)
        raise
    elapsed = time.perf_counter() - started
    print(f"[inspect] {label}完成 用时 {_format_elapsed(elapsed)}", flush=True)
    return result


def inspect_workpiece(
    output: str | Path,
    config: ScanConfig,
    calibration: str | Path,
    step_path: str | Path,
    *,
    wait_for_placement: Callable[[], str],
    wait_for_first_placement: Callable[[], str] | None = None,
    camera_config: CameraConfig | None = None,
    voxel_size: float = 0.05,
    angle_sign: float = -1.0,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
    mesh_samples: int = 300000,
    prefer_gpu: bool = False,
    storage_policy: CaptureStoragePolicy | None = None,
) -> Path:
    """交互采集任意数量放置面并完成重建合并和 STEP 对比"""

    total_started = time.perf_counter()
    print("[inspect] 检测开始", flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "inspection.json"
    state: dict[str, object] = {
        "status": "acquiring",
        "created_at": datetime.now().isoformat(),
        "model": str(step_path),
        "tolerance_mm": float(tolerance_mm),
        "voxel_size_mm": float(voxel_size),
        "placements": [],
        "stages": [],
    }

    def save_state() -> None:
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    save_state()
    manifests: list[tuple[str, Path]] = []
    placement_config = config
    try:
        if wait_for_first_placement is not None:
            command, first_frames = _parse_placement_command(
                wait_for_first_placement()
            )
            if command == "-Q":
                raise InspectionCancelled("用户取消检测")
            if command != "-C":
                raise ValueError("放置面 A 操作无效")
            if first_frames is not None:
                placement_config = replace(config, frames=first_frames)
            print(
                f"[inspect] 放置面 A 参数：采集 {placement_config.frames} 次，"
                f"每次旋转 {placement_config.step_degrees:g}°",
                flush=True,
            )

        for letter in string.ascii_uppercase:
            placement_name = f"placement_{letter}"
            manifest = _run_timed(
                f"放置面 {letter} 采集",
                lambda placement_name=placement_name,
                placement_config=placement_config: acquire_placement(
                    output / placement_name,
                    placement_config,
                    calibration,
                    camera_config=camera_config,
                    storage_policy=storage_policy,
                ),
            )
            manifests.append((letter, manifest))
            state["placements"].append(
                {
                    "name": placement_name,
                    "letter": letter,
                    "status": "captured",
                    "manifest": str(manifest),
                    "frames": placement_config.frames,
                    "step_degrees": placement_config.step_degrees,
                }
            )
            save_state()
            command, next_frames = _parse_placement_command(wait_for_placement())
            if command == "-F":
                break
            if command == "-Q":
                raise InspectionCancelled("用户取消检测")
            placement_config = replace(
                placement_config,
                frames=(
                    placement_config.frames
                    if next_frames is None
                    else next_frames
                ),
            )
            print(
                f"[inspect] 下一放置面参数：采集 {placement_config.frames} 次，"
                f"每次旋转 {placement_config.step_degrees:g}°",
                flush=True,
            )
        else:
            raise ValueError("放置面数量超过支持范围")

        state["status"] = "reconstructing"
        save_state()
        for letter, manifest in manifests:
            cloud = _run_timed(
                f"放置面 {letter} 重建",
                lambda manifest=manifest: build_placement(
                    manifest,
                    voxel_size=voxel_size,
                    angle_sign=angle_sign,
                    prefer_gpu=prefer_gpu,
                ),
            )
            state["stages"].append(
                {
                    "name": f"placement_{letter}",
                    "status": "ok",
                    "cloud": str(cloud),
                }
            )
            save_state()

        state["status"] = "merging"
        save_state()
        result = _run_timed(
            "多放置面合并和 STEP 对比",
            lambda: merge_placements(
                output,
                step_path,
                tolerance_mm=tolerance_mm,
                voxel_size=voxel_size,
                mesh_samples=mesh_samples,
                prefer_gpu=prefer_gpu,
                rebuild=False,
            ),
        )
        state["stages"].append(
            {
                "name": "merge-and-compare",
                "status": "ok",
                "cloud": str(output / "cloud.ply"),
                "report": str(result.output_dir / "report.json"),
            }
        )
        state["status"] = "ok"
        state["completed_at"] = datetime.now().isoformat()
        save_state()
        elapsed = time.perf_counter() - total_started
        print(f"[inspect] 检测完成 总用时 {_format_elapsed(elapsed)}", flush=True)
        return output / "cloud.ply"
    except BaseException as exc:
        state["status"] = "cancelled" if isinstance(exc, InspectionCancelled) else "error"
        state["error"] = str(exc) or type(exc).__name__
        state["completed_at"] = datetime.now().isoformat()
        save_state()
        elapsed = time.perf_counter() - total_started
        print(f"[inspect] 检测失败 用时 {_format_elapsed(elapsed)}", flush=True)
        raise
