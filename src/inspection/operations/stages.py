"""放置面采集重建合并和检测操作"""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from camera import (
    Camera,
    CameraConfig,
    CaptureStoragePolicy,
    profile_options,
)
from rotator import Interface, InterfaceConfig, Rotator, detect_rotator
from inspection.markers.tracking import estimate_calibration
from inspection.metrology.deviation import (
    ComparisonResult,
    DEFAULT_TOLERANCE_MM,
    merge_scans,
)
from inspection.reconstruction.fusion import (
    PointCloudProcessingConfig,
    process_cloud,
    write_result,
)
from inspection.reconstruction.observations import (
    process_capture,
)

from .config import ScanConfig


MIN_ADJACENT_OVERLAP_RATIO = 0.05


def _estimate_calibration(
    manifest_path: Path, calibration_path: Path
) -> dict[str, object]:
    """Estimate the current axis from efficient all-frame marker tracking."""

    return estimate_calibration(manifest_path, calibration_path)


def _calibration_change(previous: dict[str, object], current: dict[str, object]) -> dict[str, float]:
    old_axis = np.asarray(previous["axis"], dtype=np.float64)
    new_axis = np.asarray(current["axis"], dtype=np.float64)
    alignment = abs(float(np.dot(old_axis, new_axis) / (np.linalg.norm(old_axis) * np.linalg.norm(new_axis))))
    return {
        "axis_degrees": float(np.rad2deg(np.arccos(np.clip(alignment, -1.0, 1.0)))),
        "origin_mm": float(
            np.linalg.norm(
                np.asarray(previous["origin_mm"], dtype=np.float64)
                - np.asarray(current["origin_mm"], dtype=np.float64)
            )
        ),
    }


def _orient_axis(
    previous: dict[str, object], current: dict[str, object]
) -> dict[str, object]:
    """Resolve the axis-line sign so height and commanded rotation stay consistent."""

    old_axis = np.asarray(previous["axis"], dtype=np.float64)
    new_axis = np.asarray(current["axis"], dtype=np.float64)
    if float(np.dot(old_axis, new_axis)) < 0:
        current = dict(current)
        current["axis"] = (-new_axis).tolist()
        current["axis_aligned"] = True
    else:
        current["axis_aligned"] = False
    return current


def _capture_turntable(
    output: str | Path,
    config: ScanConfig,
    camera_config: CameraConfig | None = None,
    *,
    close_circle: bool = False,
    calibration: str | Path | None = None,
    storage_policy: CaptureStoragePolicy | None = None,
) -> Path:
    if config.frames < 1 or config.frames > 720:
        raise ValueError("frames must be between 1 and 720")
    if config.settle_seconds < 0 or config.step_degrees == 0:
        raise ValueError("step_degrees must be non-zero and settle_seconds non-negative")
    output = Path(output)
    storage_policy = storage_policy or CaptureStoragePolicy.metrology()
    per_capture_artifacts = [
        "source/points.npy",
        "source/image.png",
        "processed/cloud.npy",
        "metadata/capture.json",
    ]
    if storage_policy.save_depth:
        per_capture_artifacts.append("source/depth.npy")
    if storage_policy.save_confidence:
        per_capture_artifacts.append("source/confidence.npy")
    if storage_policy.save_normals:
        per_capture_artifacts.append("source/normals.npy")
    if storage_policy.save_image_npy:
        per_capture_artifacts.append("source/image.npy")
    if storage_policy.save_ply:
        per_capture_artifacts.append("processed/cloud.ply")
    if storage_policy.save_source_indices:
        per_capture_artifacts.append("processed/source_indices.npy")
    output.mkdir(parents=True, exist_ok=True)
    port = config.port or detect_rotator(device=config.device).port
    manifest: dict[str, object] = {
        "mode": "turntable",
        "created_at": datetime.now().isoformat(),
        "point_unit": "mm",
        "sdk_point_unit": "m",
        "return_to_start": close_circle,
        "config": {
            **asdict(config),
            "step_degrees": config.step_degrees,
            "port": port,
        },
        "storage": {
            "mode": "asynchronous",
            "workers": 1,
            "max_pending_frames": 2,
            "per_capture_artifacts": per_capture_artifacts,
            "profile": (
                "complete"
                if storage_policy == CaptureStoragePolicy.complete()
                else "metrology"
                if storage_policy == CaptureStoragePolicy.metrology()
                else "compact"
                if storage_policy == CaptureStoragePolicy.compact()
                else "custom"
            ),
        },
        "frames": [],
    }
    manifest_path = output / "run.json"
    link = Interface(InterfaceConfig(port=port), device=config.device)
    with link:
        stage = Rotator(link, ratio=config.ratio, speed=min(30, config.speed_dps), device=config.device)
        stage.check()
        error: Exception | None = None
        try:
            with Camera(camera_config) as camera:
                capture_options = profile_options(
                    camera,
                    config.capture_profile,
                    reflection_filter_threshold=config.reflection_filter_threshold,
                )
                pending: list[tuple[dict[str, object], Future[Path]]] = []
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="capture-storage") as storage:
                    for index in range(config.frames):
                        if index:
                            stage.move(angle=config.step_degrees, speed=config.speed_dps, confirm="MOVE")
                            if config.settle_seconds:
                                time.sleep(config.settle_seconds)
                        if len(pending) >= 2:
                            record, future = pending.pop(0)
                            future.result()
                            manifest["frames"].append(record)
                        angle = index * config.step_degrees
                        capture_dir = output / "captures" / f"{index + 1:03d}"
                        # 返回前复制 SDK 缓冲区
                        # 采集存储与标记定位保持独立
                        frame = camera.capture(capture_options)
                        record = {
                            "index": index,
                            "degrees": angle,
                            "directory": str(capture_dir.relative_to(output)).replace("\\", "/"),
                            "path": str(
                                (capture_dir / "processed" / "cloud.npy").relative_to(output)
                            ).replace("\\", "/"),
                        }
                        future = storage.submit(
                            camera.save,
                            frame,
                            capture_dir,
                            storage_policy=storage_policy,
                        )
                        pending.append((record, future))
                    if close_circle:
                        stage.move(angle=config.step_degrees, speed=config.speed_dps, confirm="MOVE")
                        if config.settle_seconds:
                            time.sleep(config.settle_seconds)
                    for record, future in pending:
                        future.result()
                        manifest["frames"].append(record)
                manifest["frames"].sort(key=lambda frame: int(frame["index"]))
        except Exception as exc:
            error = exc
            raise
        finally:
            try:
                stage.disable()
            finally:
                manifest["status"] = "error" if error is not None else "ok"
                if error is not None:
                    manifest["error"] = str(error)
                manifest["completed_frames"] = len(manifest["frames"])
                manifest["completed_at"] = datetime.now().isoformat()
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _write_model(
    manifest: str | Path,
    calibration: str | Path,
    output: str | Path,
    *,
    voxel_size: float = 0.05,
    angle_sign: float = -1.0,
    metrics_output: str | Path | None = None,
    processing_config: PointCloudProcessingConfig | None = None,
    prefer_gpu: bool = False,
) -> Path:
    """Build one merged point-cloud model from a completed turntable scan."""

    manifest = Path(manifest)
    calibration = Path(calibration)
    output = Path(output)
    if output.suffix.lower() != ".ply":
        raise ValueError("模型输出路径必须使用 .ply 后缀")
    config = processing_config or PointCloudProcessingConfig.production(
        voxel_size_mm=voxel_size,
        angle_sign=angle_sign,
    )
    result = process_cloud(
        manifest,
        calibration,
        config,
        prefer_gpu=prefer_gpu,
    )
    return write_result(
        result,
        output,
        report_path=metrics_output,
    )


def acquire_placement(
    output: str | Path,
    config: ScanConfig,
    calibration: str | Path,
    *,
    camera_config: CameraConfig | None = None,
    storage_policy: CaptureStoragePolicy | None = None,
) -> Path:
    """采集一个完整放置面的转台数据并保存原始证据"""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    calibration = Path(calibration)
    if not calibration.is_file():
        raise FileNotFoundError(f"转台标定文件不存在: {calibration}")
    if config.frames < 1 or config.frames > 720:
        raise ValueError("frames must be between 1 and 720")
    calibration_input = output / "input.json"
    calibration_copy = output / "calibration.json"
    calibration_text = calibration.read_text(encoding="utf-8")
    calibration_input.write_text(calibration_text, encoding="utf-8")
    calibration_copy.write_text(calibration_text, encoding="utf-8")
    total_rotation = abs(config.frames * config.step_degrees)
    if not np.isclose(total_rotation, 360.0, atol=1e-6):
        raise ValueError(
            f"整圈建模要求 frames * abs(step) == 360，当前为 {total_rotation:g} 度"
        )
    return _capture_turntable(
        output,
        config,
        camera_config,
        close_circle=True,
        calibration=calibration_input,
        storage_policy=storage_policy,
    )


def build_placement(
    source: str | Path,
    calibration: str | Path | None = None,
    output: str | Path | None = None,
    *,
    voxel_size: float = 0.05,
    angle_sign: float = -1.0,
    processing_config: PointCloudProcessingConfig | None = None,
    prefer_gpu: bool = False,
) -> Path:
    """从采集清单或目录重建一个放置面"""

    source = Path(source)
    manifest = source / "run.json" if source.is_dir() else source
    if not manifest.is_file():
        raise FileNotFoundError(f"采集清单不存在: {manifest}")
    run_dir = manifest.parent
    calibration_input = Path(calibration) if calibration else run_dir / "input.json"
    if not calibration_input.is_file():
        raise FileNotFoundError(f"找不到采集使用的转台标定: {calibration_input}")
    calibration_copy = run_dir / "calibration.json"
    previous = json.loads(calibration_input.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    capture_config = manifest_payload.get("config", {})
    # 只有这个放置面全部采集完成后才进行定位并应用定位后的 Z 轴范围
    for record in manifest_payload["frames"]:
        process_capture(
            run_dir / record["directory"],
            calibration_input,
            z_min_mm=capture_config.get("z_min_mm"),
            z_max_mm=capture_config.get("z_max_mm"),
        )
    current = _orient_axis(
        previous, _estimate_calibration(manifest, calibration_input)
    )
    change = _calibration_change(previous, current)
    calibration_copy.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    recalibrated = change["origin_mm"] > 0.25 or change["axis_degrees"] > 0.05
    if recalibrated:
        for record in manifest_payload["frames"]:
            capture_dir = run_dir / record["directory"]
            process_capture(
                capture_dir,
                calibration_copy,
                z_min_mm=capture_config.get("z_min_mm"),
                z_max_mm=capture_config.get("z_max_mm"),
            )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    measured_angles = current["measured_angles_degrees"]
    if len(measured_angles) != len(manifest_payload["frames"]):
        raise ValueError("本轮定位点没有生成完整的逐帧实测角度")
    for record, measured in zip(manifest_payload["frames"], measured_angles):
        record["measured_degrees"] = float(measured)
    manifest_payload["calibration"] = {
        "source": "current_run_fiducials",
        "axis_change_degrees": change["axis_degrees"],
        "origin_change_mm": change["origin_mm"],
        "captures_reprocessed": recalibrated,
        "axis_residual_mm": current["axis_residual_mm"],
        "max_frame_rms": max(current["frame_rms_mm"]),
        "quality_gate": current.get("quality_gate"),
    }
    manifest.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    model_output = Path(output) if output else run_dir / "cloud.ply"
    model = _write_model(
        manifest,
        calibration_copy,
        model_output,
        voxel_size=voxel_size,
        angle_sign=angle_sign,
        processing_config=processing_config,
        prefer_gpu=prefer_gpu,
        metrics_output=model_output.with_name("report.json"),
    )
    metrics_path = model_output.with_name("report.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    overlap_ratios = [
        float(pair["overlap_ratio"])
        for pair in metrics.get("pairs", [])
        if pair.get("overlap_ratio") is not None
    ]
    median_overlap = float(np.median(overlap_ratios)) if overlap_ratios else 0.0
    metrics["quality_gate"] = {
        "status": "pass" if median_overlap >= MIN_ADJACENT_OVERLAP_RATIO else "fail",
        "median_overlap": median_overlap,
        "min_overlap": MIN_ADJACENT_OVERLAP_RATIO,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if median_overlap < MIN_ADJACENT_OVERLAP_RATIO:
        raise ValueError(
            f"单面点云融合失败：相邻帧重叠率中位数 {median_overlap:.3f}，"
            f"低于最低要求 {MIN_ADJACENT_OVERLAP_RATIO:.3f}"
        )
    return model


def merge_placements(
    source: str | Path,
    step_path: str | Path | None = None,
    *,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
    voxel_size: float = 0.05,
    mesh_samples: int = 300000,
    prefer_gpu: bool = False,
    rebuild: bool = True,
    fusion_mode: str = "nominal",
) -> ComparisonResult:
    """重建所有放置面并完成多面合并和 STEP 对比"""

    source = Path(source)
    if not source.is_dir():
        raise NotADirectoryError(f"检测目录不存在: {source}")
    if fusion_mode not in {"nominal", "consensus"}:
        raise ValueError("fusion_mode must be nominal or consensus")
    if step_path is None:
        state_path = source / "inspection.json"
        if not state_path.is_file():
            raise FileNotFoundError("未提供 STEP 文件，检测目录中也没有 inspection.json")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        step_path = state.get("model")
        if not step_path:
            raise ValueError(f"检测记录中没有 STEP 模型路径: {state_path}")
    step_path = Path(step_path)
    if not step_path.is_file():
        raise FileNotFoundError(f"STEP 模型文件不存在: {step_path}")

    placement_dirs = sorted(
        (path for path in source.glob("placement_*") if path.is_dir()),
        key=lambda path: path.name,
    )
    if not placement_dirs:
        raise FileNotFoundError(f"检测目录中没有放置面目录: {source}")
    placement_clouds: list[tuple[str, Path]] = []
    for placement_dir in placement_dirs:
        if rebuild:
            build_placement(placement_dir, prefer_gpu=prefer_gpu)
        cloud = placement_dir / "cloud.ply"
        if not cloud.is_file():
            raise FileNotFoundError(f"放置面点云不存在: {cloud}")
        placement_clouds.append((placement_dir.name, cloud))

    result = merge_scans(
        step_path,
        placement_clouds,
        source,
        tolerance_mm=tolerance_mm,
        voxel_size=voxel_size,
        mesh_samples=mesh_samples,
        fusion_mode=fusion_mode,
    )
    state_path = source / "inspection.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        processing_status = result.report.get("processing_status", "ok")
        conformance = result.report.get(
            "conformance", {"status": "indeterminate"}
        )
        state["status"] = processing_status
        state["processing_status"] = processing_status
        state["status_scope"] = "processing_only"
        state["fusion_mode"] = fusion_mode
        state["conformance"] = conformance
        state.pop("error", None)
        state["completed_at"] = datetime.now().isoformat()
        state["stages"] = [
            {
                "name": placement_dir.name,
                "status": "ok",
                "cloud": str(placement_dir / "cloud.ply"),
            }
            for placement_dir in placement_dirs
        ] + [
            {
                "name": "merge-and-compare",
                "status": processing_status,
                "status_scope": "processing_only",
                "conformance": conformance,
                "cloud": str(source / "cloud.ply"),
                "report": str(result.output_dir / "report.json"),
            }
        ]
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result
