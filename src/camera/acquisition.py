"""RVC acquisition and capture persistence built on the official PyRVC API.

PyRVC is installed directly from PyPI by Pixi.  This module only owns the
inspection project's file format and lifecycle orchestration; it does not
reimplement or hide the vendor camera API.
"""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import PyRVC as RVC
import cv2

@dataclass(frozen=True, slots=True)
class CameraConfig:
    device_index: int = 0
    camera_id: str = "CameraID_Left"


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    index: int
    name: str
    serial: str
    port: str
    type: int | str
    support_x1: bool
    support_x2: bool
    firmware_match: bool


@dataclass(frozen=True, slots=True)
class Capture:
    points: np.ndarray
    captured_at: str
    device: DeviceInfo
    options: dict[str, Any]
    point_unit: str = "mm"
    image: np.ndarray | None = None
    image_type: str = "unknown"
    organized_points: np.ndarray | None = None
    sdk_points: np.ndarray | None = None
    depth: np.ndarray | None = None
    confidence: np.ndarray | None = None
    normals: np.ndarray | None = None
    sdk_artifacts: dict[str, Any] = field(default_factory=dict)
    sdk_version: str = "unknown"


@dataclass(frozen=True, slots=True)
class CaptureStoragePolicy:
    """Select which derived capture arrays are kept on disk.

    The compact policy keeps the organized point map and PNG needed to
    rebuild turntable segmentation. Sensor evidence arrays are optional and
    can be enabled for a diagnostic run without changing the capture API.
    """

    save_depth: bool = False
    save_confidence: bool = False
    save_normals: bool = False
    save_image_npy: bool = False
    save_processed_cloud_ply: bool = False
    save_source_indices: bool = False

    @classmethod
    def compact(cls) -> "CaptureStoragePolicy":
        return cls()

    @classmethod
    def complete(cls) -> "CaptureStoragePolicy":
        return cls(
            save_depth=True,
            save_confidence=True,
            save_normals=True,
            save_image_npy=True,
            save_processed_cloud_ply=True,
            save_source_indices=True,
        )

SOURCE_DIR = "source"
PROCESSED_DIR = "processed"
METADATA_DIR = "metadata"


class CaptureLayout:
    """Resolve the canonical paths for one capture frame."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.source = self.root / SOURCE_DIR
        self.processed = self.root / PROCESSED_DIR
        self.metadata = self.root / METADATA_DIR

    def ensure(self) -> None:
        self.source.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)
        self.metadata.mkdir(parents=True, exist_ok=True)

    def write_path(self, category: str, filename: str) -> Path:
        directory = {
            SOURCE_DIR: self.source,
            PROCESSED_DIR: self.processed,
            METADATA_DIR: self.metadata,
        }.get(category)
        if directory is None:
            raise ValueError(f"unknown capture category: {category}")
        return directory / filename

    def resolve(self, filename: str) -> Path:
        categories = {
            "points.npy": self.source,
            "depth.npy": self.source,
            "confidence.npy": self.source,
            "normals.npy": self.source,
            "image.npy": self.source,
            "image.png": self.source,
            "cloud.npy": self.processed,
            "cloud.ply": self.processed,
            "capture.json": self.metadata,
        }
        directory = categories.get(filename)
        if directory is None:
            raise ValueError(f"unknown capture file: {filename}")
        return directory / filename

    def relative(self, path: str | Path) -> str:
        return Path(path).relative_to(self.root).as_posix()


def _device_info(device: Any, index: int) -> DeviceInfo:
    _, info = device.GetDeviceInfo()
    return DeviceInfo(
        index=index,
        name=str(getattr(info, "name", "unknown")),
        serial=str(getattr(info, "sn", "unknown")),
        port=str(getattr(info, "port", "unknown")),
        type=getattr(getattr(info, "type", "unknown"), "name", getattr(info, "type", "unknown")),
        support_x1=bool(getattr(info, "support_x1", True)),
        support_x2=bool(getattr(info, "support_x2", False)),
        firmware_match=bool(device.IsFirmwareMatch()),
    )


def _safe_options(options: Any) -> dict[str, Any]:
    fields = (
        "capture_mode", "exposure_time_2d", "exposure_time_3d", "gain_2d",
        "gain_3d", "gamma_2d", "gamma_3d", "projector_brightness", "scan_times",
        "confidence_threshold", "use_auto_noise_removal", "noise_removal_distance",
        "noise_removal_point_number", "reflection_filter_threshold", "smooth_sigma",
        "downsample_distance", "truncate_z_min", "truncate_z_max", "calc_normal",
        "calc_normal_radius", "enable_2d_in_capture", "filter_range",
        "phase_filter_range", "light_contrast_threshold", "use_auto_bilateral_filter",
        "bilateral_filter_depth_sigma", "bilateral_filter_kernal_size",
        "bilateral_filter_space_sigma", "smoothness", "transform_to_camera",
        "use_projector_capturing_2d_image",
    )
    result = {
        field: getattr(getattr(options, field), "name", getattr(options, field))
        for field in fields
        if hasattr(options, field)
    }
    if hasattr(options, "hdr_exposure_times"):
        count = int(options.hdr_exposure_times)
        result["hdr_exposure_times"] = count
        result["hdr_exposure_time_content"] = [
            int(options.GetHDRExposureTimeContent(index)) for index in range(count)
        ]
        result["hdr_gain_content"] = [
            float(options.GetHDRGainContent(index)) for index in range(count)
        ]
        result["hdr_projector_brightness_content"] = [
            int(options.GetHDRProjectorBrightnessContent(index)) for index in range(count)
        ]
        if hasattr(options, "GetHDRScanTimesContent"):
            result["hdr_scan_times_content"] = [
                int(options.GetHDRScanTimesContent(index)) for index in range(count)
            ]
    if hasattr(options, "roi"):
        roi = options.roi
        result["roi"] = {
            field: int(getattr(roi, field))
            for field in ("x", "y", "width", "height")
            if hasattr(roi, field)
        }
    return result


def list_devices(sdk: ModuleType = RVC) -> list[DeviceInfo]:
    sdk.SystemInit()
    try:
        _, devices = sdk.SystemListDevices(sdk.SystemListDeviceTypeEnum.All)
        return [_device_info(device, index) for index, device in enumerate(devices)]
    finally:
        sdk.SystemShutdown()


class Camera(AbstractContextManager["Camera"]):
    """Own one official PyRVC X1 session for a capture workflow."""

    def __init__(self, config: CameraConfig | None = None, sdk: ModuleType = RVC):
        self.config = config or CameraConfig()
        self.sdk = sdk
        self.device: Any = None
        self.device_info: DeviceInfo | None = None
        self.handle: Any = None
        self.options: Any = None
        self._initialized = False

    def __enter__(self) -> "Camera":
        self.sdk.SystemInit()
        self._initialized = True
        try:
            _, devices = self.sdk.SystemListDevices(self.sdk.SystemListDeviceTypeEnum.All)
            if not devices:
                raise RuntimeError("未发现 RVC 设备")
            if not 0 <= self.config.device_index < len(devices):
                raise IndexError(f"device_index={self.config.device_index} 超出设备数量 {len(devices)}")
            self.device = devices[self.config.device_index]
            self.device_info = _device_info(self.device, self.config.device_index)
            if not self.device_info.firmware_match:
                raise RuntimeError("RVC 固件版本不匹配，请先用 RVCManager 升级")
            camera_id = getattr(self.sdk, self.config.camera_id)
            self.handle = self.sdk.X1.Create(self.device, camera_id)
            if not self.handle.Open() or not self.handle.IsOpen():
                raise RuntimeError("RVC 相机打开失败，可能被其他程序占用")
            loaded, options = self.handle.LoadCaptureOptionParameters()
            self.options = options if loaded else self.sdk.X1_CaptureOptions()
            return self
        except Exception:
            self.close()
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            if self.handle is not None:
                try:
                    if self.handle.IsOpen():
                        self.handle.Close()
                finally:
                    self.sdk.X1.Destroy(self.handle)
                    self.handle = None
        finally:
            if self._initialized:
                self.sdk.SystemShutdown()
            self._initialized = False

    @staticmethod
    def list_devices() -> list[DeviceInfo]:
        return list_devices()

    def capture(
        self,
        options: Any | None = None,
    ) -> Capture:
        self._require_open()
        ok = self.handle.Capture(options) if options is not None else self.handle.Capture()
        if not ok:
            raise RuntimeError(f"RVC 采集失败: {self.sdk.GetLastErrorMessage()}")
        effective_options = self.options if options is None else options
        sdk_artifacts: dict[str, Any] = {}
        point_map = self.handle.GetPointMap()
        sdk_points = np.array(point_map, dtype=np.float64, copy=True)
        organized_points = (sdk_points * 1000.0).astype(np.float32)
        points = organized_points.reshape(-1, 3)
        sdk_image = self.handle.GetImage()
        image = np.array(sdk_image, copy=True) if sdk_image is not None else None
        image_type = getattr(getattr(sdk_image, "GetType", lambda: "unknown")(), "name", "unknown")
        depth = None
        try:
            depth_map = self.handle.GetDepthMap()
            if depth_map is not None and depth_map.IsValid():
                depth = np.array(depth_map, dtype=np.float64, copy=True)
        except Exception as exc:
            sdk_artifacts["depth_error"] = str(exc)
        confidence = None
        try:
            confidence_map = self.handle.GetConfidenceMap()
            if confidence_map is not None and confidence_map.IsValid():
                confidence = np.array(confidence_map, dtype=np.float64, copy=True)
        except Exception as exc:
            sdk_artifacts["confidence_error"] = str(exc)
        normals = None
        try:
            normal_data = point_map.GetNormalDataPtr()
            if normal_data is not None and normal_data.IsValid():
                normals = np.array(normal_data, dtype=np.float64, copy=True)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # 旧版固件或关闭法向计算时可能没有这个数组
            normals = None
        sdk_artifacts.update({
            "depth_available": depth is not None,
            "confidence_available": confidence is not None,
            "normals_available": normals is not None,
        })
        return Capture(
            points=points,
            captured_at=datetime.now(timezone.utc).isoformat(),
            device=self.device_info,
            options=_safe_options(effective_options),
            organized_points=organized_points,
            sdk_points=sdk_points,
            image=image,
            image_type=image_type,
            depth=depth,
            confidence=confidence,
            normals=normals,
            sdk_artifacts=sdk_artifacts,
            sdk_version=str(getattr(self.sdk, "GetVersion", lambda: "unknown")()),
        )

    def save(
        self,
        frame: Capture,
        directory: str | Path,
        *,
        cloud_points: np.ndarray | None = None,
        extra_metadata: dict[str, Any] | None = None,
        storage_policy: CaptureStoragePolicy | None = None,
    ) -> Path:
        """Persist an in-memory capture without accessing the live SDK handle."""

        directory = Path(directory)
        policy = storage_policy or CaptureStoragePolicy.compact()
        layout = CaptureLayout(directory)
        layout.ensure()
        cloud = frame.points if cloud_points is None else np.asarray(cloud_points, dtype=np.float32)
        np.save(layout.write_path("processed", "cloud.npy"), cloud.astype(np.float32, copy=False))
        # 将 SDK 组织好的点图保存为 float32
        # 磁盘单位保持米以兼容 process_turntable_capture
        # 重建时再由流程转换为毫米
        array_files: dict[str, str] = {
            "cloud_npy": layout.relative(layout.write_path("processed", "cloud.npy")),
        }
        source_points = frame.sdk_points
        if source_points is None:
            source_points = np.asarray(frame.points).reshape(-1, 3) / 1000.0
        points_path = layout.write_path("source", "points.npy")
        np.save(points_path, np.asarray(source_points, dtype=np.float32))
        array_files["points_npy"] = layout.relative(points_path)
        for field_name, file_name, value, enabled in (
            ("depth_npy", "depth.npy", frame.depth, policy.save_depth),
            ("confidence_npy", "confidence.npy", frame.confidence, policy.save_confidence),
            ("normals_npy", "normals.npy", frame.normals, policy.save_normals),
        ):
            if enabled and value is not None:
                path = layout.write_path("source", file_name)
                np.save(path, np.asarray(value, dtype=np.float32))
                array_files[field_name] = layout.relative(path)
        point_map_path = layout.write_path("processed", "cloud.ply")
        finite = cloud.reshape(-1, 3)
        finite = finite[np.isfinite(finite).all(axis=1)]
        if policy.save_processed_cloud_ply:
            _write_ascii_ply(point_map_path, finite)
        image_path: str | None = None
        image_info: dict[str, Any] | None = None
        if frame.image is not None:
            image_target = layout.write_path("source", "image.png")
            if cv2.imwrite(str(image_target), frame.image):
                image_path = layout.relative(image_target)
                image_array_path = layout.write_path("source", "image.npy")
                if policy.save_image_npy:
                    np.save(image_array_path, frame.image)
                    array_files["image_npy"] = layout.relative(image_array_path)
                image_info = {
                    "width": int(frame.image.shape[1]),
                    "height": int(frame.image.shape[0]),
                    "type": frame.image_type,
                    "npy": array_files.get("image_npy"),
                }
        files = {
            "image": image_path,
            **array_files,
            "cloud_ply": layout.relative(point_map_path) if policy.save_processed_cloud_ply else None,
        }
        metadata: dict[str, Any] = {
            "captured_at": frame.captured_at,
            "device": asdict(frame.device),
            "options": frame.options,
            "sdk": {
                "api_version": frame.sdk_version,
                "artifacts": frame.sdk_artifacts,
                "arrays": array_files,
                "depth_unit": "m",
                "sdk_point_map_unit": "m",
                "confidence_unit": "sdk-defined",
                "normal_unit": "unit_vector",
            },
            "point_unit": frame.point_unit,
            "sdk_point_unit": "m",
            "files": files,
            "image_info": image_info,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        layout.write_path("metadata", "capture.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return point_map_path if policy.save_processed_cloud_ply else layout.write_path("processed", "cloud.npy")

    def _require_open(self) -> None:
        if self.handle is None or not self.handle.IsOpen():
            raise RuntimeError("RVC 相机尚未打开")


def _write_ascii_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(handle, points, fmt="%.7g")
