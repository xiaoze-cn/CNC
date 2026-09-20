"""Explicit RVC capture settings for repeatable acquisition."""

from __future__ import annotations

from typing import Any

from .acquisition import Camera


def _set_hdr_options(
    options: Any,
    exposures: tuple[int, ...],
    brightness: tuple[int, ...],
    *,
    scan_times: int = 3,
) -> None:
    if len(exposures) != len(brightness) or not exposures:
        raise ValueError("HDR exposure and brightness lists must be non-empty and match")
    options.hdr_exposure_times = len(exposures)
    options.exposure_time_3d = exposures[len(exposures) // 2]
    options.projector_brightness = max(brightness)
    for index, (exposure, projector) in enumerate(zip(exposures, brightness)):
        setters = (
            (options.SetHDRExposureTimeContent, exposure, "exposure"),
            (options.SetHDRGainContent, 0.0, "gain"),
            (options.SetHDRProjectorBrightnessContent, projector, "projector brightness"),
            (options.SetHDRScanTimesContent, scan_times, "scan times"),
        )
        for setter, value, label in setters:
            if not setter(index, value):
                raise ValueError(f"SDK rejected HDR {label} {value!r} at index {index}")


def capture_options_for_profile(
    camera: Camera,
    profile: str,
    *,
    reflection_filter_threshold: int | None = None,
) -> Any:
    """Build explicit options so captures never inherit stale camera state."""

    if reflection_filter_threshold is not None and not 0 <= reflection_filter_threshold <= 30:
        raise ValueError("reflection_filter_threshold must be between 0 and 30")

    loaded, options = camera.handle.LoadCaptureOptionParameters()
    if not loaded:
        options = camera.sdk.X1_CaptureOptions()
    if profile not in {
        "baseline",
        "anti-reflection",
        "hdr",
        "reflective-metal",
        "marker-low-glare",
        "hdr-anti-reflection",
    }:
        raise ValueError(f"未知相机参数组: {profile}")

    options.gain_3d = 0.0
    options.gain_2d = 0.0
    options.exposure_time_2d = 3
    options.gamma_2d = 1.0
    options.gamma_3d = 1.0
    options.scan_times = 1
    # 法向可用于边界和入射角诊断
    # 完整存储配置可以在需要时保存这些证据
    options.calc_normal = True
    options.calc_normal_radius = 5
    options.filter_range = 0
    options.phase_filter_range = 0
    options.light_contrast_threshold = 3
    options.use_auto_noise_removal = True
    # 与生产相机面板参数保持一致
    # 启用自动聚类去噪时 SDK 会忽略手动距离和点数
    # 仍然明确设置它们以便复现采集参数记录
    options.noise_removal_distance = 0.455
    options.noise_removal_point_number = 501
    options.use_auto_bilateral_filter = True
    options.bilateral_filter_depth_sigma = 0.0
    options.bilateral_filter_kernal_size = 0
    options.bilateral_filter_space_sigma = 0.0
    options.downsample_distance = 0.0
    # 保留完整的 SDK 深度输出
    # 转台标记可能位于有效测量范围之外但定位仍然需要这些数据
    options.truncate_z_min = -9999.0
    options.truncate_z_max = 9999.0
    options.transform_to_camera = True
    options.enable_2d_in_capture = True
    options.use_projector_capturing_2d_image = True
    if hasattr(camera.sdk, "SmoothnessLevel_Off"):
        options.smoothness = camera.sdk.SmoothnessLevel_Off
    if profile == "baseline":
        options.capture_mode = camera.sdk.CaptureMode_Normal
        options.confidence_threshold = 0.15
        options.reflection_filter_threshold = (
            6 if reflection_filter_threshold is None else reflection_filter_threshold
        )
        options.smooth_sigma = 5.0
        exposures = (3, 6, 10)
        brightness = (240, 240, 240)
        _set_hdr_options(options, exposures, brightness)
        return options

    options.capture_mode = (
        camera.sdk.CaptureMode_AntiInterReflection
        if "anti-reflection" in profile
        else camera.sdk.CaptureMode_Normal
    )
    options.confidence_threshold = 0.10
    profile_reflection_threshold = 6
    options.reflection_filter_threshold = (
        profile_reflection_threshold
        if reflection_filter_threshold is None
        else reflection_filter_threshold
    )
    options.smooth_sigma = 5.0
    if profile in {"hdr", "reflective-metal", "marker-low-glare", "hdr-anti-reflection"}:
        if profile == "marker-low-glare":
            exposures = (3, 8, 20)
            brightness = (60, 120, 180)
        elif profile in {"reflective-metal", "hdr-anti-reflection"}:
            # 使用三帧 HDR 曝光时间为 3 毫秒 30 毫秒和 100 毫秒
            # 对应投影亮度故意递减以保护短曝光高光帧
            exposures = (3, 30, 100)
            brightness = (40, 20, 1)
            options.confidence_threshold = 0.0
        else:
            exposures = (3, 10, 30)
            brightness = (80, 160, 240)
        _set_hdr_options(options, exposures, brightness)
    else:
        options.hdr_exposure_times = 0
        options.exposure_time_3d = 10
        options.projector_brightness = 180
    return options
