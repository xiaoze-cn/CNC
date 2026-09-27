"""RVC acquisition profiles"""

from __future__ import annotations

from typing import Any

from .acquisition import Camera


def _set_hdr(options: Any, exposures: tuple[int, ...], brightness: tuple[int, ...], *, scan_times: int = 3) -> None:
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


def profile_options(camera: Camera, profile: str, *, reflection_filter_threshold: int | None = None) -> Any:
    """Build capture options for the selected profile"""

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

    # 2D image capture
    options.exposure_time_2d = 3
    options.gain_2d = 0.0
    options.gamma_2d = 1.0
    options.enable_2d_in_capture = True
    options.use_projector_capturing_2d_image = True

    # 3D structured light capture
    options.gain_3d = 0.0
    options.gamma_3d = 1.0
    options.scan_times = 1
    options.calc_normal = True
    options.calc_normal_radius = 5

    # Stripe and phase filtering
    options.filter_range = 0
    options.phase_filter_range = 0
    options.light_contrast_threshold = 3

    # Automatic noise removal ignores its manual distance and point count
    options.use_auto_noise_removal = True
    options.noise_removal_distance = 0.455
    options.noise_removal_point_number = 501

    # Automatic bilateral filtering ignores its manual kernel and sigma values
    options.use_auto_bilateral_filter = True
    options.bilateral_filter_depth_sigma = 0.0
    options.bilateral_filter_kernal_size = 0
    options.bilateral_filter_space_sigma = 0.0
    if hasattr(camera.sdk, "SmoothnessLevel_Off"):
        options.smoothness = camera.sdk.SmoothnessLevel_Off

    # 3D point output in camera coordinates
    options.transform_to_camera = True
    options.downsample_distance = 0.0
    # Preserve turntable markers outside the subject depth range
    options.truncate_z_min = -9999.0
    options.truncate_z_max = 9999.0

    # Baseline capture profile
    if profile == "baseline":
        options.capture_mode = camera.sdk.CaptureMode_Normal
        options.confidence_threshold = 0.15
        options.reflection_filter_threshold = 6 if reflection_filter_threshold is None else reflection_filter_threshold
        options.smooth_sigma = 5.0
        exposures = (3, 6, 10)
        brightness = (240, 240, 240)
        _set_hdr(options, exposures, brightness)
        return options

    # Capture mode and quality thresholds for other profiles
    options.capture_mode = (
        camera.sdk.CaptureMode_AntiInterReflection
        if "anti-reflection" in profile
        else camera.sdk.CaptureMode_Normal
    )
    options.confidence_threshold = 0.10
    options.reflection_filter_threshold = 6 if reflection_filter_threshold is None else reflection_filter_threshold
    options.smooth_sigma = 5.0

    # 3D HDR exposure and projector brightness by profile
    if profile in {"hdr", "reflective-metal", "marker-low-glare", "hdr-anti-reflection"}:
        if profile == "marker-low-glare":
            exposures = (3, 8, 20)
            brightness = (60, 120, 180)
        elif profile in {"reflective-metal", "hdr-anti-reflection"}:
            # Longer exposure uses lower projector brightness on reflective metal
            exposures = (3, 30, 100)
            brightness = (120, 60, 10)
            options.confidence_threshold = 0.0
        else:
            exposures = (3, 10, 30)
            brightness = (80, 160, 240)
        _set_hdr(options, exposures, brightness)
    else:
        # Non HDR capture
        options.hdr_exposure_times = 0
        options.exposure_time_3d = 10
        options.projector_brightness = 180
    return options
