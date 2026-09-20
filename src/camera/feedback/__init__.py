"""Camera parameter tuning package with a hardware-independent SAC core."""

from .environment import (
    ActionSpace,
    CameraAction,
    CameraTuningEnv,
    CaptureBackend,
    CurriculumScheduler,
    FixedCleanupEvaluator,
    FrameData,
    SamplingContext,
)
__all__ = [
    "ActionSpace",
    "CameraAction",
    "CameraTuningEnv",
    "CaptureBackend",
    "CurriculumScheduler",
    "FixedCleanupEvaluator",
    "FrameData",
    "ImageSACAgent",
    "SamplingContext",
    "StructuredLightEncoder",
]


def __getattr__(name: str):
    if name in {"ImageSACAgent", "StructuredLightEncoder"}:
        from .agent import ImageSACAgent, StructuredLightEncoder

        return {"ImageSACAgent": ImageSACAgent, "StructuredLightEncoder": StructuredLightEncoder}[name]
    raise AttributeError(name)
