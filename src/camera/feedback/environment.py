"""One-direction camera tuning environment and the curriculum sampler."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True, slots=True)
class SamplingContext:
    angle_degrees: float = 0.0
    workpiece_id: str = "default"
    roi_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    name: str
    episodes: int
    angles_degrees: tuple[float, ...] = (0.0,)
    workpiece_ids: tuple[str, ...] = ("default",)


class CurriculumScheduler:
    """Start at one angle, then unlock representative and all directions."""

    def __init__(self, stages: tuple[CurriculumStage, ...], *, seed: int = 7) -> None:
        if not stages or any(stage.episodes < 1 for stage in stages):
            raise ValueError("curriculum stages must be non-empty with positive episodes")
        self.stages = stages
        self._random = random.Random(seed)
        self._ends: list[int] = []
        total = 0
        for stage in stages:
            if not stage.angles_degrees or not stage.workpiece_ids:
                raise ValueError("curriculum stages need angles and workpiece ids")
            total += stage.episodes
            self._ends.append(total)

    @classmethod
    def default(cls, *, seed: int = 7) -> "CurriculumScheduler":
        return cls(
            (
                CurriculumStage("fixed-angle", 100, (0.0,)),
                CurriculumStage("representative-angles", 300, (0.0, 45.0, 90.0, 135.0, 180.0)),
                CurriculumStage("all-directions", 600, tuple(float(v) for v in range(0, 360, 15))),
            ),
            seed=seed,
        )

    def context(self, episode: int) -> SamplingContext:
        if episode < 0:
            raise ValueError("episode must be non-negative")
        index = next((i for i, end in enumerate(self._ends) if episode < end), len(self.stages) - 1)
        stage = self.stages[index]
        return SamplingContext(
            angle_degrees=self._random.choice(stage.angles_degrees),
            workpiece_id=self._random.choice(stage.workpiece_ids),
            metadata={"curriculum_stage": stage.name, "stage_index": index},
        )


@dataclass(frozen=True, slots=True)
class CameraAction:
    exposure_scale: float
    brightness_scale: float
    base_exposures_ms: tuple[int, int, int] = (5, 20, 75)
    base_brightness: tuple[int, int, int] = (80, 140, 200)

    def sdk_values(self) -> dict[str, tuple[int, int, int]]:
        return {
            "hdr_exposure_ms": tuple(max(1, int(round(v * self.exposure_scale))) for v in self.base_exposures_ms),
            "projector_brightness": tuple(int(np.clip(round(v * self.brightness_scale), 1, 255)) for v in self.base_brightness),
        }


@dataclass(frozen=True, slots=True)
class ActionSpace:
    exposure_scale_min: float = 0.50
    exposure_scale_max: float = 2.00
    brightness_scale_min: float = 0.50
    brightness_scale_max: float = 1.25

    @staticmethod
    def _map(value: float, lower: float, upper: float) -> float:
        return lower + (float(np.clip(value, -1.0, 1.0)) + 1.0) * 0.5 * (upper - lower)

    def decode(self, normalized_action: np.ndarray | list[float]) -> CameraAction:
        values = np.asarray(normalized_action, dtype=np.float32).reshape(-1)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError("normalized action must contain two finite values")
        return CameraAction(
            self._map(values[0], self.exposure_scale_min, self.exposure_scale_max),
            self._map(values[1], self.brightness_scale_min, self.brightness_scale_max),
        )


@dataclass(frozen=True, slots=True)
class FrameData:
    image: np.ndarray
    points_mm: np.ndarray
    confidence: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class QualityMeasurement:
    stable_ratio: float
    cleaned_ratio: float
    valid_ratio: float


class CaptureBackend(Protocol):
    def probe(self, context: SamplingContext) -> FrameData: ...

    def capture(self, action: CameraAction, context: SamplingContext) -> FrameData: ...


class FixedCleanupEvaluator:
    """Fixed cleanup plus repeatability; no learned or policy-controlled thresholds."""

    def __init__(self, *, confidence_min: float = 0.40, repeat_tolerance_mm: float = 0.50, min_component_area: int = 16) -> None:
        self.confidence_min = confidence_min
        self.repeat_tolerance_mm = repeat_tolerance_mm
        self.min_component_area = min_component_area

    def _clean_mask(self, frame: FrameData) -> np.ndarray:
        points = np.asarray(frame.points_mm, dtype=np.float64)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points_mm must be HxWx3 organized data")
        mask = np.isfinite(points).all(axis=2)
        if frame.confidence is not None:
            confidence = np.asarray(frame.confidence)
            if confidence.shape != mask.shape:
                raise ValueError("confidence shape must match organized points")
            mask &= np.isfinite(confidence) & (confidence >= self.confidence_min)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if count <= 1:
            return mask
        keep = stats[:, cv2.CC_STAT_AREA] >= self.min_component_area
        keep[0] = False
        return keep[labels]

    def evaluate(self, first: FrameData, second: FrameData, context: SamplingContext) -> QualityMeasurement:
        first_points = np.asarray(first.points_mm, dtype=np.float64)
        second_points = np.asarray(second.points_mm, dtype=np.float64)
        if first_points.shape != second_points.shape:
            raise ValueError("repeat captures must have identical organized shapes")
        first_mask = self._clean_mask(first)
        second_mask = self._clean_mask(second)
        stable = first_mask & second_mask
        stable &= np.isfinite(first_points).all(axis=2) & np.isfinite(second_points).all(axis=2)
        stable &= np.linalg.norm(first_points - second_points, axis=2) <= self.repeat_tolerance_mm
        roi = np.ones(first_mask.shape, dtype=bool) if context.roi_mask is None else np.asarray(context.roi_mask, dtype=bool)
        if roi.shape != first_mask.shape:
            raise ValueError("context roi_mask must match organized points")
        denominator = max(1, int(np.count_nonzero(roi)))
        return QualityMeasurement(
            stable_ratio=float(np.count_nonzero(stable & roi) / denominator),
            cleaned_ratio=float(np.count_nonzero((first_mask | second_mask) & roi) / denominator),
            valid_ratio=float(np.count_nonzero(first_mask & roi) / denominator),
        )


class CameraTuningEnv(gym.Env):
    """Gymnasium-compatible one-step environment.

    The package does not require Gymnasium at runtime. It follows the same
    reset/step signatures so the custom SAC loop and a future standard RL
    trainer can use the identical hardware backend.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        backend: CaptureBackend,
        evaluator: FixedCleanupEvaluator,
        *,
        action_space: ActionSpace | None = None,
        observation_size: tuple[int, int] = (256, 304),
    ) -> None:
        self.backend = backend
        self.evaluator = evaluator
        self.parameter_space = action_space or ActionSpace()
        height, width = (int(value) for value in observation_size)
        if height < 16 or width < 16:
            raise ValueError("observation_size must be at least 16x16")
        self.observation_size = (height, width)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(1, height, width), dtype=np.float32)
        self.context: SamplingContext | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        context = None if options is None else options.get("context")
        if context is None:
            context = SamplingContext()
        if not isinstance(context, SamplingContext):
            raise ValueError("reset options must contain a SamplingContext under 'context'")
        self.context = context
        observation = self._image(self.backend.probe(context).image)
        return observation, {"context": context}

    def step(self, normalized_action: np.ndarray | list[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self.context is None:
            raise RuntimeError("reset must be called before step")
        action = self.parameter_space.decode(normalized_action)
        first = self.backend.capture(action, self.context)
        second = self.backend.capture(action, self.context)
        measurement = self.evaluator.evaluate(first, second, self.context)
        return self._image(first.image), measurement.stable_ratio, True, False, {"action": action, "measurement": measurement}

    def _image(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 3:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError("policy image must be grayscale HxW")
        array = array.astype(np.float32)
        if np.nanmax(array, initial=0.0) > 1.0:
            array /= 255.0
        array = np.clip(np.nan_to_num(array, nan=0.0), 0.0, 1.0)
        height, width = (int(value) for value in self.observation_size)
        source_height, source_width = array.shape
        scale = min(width / source_width, height / source_height)
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(array, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        letterboxed = np.zeros((height, width), dtype=np.float32)
        offset_y = (height - resized_height) // 2
        offset_x = (width - resized_width) // 2
        letterboxed[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
        return letterboxed[None, ...]


class MockCameraBackend:
    """Deterministic smoke backend. It never opens a camera or moves a turntable."""

    def __init__(self, *, height: int = 96, width: int = 128) -> None:
        self.height, self.width = height, width
        yy, xx = np.mgrid[:height, :width]
        self._mask = ((xx - width * 0.5) / (width * 0.42)) ** 2 + ((yy - height * 0.5) / (height * 0.40)) ** 2 < 1

    def probe(self, context: SamplingContext) -> FrameData:
        angle = np.deg2rad(context.angle_degrees)
        image = np.full((self.height, self.width), 0.40 + 0.18 * np.cos(angle), dtype=np.float32)
        return FrameData(image=image, points_mm=self._points(1.0), confidence=np.ones_like(image))

    def capture(self, action: CameraAction, context: SamplingContext) -> FrameData:
        angle = np.deg2rad(context.angle_degrees)
        ideal_exposure = 1.0 + 0.35 * np.sin(angle)
        ideal_brightness = 0.95 + 0.20 * np.cos(angle)
        quality = np.exp(-((action.exposure_scale - ideal_exposure) / 0.30) ** 2)
        quality *= np.exp(-((action.brightness_scale - ideal_brightness) / 0.18) ** 2)
        points = self._points(float(np.clip(0.12 + 0.82 * quality, 0.0, 1.0)))
        confidence = np.where(np.isfinite(points[..., 2]), 0.9, 0.0).astype(np.float32)
        return FrameData(image=self.probe(context).image, points_mm=points, confidence=confidence)

    def _points(self, valid_fraction: float) -> np.ndarray:
        points = np.full((self.height, self.width, 3), np.nan, dtype=np.float32)
        valid = self._mask.copy()
        candidates = np.flatnonzero(valid)
        valid.flat[candidates[int(len(candidates) * np.clip(valid_fraction, 0.0, 1.0)):]] = False
        yy, xx = np.mgrid[:self.height, :self.width]
        points[..., 0] = np.where(valid, xx * 0.1, np.nan)
        points[..., 1] = np.where(valid, yy * 0.1, np.nan)
        points[..., 2] = np.where(valid, 20.0, np.nan)
        return points
