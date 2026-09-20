"""Stable acquisition defaults shared by all inspection operations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScanConfig:
    port: str | None = None
    frames: int = 18
    speed_dps: float = 10.0
    settle_seconds: float = 0.25
    ratio: float = 180.0
    device: int = 1
    capture_profile: str = "reflective-metal"
    reflection_filter_threshold: int = 6
    z_min_mm: float | None = None
    z_max_mm: float | None = None

    @property
    def step_degrees(self) -> float:
        """Return the equal angular increment for one complete revolution."""

        return 360.0 / self.frames
