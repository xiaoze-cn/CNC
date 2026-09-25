"""STEP registration and geometric deviation reporting."""

from .deviation import compare_step, merge_scans, show_comparison
from .fusion import (
    LocalPlacementFusionConfig,
    PlacementFusionLayer,
    PlacementFusionResult,
    fuse_layers,
)
from .trace import TraceResult, trace_frame

__all__ = [
    "compare_step",
    "merge_scans",
    "show_comparison",
    "LocalPlacementFusionConfig",
    "PlacementFusionLayer",
    "PlacementFusionResult",
    "fuse_layers",
    "TraceResult",
    "trace_frame",
]
