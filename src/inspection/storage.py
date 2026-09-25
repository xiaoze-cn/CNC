"""Storage inspection and safe compaction for recorded captures."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RECOMPUTABLE_CAPTURE_FILES = (
    Path("source/depth.npy"),
    Path("source/confidence.npy"),
    Path("source/normals.npy"),
    Path("source/image.npy"),
    Path("processed/source_indices.npy"),
    Path("processed/cloud.ply"),
)


@dataclass(frozen=True, slots=True)
class CompactionSummary:
    captures: int = 0
    converted_points: int = 0
    deleted_files: int = 0
    reclaimed_bytes: int = 0
    converted_reclaimed_bytes: int = 0

    def add(self, other: "CompactionSummary") -> "CompactionSummary":
        return CompactionSummary(
            captures=self.captures + other.captures,
            converted_points=self.converted_points + other.converted_points,
            deleted_files=self.deleted_files + other.deleted_files,
            reclaimed_bytes=self.reclaimed_bytes + other.reclaimed_bytes,
            converted_reclaimed_bytes=(
                self.converted_reclaimed_bytes + other.converted_reclaimed_bytes
            ),
        )


def _save_atomic(path: Path) -> int:
    """Convert an array in place without leaving a partially written source."""

    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not np.issubdtype(array.dtype, np.floating) or array.dtype == np.float32:
        return 0
    original_size = path.stat().st_size
    converted = np.asarray(array, dtype=np.float32)
    # Windows 会一直锁定映射文件直到数组视图释放
    del array
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temp:
            temp_name = temp.name
            np.save(temp, converted, allow_pickle=False)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, path)
        return max(0, original_size - path.stat().st_size)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def _update_metadata(capture_dir: Path) -> None:
    metadata_path = capture_dir / "metadata" / "capture.json"
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    files = metadata.setdefault("files", {})
    for key in ("depth_npy", "confidence_npy", "normals_npy", "image_npy"):
        files.pop(key, None)
    files["cloud_ply"] = None
    if isinstance(metadata.get("image_info"), dict):
        metadata["image_info"]["npy"] = None
    metadata["storage"] = {
        "profile": "compact",
        "recomputable_files_removed": True,
        "source_points_dtype": "float32",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def compact_capture(capture_dir: str | Path, *, dry_run: bool = False) -> CompactionSummary:
    capture_dir = Path(capture_dir)
    points_path = capture_dir / "source" / "points.npy"
    metadata_path = capture_dir / "metadata" / "capture.json"
    if not metadata_path.is_file() or not points_path.is_file():
        return CompactionSummary()

    reclaimed = 0
    deleted = 0
    converted = 0
    converted_reclaimed = 0
    if not dry_run:
        converted_reclaimed = _save_atomic(points_path)
        converted = int(converted_reclaimed > 0)
        _update_metadata(capture_dir)
    elif points_path.stat().st_size:
        array = np.load(points_path, mmap_mode="r", allow_pickle=False)
        if np.issubdtype(array.dtype, np.floating) and array.dtype != np.float32:
            converted = 1
            converted_reclaimed = max(0, points_path.stat().st_size - array.size * 4)

    for relative in RECOMPUTABLE_CAPTURE_FILES:
        path = capture_dir / relative
        if not path.is_file():
            continue
        reclaimed += path.stat().st_size
        deleted += 1
        if not dry_run:
            path.unlink()
    return CompactionSummary(
        captures=1,
        converted_points=converted,
        deleted_files=deleted,
        reclaimed_bytes=reclaimed,
        converted_reclaimed_bytes=converted_reclaimed,
    )


def compact_root(root: str | Path, *, dry_run: bool = False) -> CompactionSummary:
    """Compact every capture below an inspection/acquisition root."""

    root = Path(root)
    summary = CompactionSummary()
    for metadata_path in root.glob("**/metadata/capture.json"):
        summary = summary.add(compact_capture(metadata_path.parent.parent, dry_run=dry_run))
    return summary
