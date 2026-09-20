# 采集存储约定

默认使用 `compact` 存储策略。每帧只保留：

- `source/points.npy`：SDK 点图，单位米，`float32`，可用于重新定位和重建。
- `source/image.png`：转台标记定位需要的图像。
- `processed/cloud.npy`：主体点云缓存，必要时可由上面两项和标定重新生成。
- `metadata/capture.json`：采集参数、设备信息和定位结果。

以下文件属于可重算或诊断数据，默认不保存：`depth.npy`、`confidence.npy`、`normals.npy`、`image.npy`、`source_indices.npy` 和逐帧 `cloud.ply`。

需要传感器证据调试时可使用 `-complete`。整理已有数据：

```text
just compact data/inspect --dry-run
just compact data/inspect
```

整理器会原子地把 `points.npy` 转成 `float32`，删除上述可重算文件；不会删除最终报告、STEP 文件或双面结果。
