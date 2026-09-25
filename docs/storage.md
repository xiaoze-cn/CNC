# 采集存储约定

正式检测默认使用 `metrology` 存储策略。每帧保留：

- `source/points.npy`：SDK 点图，单位米，`float32`，可用于重新定位和重建。
- `source/image.png`：转台标记定位需要的图像。
- `source/depth.npy`：邻视角遮挡判断使用的组织深度图。
- `source/confidence.npy`：SDK 点级置信度。
- `source/normals.npy`：入射角、法向一致性和同侧观察判断使用的法向。
- `processed/source_indices.npy`：主体点到原始传感器像素的索引。
- `processed/cloud.npy`：主体点云缓存，必要时可由上面两项和标定重新生成。
- `metadata/capture.json`：采集参数、设备信息和定位结果。

`metrology` 不保存重复的 `image.npy` 和逐帧 `cloud.ply`；边缘证据直接读取无损 PNG。

空间受限且不需要正式计量证据时可显式使用 `--storage-profile compact`。
需要保留所有诊断副本时使用 `--complete`。整理已有数据：

```text
just compact data/inspect --dry-run
just compact data/inspect
```

整理器会原子地把 `points.npy` 转成 `float32`，删除传感器证据和可重算文件；不会删除最终报告、STEP 文件或多摆放结果。整理后的历史数据仍可进行几何融合，但不能恢复 SDK confidence、法向和完整遮挡证据。

STEP 对比会从正式扫描的逐帧证据派生以下可见性点云：

- `visible_no_return.ply`：CAD 可见但投影邻域没有有效深度，仅作为补拍或复核候选。
- `visible_unqualified_return.ply`：投影邻域存在深度，但该 CAD 区域没有进入正式融合点云。
- `occluded.ply`：在采集视场内，但被 CAD 自身遮挡。
- `out_of_view.ply`：所有已记录相机视场均未覆盖。
- `insufficient_evidence.ply`：原始帧证据不完整，不能继续细分。

这些文件都是派生诊断层。只有 `metrology` 或 `complete` 存储策略能够重新计算完整 CAD 可见性；`visible_no_return` 不等于已经确认的材料缺失。
