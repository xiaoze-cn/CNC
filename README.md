# PCB and CNC Inspection

基于 RVC 结构光相机、电动转台和 STEP 模型的多视角三维重建与尺寸偏差分析项目。

## 从这里开始

- [系统流程与代码地图](docs/reconstruction.md)：端到端调用链、各阶段算法、坐标系、数据产物、质量门禁、当前瓶颈和后续路线。
- [相机参数说明](docs/camera.md)：PyRVC 采集参数及现场调节原则。
- [采集存储约定](docs/storage.md)：`compact`、`metrology`、`complete` 三种存储策略。
- [研发讨论记录](docs/discussion.md)：历史方案、实验结论和阶段性决策。

## 常用命令

```powershell
# 检查相机和转台
pixi run doctor

# 采集、重建、合并并与默认 STEP 比对
pixi run inspect

# 完整检测改用一致性优先融合
pixi run inspect --consensus

# 用当前代码重新处理一批已有采集
pixi run merge data/inspect/<批次目录>

# 重新处理时改用一致性优先融合
pixi run merge data/inspect/<批次目录> --consensus

# 查看最终点云
pixi run show data/inspect/<批次目录>/cloud.ply

# 查看单个放置面的局部多视角证据
pixi run show data/inspect/<批次目录>/placement_A --evidence

# 查看完整检测结果
pixi run show data/inspect/<批次目录>

# 以 0.75 mm 面片显示蓝色未观测区域
just show --0.75

# 运行回归测试
pixi run python -m unittest discover -s tests -v
```

正式检测默认使用 `metrology` 存储策略。`processing_status=ok` 只表示处理链成功，不代表工件合格；在特征级公差、测量不确定度和标准件验证闭环之前，报告中的合规结论保持 `indeterminate`。
