set positional-arguments := true
set shell := ["pwsh.exe", "-NoProfile", "-CommandWithArgs"]

[private]
default:
    @just --list

# 交互执行多放置面采集 重建 合并 STEP 对比和结果显示
inspect *args:
    pixi run inspect @($args | Select-Object -Skip 1)

# 查看指定放置面和单个采集视角对坏点的贡献
trace *args:
    pixi run trace @($args | Select-Object -Skip 1)

# 从已有多放置面采集重新定位 重建 合并并进行 STEP 对比
merge *args:
    pixi run merge @($args | Select-Object -Skip 1)

# 查看最近一次或指定的检测结果
show *args:
    pixi run show @($args | Select-Object -Skip 1)

# 清理可重算的采集中间文件
compact *args:
    pixi run compact @($args | Select-Object -Skip 1)

# 只读检查相机和转台是否在线
doctor:
    pixi run doctor
