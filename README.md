# ORCA Hand Physics-Aware Gesture Sandbox

这个项目基于 `orca_sim`、MuJoCo 和 MediaPipe，目标不是只做一个“能动起来的机械手演示”，而是搭一个更适合手势研究的中间层：

- 用 `MediaPipe` 读取手部关键点
- 用 `ORCA hand` 的关节结构和物理先验约束这些观测
- 输出更稳定、更合理的手势特征
- 为后续的机器学习任务提供更干净的输入表示

当前项目最适合的用途是：

- 手部姿态观察与可视化
- MediaPipe 关键点到 ORCA 关节空间的映射实验
- 掌向状态、掌面法向量等高层特征提取
- 中国舞手势识别前的“物理合理化”预处理

## Project Goal

这个仓库当前更推荐这样理解：

```text
camera
-> MediaPipe landmarks
-> geometric features
-> physics-aware hand representation
-> ML-friendly features
```

重点不在于把机器人手“遥操作得很像真人”，而在于把视觉观测变成：

- 更平滑
- 更少抖动
- 更符合关节结构
- 更适合分类与识别

## Environment Setup

推荐使用 Python `3.11`。

### Conda

```powershell
conda create -n orca python=3.11 -y
conda activate orca
cd "c:\D\projects\Orca robot hand\orca sim\orca_sim"
python -m pip install -U pip
python -m pip install -e .
```

### Optional MediaPipe / OpenCV extras

如果你需要运行摄像头分析与 MediaPipe 脚本：

```powershell
python -m pip install -e ".[teleop]"
```

## Quick Start

### 1. Smoke test the simulator

先确认基础仿真环境可以跑：

```powershell
python .\random_policy.py --env right --render-mode rgb_array --steps 5
```

如果输出里能看到：

- `env=right`
- `version=v2`
- `obs_shape=(34,)`
- `action_shape=(17,)`

说明基础环境已经正常。

### 2. Open the viewer

```powershell
python .\random_policy.py --env right --render-mode human
```

### 3. Run the MediaPipe hand analysis / teleop tool

```powershell
python .\mediapipe_teleop.py --sim-render-mode rgb_array --target-hand right
```

如果你安装的是新版 MediaPipe Tasks API，还需要本地 `hand_landmarker.task` 模型文件：

```powershell
python .\mediapipe_teleop.py --sim-render-mode rgb_array --target-hand right --hand-landmarker-model ".\hand_landmarker.task"
```

## What `mediapipe_teleop.py` Does

这个脚本当前同时承担三类作用：

1. 摄像头可视化
2. ORCA 右手机器人手的实时姿态驱动
3. 手势分析特征显示

窗口中会显示这些实时信息：

- `Hand`
- `Palm`
- `Palm normal xyz`
- `Base y/p/r`

其中最值得机器学习使用的是：

- `Palm`
- `Palm normal xyz`
- 各指弯曲/张开对应的关节特征

## Recommended Usage Modes

### A. Analysis mode

如果你的重点是先看观测是否合理，而不是让底座自动乱动：

```powershell
python .\mediapipe_teleop.py --sim-render-mode rgb_array --target-hand right --hand-landmarker-model ".\hand_landmarker.task" --disable-auto-base
```

这个模式更适合：

- 观察掌向分类是否稳定
- 观察 `palm_normal` 是否合理
- 为后续特征设计做分析

### B. Teleop mode

如果你想保留自动底座联动：

```powershell
python .\mediapipe_teleop.py --sim-render-mode rgb_array --target-hand right --hand-landmarker-model ".\hand_landmarker.task" --base-gain 0.8 --base-roll-gain 0.8 --smoothing 0.22
```

这更适合做演示，但不一定最适合做特征采集。

## Useful Flags

### General

- `--camera-id 1`
  选择不同摄像头

- `--target-hand right`
  只使用右手

- `--no-mirror`
  关闭自拍镜像

- `--hand-landmarker-model ".\hand_landmarker.task"`
  指定 MediaPipe hand landmarker 模型

### View / windows

- `--sim-render-mode rgb_array`
  推荐。用 OpenCV 窗口显示仿真结果，便于调试

- `--sim-render-mode human`
  使用 MuJoCo 原生 viewer

- `--sim-window-width 1400 --sim-window-height 1000`
  调整仿真窗口大小

- `--camera-window-width 1200 --camera-window-height 900`
  调整摄像头窗口大小

### Motion tuning

- `--smoothing`
  时间平滑强度。越小越稳，越大越跟手

- `--base-gain`
  底座 yaw / pitch 增益

- `--base-roll-gain`
  底座 roll 增益

- `--disable-auto-base`
  关闭自动底座控制，只保留观测分析和手动底座测试

- `--fixed-base`
  使用原始固定底座场景，而不是 teleop 专用底座场景

## Keyboard Controls

运行 `mediapipe_teleop.py` 后可用：

- `q` / `Esc`：退出
- `r`：重置
- `m`：切换镜像
- `j / l`：手动 yaw
- `i / k`：手动 pitch
- `u / o`：手动 roll

这些键在调试“底座自由度是否真的生效”时很有用。

## Why This Project Is Useful For ML

如果直接把 MediaPipe 的原始关键点喂给分类器，通常会遇到：

- 遮挡抖动
- 深度不稳定
- 掌面翻转歧义
- 手指角度不合理

这个项目更适合作为一个“中间表示层”，把原始观测变成更适合 ML 的特征，例如：

- 掌向状态
- 掌面法向量
- 指关节弯曲度
- 指间张开程度
- ORCA 关节空间表示

对中国舞手势识别来说，这类特征通常比原始 landmark 更稳。

## Suggested Research Direction

更推荐的路线不是继续死磕“连续 180 度翻腕 teleop 是否完美”，而是：

1. 采集 MediaPipe 关键点
2. 提取几何特征
3. 做物理合理化与平滑
4. 输出稳定特征
5. 再做手势分类

一个更适合当前项目的 pipeline 是：

```text
MediaPipe landmarks
-> palm state / palm normal / finger features
-> physics-aware correction
-> feature export
-> ML classifier
```

## Project Structure

主要文件如下：

- [README.md](README.md)
  项目说明

- [pyproject.toml](pyproject.toml)
  依赖与包配置

- [random_policy.py](random_policy.py)
  基础仿真 smoke test

- [mediapipe_teleop.py](mediapipe_teleop.py)
  摄像头分析、MediaPipe 特征显示、底座与手部 teleop 实验入口

- [src/orca_sim/envs.py](src/orca_sim/envs.py)
  基础环境定义

- [src/orca_sim/task_envs.py](src/orca_sim/task_envs.py)
  任务级环境，例如 cube orientation

- [src/orca_sim/scenes/v2/scene_right.xml](src/orca_sim/scenes/v2/scene_right.xml)
  原始右手场景

- [src/orca_sim/scenes/v2/scene_right_teleop.xml](src/orca_sim/scenes/v2/scene_right_teleop.xml)
  为 teleop 增加底座 yaw / pitch / roll 的场景

## Current Limitations

当前实现要特别注意这些限制：

- 单目 MediaPipe 对大幅翻腕和接近 `180°` 翻转并不稳定
- 底座姿态自动映射更适合“辅助理解”，不适合当作绝对真值
- 这个仓库现在更适合做特征工程和物理先验实验，而不是最终生产级 teleop

## Recommended Next Step

如果你要继续往“机器学习准确率提升”推进，最推荐的下一步不是继续调 viewer，而是：

1. 导出每帧特征
2. 保存标签
3. 比较：
   - 原始 MediaPipe 特征
   - 物理增强后的特征

也就是说，下一阶段最适合把这个项目用成：

**手势数据采集与物理先验特征生成工具。**
