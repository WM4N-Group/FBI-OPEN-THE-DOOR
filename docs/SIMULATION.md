# 仿真复现

## 环境

原训练服务器核验组合：Python3.11、NumPy1.26.0、MuJoCo3.12.0、Torch2.7.0+cu128、ONNX Runtime1.23.2、Gymnasium1.2.1、SB3 2.7.1、OpenCV4.11.0。`.[sim]`给出兼容包版本；CUDA wheel/驱动另按主机安装，不在真机ROS环境安装该依赖集合。

```bash
python -m pip install -e '.[sim]'
python scripts/import_reference.py --archive /absolute/path/fbi-door-reference-20261113.tgz
export MUJOCO_GL=egl
fbi-door sim-build
fbi-door check-door
fbi-door check-grasp
fbi-door check-camera
fbi-door sim-probe --record
```

CPU渲染可按主机MuJoCo后端选择OSMesa；本次验证的是EGL。录像依赖主机`ffmpeg`，不由包管理器自动安装。

## 私有资产

资产不进入Git。授权reference包包含冻结策略、params、原推理封装和模型网格。导入器核验整个包SHA256、必要文件SHA256，拒绝路径穿越、符号链接和设备文件。不要导入来历不明的PyTorch/脚本资产。

默认运行目录`.runtime`可通过`FBI_DOOR_RUNTIME`覆盖。生成的MJCF包含当前机器绝对网格路径，所以**换机器后重新sim-build**，不要直接复制生成的MJCF。

## 训练

```bash
# 梯度/接口小检查；每次用新run-name
fbi-door sim-train --envs 2 --device cuda:0 --seed 90 --run-name smoke --smoke
# 独立种子与超参实验；底层腿策略冻结
fbi-door sim-train --envs 8 --device cuda:0 --seed 42 --run-name grasp_seed42
fbi-door sim-train --envs 8 --device cuda:1 --seed 43 --run-name grasp_seed43
```

课程：teacher grasp 15万、press 30万、push 50万、full 80万请求步数；SB3按rollout批次可能略多。阶段各32回合门禁，低于30/32停止并保留模型，每1万步checkpoint。门禁主要随机初始位置，不是广泛鲁棒性或真机认证。

公开整理版**移除了自动进入未验证旧视觉控制模式的课程**。视觉闭环必须另做设计和验收；不能用已知仿真把手状态触发阶段，再宣称纯视觉。

## 相机观察训练

```bash
fbi-door camera-train --scenes 3072 --epochs 80 --device cuda:0
fbi-door camera-probe
```

场景级划分，不把同场景两幅图分到不同集合。传感RGB关闭全部debug site，语义ID只用于标签/评分。`camera-probe`的运动仍由teacher＋IK控制，网络只在线预测/记录，不是视觉开门成功证明。

## 工程边界

MuJoCo和逐步IK主要在CPU运行，给PPO指定cuda并不表示4090满载；应按吞吐和独立实验收益使用资源，不能靠无意义负载占卡。
