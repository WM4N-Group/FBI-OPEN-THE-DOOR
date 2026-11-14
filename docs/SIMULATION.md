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

## 双 A800 扩大实验

在独立项目/独立runtime上使用，不改动已有训练目录：

```bash
python scripts/launch_a800.py --runtime /absolute/path/to/validated/runtime \
  --batch-name a800_batch_01 --envs-per-run 6 --stage-scale 2 --gate-retries 2 \
  --camera-scenes 8192 --camera-epochs 160 --camera-batch-scenes 128 --dry-run
# 确认预算、设备和检查结果后，去掉 --dry-run 才会启动。
```

- 4个独立高层种子，2组/GPU；24个CPU物理环境。底层腿策略冻结。
- 每个种子的正常课程预算350万环境步，4组共1400万；失败阶段最多2次、每次原阶段50%的追加预算，总上限约2800万。SB3按rollout批次可能略多。
- 阶段仍要求30/32验证通过，不能用增加训练量替代门禁；追加预算用完仍失败则保留checkpoint并结束该候选。
- 反复用于门禁的种子属于验证集，不是最终独立测试成功率。
- 两个相机训练种子分别使用GPU0/1，第二组等待第一组完成同一场景配对数据生成；8192场景=16384张图，160epoch，128场景/批（256幅图）。相机只训练观察模型，不是视觉闭环机械臂控制。
- launcher使用脱离SSH会话的worker，保存PID、日志和状态，不终止其他任务。进程退出码0不等于任务成功。
- 支持受信任checkpoint恢复：`sim-train --resume-zip ... --start-stage press --run-name NEW_NAME`。恢复参数/优化器不保证逐步轨迹可复现；模型/观测契约必须匹配，后续阶段不可从随机策略直接起步。
- 大型GPU并不会自动加速CPU MuJoCo/IK；这里通过独立实验和相机训练使用双卡，不制造无意义负载，也不声称GPU满载。

## 共享存储故障恢复

参见[存储恢复说明](STORAGE_RECOVERY.md)。为避免共享盘的瞬时写入故障同时打断日志、状态和checkpoint，续训可使用独立系统盘runtime缓冲，并由独立镜像进程校验后同步到共享盘。未同步文件不能当作已完成共享备份；停机/迁移前必须检查pending和校验结果。恢复第一阶段支持`--initial-stage-steps`补齐剩余预算，不需要重复从头训练。

## 训练后对照评估

使用`fbi-door evaluate-paired --checkpoint <受信任PPO ZIP> --episodes 32 --record`；运行范围、数据划分、输出与证据边界见[评估说明](EVALUATION.md)。不会追加训练、改动旧模型或连接硬件。
