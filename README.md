# FBI OPEN THE DOOR

Go2＋PiPER **物理夹持开门**研究代码，以及默认拒绝自动动作的真机迁移检查工具。

> **当前是仿真研究基线，不是可直接启动的真机自主开门软件。**
> 已实现规则流程＋6D IK＋双指物理接触＋冻结的12维腿策略。高层残差PPO与双相机识别单独实验；没有完成学习型HRL管理器、可靠视觉闭环或真实接触验收。

## 先读这些

- [当前能力与证据边界](docs/STATUS.md)
- [真机迁移清单与分级测试](docs/REAL_ROBOT_MIGRATION.md)
- [软件结构与控制接口](docs/ARCHITECTURE.md)
- [仿真复现与训练](docs/SIMULATION.md)
- [相机与标定](docs/CAMERAS.md)
- [第三方来源](THIRD_PARTY_NOTICES.md)

## 安全的第一步：无需机器人或仿真依赖

```bash
git clone https://github.com/WM4N-Group/FBI-OPEN-THE-DOOR.git
cd FBI-OPEN-THE-DOOR
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m fbi_door.cli dry-run
PYTHONPATH=src python3 -m fbi_door.cli doctor
```

`dry-run` 不导入 CAN、DDS、ROS，也不发送命令。`doctor` 是离线清单检查，不会自动读取机器人；示例配置必须返回**退出码2（阻止真机开门）**，因为模型、实测夹爪力、手眼标定、停止接口等尚未验证。检查项不是安全认证。

## 代码结构

```text
src/fbi_door/
  sim/                  # 共享物理仿真、6D IK、夹持门任务、残差训练与相机观察
  deployment/           # 无硬件副作用的指令检查、标定与坐标变换、dry-run
  cli.py                # 默认安全入口
hardware/arm/           # 从原项目保留的ROS2/CAN机械臂执行层；需单独准备环境
configs/                # 不可直接上机的示例；实测配置使用*.local.json并忽略
scripts/                # 私有reference资产的校验导入
provenance/             # 资产哈希、上游来源、阶段评估摘要
licenses/               # 第三方许可证
```

## 仿真快速入口

Python 3.11／Linux x86_64 的独立环境；**不要把仿真依赖安装进Jetson/ROS Foxy环境**。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[sim]'
# 从项目所有者取得授权的私有reference包；权重和网格不放进Git。
python scripts/import_reference.py --archive /absolute/path/fbi-door-reference-20261113.tgz
export MUJOCO_GL=egl
fbi-door sim-build
fbi-door check-door
fbi-door check-grasp
fbi-door check-camera
fbi-door sim-probe --record
```

缺少资产时会明确报错，不会自动下载未知权重。私有包名和SHA256在`provenance/reference_bundle.json`中。视频和运行结果写入`.runtime/`，不进入Git；可用`FBI_DOOR_RUNTIME`指定独立运行目录。

## 真机迁移的硬性边界

1. 旧腿策略为**Piper L**；标准PiPER模型、关节范围、质量与相机外参必须另行核对。
2. 仿真夹爪±20 N是实验设置，**不是已核验的真机参数**。
3. 当前相机网络输出可见表面观察，尚未可靠估计规范抓取点／门轴，也没有控制机械臂。
4. 仿真中使用的精确模型前馈、关节/物体状态，不能在部署时当成已可获得的测量。
5. `hardware/arm`仅用于有现场操作员的分级bring-up；没有接入自动抓握开门的夹爪/底盘适配器。
6. 任何使能、复位、真实接触和底盘接管，必须在现场安全措施、实测限位/标定和独立急停验证后由操作员明确执行。
