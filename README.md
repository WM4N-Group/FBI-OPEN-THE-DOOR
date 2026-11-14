# FBI PUSH BOX

Go2＋旧 **Piper L** 推箱任务：冻结原12维腿策略，独立IK控制机械臂，锁臂后由机身推动箱子。

## 分支职责

- **`push`：推箱开发分支。** 推箱场景、碰撞修订、残差PPO、训练配置和实验记录都在这里维护。
- **`main`：开门分支。** 不再合入推箱专属改动。
- 本分支保留所依赖的共享底座、资产导入和历史开门基础代码，不代表把推箱训练混入开门任务。训练运行目录与模型分别存储；不要在活跃训练源码目录切换分支。

> 当前默认控制器为 **`clearance_v2`**，仅为仿真研究候选：特权箱体位姿、规则阶段管理、碰撞代理检查、IK和高层残差PPO。不是视觉自主推箱、学习型HRL或真机安全发布版。旧v1曾遗漏自碰门禁，仅保留用于历史复查。

## 当前动作流程

接近箱子 → 停稳 → IK及关节路径检查 → 锁臂低速建立接触 → 机身前推 → 停稳 → 保持臂姿后退脱离 → 再次停稳 → 沿检查路径收臂。

- 底层原策略仍输出12维腿控制，机械臂不混入这12维输出。
- 箱子为自由刚体，依靠真实仿真接触推动；没有焊接、吸引力、箱体执行器或rollout内位姿修改。
- v2检查自碰与箱体间隙；推箱结束后先卸力、后退至少20 cm，再收臂。
- 几何和接触检查只覆盖文档声明的仿真范围，不是连续碰撞或真实机器人安全证明。

## 从这里开始

- [长程导航任务的技能接口预留（无导航/管理器实现）](docs/NAVIGATION_SKILL_INTERFACES.md)
- [当前推箱控制器、碰撞修订与配对回归](docs/BOX_CLEARANCE.md)
- [低箱无需重训测试及15 cm适用边界](docs/LOW_BOX_TRANSFER.md)
- [推箱资产准备、训练与历史版本说明](docs/BOX_PUSH.md)
- [v2原始回归、源码指纹及训练迁移记录](provenance/box_clearance_20261114/README.md)
- [共享依赖及授权资产导入](docs/SIMULATION.md)
- [真机迁移阻断项](docs/REAL_ROBOT_MIGRATION.md)
- [第三方来源](THIRD_PARTY_NOTICES.md)

```bash
git clone --branch push https://github.com/WM4N-Group/FBI-OPEN-THE-DOOR.git
cd FBI-OPEN-THE-DOOR
# 纯标准库测试，无硬件动作，也不需要私有仿真资产。
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

仿真使用独立Python 3.11环境和独立`FBI_DOOR_RUNTIME`。按上述文档校验导入授权reference并生成场景后：

```bash
export FBI_DOOR_RUNTIME=/absolute/path/to/new-box-clearance-runtime
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/test_box_clearance_sim.py
python -m fbi_door.sim.evaluate_box --controller clearance_v2 --count 8 --seed 71000 \
  --output "$FBI_DOOR_RUNTIME/reports/v2.json" --video "$FBI_DOOR_RUNTIME/reports/v2.mp4"
python -m fbi_door.sim.train_box --controller clearance_v2 --run-name push_s200 \
  --seed 200 --envs 6 --device cuda:0 --nominal-steps 200000 --perturbed-steps 800000
```

每次使用新的输出/运行名称，保留旧模型。v2与v1的高层观测契约不同，不直接加载v1 PPO；原冻结腿权重继续复用。活跃训练先写系统盘buffer，再独立原子镜像至共享空间，需核查同步错误与待复制数。

## 安全边界

不自动使能CAN/DDS、不连接真机执行接触。旧Piper L与实际机械臂、TCP、夹爪力、相机外参、底盘停止与反力稳定性仍需现场核验。权重、网格、视频、环境和凭据不进入Git。

## 后续长程任务接口

`src/fbi_door/mission/`预留导航目标、路线候选、技能契约/请求、停止交接、来源过滤和变时长训练记录。导航是最终任务，开门/推箱是可选的环境干预，不写死技能顺序。当前只提供接口、协议测试和只读仿真结果桥接；没有实现导航器、共享世界执行器或学习型管理器，也不修改现有残差训练。

```bash
PYTHONPATH=src python3 scripts/demo_navigation_contract.py
```

此demo为显式合成协议测试，不是导航或物理技能链演示。
