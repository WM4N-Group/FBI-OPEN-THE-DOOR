# 推箱任务：冻结12维腿策略＋IK位置保持＋残差PPO

2026-11-14新增的独立仿真任务。

> **碰撞修订：默认控制器现为 `clearance_v2`。** 后续50 Hz几何/2 ms接触审计发现旧v1伸臂和收臂存在自碰，历史“8/8”仅表示旧任务判据完成，不是路径安全通过。不要继续用v1做部署候选。修订视频、同初态对照与训练迁移见[碰撞修订](BOX_CLEARANCE.md)。

以下控制分工、16秒结果和原始A800配置保留为 **v1历史记录**；重现旧控制器须明确加`--controller legacy`。当前默认v2约40秒，额外包含停稳、低速接触、卸力后退和再次停稳。

**不是视觉推箱、学习型HRL管理器或真机发布版**，不替换原开门场景、权重和实验。

## 控制分工

1. `approach`：读取仿真箱体位姿，走到箱前，停在机械臂可达范围。
2. `reach`：6D阻尼最小二乘IK。先在箱前下降，再靠近箱面；工具轴相对水平前向下倾30°，避免旧Piper L在较低目标处的关节限位及箱顶碰撞。
3. `push`：**缓存并保持机械臂六轴的关节位置指令**，由底盘向前移动推动箱子。保持的是相对机身的关节目标，不是世界系固定TCP；仍使用有力矩限制的PD，不是焊接、机械制动或无限刚度。
4. `stop`：底盘速度指令归零，保持手臂，等待停稳。
5. `retract`：底盘保持零速度指令，按限速关节目标收臂。

冻结的`arm_moving_clock/model_7999.pt`仍输出**12维腿关节控制**。原300维历史输入、观测顺序、动作映射、腿PD与模型资产不变，实际机械臂关节位置仍送入原底层观测。

新PPO输出6维**高层残差**，不是把底层换成18维整机策略：

| 阶段 | 允许的残差 | 上限 |
|---|---|---|
| 接近 | 底盘`vx/vy/wz` | ±0.04 m/s、±0.025 m/s、±0.04 rad/s |
| IK接近 | TCP位置`xyz` | ±1、±1.5、±1.5 cm |
| 锁臂前推 | 底盘`vx/vy/wz` | ±0.055 m/s、±0.025 m/s、±0.05 rad/s |
| 停步、收臂 | 无 | 全部屏蔽 |

锁臂前推时，**机械臂残差强制屏蔽**，缓存关节目标不随策略输出改变。阶段切换仍为规则，不能称为已训练HRL。原开门高层权重不用于初始化新任务。

## 物理和成功条件

- 箱子40×50×40 cm，2 kg，滑动摩擦系数0.35，六自由度自由刚体。
- 没有箱体执行器、吸引力或机器人—箱体焊接；只允许在reset初始化箱体位姿。整个rollout依靠MuJoCo接触动力学。
- 前移至少0.50 m，有至少0.4 s机械臂—箱体正法向接触，箱体侧向偏移小于0.18 m、倾斜小于0.35 rad，并完成停步收臂。
- 底盘/腿与箱体累计法向冲量达到0.5 Ns即失败，不能靠机身撞箱作弊。失稳、箱体侧偏/翻倒、接触缺失或阶段超时同样失败。
- 记录机械臂—箱体接触、非臂碰撞冲量、每个物理子步的接触力峰值、机械臂命令力矩峰值、锁臂目标变化及收臂误差。这些是**仿真诊断，不是实机限力验收**。
- 当前只随机化小幅初始位置/朝向，**没有**随机化箱重、摩擦或相机观测；不声称重箱、复杂地面、视觉或真机泛化。

## 已完成的基线与回归

公开原始数据见`provenance/box_push_20261114/`。

- 零残差规则＋IK基线，独立种子71000～71007：**8/8完成**。
- 前移0.532～0.539 m，平均约0.535 m；回合平均16.10 s。
- 机械臂—箱体接触5.122～5.214 s；非臂撞箱冲量均为0；推箱阶段保持目标的最大变化均为0 rad。
- 仿真接触法向力峰值12.20～16.78 N；不能当作硬件力传感器读数或安全标准。
- 标准库测试59项通过；带资产回归验证Gym接口、确定性reset、原始ONNX 300→12、自由箱体无执行器/焊接、锁臂残差屏蔽、真实推动与完整收臂，以及“无接触/机身撞击不算成功”的反例。
- 512环境步smoke完成10次PPO更新，并验证参数确实改变、原子保存和重载后动作一致。

这些是**基线和训练链路证据**，不是新PPO已经优于IK基线的证据。新训练结果应在预算完成后另做同条件对照，不能把训练回合成功率当作独立测试成功率。

## 复现

先按[仿真说明](SIMULATION.md)导入授权reference并生成原派生机器人。单独设置运行目录，**不要让推箱任务复用开门资产目录**：

```bash
DOOR_RUNTIME=/absolute/path/to/door-runtime
BOX_RUNTIME=/absolute/path/to/new-box-runtime
mkdir -p "$BOX_RUNTIME"
ln -s "$DOOR_RUNTIME/reference" "$BOX_RUNTIME/reference"
export FBI_DOOR_RUNTIME="$BOX_RUNTIME" MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m fbi_door.sim.build_box_scene --robot-source "$DOOR_RUNTIME/assets/robot.xml"
python scripts/test_box_clearance_sim.py
python -m fbi_door.sim.evaluate_box --count 8 --seed 71000 --randomization 1 \
  --output "$BOX_RUNTIME/reports/baseline.json" \
  --video "$BOX_RUNTIME/reports/baseline.mp4"
python -m fbi_door.sim.train_box --run-name smoke --envs 2 --device cuda:0 --smoke
python -m fbi_door.sim.train_box --run-name box_s100 --envs 6 --device cuda:0 --seed 100 \
  --controller clearance_v2 --nominal-steps 200000 --perturbed-steps 800000 --gate-retry-steps 200000
```

- 场景生成器拒绝覆盖已有场景；训练器拒绝覆盖已有运行目录。参考策略SHA256必须匹配，生成manifest和训练配置记录源文件/模型哈希。
- 从与当前任务兼容、可信的checkpoint恢复到**新运行目录**：`--resume-zip /absolute/path/to/policy_latest.zip`。从扰动阶段恢复还需`--start-stage initial_pose`；步数参数为恢复后新增预算，不能导入未知来源pickle。
- 用`python -m fbi_door.sim.evaluate_box --checkpoint /absolute/path/to/trusted.zip ...`评估新PPO。评估输出路径不能覆盖已有文件。
- `evaluate_box`默认录制只含第一个回合，视频标明特权位姿和控制器类型。视频编码取决于OpenCV；需要通用播放器兼容时另用ffmpeg转H.264。
- 只需要标准库的测试：`PYTHONPATH=src python3 -m unittest discover -s tests -v`。仿真回归需要私有资产及sim依赖，未执行不得声称通过。

## 2026-11-14 A800运行配置

两张A800各两个种子，共4组×6个环境＝24个并行MuJoCo环境：GPU0为100/101，GPU1为102/103。

每组名义预算100万环境步：固定初态20万＋小幅初始位姿扰动80万。每阶段最多追加20万步重试；阶段验证至少7/8才升级，否则停止并保留失败。最大约140万步/组（向量rollout向上取整另计），每组18小时墙钟上限。最终另评32个固定种子910000～910031，不代表硬件验收。

有效训练先写独立系统盘buffer，后台每30 s原子复制到独立共享运行目录；checkpoint校验ZIP CRC和共享回读SHA256。仍需检查镜像`pending/errors`，不能把未同步系统盘数据当成持久共享备份。

MuJoCo、IK和原冻结腿ONNX推理主要在CPU；PPO更新使用两张A800。小型MLP任务不等于显卡满载，不能用显存占满或无效计算来冒充训练吞吐。

## 真机边界

仅仿真。继承[真机迁移阻断项](REAL_ROBOT_MIGRATION.md)：旧Piper L与实际臂模型、TCP、实际负载、控制接口、反馈/力矩限制、底盘受反力稳定性、独立急停均未核验。没有使能CAN/DDS或向真机发送运动指令。
