# 长程导航优先的技能接口预留（v1）

2026-11-14，`push`研究分支。**此变更是协议、调度生命周期、记录格式和只读适配，不是导航器、共享世界执行器、学习型HRL管理器或复杂场景实验结果。** 不修改正在训练的门/箱控制器。

## 1. 任务定义：导航是主任务，操作是干预手段

任务不是“按顺序完成开门和推箱”，而是：

> 从起点到达指定导航目标；根据环境和代价，选择绕行、开门、移开障碍、重新观察、重规划或恢复。

- 操作完成只说明一个局部目标可能达成，不终止导航任务。
- 不要求每个任务都调用操作技能，也不提前写死唯一技能序列。
- 导航结束依据目标位姿、定位证据和显式收尾条件，而不是累计完成多少技能。
- v1目标是**同一具名坐标系中的2D位姿＋位置/朝向容差**；语义区域、跨楼层/3D目标需要后续目标提供器或schema扩展，不冒充已经实现。
- `deadline_s`与任务指定时钟同域；用`SkillCoordinator.for_goal()`绑定全局预算。单个技能超时是可恢复事件，不自动等于整个任务终止；全局预算耗尽另行记录。

## 2. 模块边界

```text
NavigationGoal + 持续存在的WorldReader
        ↓
NavigationPlanner.propose()          ← 接导航器/地图/可供性模块，当前无实现
        ↓ 多条路线及可能需要的干预对象
ManagerPolicy.choose()              ← 规则/搜索/BC/RL统一签名，当前无策略实现
        ↓ SkillRequest（对象、子目标、参数、路线ID、父请求/重试意图）
SkillCoordinator                    ← 校验、可执行掩码、资源租约、停止/交接协议
        ↓ start_skill_intent / stop_required（不是电机指令）
SimulationSkillExecutor             ← 共享世界技能后端，当前无实现
        ↓
现有冻结腿策略 / IK / 局部残差控制 / 独立保护
```

文件：

| 文件 | 职责 |
|---|---|
| `mission/contracts.py` | `navigation-skills/v1`、位姿/证据、任务目标、技能契约和严格请求JSON |
| `mission/runtime.py` | 候选校验、资源租约、停止确认、终态复核、来源过滤 |
| `mission/interfaces.py` | 世界读取、路线候选、导航器、管理器、仿真执行器的Protocol |
| `mission/catalogue.py` | 导航/重规划/观察/开门/推箱**契约样例**，默认没有后端 |
| `mission/evaluation.py` | 导航终态、全尝试耗时、交接记录、变时长SMDP转换 |
| `mission/boundaries.py` | 现有推箱episode信息的只读诊断证据，不是执行适配器 |

`simulation`/`test_double`注册标签仅声明接线类型，不是对实现的验证或安全认证。当前包没有CAN、DDS、ROS、SDK、仿真步进或硬件运动调用，也没有硬件profile。

## 3. 请求、证据和交接约束

### 请求

`SkillRequest`包含：

- schema、技能ID/版本、mission/world/request ID；
- 地图和观测revision，防止用旧感知结果发起新动作；
- 对象ID、具名坐标系下的2D子目标、带显式单位/范围的参数；
- 路线候选ID，以及`parent_request_id`、`normal/retry/recovery`意图。

请求JSON拒绝未知字段、缺失/不支持的schema、重复参数、NaN/Inf和bool冒充数字。目录中的参数范围**只是接口样例**，不证明现有固定0.5m推箱策略可以任意换目标；后端尚未注册时一律不可调度。

技能契约区分：

- `preconditions`：只在开始时满足，例如已停稳、手臂已收好。
- `invariants`：执行中持续成立，例如导航定位有效。
- `postconditions`：成功交接时再次验证，例如达到本次请求的对象目标、停稳并收臂。

通行性使用`request/{request}/route_clear`，不能把某条绕行路线的可通行性复用给另一个候选。对象与请求谓词分别使用`{entity}`、`{request}`，防止“上一扇门/上一次推箱成功”的布尔值被误用于下一目标。后端自报成功**不会**自动设置世界谓词、修改地图、打开门或移动箱子。

### 证据

`Fact`与`PoseSample`必须标记时间、时钟、来源和置信度。调用方另外传入当前时间，不能只让旧观测与旧请求互相匹配就当作新鲜数据。

- `SIM_TEACHER`允许仿真真值，必须保留teacher标签。
- `OBSERVED_ONLY`不把`sim_truth`事实/位姿/路线估计送给管理器；合法动作检查先检查来源，再检查真假，避免通过mask泄漏隐藏真值。
- 路线候选有world、clock、时间、map revision、来源和置信度；过期或不同世界的候选不能混用。
- **来源标签不是密码学证明。** 外部地图/候选生成器仍必须审计；不能把真值位置、最优路线成本或可供性改个标签伪装成估计。候选动作的目标坐标也必须来自允许的观测。尚未实现的候选生成/视觉闭环不能由本接口替代。
- 示例时效和置信度阈值是仿真/协议配置，不是真机已标定的安全参数。

### 控制所有权

`base/arm/gripper`表示**高层目标写入权**，不是关闭底层平衡/PD。运动技能共享一份任务协调器，由世界driver保证一个物理世界只有一个目标所有权仲裁入口。

- 互斥资源未释放，不能启动另一个争用者。
- 超时、取消、世界重置、地图失效或持续条件失败，先返回`stop_required`；**不能立即把资源交给下个技能**。
- 必须收到对应request/world的`quiescent`确认，才结束租约。它表示本技能停止发布运动目标并完成交接，不等于电机失能。
- 成功条件在停止确认时再次检查，停止延迟也计入技能耗时；重复确认幂等。
- 每个运动技能要求新鲜的`safety/motion_permitted`证据。纯观察/计算技能不占运动资源，可以在运动被禁止时请求恢复感知；带动机械臂的主动观察则必须申明`arm`资源。
- `safety/motion_permitted`是协议输入，不是硬件使能开关或真实安全保证。保护/急停必须在执行器侧独立存在，不能只靠高层奖励或此状态机。

世界重置/时钟回退会使该次耗时不可作为训练样本。协议没有强制释放未确认停止的资源，也不提供绕过机制。

## 4. 当前门/箱代码不能直接冒充共享世界技能

这是接长序任务前必须解决的边界：

1. 现有`DoorEnv`/`ClearanceBoxPushEnv`拥有独立模型、任务状态及episode reset。**不能每调用一个技能就reset环境**，否则导航位姿、门状态、箱体位置和底层历史会被重置。
2. 后续必须由同一个世界driver持续保存物理场景、定位/地图、冻结腿策略的300维历史和原始控制频率。技能开始只能重置技能局部状态，不能重建低层策略wrapper。
3. 原开门环境还包括过门，不能未经拆分就标为“只开门”；开门尚缺新版自碰验收。
4. v2推箱只读适配只导出`standalone_box_task_completed`、碰撞代理检查和收臂诊断，**不导出参数化技能`object_goal_reached`、底盘停稳、路径已清空或控制权释放**。
5. 该只读适配要求独立的`standalone:`时钟，时间必须匹配episode原记录，不能把旧记录重新盖成当前导航任务的观测。

因此目录里的`navigation.move_to/replan`、`perception.reobserve`、`manipulation.open_door/push_box`当前都是契约；导航器和共享世界执行后端未提供时，mask为不可调用。真实推箱仿真回归只能证明只读适配不改物理状态、操作完成不等于远处导航目标完成，**不能证明技能链已经跑通**。

## 5. 为后续复杂场景预留什么

以下仅为设计维度，不是已经构建/训练的场景：

| 研究因素 | 让优化真正有价值的差异 |
|---|---|
| 绕行与操作的取舍 | 短路有门、长路畅通；不能每次强制开门 |
| 非交换的操作效果 | 推箱清一条通道，却可能堵住后续路线；执行顺序/落点影响后续任务 |
| 接近/退出位姿 | 到门前的位置影响伸臂可达性；操作末态影响下一段导航 |
| 部分可观测 | 门是否锁住、推行是否可行尚未知；观察与试探有时间代价 |
| 可恢复失败 | 重新定位、换接触侧、重试、重新规划或绕行，而非失败即reset整个任务 |
| 跨技能资源/记忆 | 停止确认、历史连续性、地图更新及对象身份不能在边界丢失 |

只把“能否执行”作为物理/观测约束，不用真值把“哪个选择最优”编码到mask。路线需要的干预对象只是候选假设，操作后仍要依据新观测确认通行性并重规划。

## 6. 记录与训练接口

- 核心终态是`NavigationGoal`，不是某技能的成功位。`mission_summary()`单独报告到达目标、完成交接、独立安全证据；安全证据缺失为`None`，不能当作0事故。
- 耗时包含失败、重试、恢复与停止确认；不只比较成功回合耗时。
- 记录每次请求的对象/目标/参数、父请求、路线ID、开始/结束观测revision、map revision、状态和真实持续时间；协调器保存交接前后快照，供后续分析下一技能的可启动性/价值。
- `smdp_transition()`按奖励事件的实际时间积分，并输出`gamma_per_second ** elapsed_s`。技能失败不自动设置任务终止位；任务terminal/truncated必须由任务driver明确给出。
- 并行的观察/重规划是额外事件；不能把重叠的技能回执任意串成单条SMDP。后续学习器需先定义主技能决策时序和辅助事件的信用分配。
- **这不是新的HRL训练器。** 当前SB3残差PPO代码完全未改；不能把变时长记录直接丢给固定折扣/固定步长的PPO，声称已经实现SMDP高层学习。后续需独立的高层环境、算法接入和bootstrap处理。
- 导航里程必须来自连续定位/里程计；接口不从稀疏技能端点直线距离伪造里程。
- 没有合理的、包含交互影响的参考路径定义时，SPL保留`None`，不编造导航效率结论。

建议后续对照：导航绕行基线、规则技能调度、规划器调度、冻结技能＋学习管理器、学习管理器＋交接感知残差。所有方法需共享观测、候选/保护边界、低层技能和预算；teacher真值只作为明确标注的额外组。

## 7. 离线使用与已执行验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/demo_navigation_contract.py
```

demo只使用合成状态和`test_double`，输出`physical_navigation_executed=false`。它验证操作成功时任务仍未完成；之后的“到达”也是显式合成fixture，不是物理导航结果。示例请求见`configs/navigation_mission.example.json`，候选列表不是执行顺序。

当前98项标准库测试通过，其中34项针对新协议；包含源隔离、时钟、资源交接、延迟停止、恢复和任务终态的反例。

有授权私有资产和sim依赖时，可在**独立验证runtime**运行：

```bash
export FBI_DOOR_RUNTIME=/absolute/path/to/interface-validation-runtime
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=src python scripts/test_mission_sim_bridge.py \
  --output "$FBI_DOOR_RUNTIME/reports/bridge.json"
```

验证runtime需具备已有v2静态资产和reference；不要覆盖训练目录。只运行一条独立推箱仿真，不启动导航或管理器训练。原始验证记录见`provenance/mission_interface_20261114/`。
