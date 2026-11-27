# FBI STAIRS_UP（stair）

Go2 + Piper L 同世界仿真楼梯技能。基于自训 FBI NAV50100（310→12），楼梯策略 334→12，全腿 Actor PPO；机械臂保持冻结参考位，50 Hz 策略 / 500 Hz 物理。使用仿真状态和地形特权观测，不是视觉控制、不是真机可用策略。

## 当前证据（2026-11-27）

正式普通人行楼梯目标为 **6×17 cm，踏深 28 cm，净宽 1.2 m**。较低台阶是课程/诊断；不降低正式门。17 cm 直接训练、14 cm 课程评估均为 0/24（历史旧完成门）。

| 10 cm 权重 | 原到顶停稳门 | 加入避撞门后 |
| --- | ---: | ---: |
| 326 维课程父权重（扩展评估） | 18/24 | 0/24 |
| 334 维 GPU1 真接触惩罚 | 21/24 | 0/24 |
| 334 维 GPU0 净空+真接触惩罚 | 5/24 | 0/24 |
| NAV50100 → GPU1 STAIRS（自然地面起步） | 6/8 | 0/8 |

最后一组 NAV 合格停车 8/8，STAIRS 下一拍真实执行 8/8，2/8 上楼保护终止，8/8 首撞阶发生于接管后 1.64–1.78 s。未测试顶平台 NAV；未证明无碰撞或 17 cm 能力。GPU0 退化版不可作为已通过课程的父权重。原验收为逐级承重、四足顶面承重及停稳 ≥1 s，新验收同时拒绝任一 2 ms 已解算子步中法向力 >0.10 N 的足撞立面/侧面或非足撞台阶（允许向上踩踏面）。不是绝对零接触保证。

打包后在新仿真目录复跑相同 8 例：NAV 停稳/首拍接管均 8/8、原到顶停稳 7/8、严格无撞阶 0/8。一次终态与历史轮不同，不能声称轨迹确定性或合并分母。见公开回执 `packaging_replay.json`。

## 代码入口

- `build.py`：从原 NAV 模型建立六级连续楼梯，保留机器人质量、惯量、驱动及碰撞；核定模型 SHA（仅准许 include 路径重定位）。
- `nav_motor_bridge.py`：原 NAV 观测/动作历史/PD、脚载荷与保护；`stairs.py`：每个物理子步接触分类、逐级载荷、334 维观测、严格完成门与训练奖励。
- `network.py`：原冻结 V6 MLP 结构（复用 128×3 ELU）。
- `train.py`：PPO，默认 NAV 热启动；也可从经 SHA 核验的父权重扩展输入。`FBI_CLEARANCE_COST=0` 对应 GPU1 接触惩罚；`=1` 为额外净空惩罚的退化对照。
- `eval.py` + `registration.json`：预登记独立评估，24 例，固定例 00 全程 1× 原速视频。`handoff.py` + `handoff_registration.json`：预登记 8 例 NAV→STAIRS，无姿态或世界重置，记录首条实际积分命令、状态哈希、逐拍轨迹和原片。
- `audit_handoff.py`：核交权及终止回执、媒体帧数、SHA。`provenance/stair_20261127/`：历史评估摘要与源码对应关系。

## 环境及授权文件

Python 3.11；实测 PyTorch 2.7.0+cu128、MuJoCo 3.12.0、mujoco-warp 3.12.0、warp-lang 1.17.0、NumPy 1.26.0、opencv-python 4.11.0.86；ffmpeg/ffprobe 与 EGL。按所在服务器配置 CUDA/图形设备、先检查 GPU 与已有任务。只启动仿真，不连接设备。

沿用 [NAV 的资产合同](../nav12/README.md)：授权 FBI V6 `scene.xml`、`robot.xml` 与原网格/依赖。从 `skills/nav12/build_fbi.py` 重建 NAV 模型，不能直接用其它臂型、其它关节顺序或随意改旧 XML。Git 不包含私有资产、权重及录像。向维护人领取原 FBI 交接包与本分支的 `FBI-stair-checkpoints-internal` 补充包；以下环境变量均为接收方的实际目录。补充包所需文件见 `CHECKPOINT_SHA256SUMS`。

```bash
# 在 stair 分支仓库根目录设置授权文件位置
export FBI_NAV_MODEL_DIR=/absolute/path/to/new-nav-assets
python skills/nav12/build_fbi.py --base-assets /absolute/path/to/authorized-v6-assets --output "$FBI_NAV_MODEL_DIR"
BUNDLE=/absolute/path/to/extracted/stair-checkpoints
(cd "$BUNDLE" && sha256sum -c SHA256SUMS)
export FBI_STAIR_NAV_CHECKPOINT="$BUNDLE/nav/nav12-seed50100.pt"
export FBI_STAIR_CHECKPOINT="$BUNDLE/stair/stair-rise10-contact-gpu1.pt"
export FBI_STAIR_RISE=.10 FBI_CLEARANCE_COST=0 MUJOCO_GL=egl
# 明确设置本机获准的 CUDA_VISIBLE_DEVICES 与 MUJOCO_EGL_DEVICE_ID。
# 每次运行必须使用新目录，保持已发布源文件和历史结果不变。
RUN=$(mktemp -d /tmp/fbi-stair-run.XXXXXX)
cp skills/stair/*.py skills/stair/*registration.json "$RUN/"
cd "$RUN"
python build.py
python handoff.py
sha256sum case00/full.mp4 > resources.txt
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=avg_frame_rate,duration,nb_read_frames -of compact case00/full.mp4 >> resources.txt
ffmpeg -nostdin -v error -xerror -i case00/full.mp4 -f null - && echo decode-OK >> resources.txt
python audit_handoff.py
```

独立评估：在另一份同样构建的运行目录执行 `python eval.py "$FBI_STAIR_CHECKPOINT" independent_eval`。父权重 326 维可由该评估器扩展为 334 维零初始化附加列；不可直接喂给 334 维网络。

训练复现（诊断 10 cm，不算 17 cm 验收）：在另一个新构建的 10 cm 运行目录，使用已核对的 326 维课程父权重：

```bash
export FBI_STAIR_APPROVED_INIT="$BUNDLE/stair/stair-rise10-parent-326.pt"
export FBI_STAIR_APPROVED_SHA=3872ca9e274324db0d5d1a5c6be434a40540fe94b22c378794a4415ea82a819c
export FBI_CLEARANCE_COST=0
python train.py train --seed 57410 --n 6144 --steps 8500000
python eval.py train/policy_latest.pt gate_eval
```

该历史父权重只过旧课程门。新门严格 ≥12/24 且训练无工程停止才可进入下一高度课程；不自动晋升当前未过门的候选。运行每轮保留最终 checkpoint、原终止码、全部评估分母及未裁剪原速固定例视频。不为验收修改准备位置、保护或接触判据。
