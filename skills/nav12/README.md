# FBI NAV12（自训导航）

本分支用于 Go2 + Piper L 仿真开发，不包含 AINav，也不连接真机。本轮从原 FBI `300→12` 步态热启动，扩展为 `310→12` 并训练完整腿 Actor；六轴机械臂由独立控制保持参考位。50 Hz 策略 / 500 Hz 原仿真与 PD。独立 NAV 不代表六段链或换臂、变频通过。

- [`env.py`](env.py)：位姿目标到速度参考、5×60 历史 + 10 当前观测、12 腿动作映射、实际历史反馈与四足停稳判据。
- [`build_fbi.py`](build_fbi.py)：从授权、固定 SHA 的 `scene.xml` 与 `robot.xml` 构建原 NAV 布景，核对机器人动力学及执行器映射。
- [`train.py`](train.py)：自训 PPO 与受信任热启动权重校验。
- [`heldout.json`](heldout.json) / [`record.py`](record.py)：预登记 96 例与 1× 原速固定样本评估。原环境代码未改，移植只改模型目录、热启动位置与产物出口，不改变观测或控制节拍。

| 原独立留出，每组 32 场 | 地面 | 自由箱顶 | 固定高台 |
| --- | ---: | ---: | ---: |
| 50100 | 22/32 | 29/32 | 20/32 |
| 50101 | 18/32 | 23/32 | 19/32 |

## 复现

需要授权的冻结 FBI `climb` V6 基础资产及网格；模型/依赖没有随权重交付包公开发布。Python 3.11、PyTorch CUDA、MuJoCo 3.12、MuJoCo Warp（原项目兼容版）、Warp、NumPy、ffmpeg/ffprobe；按本组授权 GPU 配置可见设备，不启动历史服务器脚本。

```bash
export FBI_NAV_MODEL_DIR=/absolute/path/to/new-nav-assets
export FBI_NAV_WARMSTART=/absolute/path/to/authorized/model_7999.pt
python skills/nav12/build_fbi.py --base-assets /absolute/path/to/authorized-v6-assets --output "$FBI_NAV_MODEL_DIR"
python skills/nav12/train.py /absolute/path/to/new-run --seed 50100 --n 2048 --steps 12000000
# 用原训练运行目录（policy_latest.pt + status.json），或经权重 SHA 核对的等价档案。
python skills/nav12/record.py /absolute/path/to/new-run 0
```

仅评估不需要 warm-start 权重。`record.py` 的末尾 GPU 序号参数仅用于记录，不负责绑定设备。构建的 MJCF 会包含本网格路径；换机器重建，不能直接搬旧生成文件。

## 权重与交付

受控交接包分支目录 `nav/` 内有 `nav12-seed50100.pt`、`nav12-seed50101.pt` 和 SHA 清单，评估运行元数据在 `nav-evidence/`。50100 是已实测 NAV→正面 PUSH 前缀的版本。申请包时请向维护人报分支和 seed；公开 Git 只存代码和校验信息，不提交私有 checkpoint、第三方模型资产、录像、服务器配置或凭据。见[跨技能权重索引](../../docs/CHECKPOINT_HANDOFF.md)。
