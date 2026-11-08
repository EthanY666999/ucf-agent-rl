# UCF Agent RL — Qwen3.5 4B / 9B 协作发布

**这是交负责人审阅的独立发布，不是已经跑完4B/9B的实验报告。** 本次发布只做CPU检查，没有下载模型权重，也没有启动或干预任何现有GPU任务。未审阅通过前不要向合作者分发或启动正式实验。

历史复现记录见 [decision-point](https://github.com/EthanY666999/decision-point)。

## 先看可用范围

| 内容 | 本发布状态 |
| --- | --- |
| 当前2B生产训练器、SFT、GRPO4、Random K=2、UCF v2、框架内StarPO-S/GiGPO、两个网格环境 | 已提取并作便携路径适配；CPU测试，不代表4B/9B GPU验收 |
| Qwen3.5-4B / 9B | 官方tokenizer/config/模板和meta形状检查；无权重 |
| Sokoban / 难版FrozenLake数据 | 冻结地图、独立评估划分、各1000条SFT示范；CPU全量示范回放 |
| UCF v2.1 | **实验性且当前验收失败，正式入口关闭**，见下文 |
| ALFWorld | 真实文本环境、专家与动作评分原语；专家数据验收见[专用说明](docs/ALFWORLD.md)；**尚未接通完整RL训练器** |
| 4B单80G全参数RL | 不支持：静态文本状态已达78.34GiB，尚未计激活/KV等 |
| 9B多卡训练 | **分片后端尚未实现**；配置与顺序为待验收计划，不能直接启动 |
| WebShop | 后续工作，本轮不做 |

本发布不会通过修改β、优势、奖励、评估集、精度或优化器来绕过资源限制。4B可在更大显存的单卡上做后续冒烟；9B需要训练侧分片及相应权重同步，不能只把vLLM放到第二张80G卡上。

## 背景与已有发现

项目从Qwen2.5-0.5B上的PaTR/GRPO/Tree-Random复现起步，加入Golden Trace SFT，再研究有限反事实预算下的选点、后果分层和信用分配。UCF在初始四条轨迹的组相对更新之外增加反事实辅助项。目前既有正向结果，也有种子方差、无SFT冷启动与概率支持集方面的局限。

本页数字来自实际归档与只读核验，完整来源、分母、窗口和SHA256见[实验结果与问题](docs/EVIDENCE.md)和[机器可读汇总](data/evidence/summary.json)。2B快照固定截至S05第78步，后续结果不自动覆盖本发布。

- **0.5B**：历史无SFT PaTR主100题success为64.50%；SFT→UCF seed42/43/44在500题上为93.85% / 87.15% / 90.25%，有明显seed波动。无SFT UCF为57.25%，低于同seed Random K=2的62.65%。这些不是本仓库重新跑的结果。
- **原版FrozenLake饱和**：2B F02最终main100 / supplemental500 / map8的success分别93.0% / 94.4% / 98.0%，因此后续采用Sokoban和难版FrozenLake。
- **Sokoban已完成对照**：S03 SFT→GRPO4主100题75.25%、500题77.00%；S04 SFT→Random K=2为76.50%、79.30%。Random生成token为GRPO4的1.787倍，分别提高1.25和2.30个百分点。
- **S05仍进行中**：第75步固定验证success74.75%、pass@4 77.00%；同一步GRPO4为77.50% / 80.00%。没有S05最终测试结果。前75步日志KL均值S05约0.8322、GRPO4约0.1200，需继续观察。
- **信号枯竭与救回**：S05前75步，初始四条混合组平均18.13/64，加入反事实后27.16/64。前78步逐步平均救回率8.93%，奖励翻转率22.82%；“约20%”不能写成救回率。混合组增加也不自动等于最终成绩提高。
- **无SFT格式问题**：F02十步抽样离线回放中，现行v2排除97.34%的执行节点；大量节点实际能解析动作，只是动作前有解释文本。S05有SFT前58步只排除0.0278%，平均分叉2.0016/题，二者不能混推。

## v2.1为什么暂不放行

v2.1复制原始推理token前缀，在本回合第一个围栏之后定义动作节点，探测与强制采样都使用这个前缀，Part B从动作token开始。v2仍保留当前生产定义。两者不承诺在SFT数据上逐位等价。

最新独立验收的两个阻塞是：F02全量第一动作token与最终解析动作不一致率 **12.8053% > 2%**；包含残余概率p[0]=0.08的合成无偏性模拟未通过。功能单元测试通过不代表总体算法验收通过。[版本边界与反例](docs/V21_STATUS.md)保留了结果，正式入口不会静默退回v2。无SFT UCF和ALFWorld UCF均须等待后续2B验收与明确版本变更。

## 安装与CPU验收

要求Python3.12；请求的训练版本为torch2.10.0、vLLM0.19.1、transformers5.5.4。当前生产venv检查时安装的是transformers5.22.2，不能隐去这一差异，也不能据此反推运行中进程的版本。本发布5.5.4只完成CPU验收。驱动、CUDA wheel组合、补丁和具体命令见[安装说明](docs/INSTALL.md)。

```bash
git clone https://github.com/EthanY666999/ucf-agent-rl.git
cd ucf-agent-rl
python3.12 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r requirements-cpu.txt
bash scripts/cpu_smoke.sh
```

ALFWorld真实环境与专家回放另按[ALFWORLD.md](docs/ALFWORLD.md)安装固定CPU依赖并执行验收；无需Docker、视觉模拟器或GPU。模型metadata已附带，修复下载脚本只允许tokenizer/config/索引/LICENSE，不会下载权重。

4B/9B tokenizer核心与2B相同，但模板默认thinking不同。所有新prompt必须显式`enable_thinking=False`。保留采样token id、历史空think块和原old_logprob；模型自行输出`</think>`只记录，不拦截。不能把整段历史重新分词后当成原轨迹。

## 合作者按什么顺序做

[完整任务书](docs/EXPERIMENT_PLAN.md)逐项列出预算、时长依据、磁盘占用和门禁。[显存与磁盘核算](docs/MEMORY_AND_RUNTIME.md)给出actor/reference/Adam/vLLM的分项数量。所有时长估计都是规划参考，4B/9B尚无GPU吞吐实测。

1. **4B**：CPU与GPU冒烟 → Sokoban基座评估 → SFT及导出重评 → 同一起点分别GRPO4、Random K=2、UCF v2 → 难版FrozenLake同一套 → StarPO-S/GiGPO → seed43。
2. **9B**：先完成分片后端适配和GPU验收 → ALFWorld基座评估 → SFT及重评 → 同一起点GRPO4、GiGPO、Random K=2、UCF → Sokoban主线 → seed43。ALFWorld RL集成、UCF v2.1均有独立门禁。

`configs/plan.json`是静态任务清单，**不是会自动抢卡的队列**。RL默认400步，每25步验证。Sokoban/难版FrozenLake的SFT为1000条、1 epoch、batch16，约63步；ALFWorld已完成1000个episode，按history=2转换为15857个逐回合样本，batch16的一轮训练为992步，不能套用63步。seed43复用同规模、同环境的同一个seed42 SFT导出，RL独立开始。StarPO-S/GiGPO是当前框架内实现，不冒称作者完整官方训练框架。

```bash
# 仅CPU检查并打印计划，不启动GPU：
python scripts/run_experiment.py --config configs/4b/sokoban/grpo4_seed42.json

# 以下只供负责人审阅通过后，在合作者自己的合适GPU主机执行：
python scripts/patch_vllm.py --apply
python scripts/run_experiment.py --config configs/4b/sokoban/sft_seed42.json --execute
python scripts/evaluate_sft.py --config configs/4b/sokoban/sft_seed42.json --execute
python scripts/run_experiment.py --config configs/4b/sokoban/grpo4_seed42.json --smoke --execute
python scripts/run_experiment.py --config configs/4b/sokoban/grpo4_seed42.json --execute
```

权重由合作者自行下载到`models/`，固定官方revision；本仓库不含权重。正式入口检查发布/数据哈希、SFT完成与重载验收、同一SFT模型SHA、既有输出和单实验锁。需要改变资源参数时先记录新配置差异，算法参数不动。9B及未验收v2.1配置只会报告阻塞，不会偷换成较小模型或旧方法。

## 目录与交回

审阅入口：[逐文件清单](docs/FILE_LIST.md)、[CPU验收收据](docs/acceptance/CPU_RELEASE.json)、[SHA256清单](RELEASE_MANIFEST.json)。

| 路径 | 内容 |
| --- | --- |
| `code/` | 当前生产来源代码、便携加载器、v2/v2.1、外部基线、环境、vLLM补丁 |
| `data/` | 冻结地图/评估/示范、模型metadata、少量已有结果汇总；全部SHA256 |
| `configs/` | 4B/9B按环境与方法分开的配置、生产算法参数及静态顺序 |
| `scripts/` | 安装/CPU验收/受控启动/重评/轻量证据导出 |
| `tests/` | CPU合同与回归测试；实际GPU行为另行验收 |
| `docs/` | 任务书、结果、容量、安装、ALFWorld、v2.1边界和来源 |
| `results/` | 初始为空，仅`.gitkeep`；留给合作者提交轻量结果 |

使用`python scripts/export_results.py <运行目录> --output <证据包.tar.gz>`生成白名单证据包，交回`results/<model>/<environment>/<run_id>/`。包括配置、源码/数据/模型身份、train_metrics、验证与终测、ucf_logs选点/权重诊断；排除权重、checkpoint和原始rollouts。大包不得直接推送，先分片并列SHA256。结果只交回本仓库，不覆盖发布时的历史汇总。

模型加载失败、哈希不一致、KL爆炸、非有限梯度或分叉数异常时，保留失败证据，停止并向负责人确认。不要自动改参数、删除运行中checkpoint或把失败测试标为通过。ALFWorld官方环境设置有助于对照论文，但模型、提示格式、预算不同，不能直接宣称同条件超越已发表数字。

**发布时追加的控制器发现：** 当前生产v2使用整批校准，并未实现此前讨论的留一题校准；同一条轨迹后面的候选会影响前面节点的ρ。本包保留生产源码用于复核，不把它冒充已修复版本。后续修正版需单独标版本并重新验收；详见[新协议审查与控制器反例](docs/PROTOCOL_REVIEW.md)。本任务书顺序是本次4B/9B发布计划，不控制另一个聊天维护的2B队列。

第三方文件保留各自LICENSE；见[来源与版本边界](docs/SOURCE_PROVENANCE.md)。
