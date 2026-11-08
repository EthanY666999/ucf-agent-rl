# UCF Agent RL — 4B 发布审阅版

本轮以 **Qwen3.5-4B** 为主，修改后先交负责人审阅，尚未发给合作者。合作者有2张A100 80G；设备怎么摆、怎么分卡由合作者自行决定。本次不新增设备部署实现、不改设备或显存参数；[现有显存核算](docs/MEMORY_AND_RUNTIME.md)原样保留，不能据此声称现有入口已经支持双卡。

**9B 安排由合作者在 4B 之后自行决定。**

历史复现记录见 [decision-point](https://github.com/EthanY666999/decision-point)。

## 本轮发布状态

| 内容 | 状态 |
| --- | --- |
| 4B GRPO4、Random K=2、框架内StarPO-S/GiGPO、SFT及网格环境 | 保留现有实现和设备参数；CPU验收不等于GPU部署验收 |
| transformers | **固定5.22.2**，使用本版本重新进行CPU验收；torch2.10.0、vLLM0.19.1不变 |
| 4B SFT-UCF | 计划使用 **UCF-v2.2**；等待2B正式验收与对应源码同步，目前不能启动 |
| 4B无SFT UCF | 计划使用 **UCF-v2.2-ff**及首个完整动作块的token边界截断；等待对应验收与源码同步 |
| 难版FrozenLake SFT起点／4B-07 | **待定**；2B当前从原版F03起步，待H01结果后统一决定 |
| 旧UCF v2／v2.1 | 仅保留来源与历史审计；发布启动入口拒绝运行，绝不回退到旧实现 |
| ALFWorld | 现有代码、数据和文档保留，本轮不扩展、不排入4B主线 |
| WebShop | 后续工作，本轮不做 |

审阅入口：[本次差异](docs/RELEASE_REVIEW_2026-11-09.md)、[完整任务书](docs/EXPERIMENT_PLAN.md)、[文件清单](docs/FILE_LIST.md)、[5.22.2验收收据](docs/acceptance/CPU_5_22_2.json)、[SHA256清单](RELEASE_MANIFEST.json)。旧版CPU收据保留为历史记录，不代替本轮验收。

## 背景与已有发现

项目从Qwen2.5-0.5B上的PaTR/GRPO/Tree-Random复现起步，加入Golden Trace SFT，再研究有限反事实预算下的选点、后果分层和信用分配。UCF在初始四条轨迹的组相对更新之外增加反事实辅助项，目前既有正向结果，也有种子方差、冷启动与概率支持集问题。

以下数字来自已有记录，不是本次重跑结果；来源和分母见[实验结果与问题](docs/EVIDENCE.md)。S05成绩快照仍固定截至第78步，不代表它当前的实时进度。

- 0.5B历史无SFT PaTR主100题success为64.50%；SFT→UCF seed42/43/44在500题上为93.85% / 87.15% / 90.25%。无SFT UCF为57.25%，低于同seed Random K=2的62.65%。
- 原版FrozenLake在2B上饱和：F02最终main100 / supplemental500 / map8的success为93.0% / 94.4% / 98.0%，因此采用Sokoban和难版FrozenLake。
- Sokoban S03 SFT→GRPO4主100题75.25%、500题77.00%；S04 SFT→Random K=2为76.50%、79.30%。Random生成token为GRPO4的1.787倍，分别提高1.25和2.30个百分点。
- **2B的S05用的是UCF-v2。** 归档快照第75步验证success74.75%、pass@4 77.00%；同一步GRPO4为77.50% / 80.00%。前75步KL均值S05约0.8322、GRPO4约0.1200。此处没有S05最终测试结果。
- S05前75步，初始四条混合组平均18.13/64，加入反事实后27.16/64。前78步平均救回率8.93%，奖励翻转率22.82%；不能把后者写成救回率。
- F02十步离线抽样中，v2规则排除97.34%的执行节点；S05有SFT的前58步排除0.0278%，平均分叉2.0016/题。格式覆盖正常不等于控制器与采样支持均已验收。

## UCF版本与放行条件

4B的SFT-UCF配置已改为v2.2：留一题求c，留一轨迹的p̂₋ⱼ和基线保持原定义，反事实采样覆盖原动作层的完整补集。无SFT使用v2.2-ff：节点紧接首个开围栏，不跳空白；本回合在首个完整动作块结束处按**实际采样token边界**截断，训练和评估一致，不重新分词。新无SFT基线也必须使用相同协议，F02仅作历史参考。

**这些是待同步的版本要求，不是本仓库已实现／已通过的算法声明。** 必须取得2B对应版本的实际验收收据、冻结源码及配置差异，核对哈希并重跑发布测试后才能放行。当前统一启动器和直接训练入口都会拒绝UCF执行；更改版本字符串或删除配置里的gate不能调用到旧v2。具体要求见[UCF发布门禁](docs/UCF_RELEASE_STATUS.md)。

S05的v2生产源码仍保留，整批校准与仅在动作首token集合中采样的历史局限也保留。v2.1失败审计见[历史记录](docs/V21_STATUS.md)，不能把功能单测通过当成总体无偏性验收通过。

## 难版FrozenLake的统一SFT起点

**4B-07先标“待定”。** 2B当前使用原版FrozenLake的F03起点，待H01结果后统一选择SFT数据与训练协议。4B需要用自己的基座生成相应SFT模型，不能加载2B模型权重。

本仓库已有的难版Golden Trace继续保存用于审计，但**不再绑定为4B的默认SFT输入**。4B-07以及依赖该SFT起点的难版GRPO4、Random、UCF、外部基线与seed43均设置待定门禁；模型路径只是占位，不是已选定的起点。难版基座评估不依赖该SFT决定。

## 安装与CPU验收

Python3.12，固定torch2.10.0、vLLM0.19.1、transformers5.22.2。负责人提供的生产时间线表明2B自2026-11-07 02:32起使用5.22.2；本轮在独立CPU环境重跑验收，不修改2B生产环境。驱动及安装细节见[INSTALL.md](docs/INSTALL.md)。

```bash
git clone https://github.com/EthanY666999/ucf-agent-rl.git
cd ucf-agent-rl
python3.12 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r requirements-cpu.txt
bash scripts/cpu_smoke.sh
```

本仓库没有模型权重。CPU验收覆盖现有tokenizer/config、围栏与动作id、模板、SFT mask、冻结数据回放和版本门禁。所有新prompt显式传`enable_thinking=False`，保留采样token id、历史空think块与原old_logprob；模型自行生成的`</think>`只记录，不拦截。

## 4B实验顺序

**Sokoban主线 → 难版FrozenLake → 外部基线 → seed43。**

1. 冒烟验收；Sokoban基座评估 → SFT及重评 → 从同一个SFT起点分别运行GRPO4、Random K=2、UCF-v2.2。
2. 难版FrozenLake基座评估 → 待H01后确定4B-07的SFT起点 → 同一起点分别GRPO4、Random K=2、UCF-v2.2。
3. StarPO-S、GiGPO外部基线。
4. 按上述方法顺序补seed43；复用同规模、同环境的同一个SFT起点，只改变RL随机种子。

`configs/plan.json`是手动串行计划，主线只列4B；原有延期条目不属于本轮队列。RL预算仍为400步、每25步验证。Sokoban的SFT为1000条、1 epoch、batch16约63步；难版SFT预算待起点决策后确定。所有方法、数据或设备门禁未通过时不启动对应项，也不自动改参数绕过。

```bash
# CPU查看计划，不启动GPU：
python scripts/run_experiment.py --config configs/4b/sokoban/grpo4_seed42.json
python scripts/run_experiment.py --config configs/4b/sokoban/ucf_v2.2_seed42.json
```

正式GPU操作仍由合作者在负责人审阅、设备部署及对应门禁通过后决定。本次没有新增分卡、跨卡复制、优化器备选或显存参数。现有设备检查同样保留；本发布不是“现成双卡启动包”。

## 目录与结果交回

| 路径 | 内容 |
| --- | --- |
| `code/` | 生产来源实现、便携加载器、历史UCF、外部基线、环境及vLLM补丁 |
| `data/` | 冻结地图、评估、示范、模型metadata和结果汇总 |
| `configs/` | 当前4B计划、待验收门禁及保留的延期配置 |
| `scripts/` | CPU验收、受控启动、重评和轻量证据导出 |
| `tests/` | CPU合同与回归测试；GPU行为另行验收 |
| `docs/` | 当前任务书、差异、验收及历史材料 |
| `results/` | 留给合作者提交轻量结果，当前只有`.gitkeep` |

使用`python scripts/export_results.py <运行目录> --output <证据包.tar.gz>`导出配置、源码/数据/模型身份、train_metrics、验证与终测、ucf_logs；不提交权重、checkpoint或raw rollouts。结果只交回本仓库。

加载失败、哈希不一致、KL爆炸、非有限梯度或分叉数异常时保留证据并停止该组、向负责人确认。来源与许可证见[来源说明](docs/SOURCE_PROVENANCE.md)。
