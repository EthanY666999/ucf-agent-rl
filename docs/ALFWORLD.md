# ALFWorld：CPU 环境与专家数据发布

## 状态与边界

本目录提供真实 TextWorld 环境、冻结任务划分、手写专家 Golden Trace、独立回放、完整动作字符串打分原语，以及独立的 vLLM 基座评估入口。**ALFWorld 尚未接入本仓库的四方向 RL trainer；9B GRPO / GiGPO / Random / UCF 正式训练均不能从 `patr_core.py` 直接启动。** 环境与数据 CPU 通过不等于 GPU 训练通过。

UCF v2.1 在 2B 正式验收前为实验性；ALFWorld 还必须补足本环境的完整概率分层/残余质量处理和 RL 后端适配。不得以一张 CPU 原语测试收据代替 2B 或 ALFWorld 的训练验收。

## 冻结环境协议

| 项目 | 设置 |
|---|---|
| 环境 | ALFWorld 0.4.2 + TextWorld 1.7.0，纯文本；无 THOR/视觉资源 |
| 官方划分 | train 3553；valid_seen 140；valid_unseen 134 |
| 最大动作数 | 50 |
| 交互历史 | 最近 2 对观察/动作；保留任务描述和当前观察 |
| 可执行动作 | 当前状态 `admissible_commands` 原样提供 |
| 域随机化 | false |
| 回复格式 | 推理之后第一个完整的 ` ```动作``` `；不会倒找最后一个有效块 |
| 评估 | seen / unseen 分别全量评估，分别报告，不混为一个分数 |
| 专家 | ALFWorld 官方 `HandCodedTWAgent`，不是 LLM，也不是最短路 oracle |

上述 horizon/history/split 设置核对自 [verl-agent 的固定源码版本](https://github.com/langfengQ/verl-agent/tree/20bd331bdbc9026a5668e11362178e10ab7400c8)、[环境配置](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/agent_system/environments/env_package/alfworld/configs/config_tw.yaml)、[训练配置](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/verl/trainer/config/ppo_trainer.yaml)。数据来自 [ALFWorld 官方 0.4.2 release](https://github.com/alfworld/alfworld/releases/tag/0.4.2)，采用官方 solvable / task type / movable / Sliced 筛选规则。

**与发表数字比较的限制：** 原 verl-agent 使用 `<action>` XML 格式，而本任务要求围栏；本任务的 Qwen3.5 也不同于 GiGPO 原论文的 Qwen2.5。脚本提示词是同样信息窗口的本项目实现，并非官方 prompt 逐字复刻。因此可并列参考，不能把分数差异完全归因于 UCF。正式对照仍须在同一 backbone、初始化、prompt、horizon、评估集和算力预算上重跑 GRPO / GiGPO。原 GiGPO 启动脚本还有无效动作惩罚等训练设置；本发布未在 ALFWorld 中默默复制或改写它们。

## 离线数据准备与 CPU 验收

独立虚拟环境安装：

```bash
python3.12 -m venv .venv-alfworld
.venv-alfworld/bin/pip install alfworld==0.4.2 textworld==1.7.0 transformers==5.5.4
.venv-alfworld/bin/python scripts/alfworld_prepare.py --output-dir /path/to/alfworld-cpu
.venv-alfworld/bin/python scripts/alfworld_smoke.py \
  --games-root /path/to/alfworld-cpu/json_2.1.1 \
  --manifest data/alfworld/source_manifest.json \
  --count 2 --report /tmp/alfworld-smoke.json
.venv-alfworld/bin/python -m unittest discover -s tests -p 'test_alfworld*.py'
```

`alfworld_prepare.py` 只校验和展开仓库内冻结压缩包，不下载权重或视觉资源。`text_games.zip` 约 36.5 MB；展开全部游戏约 344 MB，加上约 1.1 MB 必需任务参数。无需完整 ALFRED 轨迹元数据包约 1.4 GB。

现有 CPU 收据：`data/alfworld/cpu_acceptance.json`（三种划分各 2 题的环境、克隆、专家回放）；`data/alfworld/tokenizer_cpu_acceptance.json`（4B/9B 完整动作 token 和 SFT mask）。文件哈希由仓库发布清单覆盖。GPU `prompt_logprobs` 数值一致性还未运行。

## Golden Trace

只从 3553 个官方训练任务中按 seed42 的固定顺序选取；专家在 50 步内完成且 fresh environment 逐步重放完全一致才接纳，目标为 **1000 个不同任务的完整教学轨迹**。从不使用 seen/unseen 任务训练。输出位于 `data/alfworld/golden/`，实际完成数、步数分布、失败数和 SHA256 以 `verification.json` 为准；没有完成收据时就是“进行中”。

每条记录包含冻结 game SHA、seed、观察、当时可执行动作、专家动作、下一观察、奖励和终止状态。回放验证会逐步核对这些值并要求最后成功。专家可能超时；被筛掉的任务计入失败统计，**这不是对所有训练题成功率的无偏抽样**。专家只提供动作，不捏造思维链；SFT 回复为围栏动作。`sft_messages.jsonl.gz` 按 step-independent/history=2 转为训练样本，所以 1000 个 episode 不是 1000 个单轮样本；训练步数必须据实际 turn 数和 batch size 计算。

本次已完成：**1000 条不同训练任务轨迹，全部在新环境中逐步复验成功**；1328 次尝试中 327 次专家未在 50 步内解决，另 1 次成功因达到目标数量不再保留。保留轨迹共 15,857 个动作/监督样本，平均 15.857 步（最少 4、最多 50）；4B 和 9B 分词器均完整编码检查通过，得到相同 ids/mask 的流哈希：7,003,141 个输入 token、148,320 个监督 token，最长样本 1,019 token。该数是编码后预算，不是 GPU 训练耗时。新代码另跨进程抽取 16 条再次回放成功，收据明确 `full_file=false`，不冒充第二次全量验收。

如需重建（CPU 最多 4 worker）：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv-alfworld/bin/python scripts/alfworld_generate.py \
  --games-root /path/to/alfworld-cpu/json_2.1.1 \
  --manifest data/alfworld/source_manifest.json \
  --output-dir /tmp/alfworld-golden --count 1000 --workers 4 --seed 42
```

## UCF 原语与必须通过的后续门禁

`code/alfworld_adapter/actions.py` 已实现以下 CPU 可检查的合同：

1. 节点前缀是本回合**实际采样的 ids**，截止第一个开围栏之后；不重新分词该前缀，不用后文更改围栏位置。
2. 给每个完整动作加上闭围栏后编码，检查候选 token 序列相异且没有一个是另一个的前缀。这样打分的是互不重叠的 token 事件；只打动作裸字符串的概率可能重叠。
3. 在该前缀下用 unrestricted `prompt_logprobs` 给每个 suffix 的每个 token 打分；概率为这些 logprob 之和的指数。`p[0] = 1 - sum(p[action])`。`p[0]` 包括非规范拼写/空白、不同 token 切分等剩余事件，**不等于所有事件在环境中都无效**。
4. 从限制后的动作概率用 Python RNG 抽取，保留整段强制 token 及 unrestricted per-token old_logprob；不能用归一化后的抽样概率充当 PPO old_logprob。当前打分 API 额外生成一个丢弃 token，其成本也要记录。
5. 后果类由 fresh environment + 历史动作重放的副本执行得到；成功为 GOAL，包含 `Nothing happens` 为 STAY，其他观察按完全相同的字符串分类。不能把不同观察仅因语义相似就并为一层。
6. S 按 `1 - expert_remaining / initial_expert_steps` 计算。手写专家可能绕路或超时，S 可能为负；当前接口在专家不能于剩余 horizon 内求解时明确报错，**未擅自将其置零/截断**。这条边界须在 ALFWorld UCF 正式验收时确定，β、ρ_min、F 不能因此悄悄改变。

以上不是完整 UCF estimator 的无偏证明。尤其是剩余质量 OTHER 的支持集、原样本所属事件、动作和后续生成的联合分布、token 级 loss 范围、与 v2.1 的控制器组合仍需接入后验证。不能把 OTHER 直接丢掉再宣称 HT 修正保证无偏。`require_ucf_acceptance()` 拒绝没有正式 2B v2.1 收据的启动，但收据存在也不代表 ALFWorld 后端已接通。

**实际 CPU 开销限制：** `ucf_environment_cpu_acceptance.json` 记录了一次真实测试：初始专家距离 14、剩余距离 1，33 个可执行动作的后果分类约 70 秒（包含剩余距离核查，当时另有 4 个专家生成进程）。这是 reference replay 实现，不是廉价快照。TextWorld 原始引擎含 ctypes pointers，直接 `deepcopy` 实测失败；正式 UCF 大规模训练前必须增加经过状态一致性验收的 snapshot/cache 后端并计量 CPU 时间。不能把这项开销漏算成“只增加一点 token”。

## 基座评估入口

该入口独立于四方向 trainer；只在合作者机器上显式运行，当前发布没有 GPU 评估成绩：

```bash
python scripts/alfworld_evaluate.py --model /models/Qwen3.5-9B \
  --games-root /path/to/alfworld-cpu/json_2.1.1 \
  --manifest data/alfworld/source_manifest.json --split valid_seen \
  --output-dir results/9b-alfworld-base-seen --limit 5 --execute-gpu
```

冒烟通过后移除 `--limit 5`，并单独对 `valid_unseen` 运行。脚本要求本地权重，拒绝隐式下载；记录真实生成 ids/logprobs，长度截断单独计数。环境和 expert 验收不使用 GPU。

WebShop 为后续工作，本次未适配。
