# 显存、磁盘和耗时：按实际实现核算

这是一份容量核算，不是 4B/9B 的 GPU 验收报告。本次只下载官方 tokenizer、config 和 safetensors **索引 JSON**；没有下载模型权重，没有运行 GPU。

## 精确参数数目与口径

`scripts/check_models.py` 在 `torch.device('meta')` 中按 Transformers 5.5.4 的文本模型架构构造形状、去重统计共享参数。没有分配权重存储。结果见 `data/model_metadata/cpu_acceptance.json`，下载 revision 和文件 SHA256 见同目录 `manifest.json`。

| 模型 | 文本参数 P | BF16 原始多模态 checkpoint 文件载荷 | 与 2B 文本参数比 |
|---|---:|---:|---:|
| Qwen3.5-2B | 1,881,825,088 | 4.55 GB | 1.000 |
| Qwen3.5-4B | 4,205,751,296 | 9.32 GB | 2.235 |
| Qwen3.5-9B | 8,953,803,264 | 19.31 GB | 4.758 |

4B 的 embedding / LM head 共享；9B 不共享。不能简单用模型名称乘固定字节数。原始 checkpoint 包含视觉等参数，文本 actor 不包含这些；vLLM 从多模态原始目录引导时需另留余量。官方模型来源：[4B](https://huggingface.co/Qwen/Qwen3.5-4B)、[9B](https://huggingface.co/Qwen/Qwen3.5-9B)。

## RL 常驻显存

当前生产 `train.py` 是 **FP32 actor 参数、FP32 梯度、普通 AdamW 的两个 FP32 状态、BF16 reference、BF16 vLLM 权重**。计算 autocast 为 BF16 不会把 actor 参数或 Adam 状态变成 BF16。未启用 LoRA、量化、CPU offload、FSDP 或 ZeRO。

| 项目 | 每参数字节 | 4B GiB | 9B GiB |
|---|---:|---:|---:|
| actor 参数 | 4 | 15.67 | 33.36 |
| actor 梯度 | 4 | 15.67 | 33.36 |
| AdamW 一阶、二阶状态 | 8 | 31.34 | 66.71 |
| 冻结 reference | 2 | 7.83 | 16.68 |
| vLLM 文本权重 | 2 | 7.83 | 16.68 |
| **文本静态合计** | **20** | **78.34** | **166.78** |

以上还没有计入激活、attention 工作区、vLLM KV / DeltaNet 状态、CUDA context / allocator、模型同步临时 tensor 和可能加载的视觉权重。以整个官方 checkpoint 载荷作为 vLLM 权重余量，其权重约为 4B 8.68 GiB、9B 17.98 GiB，但这不是实测 GPU 常驻量。AdamW 的 CUDA foreach 更新还可能产生约一份参数大小的临时 tensor。`gpu_memory_utilization` 是 vLLM 引擎预算，不是一个能覆盖 actor/reference/optimizer 的全局限额。

这两种config都有8层full attention、4个KV heads、head_dim256。因此BF16普通KV的理论载荷为 `8层 × K/V两份 × 4heads × 256 × 2bytes = 32KiB / resident token`：一条8192-token上下文约0.25GiB，一条32768-token上下文约1GiB；64条均长8192的上下文约16GiB，尚不含分页对齐和prefix caching。另24层linear attention的FP32循环矩阵按 `24 × 32value_heads × 128 × 128 × 4bytes` 约48MiB/状态槽，卷积状态及缓存检查点另计。具体槽数和缓存策略由vLLM决定，不能把48MiB简单当全部引擎开销。发布只允许调并发/微批这些工程参数，不降低环境步数或上下文上限来伪装显存通过。

**单 80G 结论：当前同卡全参数 RL 后端，4B 和 9B 都不能作为可运行配置交付。** 4B 静态下界已吃掉几乎全部显存；9B 仅 actor＋梯度＋Adam 就需 133.42 GiB。

| 布置 | 容量判断 | 本发布支持状态 |
|---|---|---|
| 4B，单 80G | RL 空间不足 | 不启动正式 RL |
| 4B，单 H200 约 141 GiB | 静态容量有余量；长上下文峰值仍需测量 | `high_memory_single_gpu_unverified`，先冒烟 |
| 4B，2×80G，训练与 ref/vLLM 分卡 | 训练侧静态 62.67 GiB，另一侧至少 15.67 GiB；有工程可行性 | **尚未实现和 GPU 验收**，不能仅改环境变量冒充支持 |
| 9B，单 80G 或单 H200 约141 GiB | 当前同卡 RL 不够 | blocked |
| 9B，1 张80G训练＋1 张80G rollout | 训练侧 133.42 GiB，仍装不下 | 不可用 |
| 9B，至少2张训练分片＋1张 ref/vLLM，均80G | 训练静态理想均分约66.71 GiB/卡，服务侧至少33.36 GiB；只是容量规划下界 | **需要新增 FSDP/ZeRO 及权重同步适配，目前不支持** |

3×80G 的理论下界不是实际承诺：分片聚合峰值、通信 buffer 和激活可能要求 4×80G 或更大的单卡。当前实现把 trainer 张量放在单个 CUDA 设备，vLLM 为进程内 `tensor_parallel_size=1`，并直接同步 actor 参数到引擎；单独增加 `tensor_parallel_size` 不会分片 optimizer。不要擅自切换 8-bit optimizer / BF16 主参数 / LoRA 后仍称同一精度协议。

SFT 没有 reference / vLLM，静态 16P：4B 62.67 GiB、9B 133.42 GiB，仍需激活和优化器临时空间。4B 单80G SFT只能经 GPU 冒烟后判断，9B单80G普通全参数SFT不可行。

当前网格环境的`train.py --eval-only`先调用`build_models`再进入评估分支，因此仍加载FP32 actor、BF16 reference和BF16 vLLM，只是不创建Adam状态和梯度；文本静态合计为**8P：4B 31.34 GiB、9B 66.71 GiB**，还需vLLM缓存等。不能把此入口描述为仅加载推理模型。ALFWorld的独立`alfworld_evaluate.py`才是vLLM-only评估脚本，其权重下界为2P加实际缓存开销。9B主计划仍有后端门禁，独立评估不表示RL分片已实现。

## 磁盘

生产 final_model 以 FP32 actor 导出；checkpoint 保存 FP32 actor 与两个 FP32 Adam 状态（12P）。checkpoint 原子替换会先写完整临时文件再替换旧文件，因此磁盘需同时容纳两份 checkpoint。

| 每个实验的文件 / 峰值 | 4B | 9B |
|---|---:|---:|
| 最终文本模型，4P | 16.82 GB / 15.67 GiB | 35.82 GB / 33.36 GiB |
| 一个训练 checkpoint，12P | 50.47 GB / 47.00 GiB | 107.45 GB / 100.07 GiB |
| 替换时旧＋新 checkpoint，24P | 94.01 GiB | 200.13 GiB |
| 两份 checkpoint＋一个 final_model，28P | 109.67 GiB | 233.49 GiB |
| **建议每个活动 RL run 空闲空间** | **至少130 GiB，另加轨迹/日志** | **至少275 GiB，另加轨迹/日志** |

原始模型及同环境共享的 SFT 起点还要各保留一份。多个已完成实验的 final_model、验证导出临时模型若同时存在，应逐项相加。轨迹体积受推理长度和环境步数影响，不能套用一个固定“小日志”数字；5～10步冒烟后根据实际 bytes/step 估算400步并预留50%余量。

2B 的 checkpoint 核算为22.58 GB/21.03 GiB，与历史“约22G”量级相符；不能把22G沿用到4B/9B。最终评估核验后可按证据保留协议处理 checkpoint；本发布不自动删除任何数据。

## 耗时估算与启动预算

已经测得的 2B A10080G 基准：S02 SFT（1000条、1 epoch、63步）729秒；S03 GRPO4 400步20,145秒（5.60小时）；S04 Random K=2 400步40,802秒（11.33小时）；S05 前78步11,685秒，按该阶段均值外推400步16.65小时；F02无SFT GRPO4 400步85,366秒（23.71小时）。S05外推不是最终用时。

只为给预算一个可复算起点，下面使用 `2B耗时 × 文本参数比`，假设相同 token 量、相同每参数吞吐和验证开销。实际部署会换GPU/分片、回复长度也会变化，**这些不是4B/9B测量或可靠完成时间保证**。

| 对应2B Sokoban工作负载 | 4B线性参考 | 9B线性参考 |
|---|---:|---:|
| SFT 1000条、1 epoch | 27分钟 | 58分钟 |
| SFT→GRPO4 400步 | 12.5小时 | 26.6小时 |
| SFT→Random K=2 400步 | 25.3小时 | 53.9小时 |
| SFT→UCF 400步（基于S05进行中速率） | 37.2小时 | 79.2小时 |

难版 FrozenLake、StarPO-S、GiGPO 和 ALFWorld 尚无本发布4B/9B同条件基准，不填一个伪精确小时数。特别是 ALFWorld 最多50个环境动作，不能照搬小地图 FrozenLake 的每步时间。每个新模型/环境先完成环境验收、基座评估与5～10个训练步的冒烟（正式训练另建目录）；随后使用：

`预算 = 400 × 稳态训练步秒数 + 16 × 固定验证秒数 + 最终seen/unseen等完整评估秒数 + 保存/加载时间`。

每25步验证一次；初次编译单独记录；同时报告 generated_tokens、rollout_seconds、update_seconds、峰值allocated/reserved显存、GPU实际空闲量和磁盘bytes/step。算清预算之前，blocked 的9B配置不启动正式训练。
