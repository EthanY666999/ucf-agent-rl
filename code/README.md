> 2026-11-03：所有代码已合入 main。3B/4B 只按[精简任务书](../docs/PATR_3B_4B_REPRODUCTION_TASK_2026-11-01.md)执行；内网无需上传。新增 sft.py、prepare_collaborator.py、signal_stats.py 及分片评估支持。train.py 的历史版本哈希与当前版本分别记录于 SOURCE_MANIFEST.json。下文冻结训练器描述仅指原发布。

# 当前唯一训练入口

请先看[完整复现指南](../docs/REPRODUCE.md)。从本目录执行`python scripts/run.py --method grpo --smoke`，通过后执行正式GRPO，再独立运行PaTR/Tree-Random。

- `train.py`：本轮冻结的公共训练器，未经此次打包修改。
- `patr_core.py`：冻结的树采样、打分、优势与环境接口。
- `configs/`、`artifacts/data.json`：固定配置和任务。
- `vendor/rllm_source/`：运行需要的上游环境和agent子集。
- `scripts/prepare_model.py`：固定模型revision下载和逐文件验证。
- `scripts/run.py`：便携启动、并发锁、覆盖保护、恢复入口。
- `scripts/evaluate.py`：已有最终模型的100题评估。
- `scripts/summarize_results.py`：CPU重算已发布结果及配对置信区间。
- `tests/`、`SOURCE_MANIFEST.json`：CPU合同检查及冻结代码身份。

从本目录执行以下命令，无需GPU即可重算统计：

```bash
python scripts/summarize_results.py --dataset main100
python scripts/summarize_results.py --dataset supplemental500
```

不直接执行旧归档脚本；不将DAPO/ARPO遗留分支视为本轮已经验证的方法。
