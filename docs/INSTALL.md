# 安装与CPU验收

发布环境为 Linux NVIDIA GPU 训练；本次发布检查在 macOS ARM64 上只使用 CPU / meta tensor，未下载权重，未占用当前训练 GPU。4B/9B GPU训练可用性见 [显存核算](MEMORY_AND_RUNTIME.md)，CPU测试通过不表示单80G能跑。

## 固定版本

使用 Python 3.12、torch **2.10.0**、vLLM **0.19.1**、transformers **5.22.2**，numpy 2.2.6、gymnasium 1.1.1。官方 [vLLM 0.19.1 CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.19.1/requirements/cuda.txt) 固定torch2.10.0。依赖版本与本次指定的生产版本对齐，不升级到latest后仍把结果归到本协议。

**本轮与生产环境统一为transformers5.22.2。** 2026-11-09只读核查生产venv时安装的是torch `2.10.0+cu128`、vLLM `0.19.1`、transformers `5.22.2`。负责人补充的生产时间线：自2026-11-07 02:32起使用该版本，正式实验均在此后。时间线是负责人提供的信息，不伪称本次重新读取了历史安装日志。S05已有硬件收据记录驱动`550.127.05`。

此次只升级独立发布的本地CPU验收环境，不改2B环境；新验收见`docs/acceptance/CPU_5_22_2.json`及`model_cpu_5_22_2.json`。旧5.5.4收据保留为历史记录，不能替代5.22.2结果。ALFWorld独立依赖和资料本轮冻结，不能当成本轮4B环境的安装入口。

GPU安装应使用新venv。torch与vLLM的编译CUDA组合必须共同验收；不能随意替换其中一个wheel。vLLM文档中的CUDA12.9默认说明与torch2.10官方提供的cu126/cu128/cu130索引不完全同口径，**不要写不存在的torch2.10 cu129安装命令**。来源：[vLLM安装说明](https://docs.vllm.ai/en/v0.19.1/getting_started/installation/gpu/)、[PyTorch旧版本官方命令](https://pytorch.org/get-started/previous-versions/)。

## CPU检查（没有模型权重）

```bash
python3.12 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install --upgrade pip
# Linux CPU机器可先通过官方CPU索引装torch，避免下载CUDA包；macOS使用普通pip。
# python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python scripts/check_models.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest tests/test_model_contract.py -q
```

仓库已附2B/4B/9B的官方metadata用于核对同族兼容性，每个模型仅约13MB。`scripts/download_model_metadata.py` 可以修复缺失metadata，严格allowlist五个metadata文件及原始LICENSE，并沿用manifest中的revision；不会下载`safetensors`权重。`model.safetensors.index.json`只是索引，不是权重。

验收实际覆盖：三种模型围栏71093、生产`cert_entropy.token_map`的八个动作token（`Left/Down/Right/Up`各自带/不带前导空格）、eos248046、tokenizer.json哈希一致、显式enable_thinking=False/True时文本对话fixture输出一致；生产SFT逐回合拼接保留历史空think块；mask仅覆盖动作与结束token；旧token前缀不被重新分词；config构造的文本参数形状。报告的`actual_ucf_action_ids`是实际UCF动作集合，小写词token另列`parser_diagnostic_action_ids`，不混入UCF支持集。检查需使用tokenizer的实际eos，不应从嵌套config手抄另一个id。

**模板默认值不同：**固定revision下，2B未传`enable_thinking`时默认关闭，4B/9B默认开启。4B/9B的模板文件彼此相同，但与2B不逐字相同。所有发布的环境prompt必须显式传`enable_thinking=False`以保持现有协议，不能省略该参数。这样做不拦截模型自行生成的推理或`</think>`。

官方模板重渲染完整历史会去掉先前的空think块；生产`encode_trace`与rollout累计token方式已专门避开此问题。CPU报告没有声称官方模板本身保留历史think。

## GPU主机安装（合作者审阅通过后）

建议 Ubuntu 22.04 / 24.04 x86_64，Python3.12开发头文件、C/C++编译工具。预留独立模型盘，不把权重下载到本仓库。驱动建议 **580或更新**。CUDA12.x的525.60.13只是受限minor-compatibility下界，不能据此保证vLLM/Triton/PTX工作；CUDA12.8完整支持对应570.26/更新，12.9 GA对应575.51.03/更新，CUDA13.x要求580/更新。具体以wheel和GPU架构为准。[NVIDIA兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)、[CUDA12.9驱动表](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html)。

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install 'vllm==0.19.1' 'transformers==5.22.2' 'torch==2.10.0'
python -m pip install -r requirements-gpu.txt
python -m pip check
python -c "import torch,vllm,transformers; print(torch.__version__,torch.version.cuda,vllm.__version__,transformers.__version__)"
```

此安装命令让固定vLLM依赖共同解析torch，仍必须在目标机器通过自定义kernel导入、模型加载与1步冒烟。不要在当前2B训练venv执行安装/升级。vLLM `allowed_token_ids` 修补见 `code/patches/`：仅对新venv执行，先核对上游版本/文件哈希，再应用补丁并运行补丁测试，避免给已修版本重复打补丁。

权重由合作者在其服务器准备，固定到`data/model_metadata/manifest.json`记录的revision。模型目录必须包含官方config/tokenizer，SFT导出的文本模型另需保留对应同规模原始bootstrap模型；4B/9B不能再借用2B的bootstrap config。**本次发布流程不执行权重下载。**

## 后端与放行条件

本轮不新增设备部署实现或配置：现有训练入口、模型加载、权重同步、显存检查和资源参数保持原状；不能称已实现双卡。合作者在自己的机器上决定部署方式并完成GPU验收。现有容量文档保留供其参考。发布的UCF门禁与设备无关：v2.2／v2.2-ff均须等2B验收通过且同步源码后放行，旧v2不可运行。难版SFT起点另待H01后的统一决定。

每次正式run固定记录依赖版本、驱动、数据/源码SHA256、模型revision、配置和GPU拓扑。工作目录中的训练锁用于防止同时运行多个实验；一张卡同一时刻只允许一个实验占用。CPU环境验收可以并行。
