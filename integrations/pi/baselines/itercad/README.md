# IterCAD：复用官方 Draw 交互流程

使用的作品是 [IterCAD: An Iterative Multimodal Agent for Visually-Grounded CAD Generation and Editing](https://arxiv.org/abs/2606.13368)，[官方源码](https://github.com/KnowledgeXLab/IterCAD)固定为 `55998c4a4537573f67ca241dedc52ee2840652ca`，Apache-2.0。**不是**另一篇作者不同的 `2608.24020` 同名程序修复论文。

这里直接导入官方 `eval/evalution.py` 的 `process_itercad_draw`，保留其系统提示词、CadQuery 执行器、完整对话历史、`<DONE>`、默认 5 轮与 2 次尝试。反馈图也由官方 `code_executor.render_views` 和 `gen_view.py` 产生：TOP／FRONT／RIGHT／ISO 四宫格，其中正投影视图带整体尺寸和半径标注。没有替换成我们的 View Search 或尺寸 SubAgent。

## 复现身份

- 使用作者公开的 [`KnowledgeXLab/IterCAD-RL`](https://huggingface.co/KnowledgeXLab/IterCAD-RL) 权重及该流程，才属于官方训练模型在新数据上的迁移评测。模型权重需要你在有 GPU 的环境中部署；本适配不会自动下载大模型。
- 使用 `gpt-5.6-sol` 等 API 可以运行同一官方推理协议，但应命名为 **IterCAD inference protocol + GPT-5.6 Sol**，不能称为复现了 IterCAD 的 SFT/RL 训练收益。
- 包内 `source.json` 记录原始文件哈希。`source-check` 会拒绝改动过的官方评测文件。

## 构建独立执行环境

从 CADGenBench 根目录执行：

```bash
bash integrations/pi/baselines/itercad/setup.sh
docker run --rm --read-only --tmpfs /tmp:rw,size=1g \
  cadgenbench-itercad-native:55998c4 doctor
```

脚本获取固定版本官方源码，构建独立 CadQuery 2.6.1 镜像，不向现有 build123d 环境安装冲突的 OCP／VTK 包。构建时需要访问 GitHub、PyPI 和系统软件源。若服务器的默认构建网络不可用，可指定 `CADGENBENCH_BUILD_NETWORK`。镜像中 `/opt/runtime-requirements.freeze.txt` 保存实际解析的 Python 依赖版本。

也可以按官方安装文档建立独立 CadQuery Conda 环境，然后直接调用 `native.py`；不要在现有主方法环境里混装两个 CAD 内核依赖。

## 准备工程图数据

```bash
mkdir -p artifacts/baselines results/baselines
docker run --rm --read-only --tmpfs /tmp:rw,size=1g \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/data/inputs:/inputs:ro" \
  -v "$PWD/artifacts/baselines:/out" \
  cadgenbench-itercad-native:55998c4 prepare \
  --data-inputs-dir /inputs --output-dir /out/itercad-cadgenbench-inputs-20261105-v2 \
  --all-generation
```

默认四例也可以用 `--cases 102 103 145 146`。当前工作区的 **49 个生成案例均有完整输入**。101、105 的原图已于 2026-11-05 05:07 UTC 恢复，并与原下载元数据中的 SHA 完全匹配，见[恢复核验](../../../../artifacts/experiment/gpt56-sol-low-full-method-complete-all-20261105-v1/input-recovery/input-recovery-verification.json)。新准备目录 `itercad-cadgenbench-inputs-20261105-v2` 包含全部 49 例且无排除项；旧 `v1` 的 47 例及缺图记录仍作为历史数据保留。若使用其他不完整输入，可显式加 `--skip-unavailable-inputs`，排除原因会写入 `input_manifest.json`。

多页图纸拼为单张参考画布，保留各页原始像素尺寸，记录位置，不做缩放；任务文字附加到官方初始请求。JSONL 的 `gt_view_path` 是上游对“参考工程图”字段的命名，**不代表喂入三维 GT**。

## 运行公开训练模型

先在模型服务器按你的 GPU 资源启动 OpenAI-compatible 服务，例如使用官方推荐版本的 vLLM：

```bash
vllm serve KnowledgeXLab/IterCAD-RL \
  --served-model-name KnowledgeXLab/IterCAD-RL
```

在执行 CAD 的服务器上：

```bash
export GENERATOR_API=http://127.0.0.1:8000/v1
export GEN_MODEL=KnowledgeXLab/IterCAD-RL
export OPENAI_API_KEY=EMPTY

docker run --rm --read-only --network host --tmpfs /tmp:rw,size=2g \
  --user "$(id -u):$(id -g)" --cpus 4 --memory 12g --pids-limit 512 \
  -e GENERATOR_API -e GEN_MODEL -e OPENAI_API_KEY \
  -v "$PWD/artifacts/baselines:/data:ro" \
  -v "$PWD/results/baselines:/out" \
  cadgenbench-itercad-native:55998c4 run \
  --data-jsonl /data/itercad-cadgenbench-inputs-20261105-v2/samples.jsonl \
  --output-dir /out/itercad-trained-v1 --workers 4
```

`host` 网络用于访问宿主机模型端口；远程模型服务可以使用自己的 Docker 网络。服务端模型名只是一项声明，适配器不会仅凭名称就声称已验证权重字节。

## 同骨干 API 版本

将上述三个环境变量设为你的实际网关、模型名与 API 密钥，在 `run` 命令末尾加：

```bash
--api-protocol responses --reasoning-effort low --request-timeout-seconds 600
```

上面对应本项目 GPT-5.6 Sol 网关使用的 Responses 接口；模型名由 `GEN_MODEL` 提供，适配器不会猜测或替换。传输桥按 [OpenAI 官方接口文档](https://developers.openai.com/api/docs/guides/migrate-to-responses)转换相同的文本、图像和完整可见会话历史，使用 `reasoning.effort=low`，保留原始 IterCAD 的提示词、执行器和反馈循环。每次请求的 token 用量写入 `native-work/attempt*/api_usage.json`，缺失的用量字段保留为空。未完成的响应不作为可执行结果。

默认仍为 `--api-protocol chat`，适用于原生 checkpoint 的 vLLM 服务；需要闭源 ChatCompletions 网关时也可显式选择 `chat`。`--reasoning-effort` 仅在用户提供时生效，官方 checkpoint 默认不会强加 low。Responses 模式默认不传 temperature，以兼容推理模型；如果网关允许，可显式设置 `--api-temperature`，此采样差异会记入实验合约。原始 ChatCompletions 保留官方 0.7。`--request-timeout-seconds` 作用于 Responses 桥；原始 chat 保留官方 120 秒。

网关若对输出上限有不同要求，可显式设 `--max-output-tokens`；默认沿用官方 40960。密钥只从 `OPENAI_API_KEY` 环境变量读取，不写入实验合约或命令参数。使用 GPT 版本仅复用官方推理协议，不能声称复现了 IterCAD 的训练权重或论文结果。

## 产物与评测

```text
<output>/
├── experiment_contract.json
├── summary.json
└── cases/<id>/
    ├── solution.py
    ├── output.step
    ├── result.json
    ├── attempts.json
    └── native-work/attempt*/  # 每轮原始代码、STEP与官方反馈图
```

STEP 是从官方循环刚执行的同一个 CadQuery 对象中导出，和实际代码一起保存哈希；不会为了导出重新运行可能带随机性的代码。两次尝试都保留。最后一轮失败、但之前存在有效 CAD 时，只保留先前有效产物并标 `incomplete`，不当作完整结束。

适配阶段不提供 GT 几何。无 GT 时，优先选择**已有成功代码执行，且末轮为执行成功或 `<DONE>`** 的完整尝试；同类尝试仍使用原始 `compare_results`，平局保留较早尝试。上游 CD 提取逻辑通过末轮状态区分成功轨迹，但缺少 GT/CD 后，这一区分会退化为平局。因此适配器只补充这一完成状态优先级，不增加调用、不使用几何真值或自评分，也不放宽导出检查。两次尝试均未完成时，仍保留 `incomplete`／失败结果；历史运行不会被重标。策略及每次尝试的完成状态分别写入运行合约与 `attempts.json`。

最终可用共同 CADGenBench 评测器评测 `cases/<id>/output.step`；不要把官方独立归一化后的 CD 与这里保留实际尺度的 IoU／Surface F1 混为同一协议。

当前已通过源码哈希校验和 22 项本地测试，覆盖无 GT 时完整尝试优先、同类保持原比较器、全部未完成时不升级结果等情况。此前独立 Docker 验收实际完成了 CadQuery 几何执行、官方尺寸四视图渲染与 STEP 导出；`doctor` 用固定本地回复测试五轮上限、两次尝试的循环及代码／STEP绑定，不调用模型。首次环境验收时转换了 47 例，101、105 因缺图被排除；这些历史证据保存在 `artifacts/baselines/itercad-native-verification-20261105-v1/`。恢复原图后，已另行准备上述 `v2` 的全部 49 例输入，未改动旧 `v1`。未下载作者训练权重；GPT 替代的实际运行记录与其权重复现分开报告。
