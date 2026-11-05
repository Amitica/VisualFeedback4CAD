# 工程图 → CAD：Initial、Vanilla 与论文基线

此目录提供可运行入口、固定来源和明确的适配范围。默认共同 API 配置为 **`ucloud/gpt-5.6-sol/low`**，可修改 [config.json](config.json)。训练模型与其推理协议分开标识，不把模型自评或运行完成率当作几何准确率。

| 名称 | 实际执行流程 | 复现身份 |
| --- | --- | --- |
| Initial | 原 Main 读取工程图并完成初始 CAD 生成，随后停止 | 与主方法相同的生成入口；不是“只允许一次 API 调用” |
| Vanilla | 复制 Initial 和原 Main 会话 → 原样投递历史 Direct Reflect → Main 直接修改、执行、查看普通渲染 | 无 View Search、尺寸标注 Agent 或对齐比较工具 |
| CADCodeVerify | 工程图生成 2–5 个正向问题 → 当前 CAD 四方位渲染回答 Yes/No/Unclear → 仅对未满足项产生反馈 → 修订，至多两轮 | 原仓库未提供推理器；本目录按论文协议重实现并迁移输入/执行语言 |
| Seek-CAD SVF | 同一 Initial → 无几何改动的步骤导出 → 旧几何透明、本步实体高亮 → 独立 critic 比较公开建模计划与图纸 → 一轮修订 | 仅适配 step-wise visual feedback，不冒充原 RAG/SSR/R1/Gemini 完整系统 |
| IterCAD | 官方 `process_itercad_draw` + CadQuery 执行器 + TOP/FRONT/RIGHT/ISO 四宫格 + 原多轮协议 | 官方公开训练权重迁移，或明确标记的 GPT 同骨干推理协议版本 |

**IterCAD 原反馈中已有整体尺寸和圆弧半径标注。** 适配保留这一设计，没有将它替换为普通等轴测图，也不将它当作“无尺寸”基线。

## 一键运行全部五种方法

需要 Linux、Node ≥22.19、npm、Python 3 和可用的 Docker daemon。先按[认证说明](../portable-full-method/credentials.md)配置本机 Pi provider 与密钥。一键运行中的 IterCAD 使用 Responses；新环境应采用本目录的 [Responses 配置模板](models.example.json)，将占位 `baseUrl` 改为你的实际 API 根地址，密钥仍由 `UCLOUD_API_KEY` 提供。已有可用的 `ucloud/gpt-5.6-sol` Responses 配置可直接沿用。

将官方公开输入下载到 `data/inputs/<id>/`。若尚未下载，可在独立 Python 环境安装 `huggingface_hub`，然后执行：

```bash
hf download HuggingAI4Engineering/cadgenbench-data --repo-type dataset \
  --revision f76f965585817c621d6ea0d150d745adf670e66e \
  --local-dir data/inputs
```

这里只下载公开工程图与任务描述，不需要三维 GT。不要将 `data/inputs` 替换为 GT 目录。

从仓库根目录执行以下命令，首次加 `--setup` 自动安装 Node 依赖、构建共同 CAD 镜像／评测镜像及独立 IterCAD 镜像：

```bash
bash integrations/pi/baselines/run-all.sh --setup \
  --cases 140 --parallel 1 \
  --output-dir results/baselines/all-methods-case140-v1
```

后续运行不需要重建，去掉 `--setup` 即可。脚本先实际生成一次 Initial；Vanilla、CADCodeVerify 与 Seek-CAD SVF 随后并行读取**同一份初始代码、STEP 和 Main 会话**。IterCAD 同时运行官方独立的 CadQuery 生成与迭代流程，保留 5 轮／2 次尝试，不强行套用 build123d Initial。其 Responses 客户端自动沿用 Pi 的同一 API 地址、模型与密钥，密钥只通过环境变量传入容器，不写入命令或运行合约。

运行全部生成案例，只需将 `--cases 140` 替换为 `--all-generation`；`--parallel all` 表示共同方法中每个案例全部并行，IterCAD 官方进程池最多 4 个。方法间也会并行，因此需预留足够 CPU 与内存。新输出目录不能已存在：

```bash
bash integrations/pi/baselines/run-all.sh --all-generation --parallel all \
  --output-dir results/baselines/all-methods-generation-v1
```

本机 Docker 网络需增加 `--network cadgenbench-retry12-20261105`，正常新服务器可保留默认 `bridge`。当 API 地址或 HTTP(S) 代理使用主机的 `localhost`／`127.x.x.x` 时，IterCAD 容器自动采用 Linux `host` 网络，以保留现有代理及 TLS 校验；共同 build123d 方法仍使用指定网络。可用 `--itercad-network <名称>` 显式覆盖，实际路由会保存到 `run-plan.json` 的 `itercad_transport`。自定义公共设置通过 `--config your-config.json` 指定；默认仍为 `ucloud/gpt-5.6-sol/low`。

如要从已有、可核验的原始 Main 继续，而不再次生成 Initial：

```bash
bash integrations/pi/baselines/run-all.sh \
  --methods vanilla,cadcodeverify,seek-cad \
  --initial-root results/baselines/initial-full-v1/generation \
  --cases 131 --parallel 1 \
  --output-dir results/baselines/shared-initial-repairs-v1
```

统一入口保存 `run-plan.json`、`resolved-config.json`、各方法日志与 `summary.json`。任一方法未完成会以非零状态退出，已失败的方法不会被 Initial 替代。单个 Initial 失败时，其余有效 Initial 仍可继续反思；失败案例保留在各方法评测分母中。`timeout_minutes` 约束共同 build123d 方法的 fresh Initial 生成命令及反思控制器；IterCAD 保留独立的 600 秒 API 请求时限和官方轮次预算。加 `--dry-run` 可先检查计划，不调用模型、不创建结果目录；首次使用前需先构建 Node 入口（`npm --prefix integrations/pi ci`、`npm --prefix integrations/pi run build`）。

## 1. 共同环境与认证

从 CADGenBench 根目录运行：

```bash
bash integrations/pi/baselines/setup.sh
```

需要 Node ≥22.19、npm 和 Docker。此脚本为 Initial／Vanilla／CADCodeVerify／Seek-CAD SVF 构建共同 build123d 镜像，并构建包含 OpenMP 依赖的离线评测镜像。IterCAD 使用单独的 CadQuery 镜像，安装步骤见其 [README](itercad/README.md)。

模型认证沿用 Pi 的 `PI_CODING_AGENT_DIR`，UCloud 配置示例见 [认证说明](../portable-full-method/credentials.md)。真实网关地址与密钥在运行机器本地配置；不写入运行合约。

本机默认 Docker bridge 曾缺少设备，因此当前实验采用独立网络 `cadgenbench-retry12-20261105`。在本机运行下述命令时增加 `--network cadgenbench-retry12-20261105`；健康新服务器可用默认 bridge。也可以在自己的 `--config` 文件中设置 `docker_network`。共同镜像已在本机构建并验证；若在本机重新构建，可使用 `CADGENBENCH_BUILD_NETWORK=host bash integrations/pi/baselines/setup.sh`，避免受损的默认构建网络。

## 2. Initial：先生成一次，再共享给各反思基线

```bash
node integrations/pi/baselines/run.mjs --method initial \
  --all-generation --skip-unavailable-inputs --parallel all \
  --output-dir results/baselines/initial-full-v1
```

`--skip-unavailable-inputs` 显式跳过缺失的输入，并将原因写入合约；没有该参数时，缺图会报错。当前工作区的 **49 个生成案例均可用**（101–143、145–150）。101、105 的原图已于 2026-11-05 05:07 UTC 从官方数据源恢复，SHA 与原下载元数据一致，见[恢复核验](../../../artifacts/experiment/gpt56-sol-low-full-method-complete-all-20261105-v1/input-recovery/input-recovery-verification.json)。此前的 47 例清单是恢复前的历史记录。

结果目录：

```text
initial-full-v1/
├── generation/<id>/       # 完整原Main会话、代码、STEP、执行记录
├── evaluation/<id>/       # 可交给共同评测器的初始代码、STEP及结果
├── initial_bindings.json  # 原始程序/STEP哈希与来源
└── experiment_contract.json
```

如已有主方法生成目录，可以只导出 Initial，**不再调用模型**：

```bash
node integrations/pi/baselines/run.mjs --method initial \
  --initial-root results/pi/gpt56-sol-low-full-method-retry12-20261105-v1 \
  --output-dir results/baselines/initial-existing-v1
```

`--initial-root` 支持本项目完整方法根目录或直接包含 `<id>/` 的 Main 工作目录；未指定案例时自动选取该目录中可核验的原始 Main 案例。只有最终 STEP、没有程序和原 Main 会话的结果压缩包不能用于恢复 Vanilla。

当前已有12例 Initial 的独立导出：`final_results/initial-gpt56-sol-low-retry12-20261105-v1/evaluation`。它只对应这12例新生成模型，不能冒充旧47例完整批次的 Initial。

## 3. Vanilla：历史 Direct Reflect 原样继续

```bash
node integrations/pi/baselines/run.mjs --method vanilla \
  --initial-root results/baselines/initial-full-v1/generation \
  --parallel all --output-dir results/baselines/vanilla-full-v1
```

这是原 Main 会话的继续，不是让另一个 critic 先生成报告。提示词来自历史实验 `gpt56-four-arm-5case-20260821-v1/02-self-reflect`，完整文字与来源哈希见 [vanilla/](vanilla/)。既有的 `mainSelfReflectPrompt` 原样复用；没有添加视图搜索、尺寸标注或结构化反馈。

为复现历史配置，其“源码与 STEP 均需改变”的完成条件也原样保留。这与允许判断无需修改的另一种 Vanilla 定义不同；不要混合后当作同一实验。提示词会复制到新运行的 `direct-reflect-prompt.txt`。

## 4. CADCodeVerify：真正分开的问答反馈

```bash
node integrations/pi/baselines/run.mjs --method cadcodeverify \
  --initial-root results/baselines/initial-full-v1/generation \
  --parallel all --output-dir results/baselines/cadcodeverify-full-v1
```

每个阶段是独立模型调用，保存问题、固定问题的回答、未满足项反馈、修订代码及执行结果；全 Yes 提前停止，两轮后停止增加修订。四个固定方向不针对图纸搜索或对齐。原论文未公开精确相机仰角，本适配的方向/上方向在 `adapter.mjs` 明确固定。

原方法使用文本输入、CadQuery；这里使用图纸输入和共同 build123d 执行环境。应报告为 **CADCodeVerify (drawing-adapted)**，不是原 CADPrompt 分数复现。来源与差异见 [方法说明](cadcodeverify/README.md)。

## 5. Seek-CAD：步骤视觉反馈的受控适配

```bash
node integrations/pi/baselines/run.mjs --method seek-cad \
  --initial-root results/baselines/initial-full-v1/generation \
  --parallel all --output-dir results/baselines/seek-cad-svf-full-v1
```

控制器先将同一 Initial 程序添加步骤快照；导出的最终几何必须通过世界坐标下 BRep 双向差集与边界检查，才允许 critic 使用该步骤证据。不能通过只比较体积宣称几何相同，也不能在步骤导出失败时偷偷退化为最终渲染图反思。步骤高亮图、公开计划、正/负反馈和修订记录全部保留。

原完整 Seek-CAD 的 SSR、RAG、DeepSeek-R1/Gemini 组合没有在这里复现；应使用 **Seek-CAD SVF (drawing-adapted, shared Initial)** 名称。原仓库只有数据与转换工具，没有完整推理器。完整来源边界见 [方法说明](seek-cad/README.md)。

## 6. IterCAD：官方推理器的工程图适配

详见 [IterCAD 运行说明](itercad/README.md)。此处选定 EMNLP2026／arXiv `2606.13368`，不是另一篇同名程序修复工作。

```bash
bash integrations/pi/baselines/itercad/setup.sh
```

当前 **49 例**转换数据为 `artifacts/baselines/itercad-cadgenbench-inputs-20261105-v2/samples.jsonl`；新机器上的 `native.py prepare` 命令见 [IterCAD 运行说明](itercad/README.md)。旧 `itercad-cadgenbench-inputs-20261105-v1` 保留恢复前的 47 例及其排除记录，未被覆盖。

官方权重是 `KnowledgeXLab/IterCAD-RL`，需在模型服务器部署；也可连接自己的 API 网关，明确指定模型与传输协议。训练权重迁移与 GPT 替代模型是两种实验，不能把后者当成复现了作者 SFT/RL 的收益。

## 7. 同一评测入口

每个共同基线的最终模型位于 `<output>/evaluation/<id>/output.step`，程序为同目录 `solution.py`。IterCAD 位于 `<output>/cases/<id>/`。只有尺寸、渲染自评分、编译成功，均不代表 IoU／Surface F1 已计算。

将 GT 放在独立目录，布局 `<GT-dir>/<id>/ground_truth.step`，再执行：

```bash
bash integrations/pi/baselines/evaluate.sh \
  results/baselines/initial-full-v1 /absolute/path/to/GT 8
bash integrations/pi/baselines/evaluate.sh \
  results/baselines/vanilla-full-v1 /absolute/path/to/GT 8
```

该入口对所有方法使用 CADGenBench 原评测器，并保留真实尺度；GT 只以只读方式挂载到离线评测容器，没有进入生成、问题、反馈或反思。无 GT 时只导出候选模型及有效性元数据，不生成虚构精度。

比较反思策略时，检查 `initial_bindings.json` 的源程序与 STEP 哈希一致。失败状态保留在各运行的 `summary.json`／`case_status.json` 中。共同入口会为每个请求的案例保留评测目录，失败项没有替代 STEP，离线评测时按 missing 处理；不要删除这些目录后仅对成功子集求均值。先等运行的 `summary.json` 写出，再进行最终评测。

## 8. 检查与复现范围

所有共同入口均支持 `--dry-run`；使用新输出目录，不覆盖历史结果。`--cases 146` 可限定一个案例，`--parallel all` 将选定案例全部并行。`max_compile_attempts=3` 是本适配的明确工程预算，不是宣称原论文给出的编译次数。

CADCodeVerify 与 Seek-CAD 的 build123d 修订会收到统一的 API 用法说明。若修订执行失败，控制器将实际执行错误交回模型；网格拓扑错误另附出错边的三维坐标及修改前可执行代码，以便定位新引入的布尔问题。这些诊断只读取 STEP，不修改或修复几何，不增加编译或语义修订预算。该执行辅助属于本适配环境，不宣称来自原论文；完整提示词与实际诊断会保存到 `protocol/`，实现哈希记录在运行合约中。

方法代码与数据输入适配的可运行验证，不等于已复现整套论文精度。模型服务、输入缺失、编译失败、步骤证据无法保持原几何等错误均显式保存，不用其他方法输出替代。

2026-11-05 已在本机通过一次完整公开入口验收：**case 140，`ucloud/gpt-5.6-sol/low`，五种方法全部完成，命令退出码 0**。五份最终 STEP 独立通过有效、水密检查，三种反思方法的 Initial 绑定相同且原始 Initial 未被修改。[仓库内核验摘要](verification/case140-20261105.json)保存运行代码、最终程序和 STEP 的哈希，以及模型调用与协议核验信息。原始结果保存在验收机器的 `results/baselines/one-click-case140-20261105-v4`，不随源码仓库分发；重新运行后会生成自己的完整记录。历史失败运行 v1–v3 保留原状态，没有用旧结果替代本轮候选。

验收中修复了输出父目录、超时和部分 Initial 失败处理、主机回环代理路由，并补充编译诊断。IterCAD 在无 GT 距离分数时优先选择正常结束的尝试，同类仍使用官方比较器；该适配及完整轨迹均写入合约。修改 native adapter 后需重新构建其镜像（`bash integrations/pi/baselines/itercad/setup.sh`），本机可加 `CADGENBENCH_BUILD_NETWORK=host`。基线全集尚未运行；本次验证运行与产物有效性，未计算真实 GT 的几何准确率。
