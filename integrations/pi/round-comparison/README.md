# Initial、Vanilla 与完整方法的多轮比较

此入口统一运行三组实验，并为每个案例保存可单独评测的结果：

| 实验 | 实际流程 | 保存结果 |
| --- | --- | --- |
| Initial | Main 根据工程图生成并执行初始 CAD，随后停止 | 一份初始结果 |
| Vanilla | 从同一 Initial 的 Main 会话继续，使用历史 Direct Reflect 提示词直接反思 | 一轮结果 |
| Ours | 从同一 Initial 搜索并锁定视图；尺寸 Agent 标注当前 CAD，Main 比较并反思；随后继续更新尺寸和反思 | 默认三轮，逐轮保存 |

默认模型为 **`ucloud / gpt-5.6-sol / low`**。`--rounds N` **只控制 Ours 的轮数**，不增加 Vanilla 的反思轮数。Initial 始终保存，Vanilla 和 Ours 均从它独立分支；Vanilla 的修订不会成为 Ours 的输入。

## 一轮如何定义

先对 Initial 执行一次视图搜索并锁定观察条件；这是准备阶段，不计为反思轮数。Ours 的第 1 轮为：**尺寸 Agent 为当前 CAD 标注尺寸 → Main 根据对比视图反思并修复**。

第 2 轮及后续轮次为：**继承上一轮的 CAD 和 Main 会话 → 尺寸 Agent 重新检查当前 CAD 并更新标注 → Main 继续反思并修复**。视图搜索只运行一次；后续轮次沿用同一投影、剖切参数、二维变换和参考坐标框架。

一轮不是一次模型 API 调用，也不是一次 `execute_build123d`。Main 在一轮内可以多次修改、执行和检查；每次成功执行后，尺寸 Agent 都会检查当前 CAD，避免 Main 使用旧尺寸。每轮结束时，保存的程序、STEP、尺寸状态和最终视图检查必须对应同一份 CAD。

若没有证据支持进一步修改，Ours 可以保留上一轮的几何；下一轮仍会启动新的尺寸检查，不能把上一轮的缓存当作本轮结果。Vanilla 保留历史 Direct Reflect 的完成条件：源代码和 STEP 均需改变，并保持有效、水密。两者的这一差异应在分析结果时明确报告。

## 目标服务器准备

需要 Linux、Node ≥22.19、npm、Git 和可用的 Docker daemon。几何执行、渲染和标注在 Docker 中完成，宿主机不需要安装 CAD 内核或 GPU。首次 `--setup` 会安装 Node 依赖、编译入口并构建运行镜像。

仓库包含 49 个公开生成输入（101–143、145–150），不包含 API 密钥或三维 GT。先创建目标机器自己的模型配置：

```bash
export PI_CODING_AGENT_DIR="$HOME/.pi/visualfeedback4cad"
mkdir -p "$PI_CODING_AGENT_DIR"
cp -n integrations/pi/baselines/models.example.json "$PI_CODING_AGENT_DIR/models.json"
```

将该 `models.json` 的占位 `baseUrl` 改为自己的 API 根地址，确认支持图像输入、Responses 和模型标识 `gpt-5.6-sol`；然后在同一终端设置密钥：

```bash
read -rsp 'UCLOUD_API_KEY: ' UCLOUD_API_KEY
echo
export UCLOUD_API_KEY
```

已有可用配置时，直接将 `PI_CODING_AGENT_DIR` 指向对应目录即可。认证说明见 [credentials.md](../portable-full-method/credentials.md)；此入口可使用上面的 Responses 模板，不要将真实密钥写入仓库配置。

## 运行命令

首次先跑一个案例：

```bash
bash integrations/pi/round-comparison/run.sh --setup \
  --cases 140 --rounds 3 --parallel 1 \
  --output-dir results/comparison/case140-v1
```

之后无需重复构建；运行全部 49 个生成案例：

```bash
bash integrations/pi/round-comparison/run.sh --all-generation \
  --rounds 3 --parallel 4 \
  --output-dir results/comparison/all49-v1
```

`--parallel` 控制案例并发；同一条 Ours 轨迹中的轮次依次执行。修改轮数只需改变 `--rounds`，例如 `--rounds 5`；每个输出目录必须使用新名称。

只选择部分方法：

```bash
bash integrations/pi/round-comparison/run.sh \
  --cases 140 --methods initial,ours --rounds 3 --parallel 1 \
  --output-dir results/comparison/ours-case140-v1
```

`--methods` 支持 `initial,vanilla,ours`，Initial 始终保存。该选项用于选择实验分支，不会将另一种方法的输出替换为当前方法的结果。

复用已有、未经过反思的原始 Main Initial，避免重新生成：

```bash
bash integrations/pi/round-comparison/run.sh \
  --initial-root results/baselines/initial-full-v1/generation \
  --cases 140 --rounds 3 --parallel 1 \
  --output-dir results/comparison/shared-initial-case140-v1
```

原始 Initial 必须包含匹配的程序、STEP、执行记录和可恢复 Main 会话，模型与推理档位也须一致。只有 STEP 的结果包，或已经完成反思的某轮结果，不能被重新命名为 Initial。

入口已编译后，可先查看计划、不调用模型：

```bash
bash integrations/pi/round-comparison/run.sh --all-generation \
  --rounds 3 --parallel 4 \
  --output-dir results/comparison/plan-check --dry-run
```

## 结果目录

以 case 140、Ours 三轮为例：

```text
<output>/
├── cases/140/
│   ├── 01-generate/140/          # 原始 Initial 和 Main 会话
│   ├── 02-view-search/140/       # 只搜索一次，保留锁定视图
│   ├── vanilla/round-01/run/140/ # Vanilla 唯一一轮的完整记录
│   └── ours/
│       ├── round-01/run/140/
│       ├── round-02/run/140/
│       └── round-03/run/140/
└── evaluation/
    ├── initial/140/
    ├── vanilla/round-01/140/
    └── ours/
        ├── round-01/140/
        ├── round-02/140/
        └── round-03/140/
```

`evaluation/` 各案例目录包含对应轮次的 `solution.py`、`output.step` 和来源信息，可分别交给同一评测器。完整运行目录保留会话、执行、对比图和尺寸标注证据；每轮 `after/dimension_annotations_state.json` 记录该轮最终尺寸状态。

顶层 `summary.json` 保存各案例、分支和轮次的状态；`rounds.csv` 提供逐轮程序、STEP、continuation ID 与评测目录索引。每个轮次的 `round.json` 还记录前驱绑定、是否改变几何、尺寸 revision、各视图的尺寸状态及实际检查过的视图。尺寸流程 `completed` 表示该轮检查结束，部分尺寸仍可能是 `unresolved` 或 `unsupported`；这些状态会原样保留，不能解释为所有图纸尺寸都已正确标注。

提供 GT 后，可对每个固定轮次分别调用现有评测器，例如：

```bash
bash integrations/pi/baselines/evaluate.sh results/comparison/all49-v1/evaluation/initial /absolute/path/to/GT 4
bash integrations/pi/baselines/evaluate.sh results/comparison/all49-v1/evaluation/vanilla/round-01 /absolute/path/to/GT 4
bash integrations/pi/baselines/evaluate.sh results/comparison/all49-v1/evaluation/ours/round-01 /absolute/path/to/GT 4
bash integrations/pi/baselines/evaluate.sh results/comparison/all49-v1/evaluation/ours/round-03 /absolute/path/to/GT 4
```

分析多轮结果时，应按固定案例集合分别评测 Initial、Vanilla 第 1 轮、Ours 第 1–N 轮；默认输出是指定轮次的结果，不按未知 GT 选择最好的一轮。若某轮失败，其后依赖轮次会记录为 blocked 或 missing，不能使用上一轮或其他方法的模型冒充该轮成功结果。案例的失败状态应保留在评测分母中。

本入口只整理执行流程和逐轮产物，不自动生成 IoU、Surface F1 或 CAD Score。几何精度评测需要另行提供三维 GT，并保持各轮使用相同指标及真实尺度；GT 不进入生成、视图搜索、尺寸标注或反思。

2026-11-06 已通过编译、离线调度与证据检查、49 例运行计划及已有真实单轮产物的只读验证，见[核验记录](verification/20261106.json)。本轮整理没有启动新的付费多轮批次；历史单轮结果不会被标记为新多轮结果。首次采用此入口应使用 `--setup` 构建包含链式续接能力的新镜像。
