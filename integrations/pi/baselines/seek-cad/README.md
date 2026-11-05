# Seek-CAD SVF：工程图适配基线

本实现保留 Seek-CAD 的**分步视觉反馈（SVF）**：将每一步的实际特征突出显示、将先前几何透明化，再结合完整 CAD 图和公开建模计划，由独立 VLM 给出正/负反馈；只有负反馈触发一轮代码修正。

它是 **`seek-cad-svf-drawing-adapted`**，用于在相同 Initial 上比较反馈策略。不能把它写成 Seek-CAD 官方完整系统的原样复现。

## 官方来源及差异

- [论文（ICLR 2026，arXiv v3）](https://arxiv.org/html/2505.17702v3)，重点见 3.1、3.2、5.1 和 5.2。
- [官方仓库](https://github.com/Sunny-Hack/Seek-CAD)，核查提交 `230947d70960cb58bea76fe75d46d3796c28ca34`，MIT 许可。
- 本地源码：`external_baselines/seek-cad`。该提交公开了数据集、SSR 转换与可视化工具，没有完整推理/反馈驱动程序。获取时使用稀疏检出，没有下载额外 Google Drive 数据。

原系统使用 DeepSeek-R1-32B-Q4、BGE-M3/Dify 混合检索、SSR/CapType 表达以及 Gemini-2.0 反馈。图纸适配采用共享 build123d Initial 和统一配置的模型，**没有将这些不同条件藏在同一个“原方法复现”标签下**。原文实验设置选择一轮 refinement，本基线也默认一轮。

原论文比较其他反馈策略时也固定生成前端，仅替换 refinement。这里采用相同的控制思路，但输入换成工程图，且模型的私有推理不可获取，因此使用可公开的建模步骤描述。原 RAG 前端、SSR、原生成器/评审器、作者精确的检索子集和提示词均不宣称已复现。完整差异见 [source_contract.json](source_contract.json)。

## 运行机制

1. 从已正式执行的 Initial 保存 `solution.py` 与 `output.step` 的 SHA256。
2. 在独立工作目录中，对代码做仅增加快照和导出的 instrumentation，生成 `semantic_steps/manifest.json`。步骤来自执行的实际几何，不能把最终模型重复导出伪装成分步过程。
3. 将 instrumentation 生成的最终实体与 Initial 在原始世界坐标中进行**双向 BRep 差集体积和包围盒检查**。不做配准、缩放或 ICP。几何不一致时拒绝反馈流程，不能将这次 instrumentation 当作已完成修复。
4. 用固定的透视相机渲染每一步：先前实体透明灰色，本步添加/细化实体蓝色、切除实体或工具体橙色；最后单独显示完整 CAD。没有视图搜索，也没有自动尺寸标注。
5. 独立 critic 同时查看原图纸、公开建模计划、实际分步图和最终图，输出明确的正反馈或带步骤编号的差异。负反馈连同原始代码和工程图交给代码生成阶段，进行一轮修订；此阶段是独立调用，不继续原 Main 会话。

Instrumentation 由模型产生时，最终几何会独立校验，但不能据此保证每条语义归因都正确；这些分步归因仍是 critic 要检查的证据，而不是 GT。

## 适配接口

`adapter.mjs` 提供纯接口，供统一实验 runner 调用：

| 接口 | 用途 |
| --- | --- |
| `METHOD` | 方法身份、来源与一轮修正预算 |
| `buildInstrumentationPrompt(ctx)` / `parseInstrumentation(text)` | 产生公开计划和真实步骤导出程序 |
| `validateStepManifest({...})` | 校验 Initial 绑定、步骤顺序和文件路径；不替代几何检查 |
| `buildCritiquePrompt(ctx)` / `parseCritique(text)` | 独立的正/负分步反馈 |
| `buildRevisionPrompt(ctx)` | 仅对明确负反馈执行代码修正 |
| `assertGeometryEquivalent(report)` | 拒绝未通过实体一致性验证的 instrumentation |

Instrumentation `ctx` 包含 `fixtureId`、`source`、`sourceSha256`、`initialStepSha256`，可选 `maxSteps`。Critic `ctx` 包含 `steps`、`geometryEquivalent: true` 和 `drawingDescription`；调用模型时还需附上原图、按顺序排列的 `step_images` 和 `final_image`。模型不能用一句“看起来一样”替代几何验证。

## 步骤文件格式

```json
{
  "version": 1,
  "source_sha256": "<Initial source SHA256>",
  "initial_step_sha256": "<Initial STEP SHA256>",
  "modeling_plan": [
    {"step_id": "S001", "description": "Create the base plate"},
    {"step_id": "S002", "description": "Cut the mounting hole"}
  ],
  "steps": [
    {"step_id": "S001", "description": "Create the base plate", "operation": "add", "cumulative_step_path": "semantic_steps/S001.step", "focus_step_path": "semantic_steps/S001-focus.step"},
    {"step_id": "S002", "description": "Cut the mounting hole", "operation": "remove", "cumulative_step_path": "semantic_steps/S002.step", "focus_step_path": "semantic_steps/S002-focus.step", "prior_step_path": "semantic_steps/S001.step"}
  ],
  "final_step_path": "semantic_steps/final.step"
}
```

`focus_step_path` 必须是执行生成的特征或操作实体，不是任意图片。切除步骤显示的是被切除的实体/工具区域，橙色不表示最终 CAD 仍有该实体。所有步骤采用同一最终模型的包围盒设置相机。

## 几何验证与渲染命令

在 CAD Docker 运行环境内执行，Initial 参考文件建议以只读方式挂载：

```bash
python integrations/pi/baselines/seek-cad/render_steps.py \
  --workspace /workspace/trace \
  --manifest semantic_steps/manifest.json \
  --output-dir svf-rendered \
  --source-sha256 '<Initial source SHA256>' \
  --initial-step-sha256 '<Initial STEP SHA256>' \
  --reference-step /initial/output.step
```

成功后返回 `svf-rendered/render_manifest.json`，包含分步图、最终图、图像哈希、实际几何一致性报告和相机设置。几何不一致时直接失败，不退化成只检查最终图。

## 不调用模型的验证

```bash
node integrations/pi/baselines/seek-cad/adapter.test.mjs
python -m pytest integrations/pi/baselines/seek-cad/test_render_steps.py -q
```

已通过 6 个 Node 测试和 3 个真实几何/渲染测试。其中一项构造**体积和包围盒相同、孔位置不同**的两个模型，确认一致性检查会拒绝；另一项真实导出两个步骤并渲染透视图。这些实现测试不构成论文分数复现。

## Case 131 首次 smoke 的失败与防护

`results/baselines/seek-cad-svf-drawing-smoke-131-20261105-v1` 没有完成 SVF，不应记作成功。两份模型插桩程序沿用了 `/workspace/output.step` 和 `/workspace/semantic_steps` 绝对路径，而实际执行目录是独立的 `instrumentation/1`、`instrumentation/2`，因此写入了主工作区，子目录的执行结果没有正式输出。

`parseInstrumentation` 现在只将明确的标准输出字符串常量转换为相对当前执行目录的路径，记录替换次数；无法明确归一化的 `/workspace` 引用和含父目录跳转的标准输出会在执行前拒绝。提示词也明确允许仅调整导出目的路径，要求全部输出相对当前目录。此静态检查不是通用 Python 文件系统沙箱。

无模型重放还揭示第二个问题：模型用 `Compound(children=boss_features)` 生成 focus 快照，将已有六个凸台从最终装配重新挂到临时装配，改变了最终实体。两份插桩结果都从原始 8 个实体、体积 199734.04 mm³ 变成 2 个实体、180413.24 mm³，被独立几何等价检查正确拒绝。直接重新执行原始 Initial 源码则复现原来的几何，排除了 Initial 代码／STEP 绑定错误。

插桩提示词现在明确要求快照聚合使用 `deepcopy(shape)`，避免改变原对象的父子归属。该提示不能替代几何验证：每次仍须通过世界坐标下的 B-Rep 对称差和包围盒检查，才可提供步骤图给 critic。所有既有回复哈希、路径规范化后的源码哈希、执行记录和几何差异保存在 `artifacts/baselines/seek-cad-instrumentation-replay-131-20261105-v1/`；此重放未调用模型，也没有将失败轨迹改成成功结果。
