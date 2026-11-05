# CAD 工程图视图匹配工具

本目录包含面向 LLM Agent 的 CAD 工程图投影、配准和比较工具。它服务于以下闭环：

```text
输入工程图 + 当前 CAD 模型
        ↓
恢复工程图对应的 CAD 观察视图
        ↓
生成一个几何分数和一张输入/CAD comparison
        ↓
Agent 判断 CAD 中缺少、多出或位置错误的几何
        ↓
修改 CAD，重新生成并比较
```

本文首先描述完整的目标工具，然后说明当前代码已经实现的部分。两者必须区分：当前实现可以验证、对齐并排序一组由 LLM 给出的候选视图，但还不能仅根据一张工程图自动生成任意三维视角或解析完整图纸中的全部视图。

## 1. 工具要解决什么问题

输入：

- 一张工程图图像，可能是单个视图，也可能是包含多个视图、尺寸和标题栏的完整图纸；
- 一个当前 CAD 模型，当前代码使用 STEP/B-Rep；
- 可选的 LLM 语义提示，例如 `top`、`isometric` 或 `section A-A`。

期望输出：

- 与输入工程图对应的三维观察参数 `ViewSpec`；
- CAD 投影到工程图页面上的二维变换 `PaperTransform`；
- 对齐后的 CAD 工程图；
- 匹配边、CAD 缺失边和 CAD 多余边；
- 最佳候选分数、第二候选差距和置信度；
- 供 Agent 观察的一张总图：原始输入与当前 CAD 对应视图并排显示。

这个问题可以写成受约束逆向渲染：

\[
(\theta^*, T^*) =
\arg\min_{\theta,T}
D\left(G(I),\;T\left(R(M;\theta)\right)\right)
\]

其中：

- \(I\) 是输入工程图；
- \(M\) 是 CAD 模型；
- \(G(I)\) 是从工程图中提取的实体几何线，而不是所有黑色像素；
- \(\theta\) 是三维视图参数；
- \(R(M;\theta)\) 是 CAD 内核生成的隐藏线消除投影；
- \(T\) 是二维页面相似变换；
- \(D\) 是工程几何差异。

最终视角不是由 LLM 单独决定的。LLM 负责提出小规模、具有语义意义的候选；CAD 投影和几何评分负责验证并最终裁决。

## 2. 两类变量不能混在一起

### 2.1 三维视图参数 `ViewSpec`

```python
@dataclass
class ViewSpec:
    projection: Literal["orthographic", "perspective"]
    direction: tuple[float, float, float]
    up: tuple[float, float, float]
    section_plane: SectionPlane | None
    show_hidden_lines: bool
```

工程图通常采用正投影，因此默认应固定为 `orthographic`。`direction` 表示从模型指向观察者的方向，`up` 定义图像上方。剖视图还需要剖切平面。

### 2.2 二维页面变换 `PaperTransform`

```python
@dataclass
class PaperTransform:
    uniform_scale: float
    roll_deg: float
    translation_x: float
    translation_y: float
    reflected: bool = False
```

二维变换只处理模型单位与像素比例、图纸位置以及扫描倾斜。默认只允许：

- 等比例缩放；
- 平移；
- 小角度平面旋转；
- 明确允许时的左右反射。

禁止使用独立 X/Y 缩放、剪切和单应变换。否则错误的 CAD 长宽比也可能被强行拉伸成与输入图相似，残差将失去几何意义。

## 3. 总体流程

```text
                      ┌────────────────────┐
输入工程图 ──────────→│ 视图区域与线型解析 │
                      └─────────┬──────────┘
                                │ drawing primitives
                                ↓
                      ┌────────────────────┐
                      │ LLM 生成语义候选   │
                      └─────────┬──────────┘
                                │ top-k ViewSpec priors
                                ↓
CAD B-Rep ───────────→┌────────────────────┐
                      │ CAD 投影候选生成    │
                      │ OCCT HLR / section │
                      └─────────┬──────────┘
                                │ vector edges
                                ↓
                      ┌────────────────────┐
                      │ 受约束二维配准      │
                      └─────────┬──────────┘
                                ↓
                      ┌────────────────────┐
                      │ 几何评分与候选排序  │
                      └─────────┬──────────┘
                                │
                    高置信度 ───┤─── 低置信度
                       ↓        │        ↓
                    输出结果    │  扩大/连续视角搜索
                                │        │
                                └────────┘
```

普通视图走快速路径；只有特殊视图、LLM 低置信度或几何验证失败时，才扩大三维方向搜索。

## 4. 完整目标工具的伪代码

```python
def find_matching_views(drawing_image, cad_model, llm_hint=None):
    """在一张图纸中寻找一个或多个与 CAD 对应的视图。"""

    # 1. 如果输入是完整图纸，先拆出主视图、俯视图、剖视图等区域。
    regions = detect_view_regions(drawing_image)

    matches = []

    for region in regions:
        # 2. 分离实体轮廓、隐藏线、中心线、剖面线和标注。
        primitives = parse_drawing_primitives(region)

        # 3. LLM 只生成语义先验，不直接决定最终浮点相机。
        hypotheses = llm_propose_view_hypotheses(
            image=region,
            context=drawing_image,
            hint=llm_hint,
            top_k=3,
        )

        # 4. 将语义候选展开为 CAD 坐标系中的可执行 ViewSpec。
        candidates = expand_hypotheses(hypotheses, cad_model)

        ranked = []
        for view_spec in candidates:
            # 5. CAD 内核生成可见线和隐藏线。
            projection = project_cad_hlr(cad_model, view_spec)

            # 6. 对每个三维候选，独立求解二维页面变换。
            transform = register_similarity_transform(
                cad_edges=projection.visible_edges,
                drawing_edges=primitives.visible_geometry,
            )

            # 7. 比较轮廓、内部边、线型和拓扑，而非比较所有黑像素。
            score = score_geometric_consistency(
                projection=transform.apply(projection),
                drawing=primitives,
            )

            ranked.append(
                CandidateMatch(
                    score=score,
                    view_spec=view_spec,
                    transform=transform,
                )
            )

        ranked.sort(key=lambda item: item.score)
        best = ranked[0]

        # 8. 普通候选解释不了输入时，才在观察球面上局部或全局搜索。
        if not is_confident(best, ranked[1:]):
            best = refine_special_view(
                seed=best,
                cad_model=cad_model,
                drawing=primitives,
            )

        # 9. 最终结果重新使用精确 HLR 生成，不以快速搜索渲染代替。
        exact_projection = project_cad_hlr(cad_model, best.view_spec)
        aligned_projection = best.transform.apply(exact_projection)

        # 10. 产生适合 Self-Reflection 的结构化差异。
        residual = classify_residual_edges(
            drawing=primitives,
            cad_projection=aligned_projection,
        )

        matches.append(
            ViewMatch(
                region=region,
                view_spec=best.view_spec,
                paper_transform=best.transform,
                score=best.score,
                confidence=score_margin(best, ranked[1:]),
                residual=residual,
            )
        )

    return matches
```

Agent 面向的公共接口应尽量保持为一个：

```python
find_matching_views(
    drawing_image="drawing.png",
    cad_model="candidate.step",
    llm_hint={"likely_view": "top"},  # 可选
) -> list[ViewMatch]
```

候选生成、HLR、配准和评分是内部模块，不需要暴露成大量由 LLM 手工串联的微工具。

## 5. 候选视角如何生成

### 5.1 标准正投影视图

即使 LLM 判断输入是“俯视图”，也不能直接假设 CAD 的 `+Z` 就是图纸的俯视方向。CAD 建模坐标系可能与制图坐标系不一致。

稳妥做法是让 LLM 缩小语义范围，再由工具验证 CAD 轴向候选：

```python
STANDARD_DIRECTIONS = [
    (+1, 0, 0), (-1, 0, 0),
    (0, +1, 0), (0, -1, 0),
    (0, 0, +1), (0, 0, -1),
]
```

如果 CAD 坐标系可靠，LLM 的 `top/front/left` 提示可以把候选减少到一至两个；如果坐标系不可靠，则检查六个方向。

每个 `direction` 还要选择合法的 `up`。自动搜索时，可以把另外两个 CAD 主轴投影到观察平面，枚举离散的 0°、90°、180° 和 270° 页面朝向；选定离散朝向后，二维配准只修正几度的扫描或页面倾斜。不能指望当前默认的 ±6° 小角度配准修正一个 90° 的 `up` 错误。

### 5.2 轴测图

常见等轴测图可以先检查八个立方体对角方向：

```python
ISOMETRIC_DIRECTIONS = [
    normalize((sx, sy, sz))
    for sx in (-1, +1)
    for sy in (-1, +1)
    for sz in (-1, +1)
]
```

LLM 可以根据“看到了顶面、左面还是右面”进一步减少候选。

### 5.3 特殊斜视图或辅助视图

特殊视图不适合让 LLM 直接输出高精度三维向量。推荐策略是：

1. LLM 给出粗方向或可见面关系；
2. 在粗方向附近采样观察球面；
3. 用快速轮廓投影进行粗排序；
4. 对前几个方向做无梯度局部优化；
5. 最后用 OCCT HLR 精确验证。

HLR 随视角变化可能出现边的离散出现和消失，目标函数并不光滑，因此不应假设普通梯度优化始终可靠。可以使用球面网格、Powell、CMA-ES 或其他无梯度搜索，但搜索必须受 LLM 先验和工程制图约束限制。

### 5.4 剖视图

剖视图除了观察方向，还需要恢复剖切平面：

```python
@dataclass
class SectionPlane:
    origin: tuple[float, float, float]
    normal: tuple[float, float, float]
```

优先从同一张完整图纸上的剖切线、箭头和 `A-A` 标签推断平面。仅给出孤立剖视图时，剖切位置可能不唯一，工具应搜索候选平面并报告不确定性，而不是假装得到了唯一答案。

当前代码只支持法向与观察方向一致的直剖视图，并通过 `section_origin` 指定平面上的一点。

## 6. 工程图内容必须分层

输入图中的所有黑色线条不等于 CAD 几何。理想的解析结果应为：

```python
@dataclass
class DrawingPrimitives:
    visible_geometry: EdgeSet
    hidden_geometry: EdgeSet
    centerlines: EdgeSet
    section_hatches: EdgeSet
    dimensions: EdgeSet
    text: list[TextRegion]
    borders_and_title_block: EdgeSet
```

用于 CAD 几何匹配的主要是前三类；尺寸、文字和标题栏应从几何损失中排除。剖面线主要用于识别剖视区域和剖切语义，不应与普通实体边等权比较。

可以结合以下信息分层：

- 线宽、灰度和虚实线型；
- 连通域、OCR 文本框和尺寸箭头；
- 长直线与图框关系；
- 闭环、交点和轮廓拓扑；
- LLM 对区域语义的判断。

LLM 适合判断“这条线像尺寸线还是剖切线”，但像素级边缘和精确端点应由图像处理或视觉模型产生。

## 7. 二维配准细节

对每一个固定的三维候选视图，只优化：

\[
T = (s, \alpha, t_x, t_y)
\]

其中 \(s\) 是统一比例，\(\alpha\) 是小角度页面旋转。当前实现使用：

1. 将 CAD 曲线按弧长近似均匀采样，避免圆弧因离散点多而获得过大权重；
2. 对输入黑色像素计算距离变换；
3. 使用 Differential Evolution 做全局初始化；
4. 使用 Powell 做局部细化；
5. 对远距离误差截断，允许不完整 CAD 依靠共同结构完成配准；
6. 对投影到图像外的点施加额外惩罚。

ROI 是“零件大致包围框”，不是任意搜索窗口。限制投影至少占据 ROI 的一定比例，可以防止完整 CAD 投影缩小后错误吸附到一个孔、尺寸线或局部密集区域。

## 8. 几何评分

### 8.1 当前实现的评分

当前 `align_projection()` 的优化目标仍使用单向距离：

```text
CAD 可见边采样点 → 最近的输入黑色像素
```

这种做法能降低尺寸文字对“缺失 CAD 几何”的直接惩罚，适合先把一个已知候选视图配准到图纸上。

配准完成后，`score_aligned_geometry()` 另外计算一个保守的双向候选分数：

```text
80% CAD → reference robust distance
20% reference ink → CAD robust distance
```

最终只对外返回归一化到 `[0, 1]` 的一个分数。距离图和双向距离只在函数内部参与计算，不保存为残差图片或结构化报告。

### 8.2 目标评分

完整自动搜索需要分层、双向和拓扑感知的评分：

\[
L =
L_{\mathrm{silhouette}}
+ \lambda_v L_{\mathrm{visible}}
+ \lambda_h L_{\mathrm{hidden}}
+ \lambda_t L_{\mathrm{topology}}
\]

边距离应同时计算：

```text
CAD → Drawing：检测 CAD 多出来的边；
Drawing → CAD：检测 CAD 缺失的边。
```

反向距离只能对已经去除尺寸、文本、中心线和图框的几何层计算。否则所有标注都会被错误解释为 CAD 缺失几何。

未来如需研究误差分解，工具内部可以重新引入：

```python
@dataclass
class GeometricResidual:
    matched_edges: EdgeSet
    missing_in_cad: EdgeSet
    extra_in_cad: EdgeSet
    line_type_mismatches: EdgeSet
```

当前简化接口不公开这些字段，以保持 Agent 工具调用为 `score + comparison.png`。

## 9. 置信度和不可辨识情况

最佳分数本身不能代表视图唯一。置信度还应考虑第一名与第二名的分数差：

```python
confidence = combine(
    absolute_fit=best.score,
    score_margin=second.score - best.score,
    geometry_coverage=best.matched_edge_fraction,
)
```

以下情况可能没有唯一答案：

- CAD 关于某个轴完全对称；
- 正视和后视的轮廓相同，而且输入没有隐藏线；
- 工程图裁剪掉了具有判别力的局部结构；
- 当前 CAD 缺失过多，多个方向都只能局部匹配；
- 孤立剖视图无法确定唯一剖切位置。

当前紧凑工具不返回候选列表或结构化置信度；调用 Agent 如果需要处理歧义，应通过再次提供不同候选并比较返回的 score 完成外层搜索。

## 10. 当前代码的真实能力

当前实现位于 [`drawing_view.py`](drawing_view.py)，包括：

### `project_step()`

- 读取 STEP/B-Rep；
- 使用 build123d/OpenCascade HLR 生成精确可见线和隐藏线；
- 支持预设观察方向或显式 `direction/up`；
- 支持由 `section_origin` 指定的简单直剖；
- 输出精确形状和采样折线。

它**不会猜测或搜索视角**。

### `align_projection()`

- 将一个已知 CAD 投影配准到输入图；
- 只允许等比例缩放、平移、小角度旋转和可选反射；
- 支持 ROI、尺度占比限制和稳健距离；
- 返回变换与像素距离诊断。

它优化的是二维页面位置，**不会改变三维观察方向**。

### `compare_step_to_drawing()`

- 串联投影和配准；
- 内部计算几何误差；
- 对外只返回一个 `score` 和一张 `comparison.png`；
- comparison 只包含原始输入和当前 CAD，不包含残差面板。

它验证的是调用方给出的单个候选 `view/direction`。

### `compare_view_candidates()`

- 接受两个或更多由 LLM 提出的命名或自定义 `ViewCandidate`；
- 对每个候选执行完全相同的 HLR、二维配准与几何评分；
- 在系统临时目录中完成排序；
- 搜索结束后只返回最佳候选的一个 `score` 和一张 `comparison.png`；
- 不输出候选文件夹、排名报告或残差图。

它实现“LLM 提候选、工具内部验证”，但保持最小 Agent 接口。

### `compare_step_to_drawings()`

- 接受多个已经选定的 `DrawingViewSpec`；
- 分别完成投影、配准和评分；
- 将每个视图的“输入/CAD”双联图拼成一张统一总图；
- 返回所有视图几何分数的平均值和唯一的 `comparison.png`；
- 中间视图只存在于临时目录，不产生公开文件。

当前命令行入口位于 [`_cli.py`](_cli.py)：

```bash
cadgenbench drawing-view candidate.step reference.png \
  --view top \
  --roi 145 200 500 380 \
  --output-dir feedback/top
```

多个候选的 Agent 调用：

```bash
cadgenbench drawing-view candidate.step reference.png \
  --candidates top bottom front rear left right \
  --roi 145 200 500 380 \
  --output-dir feedback/view-search
```

特殊方向可以显式提供：

```bash
cadgenbench drawing-view candidate.step reference.png \
  --view custom \
  --direction -1 1 1 \
  --up 0 0 1 \
  --roi 35 35 835 650 \
  --output-dir feedback/isometric
```

多视图总图使用一个 JSON manifest：

```json
{
  "views": [
    {
      "name": "main orthographic",
      "reference": "04_main_orthographic.png",
      "view": "top",
      "roi": [145, 200, 500, 380]
    },
    {
      "name": "side view",
      "reference": "05_side_view.png",
      "view": "left",
      "roi": [150, 45, 480, 155]
    }
  ]
}
```

```bash
cadgenbench drawing-sheet candidate.step views.json \
  --output-dir feedback
```

工具的公开文件输出只有：

```text
<output-dir>/
└── comparison.png
```

命令标准输出只包含 `score` 和 `comparison_png`。工具不生成结构化报告。

## 11. 当前限制和最常见失败原因

1. **调用方提供了错误的三维视图。** 二维配准无法修正真正的观察方向错误。
2. **输入是完整图纸但没有 ROI。** 标题栏、尺寸和其他视图会形成错误吸附目标。
3. **尺寸和文本尚未被完整分离。** 当前黑像素掩码可能把标注当成几何线。
4. **输入是阴影渲染而不是线稿。** Otsu 阈值可能把大面积阴影识别为工程边。
5. **CAD 坐标系与制图坐标系不同。** `top` 等语义标签不能直接等价于固定世界轴。
6. **视图具有对称性。** 单个视图可能不足以区分正反方向。
7. **CAD 本身缺失过多。** 共同结构太少时，配准结果不稳定。
8. **剖切平面未知或复杂。** 当前只支持观察方向法向的简单直剖，不支持阶梯剖、旋转剖和任意剖切程序。
9. **候选分数不是最终 CAD 正确性指标。** 它可以比较视角，但反向残差仍混有尺寸、文字和中心线；只有完成图层分离后才能可靠命名为 CAD 缺失几何。

## 12. 推荐的实现顺序

为了从当前“候选验证器”演进到完整“自动视图匹配器”，建议按以下顺序实现：

1. **已实现：** 对 LLM 提出的标准或自定义候选执行排序；
2. **已实现：** 保守双向评分、候选间距和结构化 CAD 侧残差；
3. 工程图视图区域检测和 ROI 自动提取；
4. 标注、文本、中心线和实体几何的分层；
5. 低置信度时的观察球面搜索与局部细化；
6. 从完整图纸解析剖切线和剖视方向；
7. 将 `unexplained reference ink` 可靠升级为 `missing_in_cad`。

当前版本已经能形成由 LLM 控制的普通视图搜索闭环；后续搜索和剖视图模块只负责处理特殊情况，不应降低普通视图路径的确定性。
