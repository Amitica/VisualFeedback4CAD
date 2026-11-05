# CADCodeVerify：工程图输入适配

依据 [ICLR 2025 论文](https://openreview.net/forum?id=BLWaTeucYX)与[官方仓库](https://github.com/Kamel773/CAD_Code_Generation)，固定源码 commit 为 `33dcecd6087ff7b8a4454b1ba4d56504a6364399`。

该 commit 的仓库提供 CADPrompt 数据和方法图，没有发布问答／修复推理程序，也没有 LICENSE 文件。本目录是 **paper-protocol reimplementation**，不能标作直接运行官方推理代码。未复制 CADPrompt 的 GT 或示例到生成输入中。

共同 runner 必须分开执行以下调用，不能只给 Main 添加一句“自我验证”：

1. 从参考工程图生成固定的 2–5 个正向二元问题。
2. 给定工程图与当前 CAD 四个固定方位渲染，逐题回答 Yes／No／Unclear。
3. 全部 Yes 时停止；否则只以 No／Unclear 项生成反馈。
4. 依据反馈修订并正式执行代码。最多两轮修订，编译错误反馈另外记录。

`adapter.mjs` 提供各阶段提示词及严格 JSON 解析。四视图绕 Z 轴相对旋转 0／90／180／270 度；这里明确选用等轴测仰角，因为论文没有公开具体仰角和渲染程序。

输入从自然语言变为工程图、执行语言从 CadQuery 改为共同 build123d 后，属于受控骨干和任务适配。模型、初始程序、轮数、失败与每个中间状态应写入运行合约。不能把当前 CAD 的几何有效性、渲染匹配分数或模型自评当作最终 CAD 准确率。

身份、差异和原始参数见 `source.json`。最终统一使用 CADGenBench 的评测路径，GT 只进入离线评估，不进入问题、答案或修复反馈。
