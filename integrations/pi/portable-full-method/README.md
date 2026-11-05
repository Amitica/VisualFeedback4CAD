# 完整 CAD 方法：跨服务器运行包

配置：所有智能体均为 **`ucloud / gpt-5.6-sol / low`**。运行顺序为：

```text
Main 初始生成并执行 CAD（base-only）
  → 图纸分区 + 每个视图的自动 View Search Agent
  → 固定自动搜索得到的相机、剖切及二维配准结果
  → 尺寸标注 Agent：选择对应边/面，测量并标注实际尺寸
  → 原 Main 会话继续检查、修复
       → 每次成功生成新的 STEP 后，尺寸 Agent 重新检查、测量、更新
       → Main 比较更新后的图像并继续修复，直至通过最终检查
```

这里使用新生成的 CAD 和自动搜索视图。没有导入历史 Direct 结果或人工 Oracle 视图；没有启用旧的 Shape Feedback、GAGR 或额外修复 SubAgent。图纸分区是已有 View Search 的准备阶段，同样继承 `low`。

## 基线实验入口

新版迁移包同时包含 [Initial、Vanilla、CADCodeVerify、Seek-CAD SVF 与 IterCAD 的一键入口](../baselines/README.md)。在目标服务器配置模型与密钥后，首次运行单例：

```bash
bash integrations/pi/baselines/run-all.sh --setup --cases 140 --parallel 1 \
  --output-dir results/baselines/all-methods-case140-v1
```

构建完成后运行包内全部生成案例：

```bash
bash integrations/pi/baselines/run-all.sh --all-generation --parallel 4 \
  --output-dir results/baselines/all-methods-generation-v1
```

基线套件需要宿主机 Git 和 Python 3；其构建脚本自动获取并校验固定版本的 IterCAD 官方源码。统一入口中的 IterCAD 使用 Responses 协议，新环境请采用 [基线模型配置模板](../baselines/models.example.json)，并填入自己的实际网关地址。凭证和三维 GT 不包含在迁移包中；几何精度评测需另行提供 GT。

## 1. 解压与构建

迁移整个 `.tar.gz` 和对应 `.sha256` 到目标服务器，在压缩包所在目录执行：

```bash
sha256sum -c cad-full-method-gpt56-sol-low-20261019-v1.tar.gz.sha256
tar -xzf cad-full-method-gpt56-sol-low-20261019-v1.tar.gz
cd cad-full-method-gpt56-sol-low-20261019-v1
bash integrations/pi/portable-full-method/setup.sh
```

目标服务器需要 Linux、Node **≥22.19**、npm、Docker，以及 Docker daemon 的访问权限。首次构建需要访问 npm、PyPI、Debian 软件源与 Docker Hub。宿主机不需要安装 Python、OpenCascade、build123d 或 GPU：几何执行、渲染和自动搜索全部在镜像中进行。

构建使用包内源码与 npm 锁文件，不依赖原服务器的代码路径或本地 `node_modules`。默认镜像为 `cadgenbench-pi-agent:full-method-gpt56-low-v1`。如需其他标签，在构建和运行时一致设置 `CAD_FULL_METHOD_IMAGE`。

## 2. 配置目标服务器的模型与密钥

如果已有可用的 Pi 模型配置：

```bash
export PI_CODING_AGENT_DIR=/absolute/path/to/pi-agent-directory
```

否则：

```bash
export PI_CODING_AGENT_DIR="$HOME/.pi/cadgenbench-full-method"
mkdir -p "$PI_CODING_AGENT_DIR"
cp -n integrations/pi/portable-full-method/models.example.json "$PI_CODING_AGENT_DIR/models.json"
```

在该 `models.json` 中填入实际网关 `baseUrl`，确认其接受模型标识 `gpt-5.6-sol`。模板中的 `.invalid` 地址是占位符，不能直接调用。协议、环境变量引用与可选模型规格见 [credentials.md](credentials.md)。迁移包不包含原服务器的密钥或私人模型配置。

```bash
read -rsp 'UCLOUD_API_KEY: ' UCLOUD_API_KEY
echo
export UCLOUD_API_KEY
```

`PI_CODING_AGENT_DIR` 与 `UCLOUD_API_KEY` 必须存在于真正启动实验的进程环境中。若使用 tmux，在对应会话内设置。

## 3. 检查后启动

先检查镜像、模型注册、认证和几何工具；此命令不进行模型推理：

```bash
node integrations/pi/portable-full-method/run.mjs --doctor
```

查看完整命令计划；不调用模型、不启动 Docker，也不创建实验目录：

```bash
node integrations/pi/portable-full-method/run.mjs --dry-run
```

默认依次运行 **102、103、145、146** 四例：

```bash
node integrations/pi/portable-full-method/run.mjs \
  --output-dir "$PWD/results/pi/full-method-gpt56-low-v1"
```

仅运行一例，便于先检查流程：

```bash
node integrations/pi/portable-full-method/run.mjs --cases 146 \
  --output-dir "$PWD/results/pi/full-method-gpt56-low-case146-v1"
```

运行包内全部 **47 个可用生成案例**：

```bash
node integrations/pi/portable-full-method/run.mjs --all-generation \
  --output-dir "$PWD/results/pi/full-method-gpt56-low-all-v1"
```

这里的 47 例指上述 `cad-full-method-gpt56-sol-low-20261019-v1` 原发布包。当前工作区已于 2026-11-05 05:07 UTC 从官方数据源恢复 101、105 的原始输入，现有 49 个可用生成案例（101–143、145–150）；恢复图像的 SHA 与原下载元数据完全一致，见[输入恢复核验](../../../artifacts/experiment/gpt56-sol-low-full-method-complete-all-20261105-v1/input-recovery/input-recovery-verification.json)。这不改变原发布包的 47 例内容、归档文件或其校验哈希。

每个命令都从初始生成开始，会实际调用模型。输出目录必须是新目录；程序拒绝覆盖已有结果。案例之间串行执行，单个案例内部的多个视图搜索遵循原实现并发运行。

## 4. 配置与保存的证据

[config.json](config.json) 固定模型、推理档位、默认案例和阶段时限：初始生成 120 分钟、视图搜索 180 分钟、Main 修复 120 分钟。Main 容器沿用每例 2 CPU / 6 GB，搜索容器默认也为 2 CPU / 6 GB，可在配置中调整搜索资源。Main 修复内部原有的尺寸 SubAgent 时限仍然生效。

```text
<output-dir>/
├── experiment_contract.json           # 完整命令、参数与阶段合约
├── full_method_summary.json           # 各案例状态、失败阶段、最终维度状态
└── cases/<id>/
    ├── generation.log
    ├── view_search.log
    ├── repair.log
    ├── 01-generate/<id>/              # 初始程序、STEP、原Main会话
    ├── 02-view-search/<id>/           # 自动分区、搜索会话、候选及评分
    │   ├── automatic_frame_lock.json
    │   ├── view_search_only_result.before-frame-lock.json
    │   └── drawing_feedback/
    │       ├── .../view_manifest.json # 原始自动搜索结果，保持不变
    │       └── automatic-fixed-frame/
    │           ├── anchor.step       # 初始CAD，固定投影物理原点
    │           └── view_manifest.json
    └── 03-repair/
        ├── experiment_contract.json
        ├── continuation_summary.json
        └── run/<id>/
            ├── solution.py           # 最终程序
            ├── output.step           # 最终CAD
            ├── agent_result.json
            ├── result.json           # 无GT的候选评估
            ├── sessions/             # 继续原Main会话
            └── drawing_feedback/
                └── dimension-annotations/ # 选择、测量、标注及更新记录
```

固定投影操作仅保存自动搜索选出的相机、剖切参数、二维变换和原始 STEP，不进行人工校准，也不改变候选评分或提升其对齐质量。后续 CAD 修改保留这一投影基准，避免重新缩放掩盖尺寸变化。若搜索有失败视图或缺少有效候选，当前案例会明确失败，不会用人工参数或默认相机替代。

Main 修复入口要求初始会话的模型和推理档位确实为 `low`；不会将历史 `high` 会话改写为 `low`。完成时还检查最终源码、STEP 与尺寸证据的哈希绑定。尺寸 Agent 的 `annotated`、`not_needed`、`unresolved`、`unsupported` 等覆盖状态均保留；流程完成不等于所有尺寸都已成功对应。

## 5. 范围与恢复

- 包内包含当前服务器可用的 47 个生成案例公开输入、当前源码、配置、构建文件和逐文件 SHA256 清单。101、105 的源输入缺少 `input.png`，因此未收录；遗漏及原因在打包清单中明确记录。没有 GT、真实凭证、历史结果或历史容器。
- 没有 GT 时仅报告流程完成、有效性和候选几何统计。IoU、Surface F1 等需另行提供评估端 GT，不能将候选有效性当作 CAD Score。
- 当前尺寸工具不支持剖切新生成的拓扑实体、GD&T 或螺纹规格的完整检查，相关状态会记录在尺寸结果中。
- Ctrl-C 会传递给当前阶段并保留已写入的日志与结果。此完整流程入口不自动覆盖或重跑中断的阶段。新实验应使用新输出目录；不要跨服务器复制 `container.json` 来恢复旧容器。
- 初次运行只需构建一次镜像。已有实验绑定其镜像 ID 与原机挂载路径，请勿在运行过程中移动目录。

打包者本次只做不调用模型的验证。实际 API 可达性、模型对齐效果与最终 CAD 精度需在目标服务器运行后确认。
