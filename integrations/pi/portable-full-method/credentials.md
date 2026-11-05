# 模型配置与认证

本实验的所有智能体使用 `provider=ucloud`、`model=gpt-5.6-sol`、`thinking_level=low`。`ucloud` 是本项目使用的自定义 Pi provider 名称，不是 Pi 0.84.1 的内置 provider；因此另一台服务器除了密钥，还需要对应的 `models.json`。

## 在目标服务器配置

如果目标服务器已有可用的 `ucloud/gpt-5.6-sol` 配置，可直接指定该目录：

```bash
export PI_CODING_AGENT_DIR=/absolute/path/to/your/pi-agent-directory
```

否则，在本机终端执行以下操作。示例不会覆盖已有的 `models.json`：

```bash
export PI_CODING_AGENT_DIR="$HOME/.pi/cadgenbench-full-method"
mkdir -p "$PI_CODING_AGENT_DIR"
cp -n integrations/pi/portable-full-method/models.example.json "$PI_CODING_AGENT_DIR/models.json"
```

编辑目标目录下的 `models.json`，将 `baseUrl` 中的 `.invalid` 占位地址替换为你实际使用的 UCloud 网关地址。仓库未提供可核实的统一网关地址，迁移包也不读取原服务器的私人配置。`baseUrl` 必须是 API 根地址，例如 Chat Completions 服务常用的 `https://<your-gateway>/v1`，不要添加 `/chat/completions`。

模板使用 Pi 的 `openai-completions` 适配器，即 OpenAI 兼容的 Chat Completions 协议。若你的网关使用 Responses 协议，请沿用其实际可用配置：把 `api` 设为 `openai-responses`，删除模板中仅用于 Chat Completions 的 `compat` 对象，并使用该网关对应的 API 根地址。`gpt-5.6-sol` 是本仓库的实验模型标识；目标网关需要接受这一标识，否则应在实验配置与模型配置中一起更换。

通过当前终端的环境变量提供密钥；以下命令不会把密钥内容写入命令历史：

```bash
read -rsp 'UCLOUD_API_KEY: ' UCLOUD_API_KEY
echo
export UCLOUD_API_KEY
```

`models.json` 中的 `"apiKey": "$UCLOUD_API_KEY"` 是 Pi 支持的环境变量引用，必须保留 `$`。单写 `"UCLOUD_API_KEY"` 会被当作字面密钥。模板不含真实密钥，也不应把真实值写回模板。

模板声明文本和图像输入、推理支持，并将 Pi 的 `low` 映射为服务端的 `low`。实际请求级别由实验配置设置；该声明不会自动强制其他运行都使用 `low`。若网关不接受 `reasoning_effort=low`，需要先确认其兼容协议，不能将关闭 reasoning 参数的运行算作本次 low 实验。

模板未臆测模型的实际上下文长度、最大输出长度或价格。省略时 Pi 0.84.1 分别使用 `128000`、`16384` 和零费率默认值；这些是客户端默认值，不是经过验证的模型规格或实际费用。若已有原实验的非敏感规格配置，应使用相同值。未配置费率时，统计中的 token 数仍有效，费用字段不代表真实账单。

## 主机与容器使用同一份配置

主机编排器将 `$PI_CODING_AGENT_DIR` 以可读写方式挂载到容器的 `/pi-agent`，并设置容器内 `PI_CODING_AGENT_DIR=/pi-agent`。Main、视图搜索和尺寸标注三个阶段因而读取同一份 `models.json` 和认证配置；视图搜索的 Python 几何工作也在容器内执行。

本版本为自定义 `ucloud` provider 显式转发 `UCLOUD_API_KEY`。Docker 参数仅包含 `--env UCLOUD_API_KEY`，不包含密钥值；值由当前主机进程环境提供。直接生成、容器内视图搜索和后续修复都需在同一认证环境中启动。若通过 `tmux`、`systemd` 或批处理系统运行，需在实际运行进程中设置这两个环境变量。

不要将真实的 `auth.json`、`.env`、完整的 `~/.pi/agent`、OAuth 缓存或其他服务凭证放入迁移压缩包。只在目标服务器本地配置凭证。运行时的容器能够访问挂载的 Pi 目录，因此建议为此次实验使用单独的配置目录。
