# 医疗 AI 助手（Medical AI Assistant）

基于 **FastAPI + Ollama + MCP 协议** 构建的智能医疗问诊系统，支持本地大模型对话、周边医院查询和就医建议邮件发送。

---

## 项目架构

```
yiliao/
├── main.py          # FastAPI 后端服务（Ollama 对话 + MCP 客户端 + MySQL 持久化）
├── test.py          # MCP 工具服务器（高德地图医院查询 + QQ邮箱发送）
└── 1.html           # 前端聊天界面（纯 HTML/CSS/JS，单文件）
```

```
┌──────────────┐       HTTP/SSE        ┌──────────────┐       MCP(SSE)       ┌──────────────┐
│   1.html     │ ──────────────────→   │   main.py    │ ───────────────────→ │   test.py    │
│  前端界面    │ ←── JSON/Stream ───   │  FastAPI     │ ←── 工具调用 ──────  │  MCP Server  │
└──────────────┘                       │  Port 8000   │                      │  Port 8001   │
                                       └──────┬───────┘                      └──────┬───────┘
                                              │                                     │
                                         ┌────▼─────┐                     ┌─────────▼─────────┐
                                         │  Ollama   │                     │  高德地图 API     │
                                         │ qwen3:4b  │                     │  QQ邮箱 SMTP      │
                                         └──────────┘                     └───────────────────┘
                                              │
                                         ┌────▼─────┐
                                         │  MySQL    │
                                         │  对话历史  │
                                         └──────────┘
```

---

## 功能模块

### 1. 智能问诊对话（main.py + Ollama）

- 调用本地 **Ollama** 运行 `qwen3:4b` 模型进行医疗问答
- 支持 **Function Calling**：模型可自主决定是否调用 MCP 工具
- 支持流式/非流式两种响应模式
- 完整的对话上下文管理，可配置历史消息长度

### 2. MCP 工具集成（test.py）

MCP 服务器提供两个可被大模型调用的工具：

| 工具名称 | 功能 | 依赖 |
|---------|------|------|
| `get_nearby_hospitals` | 根据经纬度查询周边 5km 内医院 | 高德地图 Web API |
| `send_medical_advice` | 将就医建议通过邮件发送给用户 | QQ 邮箱 SMTP |

### 3. 对话历史管理（MySQL）

- 自动持久化所有对话消息（user / assistant / tool）
- 支持按会话 ID 查询历史记录
- 支持删除会话或单条消息
- 会话列表展示（预览、消息数、最后活跃时间）

### 4. 前端界面（1.html）

- 左侧边栏：会话列表，支持新建/切换/删除会话
- 中央聊天区：Markdown 渲染的对话界面
- 右侧面板：附近医院地图展示
- 响应式设计，适配桌面端

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务状态、模型信息、已加载工具列表 |
| GET | `/api/status` | 健康检查（模型、工具数） |
| POST | `/chat` | 核心对话接口，支持流式输出 |
| GET | `/conversations` | 获取会话列表 |
| GET | `/history/{conversation_id}` | 获取指定会话的历史消息 |
| DELETE | `/conversation/{conversation_id}` | 删除整个会话 |
| DELETE | `/message/{message_id}` | 删除指定消息及后续消息 |
| POST | `/api/nearby_hospitals` | 代理查询附近医院（前端直调用） |
| POST | `/api/send_advice` | 代理发送就医建议邮件（前端直调用） |
| GET | `/health` | 综合健康检查（数据库 + MCP） |

---

## 环境依赖

### 基础环境

- Python 3.9+
- MySQL 8.0+
- [Ollama](https://ollama.com/)（已安装 `qwen3:4b` 模型）

### Python 包

```bash
pip install fastapi uvicorn httpx databases aiomysql pydantic mcp python-dotenv
```

### 外部服务配置

1. **Ollama**：确保本地运行并已拉取模型
   ```bash
   ollama pull qwen3:4b
   ```

2. **MySQL**：创建数据库
   ```sql
   CREATE DATABASE agenthistorl;
   ```

3. **高德地图 API**（test.py）：在 [高德开放平台](https://lbs.amap.com/) 申请 Web 服务 Key，填入 `AMAP_SERVER_KEY`

4. **QQ 邮箱 SMTP**（test.py）：在 QQ 邮箱设置中开启 SMTP 服务，获取授权码填入 `SMTP_PASSWORD`

---

## 启动方式

按顺序启动三个服务：

### 1. 启动 MCP 工具服务器

```bash
python test.py
# 输出：MCP Server running on http://127.0.0.1:8001/sse
```

### 2. 启动 FastAPI 后端

```bash
python main.py
# 输出：API文档 http://127.0.0.1:8000/docs
```

启动时自动完成：
- 连接 MySQL 数据库并建表
- 通过 SSE 连接 MCP 服务器，加载可用工具列表
- 将工具注册到 Ollama 的 Function Calling 中

### 3. 打开前端

浏览器直接打开 `1.html` 即可使用。

---

## 对话流程

```
用户输入问题
    │
    ▼
main.py 将 query + history 发送给 Ollama
    │
    ▼
Ollama 返回：纯文本回复 或 tool_calls
    │
    ├── 纯文本 → 直接流式返回前端
    │
    └── tool_calls → main.py 通过 MCP 调用对应工具
                        │
                        ▼
                    test.py 执行工具逻辑
                    （查医院 / 发邮件）
                        │
                        ▼
                    结果传回 Ollama 再次推理
                        │
                        ▼
                    最终回复返回前端
```

每轮对话的消息（user / assistant / tool）都会写入 MySQL 的 `agentmessages` 表。

---

## 配置项

| 配置项 | 位置 | 默认值 | 说明 |
|--------|------|--------|------|
| `OLLAMA_URL` | main.py | `http://localhost:11434/api/chat` | Ollama API 地址 |
| `DEFAULT_MODEL` | main.py | `qwen3:4b` | 默认大模型 |
| `MCP_SERVER_URL` | main.py | `http://127.0.0.1:8001/sse` | MCP 服务器地址 |
| `MYSQL_PASSWORD` | main.py | 环境变量或 `root123` | MySQL 密码 |
| `AMAP_SERVER_KEY` | test.py | 空 | 高德地图 Web 服务 Key |
| `SMTP_USER` / `SMTP_PASSWORD` | test.py | 空 | QQ 邮箱账号和授权码 |

---

## 注意事项

- MCP 工具服务器（test.py）必须在 FastAPI 后端启动**之前**运行，否则后端无法加载工具列表。
- 高德 API Key 和邮箱 SMTP 授权码属于敏感信息，生产环境建议使用环境变量管理。
- Ollama 模型回复质量取决于所使用模型的能力，`qwen3:4b` 为轻量模型，可根据硬件条件替换为更大模型。
- 前端 `1.html` 中的 API 地址默认指向 `http://127.0.0.1:8000`，如需修改请调整文件中的 `API_BASE` 变量。