# Python LangChain 入门项目

这个项目是一个适合新手的最小 LangChain 智能体骨架，目标是先跑通：

- 读取环境变量
- 初始化模型
- 创建 agent
- 调用多个工具

## 目录结构

```text
mylangchain/
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  └─ tools.py
├─ data/
│  └─ notes.txt
├─ .env.example
├─ requirements.txt
└─ README.md
```

## 1. 创建虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. 安装依赖

```powershell
pip install -r requirements.txt
```

## 3. 配置环境变量

把 `.env.example` 复制为 `.env`，然后填写你的模型配置：

这个项目现在默认按智谱 GLM-5.1 配置：

```env
ZAI_API_KEY=你的智谱密钥
ZAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
MODEL_NAME=glm-5.1
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIMENSIONS=1024
RAG_VECTOR_DIR=.rag_chroma
RAG_COLLECTION_NAME=mylangchain_knowledge
```

如果你后面想切到其他兼容 OpenAI API 的模型，也可以继续使用：

```env
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=兼容接口地址
MODEL_NAME=对应模型名
```

## 4. 运行项目

```powershell
python -m app.main
```

启动后默认是普通对话模式。如果你想体验结构化输出，可以输入：

```text
/json 上海今天天气怎么样？
```

这样 agent 会尽量返回稳定字段的 JSON 结果，方便后续接前端或程序处理。

可以先试这些问题：

- `现在几点？`
- `读取 notes.txt 并帮我总结一下`
- `上海今天天气怎么样？`
- `帮我查一下北京的实时天气`
- `列出当前知识库文件`
- `RAG 是什么？`
- `根据本地知识库解释一下为什么需要 RAG`
- `根据本地知识库解释 RAG 和微调有什么区别`
- `/json 上海今天天气怎么样？`
- `/json 根据本地知识库总结 RAG 的核心流程`
- `/json 读取 notes.txt 后帮我生成三个学习建议`
- `先读 notes.txt，再告诉我下一步应该学什么`

## 5. 这个项目里有什么

`app/tools.py` 里现在放了五个工具，其中前四个默认注册给 agent 使用：

- `get_current_time()`：返回指定时区的当前时间
- `read_local_note()`：读取 `data/` 目录下的 txt 文件
- `get_weather_by_city()`：查询指定城市的当前天气
- `list_knowledge_base_files()`：列出本地知识库文件
- `search_local_knowledge()`：保留的工具版 RAG 示例，方便对比 ToolMessage 写法

`app/main.py` 里完成了最小 agent 的初始化和调用，并在调用模型前主动执行本地 RAG 检索。

天气工具默认使用 Open-Meteo 的公开接口，不需要额外配置天气 API Key，适合入门学习。
结构化输出模式使用 Pydantic schema 约束返回字段，适合继续往“可被程序消费”的方向演进。
本地 RAG 模块会读取 `data/` 目录下的 `.txt` 和 `.md` 文件，切分文本后调用 embedding 模型生成向量，再放进 Chroma 做语义检索。
当前版本已经支持本地持久化，默认会把向量索引保存到 `.rag_chroma/`，下次启动时如果知识库文件没有变化，会直接复用已有索引。
默认 RAG 链路已经改成直接 Prompt 形式：程序先检索知识库，再把命中的片段拼进本轮用户消息中，模型不会再通过 `ToolMessage` 接收知识库内容。

如果你想看“新建一个工具应该怎么写”，可以直接打开：

- `app/tool_examples.py`

里面放了：

- 一个最小工具模板 `tool_template()`
- 一个无参工具示例 `read_todo_list()`
- 一个带参工具示例 `get_weather_example()`

## 6. 你接下来可以怎么学

建议按这个顺序继续扩展：

1. 先看懂 `build_agent()` 是怎么把模型和工具接起来的
2. 看懂 `app/rag.py` 里“文档加载 -> 切分 -> embedding -> Chroma 持久化向量库 -> 检索”的 RAG 流程
3. 自己新增知识文件，观察检索结果怎么变化
4. 再进入更正式的生产向量数据库 / LangGraph

## 7. 常见问题

### 启动时报缺少 API Key

说明你还没有创建 `.env` 或没有填写 `ZAI_API_KEY`。

### 工具没有被调用

这是正常现象。Agent 会自己判断是否真的需要工具。你可以问得更明确一点，例如：

- `请调用工具读取 notes.txt`
- `请先读取 notes.txt 再回答`
