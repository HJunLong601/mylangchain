# Python LangChain 入门项目

这个项目是一个适合新手的最小 LangChain 智能体骨架，目标是先跑通：

- 读取环境变量
- 初始化模型
- 创建 agent
- 调用两个本地工具

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

可以先试这些问题：

- `现在几点？`
- `读取 notes.txt 并帮我总结一下`
- `先读 notes.txt，再告诉我下一步应该学什么`

## 5. 这个项目里有什么

`app/tools.py` 里先放了两个工具：

- `get_current_time()`：返回指定时区的当前时间
- `read_local_note()`：读取 `data/` 目录下的 txt 文件

`app/main.py` 里完成了最小 agent 的初始化和调用。

如果你想看“新建一个工具应该怎么写”，可以直接打开：

- `app/tool_examples.py`

里面放了：

- 一个最小工具模板 `tool_template()`
- 一个无参工具示例 `read_todo_list()`
- 一个带参工具示例 `get_weather_example()`

## 6. 你接下来可以怎么学

建议按这个顺序继续扩展：

1. 先看懂 `build_agent()` 是怎么把模型和工具接起来的
2. 自己新增一个工具，比如“读取待办事项”或“查询天气”
3. 给 agent 增加结构化输出
4. 再进入 RAG 和 LangGraph

## 7. 常见问题

### 启动时报缺少 API Key

说明你还没有创建 `.env` 或没有填写 `ZAI_API_KEY`。

### 工具没有被调用

这是正常现象。Agent 会自己判断是否真的需要工具。你可以问得更明确一点，例如：

- `请调用工具读取 notes.txt`
- `请先读取 notes.txt 再回答`
