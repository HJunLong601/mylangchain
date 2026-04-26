# 从零理解 LangChain Agent：一篇写给初学者的入门指南

## 前言

如果你刚开始接触大模型应用开发，很容易被一堆名词绕晕：`LLM`、`Prompt`、`Tool`、`Agent`、`RAG`、`Memory`、`LangGraph`。

本文配套项目地址：

- GitHub: <https://github.com/HJunLong601/mylangchain>

我自己在入门时，也最关心几个非常实际的问题：

- LangChain 到底是做什么的？
- Agent 和普通对话调用有什么区别？
- 我应该先学 Python、TypeScript，还是继续留在自己熟悉的技术栈？
- 一个最小可运行的 Agent，代码到底长什么样？

这篇文章就围绕这些问题展开，用一套能实际跑起来的最小示例，把 LangChain Agent 的入门路径梳理清楚。

---

## 一、先搞清楚几个核心名词

### 1. LLM

`LLM`，也就是大语言模型，是整个智能体系统里的“大脑”。

它负责：

- 理解用户问题
- 决定是否需要调用工具
- 根据工具结果组织最终回答

比如你接入的 `GLM-5`、`GLM-5.1`、`GPT` 系列，本质上都是模型。

### 2. Prompt

`Prompt` 就是你给模型的提示信息。

它不只是“用户问的一句话”，还包括：

- 系统提示词
- 用户输入
- 历史对话
- 工具返回结果

在一个 Agent 场景里，Prompt 实际上是“整段上下文”。

### 3. Tool

`Tool` 是给模型使用的外部能力。

模型本身只擅长推理和生成，它并不知道：

- 你电脑当前几点
- 某个本地文件内容是什么
- 某个接口的实时数据是什么

这时候就需要 Tool。

在 LangChain 里，Tool 最常见的形态就是一个普通 Python 函数。  
例如：

- 获取当前时间
- 读取本地文本文件
- 调用天气接口
- 查询数据库

### 4. Agent

`Agent` 可以理解成：

**模型 + 工具 + 决策循环**

和普通问答最大的区别在于，Agent 不只是“你问我答”，而是会在回答前先判断：

- 要不要调用工具
- 该调用哪个工具
- 传什么参数
- 工具返回后如何继续完成回答

这也是 Agent 的核心价值。

### 5. Memory

`Memory` 指的是让系统保留上下文，支持多轮对话。

单轮对话是：

- 用户问一次
- 模型答一次

多轮对话则是：

- 保留历史消息
- 下一轮继续把历史传给模型

这样模型才能理解“刚才提到的那个文件”“你上一轮说的时间”等上下文指代。

### 6. RAG

`RAG` 是“检索增强生成”。

简单说，就是：

- 先从知识库里检索相关内容
- 再把检索结果交给模型生成答案

RAG 解决的是“模型不知道你私有数据”的问题，特别适合：

- 企业文档问答
- 本地知识库助手
- 项目代码问答

### 7. LangGraph

`LangGraph` 是 LangChain 体系里用来做更复杂工作流和状态编排的框架。

如果 LangChain 的高层 Agent 适合快速上手，那么 LangGraph 更适合：

- 多步骤流程控制
- 多 Agent 协作
- 人工审核
- 可恢复、可持久化的执行状态

对初学者来说，建议顺序是：

**先学 LangChain Agent，再学 LangGraph。**

---

## 二、LangChain 到底是什么

LangChain 不是模型，也不是某一家厂商的 API。

它更像一个“应用层框架”，帮你把这些东西连起来：

- 模型
- Prompt
- Tool
- 对话消息
- 结构化输出
- 工作流

你可以把它理解成：

> 用统一的方式，把“大模型能力”接进你的程序。

它最大的价值不是替你发一个 HTTP 请求，而是帮你组织“大模型应用开发”这件事。

对于入门者来说，LangChain 的意义主要体现在三点：

1. 帮你快速搭出一个最小 Agent
2. 帮你把普通 Python 函数变成模型可调用工具
3. 帮你把消息、工具调用、模型输出串成一条完整链路

---

## 三、Agent 是怎么知道要调用哪个工具的

这是很多人一开始最困惑的点。

答案是：

**不是你用 `if/else` 写死的，而是模型根据工具描述自己判断的。**

当你把工具注册给 Agent 时，模型能看到这些信息：

- 工具名
- 参数定义
- 方法注释

例如一个读取文件的工具：

```python
def read_local_note(filename: str = "notes.txt") -> str:
    """读取 data 目录下的 txt 文件内容，避免 agent 直接访问任意路径。"""
```

模型看到这个描述后，就会判断：

- 如果用户说“读取 notes.txt”
- 或者“看看 data 目录里的笔记”

那么这个工具很可能就是最合适的。

所以，Agent 的决策逻辑更像：

```text
用户问题 -> 模型阅读可用工具说明 -> 选择工具 -> 调用工具 -> 生成最终回答
```

这也是为什么一个好用的 Tool 需要：

- 清晰的函数名
- 明确的参数名
- 好理解的 docstring

---

## 四、入门学习路径应该怎么走

如果你刚开始学 LangChain，我建议按下面这个顺序来。

### 第一阶段：理解最小闭环

目标不是做复杂系统，而是先跑通：

- 模型接入
- Tool 注册
- Agent 调用
- 终端输入输出

这一步最重要。

因为一旦你跑通最小闭环，后面的多轮对话、RAG、工作流，都会顺很多。

### 第二阶段：做两个简单工具

先不要碰太复杂的业务，建议先做这些类型的工具：

- 获取当前时间
- 读取本地文本
- 返回固定待办事项

你需要先真正理解：

- Tool 本质上就是函数
- Agent 通过描述选择工具
- 工具只负责执行，不负责聊天

### 第三阶段：改成多轮对话

当你完成单轮 Agent 之后，下一步就应该把它改成支持连续对话。

本质上就是：

- 保留 `messages`
- 下一轮继续把历史消息传给模型

这样你就能理解 `Memory` 的最基础形态。

### 第四阶段：增加结构化输出

这一步非常重要，因为实际项目往往不只是“生成一段话”，而是要：

- 输出 JSON
- 输出固定字段
- 生成表单数据

结构化输出会让你的 Agent 更像一个系统组件，而不只是一个聊天机器人。

### 第五阶段：再进入 RAG 和 LangGraph

这时候再去学：

- 向量检索
- 文档问答
- 工作流编排
- 多 Agent 系统

会更容易吃透。

一句话总结学习路径：

**先最小 Agent，再 Tool，再多轮对话，再结构化输出，最后 RAG / LangGraph。**

---

## 五、技术栈应该怎么选

不同背景的人，适合的路线不一样。

### 1. Python：最推荐的入门路线

如果你的目标是“最快学会 Agent 开发”，首选通常还是 Python。

原因很简单：

- 资料最多
- 示例最多
- 社区讨论最多
- AI 生态最成熟

像 LangChain、LangGraph、RAG、向量数据库、数据处理，这些在 Python 里都非常顺手。

### 2. TypeScript：适合前端 / RN 背景

如果你本身做前端，或者有 React Native 背景，那么 `TypeScript` 也是非常好的选择。

适合的场景：

- Web + Agent 一起开发
- RN 前端 + Node.js 后端
- 与前端工程体系更自然衔接

### 3. Kotlin / Java：适合 JVM 方向开发者

如果你是 Android 开发者，想尽量留在 JVM 技术栈里，也可以考虑 `LangChain4j`。

优点是：

- Kotlin / Java 更熟悉
- 更容易接入已有后端体系

但从入门效率看，示例和资料通常没有 Python 丰富。

### 4. 实际建议

如果你是 Android 开发者、会一点 RN，那么比较实用的建议是：

- 想最快理解 Agent 原理：选 `Python`
- 想以后做 App + 后端联动：选 `TypeScript`
- 想尽量留在 JVM：选 `Kotlin + LangChain4j`

而对于“第一次学 Agent”这件事，我仍然更推荐：

**先用 Python 跑通最小模型和工具调用。**

因为这条路径阻力最小。

---

## 六、OpenAI 兼容 API 是什么意思

很多模型服务商不是 OpenAI，但会提供“OpenAI 兼容接口”。

它的意思是：

- 请求格式和 OpenAI 类似
- 返回格式和 OpenAI 类似
- 工具调用方式也尽量兼容

于是你就可以继续使用类似这样的封装：

```python
from langchain_openai import ChatOpenAI
```

然后只改三个东西：

- `api_key`
- `base_url`
- `model`

例如接入智谱：

```python
model = ChatOpenAI(
    api_key="你的智谱 Key",
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    model="glm-5.1",
)
```

这并不是在调用 OpenAI，而是在调用智谱自己的服务，只是它兼容 OpenAI 风格接口，所以可以复用同一套 LangChain 封装。

---

## 七、一个最小 Agent 长什么样

下面是一个最小可运行的 Agent 核心代码。

它做了三件事：

- 创建模型
- 注册工具
- 用 `create_agent` 组装成 Agent

### 1. 工具定义

```python
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def get_current_time(timezone_name: str = "Asia/Shanghai") -> str:
    """返回指定时区的当前时间，给 agent 作为可调用工具使用。"""
    current_time = datetime.now(ZoneInfo(timezone_name))
    return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")


def read_local_note(filename: str = "notes.txt") -> str:
    """读取 data 目录下的 txt 文件内容。"""
    note_path = Path("data") / Path(filename).name
    return note_path.read_text(encoding="utf-8").strip()
```

### 2. Agent 组装

```python
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.tools import get_current_time, read_local_note


SYSTEM_PROMPT = """
You are a beginner-friendly AI assistant for learning LangChain.
Use tools when they help.
When you answer, explain briefly and clearly in Chinese unless the user asks otherwise.
""".strip()


def build_agent():
    load_dotenv()

    model = ChatOpenAI(
        api_key=os.getenv("ZAI_API_KEY"),
        base_url=os.getenv("ZAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
        model=os.getenv("MODEL_NAME", "glm-5.1"),
    )

    return create_agent(
        model=model,
        tools=[get_current_time, read_local_note],
        system_prompt=SYSTEM_PROMPT,
    )
```

### 3. 多轮对话入口

```python
def main():
    agent = build_agent()
    messages = []

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]

        final_message = messages[-1]
        print(final_message.content)
```

这个示例已经足够帮助你建立最重要的几个认知：

- Tool 就是普通函数
- Agent 会自己决定要不要调用工具
- 多轮对话的本质是保留 `messages`

---

## 八、消息类型怎么理解

在 LangChain 里，初学者最需要理解的核心消息类型是 4 种：

### 1. SystemMessage

系统消息，定义角色和规则。

例如：

- 你是一个有帮助的助手
- 回答要用中文
- 优先使用工具

### 2. HumanMessage

用户消息，也就是用户输入。

例如：

- 现在几点？
- 读取 notes.txt

### 3. AIMessage

模型消息。

它既可能是正常回答，也可能是“准备调用工具”的消息。

### 4. ToolMessage

工具执行之后返回的结果。

例如：

- 时间工具返回当前时间
- 文件读取工具返回笔记内容

你可以把一次完整调用理解成：

```text
HumanMessage -> AIMessage(决定调用工具) -> ToolMessage -> AIMessage(最终回答)
```

这条链路看懂了，Agent 的运行过程就不神秘了。

---

## 九、初学者最容易踩的坑

### 1. 一开始就做太复杂

很多人一上来就想做：

- 联网搜索
- 多工具协作
- RAG
- 多 Agent

结果是概念堆太多，反而搞不清 Agent 的核心。

最好的方式是先跑通一个只有两个工具的最小示例。

### 2. 工具函数写得太复杂

Tool 最好遵循一个原则：

**一个工具只做一件事。**

比如：

- 取时间就只取时间
- 读文件就只读文件

不要把“读取文件 + 总结内容 + 分类结果”全塞进一个工具。

### 3. 不重视 docstring

模型是否能正确使用工具，和方法注释关系非常大。

尤其是：

- 工具用途
- 参数含义
- 限制条件

这些都应该写清楚。

### 4. 把工具当成聊天逻辑

聊天是模型负责的，工具只负责执行。

工具不应该负责：

- 寒暄
- 大段解释
- 聊天式回答

工具应该尽量只返回结果。

---

## 十、给初学者的一个实践建议

如果你今天就要开始，我建议先完成下面这个小目标：

### 第一步

做一个最小 Agent，包含两个工具：

- 获取当前时间
- 读取本地笔记

### 第二步

把它改成多轮对话。

### 第三步

新增一个自己的工具，例如：

- 读取待办事项
- 查询天气
- 读取某个固定 JSON

### 第四步

观察每一轮消息里：

- 用户输入了什么
- 模型是否发起工具调用
- 工具返回了什么
- 最终回答怎么生成

只要你把这一套真正跑通，LangChain Agent 的基础就算入门了。

---

## 结语

LangChain Agent 入门没有想象中那么难。

对于初学者来说，真正重要的不是一开始学会所有高级概念，而是先建立这几个扎实认知：

- LangChain 是把模型、工具和消息组织起来的框架
- Agent 的关键在于“模型能决定是否调用工具”
- Tool 本质上就是普通函数
- 多轮对话本质上就是保留消息历史

当你把最小 Agent 跑通之后，再去看 Memory、RAG、LangGraph，会顺畅很多。

所以最好的起点不是“把所有概念都看懂”，而是：

**先做一个小而完整、能真实运行的 Agent。**

这一步走通了，后面的学习曲线会轻松很多。
