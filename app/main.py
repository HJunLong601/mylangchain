from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.tools import get_current_time, read_local_note


# 这是给模型的系统提示词。
# 它决定了 agent 的基础角色、回答风格，以及是否倾向于使用工具。
# 从“消息类型”角度看，这个小项目里最常见的是 4 种：
# 1. SystemMessage: 系统提示词，负责定义助手角色和整体规则
# 2. HumanMessage: 用户输入的问题
# 3. AIMessage: 模型返回的回答，或者模型发起的工具调用请求
# 4. ToolMessage: 工具执行后的返回结果
# 严格来说，LangChain 生态里不止这几种消息对象；
# 但对当前这个入门项目，先理解这 4 种就足够了。
SYSTEM_PROMPT = """
You are a beginner-friendly AI assistant for learning LangChain.
Use tools when they help.
When you answer, explain briefly and clearly in Chinese unless the user asks otherwise.
""".strip()


def format_message_content(content) -> str:
    """把 LangChain 返回的消息内容整理成适合终端打印的纯文本。"""
    # 最简单的情况：模型直接返回纯字符串。
    if isinstance(content, str):
        return content

    # 某些模型返回的是分段结构，比如文本块列表。
    # 这里把能提取到的 text 字段拼起来，方便在终端里直观看结果。
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    return str(content)


def build_agent():
    """读取配置并创建一个带本地工具的 LangChain agent。"""
    # 从 .env 文件加载环境变量，这样我们就不用把密钥直接写在代码里。
    load_dotenv()

    # 优先读取智谱配置，同时保留对 OpenAI 风格变量名的兼容。
    api_key = os.getenv("ZAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set ZAI_API_KEY in .env before running the project."
        )

    # 先整理出模型初始化参数。
    # 这里的 model 是模型名，不是 Python 对象。
    model_kwargs: dict[str, str] = {
        "model": os.getenv("MODEL_NAME", "glm-5.1"),
        "api_key": api_key,
    }

    base_url = (
        os.getenv("ZAI_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://open.bigmodel.cn/api/paas/v4/"
    ).strip()
    # 这里兼容 OpenAI 风格接口，所以既可以连智谱，也可以切到其他兼容服务。
    if base_url:
        model_kwargs["base_url"] = base_url

    # ChatOpenAI 是 LangChain 提供的“OpenAI 风格聊天模型”封装。
    # 因为智谱提供兼容接口，所以这里同样可以复用。
    model = ChatOpenAI(**model_kwargs)

    # create_agent 会把“模型 + 工具 + 系统提示词”组装成一个可调用的 agent。
    # tools 里传入的就是普通 Python 函数，LangChain 会把它们注册成模型可调用的工具。
    return create_agent(
        model=model,
        tools=[get_current_time, read_local_note],
        system_prompt=SYSTEM_PROMPT,
    )


def main() -> None:
    """命令行入口：循环接收用户输入，保留上下文，并打印每轮回复。"""
    # 第一步：先把 agent 组装好。
    agent = build_agent()
    # messages 用来保存整段对话历史，多轮对话的关键就是把历史消息继续传给模型。
    # 在这个示例里，你可以把它理解成一个“消息列表”：
    # - user: 用户消息，对应 HumanMessage
    # - assistant: 模型消息，对应 AIMessage
    # - tool: 工具返回，对应 ToolMessage
    # system 提示词这次没有手动放进列表里，而是通过 create_agent(..., system_prompt=...)
    # 交给 LangChain 管理，所以你在 messages 变量里通常先看到的是 user / assistant / tool。
    messages: list[dict[str, str]] = []

    print("多轮对话已开启，输入 exit 或 quit 结束。")

    while True:
        # 第二步：从终端持续读取用户问题。
        user_input = input("\n你: ").strip()
        if not user_input:
            print("你还没有输入内容。")
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("对话结束。")
            return

        # 第三步：把用户问题追加到历史消息中。
        # 这里的 role='user' 是最常见的消息类型之一，表示这条消息来自用户。
        # [
        #   {
        #     "role": "user",
        #     "content": "data目录下的文件"
        #   },
        #   {
        #     "role": "assistant",
        #     "content": "我来帮您查看 data 目录下的文件。我可以读取 data 目录下的 .txt 文件。\n\n让我先尝试读取默认的 notes.txt 文件：",
        #     "tool_calls": [
        #       {
        #         "name": "read_local_note",
        #         "args": {
        #           "filename": "notes.txt"
        #         }
        #       }
        #     ]
        #   },
        #   {
        #     "role": "tool",
        #     "name": "read_local_note",
        #     "content": "LangChain 学习笔记：\n\n1. 模型负责推理和生成。\n2. Tool 是给模型调用的外部能力。\n3. Agent 会根据问题决定是否调用工具。\n4. 先跑通最小例子，再继续学 memory、RAG 和 LangGraph。"
        #   },
        #   {
        #     "role": "assistant",
        #     "content": "根据读取结果，data 目录下有一个文件：notes.txt\n\n文件内容如下：\nLangChain 学习笔记：\n\n1. 模型负责推理和生成。\n2. Tool 是给模型调用的外部能力。\n3. Agent 会根据问题决定是否调用工具。\n4. 先跑通最小例子，再继续学 memory、RAG 和 LangGraph。"
        #   },
        #   {
        #     "role": "user",
        #     "content": "读取一下"
        #   }
        # ]
        
        messages.append({"role": "user", "content": user_input})


        # 第四步：把完整历史消息传给 agent，这样模型就能记住上下文。
        result = agent.invoke({"messages": messages})

        # 第五步：LangChain 会返回更新后的完整消息列表，里面通常会包含：
        # - HumanMessage: 本轮用户输入
        # - AIMessage: 模型的回答，或者模型准备调用工具时的请求
        # - ToolMessage: 工具执行后的结果
        # 如果本轮没有调用工具，可能就只有 HumanMessage + AIMessage。
        updated_messages = result.get("messages", [])
        if not updated_messages:
            print(result)
            continue

        # 用最新消息列表覆盖本地历史，保证下一轮还能延续上下文。
        messages = updated_messages

        # 只取最后一条作为当前这一轮的最终答复。
        final_message = messages[-1]
        content = getattr(final_message, "content", final_message)
        print("\nAgent 回复:\n")
        print(format_message_content(content))


# 只有直接运行这个文件时，main() 才会执行。
# 如果这个文件被别的模块 import，就不会自动启动。
if __name__ == "__main__":
    main()
