from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI

from app.schemas import AssistantStructuredReply
from app.tools import (
    get_current_time,
    get_weather_by_city,
    list_knowledge_base_files,
    read_local_note,
    search_local_knowledge,
)


# 这是给模型的系统提示词。
# 它决定了 agent 的基础角色、回答风格，以及是否倾向于使用工具。
# 从“消息类型”角度看，这个小项目里最常见的是 4 种：
# 1. SystemMessage: 系统提示词，负责定义助手角色和整体规则
# 2. HumanMessage: 用户输入的问题
# 3. AIMessage: 模型返回的回答，或者模型发起的工具调用请求
# 4. ToolMessage: 工具执行后的返回结果
# 严格来说，LangChain 生态里不止这几种消息对象；
# 但对当前这个入门项目，先理解这 4 种就足够了。
# 当前程序在运行时支持两种回答模式：
# 1. 普通模式：返回自然语言，适合直接聊天
# 2. 结构化模式：返回固定字段，适合继续接前端或后端逻辑
SYSTEM_PROMPT = """
You are a beginner-friendly AI assistant for learning LangChain.
Use tools when they help.
When you answer, explain briefly and clearly in Chinese unless the user asks otherwise.
If a question is about local notes, learning materials, or knowledge-base content, prefer using knowledge search tools first.
When you answer based on retrieved knowledge, mention the source file names when possible.
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


def format_structured_response(data: AssistantStructuredReply) -> str:
    """把结构化输出格式化成终端里更好读的 JSON 文本。"""
    # model_dump() 会把 Pydantic 对象转成普通字典。
    # 例如 data 原本是 AssistantStructuredReply(...) 这样的对象，
    # 调用 data.model_dump() 后，就会变成 {"answer": "...", "key_points": [...]} 这种普通 dict。
    # 之所以先转 dict，是因为 json.dumps() 更适合处理基础 Python 数据结构。
    # ensure_ascii=False 是为了让中文直接显示，而不是变成 \uXXXX。
    # indent=2 则让 JSON 在终端里更容易阅读。
    return json.dumps(
        data.model_dump(),
        ensure_ascii=False,
        indent=2,
    )


def print_new_tool_messages(new_messages: list) -> None:
    """
    打印本轮新增的 ToolMessage 内容。

    这个函数的目的，是把“工具返回给模型看的内容”单独展示出来。
    对学习 Agent / RAG 很有帮助，因为你能直接观察：
    - 工具到底返回了什么
    - 返回的是自然语言、结构化文本，还是检索片段
    - 模型后续的最终回答是不是建立在这些内容之上
    """
    for message in new_messages:
        message_type = getattr(message, "type", "")
        if message_type != "tool":
            continue

        tool_name = getattr(message, "name", "unknown_tool")
        content = getattr(message, "content", "")
        print(f"\nToolMessage [{tool_name}]:\n")
        print(format_message_content(content))


def build_model() -> ChatOpenAI:
    """读取配置并创建模型对象，供不同模式复用。"""
    # 从 .env 文件加载环境变量，这样我们就不用把密钥直接写在代码里。
    load_dotenv()

    # 优先读取智谱配置，同时保留对 OpenAI 风格变量名的兼容。
    api_key = os.getenv("ZAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set ZAI_API_KEY in .env before running the project."
        )

    # 先整理出模型初始化参数。
    # 这里的 model 是“模型名字”，比如 glm-5.1，不是 Python 里的模型实例。
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
    return ChatOpenAI(**model_kwargs)


def build_agent():
    """创建普通文本模式的 LangChain agent。"""
    model = build_model()

    # create_agent 会把“模型 + 工具 + 系统提示词”组装成一个可调用的 agent。
    # tools 里传入的就是普通 Python 函数，LangChain 会把它们注册成模型可调用的工具。
    # 这个模式下，agent 的目标是给用户返回一段自然语言回答。
    return create_agent(
        model=model,
        tools=[
            get_current_time,
            read_local_note,
            get_weather_by_city,
            list_knowledge_base_files,
            search_local_knowledge,
        ],
        system_prompt=SYSTEM_PROMPT,
    )


def build_structured_agent():
    """创建结构化输出模式的 agent，返回稳定字段而不是自由文本。"""
    model = build_model()
    # 结构化模式和普通模式用的是同一批工具，区别只在于“最终输出格式”。
    # 这里额外补一条系统提示，是为了提醒模型：当前目标不是自由发挥，
    # 而是认真把 schema 要求的每个字段都填完整。
    return create_agent(
        model=model,
        tools=[
            get_current_time,
            read_local_note,
            get_weather_by_city,
            list_knowledge_base_files,
            search_local_knowledge,
        ],
        system_prompt=(
            SYSTEM_PROMPT
            + "\nWhen structured output mode is enabled, fill every field carefully."
        ),
        response_format=ToolStrategy(
            # ToolStrategy 会让模型按给定 schema 产出结构化结果。
            # 你可以把它理解成：“最后不是返回一段散文，而是返回一个对象。”
            schema=AssistantStructuredReply,
            # 这是结构化输出阶段在工具消息里显示的提示文本。
            # 它主要是帮助调试和观察，不影响 schema 本身。
            tool_message_content="已生成结构化响应。"
        ),
    )


def main() -> None:
    """命令行入口：循环接收用户输入，保留上下文，并打印每轮回复。"""
    # 第一步：先把 agent 组装好。
    # 这里同时准备两个 agent，是因为它们的“行为目标”不同：
    # - agent: 面向自然语言回答
    # - structured_agent: 面向固定字段输出
    agent = build_agent()
    structured_agent = build_structured_agent()
    # messages 用来保存整段对话历史，多轮对话的关键就是把历史消息继续传给模型。
    # 在这个示例里，你可以把它理解成一个“消息列表”：
    # - user: 用户消息，对应 HumanMessage
    # - assistant: 模型消息，对应 AIMessage
    # - tool: 工具返回，对应 ToolMessage
    # system 提示词这次没有手动放进列表里，而是通过 create_agent(..., system_prompt=...)
    # 交给 LangChain 管理，所以你在 messages 变量里通常先看到的是 user / assistant / tool。
    messages: list[dict[str, str]] = []

    print("多轮对话已开启，输入 exit 或 quit 结束。")
    print("输入 /json 加空格再提问，可以查看结构化输出。")

    while True:
        # 第二步：从终端持续读取用户问题。
        user_input = input("\n你: ").strip()
        if not user_input:
            print("你还没有输入内容。")
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("对话结束。")
            return

        # 约定：如果用户输入以 /json 开头，就切换到结构化输出模式。
        # 这样我们不需要额外做菜单，也能快速对比两种模式的差异。
        use_structured_output = user_input.startswith("/json ")
        normalized_input = user_input[6:].strip() if use_structured_output else user_input
        if not normalized_input:
            print("请在 /json 后面输入问题。")
            continue

        # 第三步：把用户问题追加到历史消息中。
        # 这里的 role='user' 是最常见的消息类型之一，表示这条消息来自用户。
        # 注意：即使是 /json 模式，真正存进消息历史里的也是“去掉命令前缀后的问题正文”。
        # 这样可以避免把 /json 当成用户语义的一部分污染后续上下文。
        messages.append({"role": "user", "content": normalized_input})
        previous_message_count = len(messages)

        # 第四步：把完整历史消息传给 agent，这样模型就能记住上下文。
        # 普通模式和结构化模式的切换点就在这里。
        # 如果模型在这一轮决定调用 search_local_knowledge 之类的工具，
        # LangChain 会先执行工具函数，再把工具返回值包装成 ToolMessage 放回消息链，
        # 然后模型会继续读取这条 ToolMessage，最后产出面向用户的 AIMessage。
        current_agent = structured_agent if use_structured_output else agent
        result = current_agent.invoke({"messages": messages})

        # 第五步：LangChain 会返回更新后的完整消息列表，里面通常会包含：
        # - HumanMessage: 本轮用户输入
        # - AIMessage: 模型的回答，或者模型准备调用工具时的请求
        # - ToolMessage: 工具执行后的结果
        # 如果本轮没有调用工具，可能就只有 HumanMessage + AIMessage。
        # 如果本轮调用了 RAG 检索工具，那么 ToolMessage 里放的就是
        # “命中的知识片段整理文本”，而不是向量本身。
        updated_messages = result.get("messages", [])
        if not updated_messages:
            print(result)
            continue

        # 这里只取“本轮新增的消息”，避免把前几轮已经出现过的 ToolMessage 重复打印。
        # 常见新增内容可能是：
        # - AIMessage: 模型发起工具调用请求
        # - ToolMessage: 工具执行结果
        # - AIMessage: 模型基于工具结果给出的最终回答
        new_messages = updated_messages[previous_message_count:]
        print_new_tool_messages(new_messages)

        # 用最新消息列表覆盖本地历史，保证下一轮还能延续上下文。
        # 这里很关键：不是简单地只追加最后一句，而是直接用 LangChain 返回的完整消息链覆盖。
        # 这样工具调用过程中的 AIMessage / ToolMessage 也会一起保留下来。
        messages = updated_messages

        structured_response = result.get("structured_response")
        if structured_response is not None:
            # 结构化模式下，除了 messages 之外，LangChain 还会单独返回 structured_response。
            # 这里优先使用它，因为这是已经按 schema 校验过的结果，最适合程序继续消费。
            # 这个字段的“格式定义”不是在这里写死的，而是在 build_structured_agent() 里：
            # response_format=ToolStrategy(schema=AssistantStructuredReply, ...)
            # 其中 AssistantStructuredReply 定义在 app/schemas.py。
            # 所以这里拿到的 structured_response，本质上就是一个
            # AssistantStructuredReply 的 Pydantic 对象实例，而不是随意拼出来的 dict。
            print("\nAgent 结构化回复:\n")
            print(format_structured_response(structured_response))
            continue

        # 只取最后一条作为当前这一轮的最终答复。
        # 在普通模式里，我们关心的是最终那条 AIMessage 的 content。
        final_message = messages[-1]
        content = getattr(final_message, "content", final_message)
        print("\nAgent 回复:\n")
        print(format_message_content(content))


# 只有直接运行这个文件时，main() 才会执行。
# 如果这个文件被别的模块 import，就不会自动启动。
if __name__ == "__main__":
    main()
