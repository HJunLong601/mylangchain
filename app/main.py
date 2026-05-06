from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI

from app.rag import format_scored_search_results, search_knowledge_with_scores
from app.schemas import AssistantStructuredReply
from app.tools import (
    get_current_time,
    get_weather_by_city,
    list_knowledge_base_files,
    read_local_note,
)


# main 模块自己的日志，主要用来观察“直接 Prompt RAG”这条链路。
# 之前 RAG 通过 ToolMessage 返回时，你能在终端看到 ToolMessage；
# 现在改成直接拼 Prompt 后，我们用日志把“检索 -> 拼 Prompt -> 调模型”的过程展示出来。
LOGGER = logging.getLogger("app.main")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[MAIN] %(levelname)s: %(message)s")
    )
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


# 直接 Prompt RAG 每次最多塞入多少个检索片段。
# 数量太少：可能漏掉重要上下文。
# 数量太多：Prompt 会变长，成本更高，也更容易干扰模型。
DIRECT_RAG_MAX_RESULTS = 3
DEFAULT_RAG_QUERY_REWRITE_ENABLED = True
DEFAULT_RAG_REWRITE_HISTORY_MESSAGES = 6


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
If the user's message contains local knowledge-base context, treat it as retrieved reference material.
When local knowledge-base context is relevant, answer based on that context first.
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


def clean_user_text(text: str) -> str:
    """
    清理用户输入里的边界空白和 BOM 字符。

    正常手动输入时一般不会遇到 BOM。
    但在 Windows PowerShell 里用 here-string 或管道喂数据时，
    第一行偶尔会带上 \ufeff，导致日志里看到奇怪的隐藏字符。
    这里统一清掉，避免影响 Query Rewrite 和向量检索。
    """
    return text.lstrip("\ufeff").strip()


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


def env_bool(name: str, default: bool) -> bool:
    """
    从环境变量读取布尔开关。

    Python 的 os.getenv(...) 返回的是字符串，而不是 bool。
    所以这里统一支持几种常见写法：
    - true / 1 / yes / on: 开启
    - false / 0 / no / off: 关闭

    如果环境变量没有配置，就使用 default。
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized_value = raw_value.strip().lower()
    if normalized_value in {"true", "1", "yes", "on"}:
        return True
    if normalized_value in {"false", "0", "no", "off"}:
        return False

    raise ValueError(f"{name} 必须是布尔值，当前值为: {raw_value}")


def env_int(name: str, default: int) -> int:
    """
    从环境变量读取整数。

    这里用于控制 Query Rewrite 参考多少条历史消息。
    消息太少：可能无法理解“它、这个、上面那个”指什么。
    消息太多：改写 prompt 会变长，成本更高。
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数，当前值为: {raw_value}") from exc

    if value < 0:
        raise ValueError(f"{name} 不能小于 0。")

    return value


def extract_original_user_question(content: str) -> str:
    """
    从直接 Prompt RAG 的增强 Prompt 里取回原始用户问题。

    当前 main.py 存进 messages 的 user 内容不是原始问题，
    而是“检索结果 + 回答规则 + 用户问题”组成的增强 Prompt。
    如果 Query Rewrite 直接读取这段增强 Prompt，会把知识片段也当成历史用户输入，
    反而污染改写效果。

    所以这里做一个小清洗：
    - 如果发现【用户问题】标记，就只取它后面的原始问题
    - 如果没有这个标记，说明这不是 RAG 增强 Prompt，就原样返回
    """
    marker = "【用户问题】"
    if marker not in content:
        return content.strip()

    return clean_user_text(content.split(marker, maxsplit=1)[1])


def get_message_role(message) -> str:
    """
    从 LangChain 消息对象或普通 dict 中提取角色。

    当前项目里 messages 可能混合两种形态：
    - 我们手动追加的 dict: {"role": "user", "content": "..."}
    - LangChain 返回的消息对象: HumanMessage / AIMessage / ToolMessage

    为了让 Query Rewrite 能稳定读取历史，这里统一做一层兼容。
    """
    if isinstance(message, dict):
        return str(message.get("role", ""))

    message_type = getattr(message, "type", "")
    if message_type == "human":
        return "user"
    if message_type == "ai":
        return "assistant"
    if message_type == "tool":
        return "tool"

    return str(message_type)


def get_message_content(message) -> str:
    """从 LangChain 消息对象或普通 dict 中提取文本内容。"""
    if isinstance(message, dict):
        return format_message_content(message.get("content", ""))

    return format_message_content(getattr(message, "content", ""))


def format_recent_history_for_rewrite(
    messages: list,
    max_messages: int,
) -> str:
    """
    整理最近几条对话历史，供 Query Rewrite 使用。

    注意这里不是把完整 messages 原样塞给改写模型。
    原因有两个：
    - 历史里可能包含很长的 RAG 增强 Prompt，直接塞进去成本高、噪音大
    - Query Rewrite 只需要知道最近上下文，不需要读取全部知识库片段

    所以这里会：
    - 跳过 ToolMessage
    - user 消息只保留原始用户问题
    - assistant 消息做长度截断
    """
    if max_messages <= 0:
        return "无历史对话。"

    lines: list[str] = []
    for message in messages[-max_messages:]:
        role = get_message_role(message)
        if role == "tool":
            continue

        content = get_message_content(message)
        if not content:
            continue

        if role == "user":
            content = extract_original_user_question(content)
            label = "用户"
        elif role == "assistant":
            label = "助手"
        else:
            label = role or "未知"

        # 改写问题只需要最近语义，不需要助手回答的完整长文本。
        if len(content) > 300:
            content = content[:300] + "..."

        lines.append(f"{label}: {content}")

    if not lines:
        return "无历史对话。"

    return "\n".join(lines)


def is_query_rewrite_enabled() -> bool:
    """读取是否开启 RAG Query Rewrite。"""
    load_dotenv()
    return env_bool(
        "RAG_QUERY_REWRITE_ENABLED",
        DEFAULT_RAG_QUERY_REWRITE_ENABLED,
    )


def get_rewrite_history_message_count() -> int:
    """读取 Query Rewrite 最多参考多少条历史消息。"""
    load_dotenv()
    return env_int(
        "RAG_REWRITE_HISTORY_MESSAGES",
        DEFAULT_RAG_REWRITE_HISTORY_MESSAGES,
    )


def rewrite_query_for_rag(
    *,
    user_question: str,
    conversation_messages: list,
    model: ChatOpenAI,
) -> str:
    """
    把用户当前问题改写成适合向量检索的独立问题。

    为什么需要 Query Rewrite：
    用户在多轮对话里经常会问：
    - “它和微调有什么区别？”
    - “那下一步呢？”
    - “这个有什么问题？”

    人类能从上下文知道“它 / 那 / 这个”指什么，
    但向量检索只拿当前句子去查知识库，可能完全不知道指代对象。

    所以这里先让模型做一次轻量改写：
    - 输入：最近几轮对话 + 当前问题
    - 输出：一个完整、独立、适合搜索的 query
    - 检索：使用改写后的 query
    - 回答：仍然面向用户原问题
    """
    question = clean_user_text(user_question)
    if not question:
        return user_question

    if not is_query_rewrite_enabled():
        LOGGER.info("Query Rewrite 已关闭，直接使用原始问题检索 query=%r", question)
        return question

    history_text = format_recent_history_for_rewrite(
        conversation_messages,
        max_messages=get_rewrite_history_message_count(),
    )
    rewrite_prompt = f"""
你是 RAG 检索问题改写器。

你的任务：
根据【最近对话历史】和【当前用户问题】，把当前问题改写成一个独立、明确、适合向量检索的搜索问题。

规则：
1. 只输出改写后的问题，不要解释。
2. 如果当前问题已经足够独立清楚，就原样输出。
3. 如果当前问题里有“它、这个、那个、上面、下一步”等指代，要根据历史补全指代对象。
4. 不要回答问题，只改写检索 query。
5. 不要引入历史里没有出现的新实体。

【最近对话历史】
{history_text}

【当前用户问题】
{question}
""".strip()

    LOGGER.info("Query Rewrite: 原始问题=%r", question)
    LOGGER.info("Query Rewrite: 参考历史如下\n%s", history_text)
    response = model.invoke([
        {
            "role": "system",
            "content": "你只负责把用户问题改写成适合 RAG 检索的独立 query。",
        },
        {
            "role": "user",
            "content": rewrite_prompt,
        },
    ])
    rewritten_query = format_message_content(
        getattr(response, "content", response)
    ).strip()

    if not rewritten_query:
        LOGGER.info("Query Rewrite: 模型返回为空，回退使用原始问题")
        return question

    # 有些模型会自作聪明加引号或标签，这里做一点轻量清理。
    rewritten_query = rewritten_query.strip().strip('"').strip("'").strip()
    LOGGER.info("Query Rewrite: 改写后问题=%r", rewritten_query)
    return rewritten_query


def build_direct_rag_prompt(
    user_question: str,
    *,
    conversation_messages: list | None = None,
    rewrite_model: ChatOpenAI | None = None,
    max_results: int = DIRECT_RAG_MAX_RESULTS,
) -> str:
    """
    构建“直接 Prompt 版 RAG”的用户消息。

    之前的 Agent 工具版 RAG 是这样传递知识库内容的：
    用户问题 -> 模型决定调用工具 -> 工具返回知识片段 -> LangChain 生成 ToolMessage -> 模型读取 ToolMessage

    现在改成更常见的普通 RAG Chain 写法：
    用户问题 -> 程序先检索知识库 -> 把知识片段直接拼进 Prompt -> 模型基于这个 Prompt 回答

    也就是说：
    - 检索动作由我们的 Python 代码主动执行，不再等模型自己决定是否调用 RAG 工具。
    - 检索结果会成为本轮 HumanMessage 的一部分，而不是 ToolMessage。
    - 好处是链路更简单、更可控；代价是每轮都会先检索，Prompt 也会变长。

    这一版又补上了“检索质量控制”：
    - 检索时拿到每个 chunk 的 distance
    - distance 小于等于阈值的 chunk 才会进入 Prompt
    - 如果没有 chunk 通过阈值，就让模型明确说明知识库证据不足

    现在还加入了 Query Rewrite：
    - 先根据最近对话历史把用户问题改写成独立检索 query
    - 用改写后的 query 去查知识库
    - 最终 Prompt 里同时保留“原始用户问题”和“实际检索 query”
    """
    question = clean_user_text(user_question)
    if not question:
        return user_question

    retrieval_query = question
    if rewrite_model is not None:
        retrieval_query = rewrite_query_for_rag(
            user_question=question,
            conversation_messages=conversation_messages or [],
            model=rewrite_model,
        )

    LOGGER.info(
        "直接 Prompt RAG: 开始检索 original_question=%r retrieval_query=%r max_results=%s",
        question,
        retrieval_query,
        max_results,
    )
    results = search_knowledge_with_scores(query=retrieval_query, max_results=max_results)
    accepted_count = sum(
        1
        for result in results
        if result.passed_threshold
    )
    LOGGER.info(
        "直接 Prompt RAG: 检索完成 raw_hits=%s accepted_hits=%s rejected_hits=%s",
        len(results),
        accepted_count,
        len(results) - accepted_count,
    )

    # format_scored_search_results(...) 会把通过阈值的 Document 分片整理成文本。
    # 这里得到的 retrieved_context 会被直接拼到 Prompt 里，
    # 所以模型看到它时，并不知道“这是工具消息”，只会把它当成本轮输入的一部分。
    #
    # 如果没有任何 chunk 通过阈值，这里也不会强行塞入低质量内容，
    # 而是返回一段“证据不足”的说明，让模型基于这个信号做兜底回答。
    retrieved_context = format_scored_search_results(results)

    prompt = f"""
你正在使用“直接 Prompt 版 RAG”回答问题。

请遵守这些规则：
1. 优先根据【本地知识库检索结果】回答。
2. 只有通过阈值的检索片段才算可靠证据。
3. 如果检索结果提示“没有通过相关性阈值的知识片段”，要明确说明“知识库里没有找到足够相关的内容”。
4. 如果使用了检索结果，请尽量带上来源文件名。
5. 不要编造知识库中没有出现的来源。

【实际用于检索的问题】
{retrieval_query}

【本地知识库检索结果】
{retrieved_context}

【用户问题】
{question}
""".strip()

    LOGGER.info(
        (
            "直接 Prompt RAG: 已生成增强 Prompt chars=%s context_chars=%s "
            "accepted_hits=%s"
        ),
        len(prompt),
        len(retrieved_context),
        accepted_count,
    )
    LOGGER.info(
        "直接 Prompt RAG: 即将发送给模型的本轮 Prompt 如下\n%s",
        prompt,
    )
    return prompt


def print_new_tool_messages(new_messages: list) -> None:
    """
    打印本轮新增的 ToolMessage 内容。

    这个函数的目的，是把“工具返回给模型看的内容”单独展示出来。
    对学习 Agent 工具调用很有帮助，因为你能直接观察：
    - 工具到底返回了什么
    - 返回的是自然语言、结构化文本，还是检索片段
    - 模型后续的最终回答是不是建立在这些内容之上

    注意：当前默认 RAG 已经改成“直接 Prompt 版”，所以知识库内容不会再从这里打印。
    如果这里还能看到 ToolMessage，通常来自天气、时间、列文件等其他工具。
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
    # 注意：这里不再注册 search_local_knowledge。
    # 因为当前 RAG 已经改成“先检索，再直接拼 Prompt”，不需要模型通过工具调用获取知识库内容。
    return create_agent(
        model=model,
        tools=[
            get_current_time,
            read_local_note,
            get_weather_by_city,
            list_knowledge_base_files,
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
    # rewrite_model 专门用于 Query Rewrite。
    # 它不绑定工具，也不负责最终回答，只做一件事：
    # 把“用户聊天式问题”改写成“适合向量检索的问题”。
    rewrite_model = build_model()
    # messages 用来保存整段对话历史，多轮对话的关键就是把历史消息继续传给模型。
    # 在这个示例里，你可以把它理解成一个“消息列表”：
    # - user: 用户消息，对应 HumanMessage
    # - assistant: 模型消息，对应 AIMessage
    # - tool: 工具返回，对应 ToolMessage
    # system 提示词这次没有手动放进列表里，而是通过 create_agent(..., system_prompt=...)
    # 交给 LangChain 管理，所以你在 messages 变量里通常先看到的是 user / assistant / tool。
    # 当前 RAG 采用直接 Prompt 形式，所以本地知识库检索结果会被拼到 user 消息里，
    # 不再额外产生 search_local_knowledge 对应的 ToolMessage。
    messages: list[dict[str, str]] = []

    print("多轮对话已开启，输入 exit 或 quit 结束。")
    print("输入 /json 加空格再提问，可以查看结构化输出。")

    while True:
        # 第二步：从终端持续读取用户问题。
        user_input = clean_user_text(input("\n你: "))
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

        # 第三步：先做 Query Rewrite，再做本地知识库检索，
        # 最后把检索结果直接拼进本轮用户 Prompt。
        # 这是“直接 Prompt 版 RAG”的核心改动。
        # normalized_input 是用户真正输入的问题；
        # Query Rewrite 会参考 messages 里的最近历史，把它改写成 retrieval_query；
        # rag_prompt 是增强后的问题，里面包含“检索 query + 检索结果 + 用户问题 + 回答规则”。
        rag_prompt = build_direct_rag_prompt(
            normalized_input,
            conversation_messages=messages,
            rewrite_model=rewrite_model,
        )

        # 第四步：把增强后的用户问题追加到历史消息中。
        # 这里的 role='user' 是最常见的消息类型之一，表示这条消息来自用户。
        # 注意：即使是 /json 模式，真正存进消息历史里的也是“去掉命令前缀后的问题正文”。
        # 这样可以避免把 /json 当成用户语义的一部分污染后续上下文。
        # 当前版本为了方便观察，存进去的是已经拼好知识库上下文的 rag_prompt。
        # 这也意味着多轮对话时，历史里会保留每轮当时检索到的上下文。
        messages.append({"role": "user", "content": rag_prompt})
        previous_message_count = len(messages)

        # 第五步：把完整历史消息传给 agent，这样模型就能记住上下文。
        # 普通模式和结构化模式的切换点就在这里。
        # RAG 内容已经在上一步进入了 Prompt，所以这里不会再通过 ToolMessage 传递知识库内容。
        # 不过如果模型需要查询时间、天气或列出知识库文件，它仍然可以调用其他工具。
        current_agent = structured_agent if use_structured_output else agent
        result = current_agent.invoke({"messages": messages})

        # 第六步：LangChain 会返回更新后的完整消息列表，里面通常会包含：
        # - HumanMessage: 本轮用户输入
        # - AIMessage: 模型的回答，或者模型准备调用工具时的请求
        # - ToolMessage: 工具执行后的结果
        # 如果本轮没有调用工具，可能就只有 HumanMessage + AIMessage。
        # 当前默认 RAG 不走工具，所以不会再出现“RAG 检索结果 ToolMessage”。
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
