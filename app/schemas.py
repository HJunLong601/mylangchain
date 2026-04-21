from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# schemas.py 里定义数据结构
# -> main.py 里把这个结构传给 response_format
# -> agent.invoke(...) 时 LangChain 要求模型按这个结构输出
# -> 最后结果放进 result["structured_response"]
class AssistantStructuredReply(BaseModel):
    """
    结构化输出示例：
    - 让 agent 不只返回自然语言，还返回稳定字段，方便程序继续处理。
    - 这个 schema 设计成通用格式，既能承载天气问题，也能承载笔记总结类问题。
    - 你可以把这个类理解成“AI 回答的数据合同”。
      只要前后端都遵守这份合同，后面接页面、接口、数据库都会更稳定。
    """

    request_type: Literal["weather", "note", "general", "mixed"] = Field(
        description="用户请求的类型。天气问题填 weather，笔记/文件问题填 note，混合问题填 mixed，其余填 general。"
    )
    answer: str = Field(
        description="给用户看的最终回答，用简洁中文表达。"
    )
    key_points: list[str] = Field(
        description="回答中的核心信息点列表。每一项尽量简短明确。"
    )
    used_tools: list[str] = Field(
        description="本轮回答中实际使用过的工具名列表；如果没有使用工具则返回空列表。"
    )
    follow_up_suggestions: list[str] = Field(
        description="建议用户下一步可继续追问的方向。"
    )
    caution: str | None = Field(
        default=None,
        description="需要额外提醒用户的事项；如果没有则为 null。"
    )


# 设计这个 schema 时，可以重点观察这几个思路：
# 1. answer: 给人看，是最终自然语言结论
# 2. key_points: 给程序或 UI 做列表展示
# 3. used_tools: 帮助你理解这一轮到底调用了什么能力
# 4. follow_up_suggestions: 让 agent 更像“可继续交互的助手”
# 5. caution: 单独承载风险提示，避免和 answer 混在一起
