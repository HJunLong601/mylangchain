from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document


# 这里单独定义 RAG 使用的数据目录和可支持的文件类型。
# 当前最小版本只做本地 txt / md 检索，目的是先把“检索增强生成”的主流程跑通。
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_EXTENSIONS = {".txt", ".md"}


@dataclass
class KnowledgeChunk:
    """
    表示知识库中的一个文本分片。

    为什么要分片：
    - 原始文档通常比较长，不适合一次性整篇拿去做检索。
    - RAG 的常见做法是先把文档切成小块，再只取最相关的几块喂给模型。
    """

    source: str
    path: str
    chunk_index: int
    content: str
    search_terms: set[str]


def list_knowledge_file_paths() -> list[Path]:
    """列出 data 目录下可用于本地 RAG 的知识库文件。"""
    if not DATA_DIR.exists():
        return []

    return sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in KNOWLEDGE_EXTENSIONS
    )


def load_knowledge_documents() -> list[Document]:
    """
    把知识库文件读取成 LangChain 的 Document 对象。

    这里刻意使用 Document，是为了让你从一开始就熟悉 LangChain 常见的数据形态：
    - page_content: 文本正文
    - metadata: 附加信息，比如来源文件名、绝对路径等
    """
    documents: list[Document] = []

    for path in list_knowledge_file_paths():
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "path": str(path),
                },
            )
        )

    return documents


def split_text_into_chunks(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> list[str]:
    """
    按固定窗口把长文本切成多个重叠分片。

    这是一个“自己实现的最小切分器”，作用和文本切分器类似：
    - chunk_size: 每块最多多少字符
    - chunk_overlap: 相邻块之间保留多少重叠内容

    为什么要 overlap：
    - 如果一个知识点恰好落在边界处，完全不重叠很容易把上下文切断。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0。")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0。")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size。")

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        # 下一块从“当前块尾部往前回退 overlap 个字符”的位置重新开始。
        # 这样相邻块就能共享一小段上下文。
        start = end - chunk_overlap

    return chunks


def extract_search_terms(text: str) -> set[str]:
    """
    从查询或文本中提取用于检索的关键词集合。

    这是一个非常适合教学的“最小检索词提取器”：
    - 英文 / 数字：按单词提取
    - 中文：按单字和相邻双字组合提取

    这样做的原因是：
    - 英文天然有空格，按单词拆分很自然
    - 中文没有空格，如果只按整句匹配，召回会很差
    - 加上双字组合后，对“学习路径”“结构化输出”这类短词会更友好
    """
    terms: set[str] = set()
    segments = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", text.lower())

    for segment in segments:
        if re.fullmatch(r"[a-z0-9_]+", segment):
            if len(segment) >= 2:
                terms.add(segment)
            continue

        # 中文段落主要按双字组合建索引，尽量减少单字带来的噪声。
        # 例如“是、的、了”这种单字在检索里区分度很低，容易把结果带偏。
        if len(segment) == 1:
            terms.add(segment)

        # 同时把较短的整段词也放进去，方便“学习路径”“结构化输出”这种短语直接命中。
        if len(segment) <= 8:
            terms.add(segment)

        for index in range(len(segment) - 1):
            terms.add(segment[index : index + 2])

    return terms


def build_knowledge_index() -> list[KnowledgeChunk]:
    """
    构建最小本地知识索引。

    索引构建流程：
    1. 读取所有知识库文件
    2. 对每个文件做文本切分
    3. 为每个分片提前提取检索词，方便后续计算匹配分数

    在生产场景里，这里常常会换成 embedding + 向量库。
    但对于学习阶段，先把“文档 -> 分片 -> 检索”这条链路理解透更重要。
    """
    chunks: list[KnowledgeChunk] = []

    for document in load_knowledge_documents():
        source = document.metadata["source"]
        path = document.metadata["path"]
        split_chunks = split_text_into_chunks(document.page_content)

        for chunk_index, chunk_text in enumerate(split_chunks, start=1):
            chunks.append(
                KnowledgeChunk(
                    source=source,
                    path=path,
                    chunk_index=chunk_index,
                    content=chunk_text,
                    search_terms=extract_search_terms(chunk_text),
                )
            )

    return chunks


def score_chunk(query: str, query_terms: set[str], chunk: KnowledgeChunk) -> float:
    """
    计算查询和某个文本分片的相关度分数。

    当前用的是一个很容易理解的“词项重叠评分”：
    - 命中的关键词越多，分数越高
    - 命中的双字 / 单词越多，说明越相关

    它不如 embedding 检索语义强，但非常适合入门：
    - 容易调试
    - 容易观察为什么某段被召回
    - 不依赖额外模型或向量数据库
    """
    if not query_terms:
        return 0.0

    overlap = query_terms & chunk.search_terms
    if not overlap:
        return 0.0

    # 基础分：较长的词通常信息量更高，这里给它们更多权重。
    # 例如 “结构化输出” 比单个字 “输” 更有区分度。
    score = float(sum(max(len(term), 1) for term in overlap))

    # 覆盖率分：如果查询中的关键词大部分都能在同一分片里命中，
    # 说明这个分片更可能是“围绕这个问题展开”的，而不只是顺带提了一下。
    score += (len(overlap) / len(query_terms)) * 10

    normalized_query = re.sub(r"\s+", "", query.lower())
    normalized_content = re.sub(r"\s+", "", chunk.content.lower())

    # 精确片段加分：如果用户的问题原句（去掉空白后）直接出现在文本里，
    # 这通常意味着它是解释性很强、很直接相关的片段。
    if normalized_query and normalized_query in normalized_content:
        score += max(len(normalized_query), 10)

    return score


def search_knowledge(
    query: str,
    max_results: int = 3,
) -> list[KnowledgeChunk]:
    """
    在本地知识库中搜索与 query 最相关的文本分片。

    返回值是 KnowledgeChunk 列表，而不是直接拼好的字符串，
    因为这样更利于后续扩展：
    - 你可以继续做结构化返回
    - 可以单独显示 source / chunk_index
    - 也可以把 chunk 原文直接传给模型
    """
    query_terms = extract_search_terms(query)
    if not query_terms:
        return []

    scored_chunks: list[tuple[float, KnowledgeChunk]] = []
    for chunk in build_knowledge_index():
        score = score_chunk(query, query_terms, chunk)
        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(
        key=lambda item: (item[0], -len(item[1].content)),
        reverse=True,
    )

    return [chunk for _, chunk in scored_chunks[:max_results]]


def format_search_results(results: list[KnowledgeChunk]) -> str:
    """
    把检索结果格式化成适合 agent 使用的文本。

    这个函数的目标不是直接代替模型回答问题，
    而是把“最相关的知识片段”整理清楚，再交给模型做最后的解释和组织。
    """
    if not results:
        return "没有检索到相关知识片段。"

    lines = ["已从本地知识库检索到以下相关片段："]
    for item in results:
        lines.extend(
            [
                "",
                f"[来源] {item.source} (chunk #{item.chunk_index})",
                "[内容]",
                item.content,
            ]
        )

    return "\n".join(lines)
