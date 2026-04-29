from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


# 这里单独定义 RAG 使用的数据目录和可支持的文件类型。
# 当前版本是“最小向量检索版 RAG”：
# - 文档来源：data 目录下的 .txt / .md 文件
# - 文本切分：手写的固定窗口切分
# - 向量模型：通过 OpenAI 兼容接口接入智谱 embedding 模型
# - 向量库：LangChain 提供的 InMemoryVectorStore
#
# 这条链路已经非常接近真实项目，只是向量库存储仍然放在内存里，
# 适合本地学习和小型 demo，不适合生产环境持久化。
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_EXTENSIONS = {".txt", ".md"}


# 下面这几个模块级变量用于做“懒加载缓存”。
# 含义是：第一次真的用到 RAG 检索时，我们才去构建 embedding 模型和向量库；
# 后续如果 data 目录内容没变，就直接复用，避免每次问答都重新向量化整套文档。
_VECTOR_STORE_CACHE: InMemoryVectorStore | None = None
_VECTOR_STORE_SIGNATURE: tuple[tuple[str, int], ...] | None = None


LOGGER = logging.getLogger("app.rag")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[RAG] %(levelname)s: %(message)s")
    )
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def log_chunk_structure(
    *,
    source: str,
    chunk_index: int,
    chunk_text: str,
    metadata: dict,
) -> None:
    """
    输出单个 chunk 的结构日志。

    这里专门单独封装一个函数，是因为“看清楚每个 chunk 长什么样”
    正是学习 RAG 时最重要的观察点之一。
    """
    LOGGER.info(
        "分片结构 source=%s chunk=%s chars=%s metadata=%s preview=%r",
        source,
        chunk_index,
        len(chunk_text),
        metadata,
        chunk_text[:120],
    )


def list_knowledge_file_paths() -> list[Path]:
    """列出 data 目录下可用于本地 RAG 的知识库文件。"""
    if not DATA_DIR.exists():
        LOGGER.info("知识库目录不存在: %s", DATA_DIR)
        return []

    file_paths = sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in KNOWLEDGE_EXTENSIONS
    )
    LOGGER.info(
        "发现知识库文件: %s",
        [path.name for path in file_paths],
    )
    return file_paths


def get_knowledge_signature() -> tuple[tuple[str, int], ...]:
    """
    为当前知识库生成一个“签名”。

    这个签名只包含两类信息：
    - 文件名
    - 文件最后修改时间

    它的作用是帮助我们判断：data 目录里的内容有没有变化。
    如果签名没变，就说明当前内存里的向量库可以继续复用。
    """
    signature = tuple(
        (path.name, path.stat().st_mtime_ns)
        for path in list_knowledge_file_paths()
    )
    LOGGER.info("当前知识库签名: %s", signature)
    return signature


def load_knowledge_documents() -> list[Document]:
    """
    把知识库文件读取成 LangChain 的 Document 对象。

    这里继续沿用 LangChain 常见的数据形态：
    - page_content: 文本正文
    - metadata: 附加信息，比如来源文件名、绝对路径等

    这样做的好处是：
    - 后面无论接向量库、检索器还是链式调用，都能更自然对接
    - metadata 可以一直跟着文档流转，方便最终回答时回溯来源
    """
    documents: list[Document] = []

    for path in list_knowledge_file_paths():
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            LOGGER.info("跳过空知识文件: %s", path.name)
            continue

        LOGGER.info(
            "已加载知识文件 file=%s chars=%s path=%s",
            path.name,
            len(text),
            path,
        )
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

    为什么还保留这个手写切分器：
    - 它比直接引入更复杂的 splitter 更容易读懂
    - 对学习 RAG 主流程已经足够
    - 你能清楚看到 chunk_size / overlap 对检索效果的影响

    参数说明：
    - chunk_size: 每块最多多少字符
    - chunk_overlap: 相邻块之间保留多少重叠内容
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
            LOGGER.info(
                "文本切分产生分片 index=%s start=%s end=%s chars=%s preview=%r",
                len(chunks),
                start,
                end,
                len(chunk),
                chunk[:120],
            )

        if end == text_length:
            break

        # 下一块从“当前块尾部往前回退 overlap 个字符”的位置重新开始，
        # 这样相邻块可以共享少量上下文，减少知识点刚好被切断的问题。
        start = end - chunk_overlap

    LOGGER.info(
        "文本切分完成 total_chunks=%s chunk_size=%s overlap=%s",
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def split_documents_into_chunks(
    documents: list[Document],
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> list[Document]:
    """
    把原始文档切成多个 Document 分片。

    这里不仅切正文，还会把 metadata 一起继承下来，并补一个 chunk_index。
    这样后面检索回来时，我们不仅知道“来自哪个文件”，还知道“来自第几块”。
    """
    chunked_documents: list[Document] = []

    for document in documents:
        LOGGER.info(
            "开始切分文档 source=%s chars=%s",
            document.metadata.get("source"),
            len(document.page_content),
        )
        chunks = split_text_into_chunks(
            document.page_content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk_text in enumerate(chunks, start=1):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = chunk_index
            log_chunk_structure(
                source=metadata.get("source", "unknown"),
                chunk_index=chunk_index,
                chunk_text=chunk_text,
                metadata=metadata,
            )
            chunked_documents.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata,
                )
            )

    LOGGER.info("全部文档切分完成 total_chunk_documents=%s", len(chunked_documents))
    return chunked_documents


def build_embeddings_model() -> OpenAIEmbeddings:
    """
    创建用于向量化文本的 Embeddings 模型。

    这里使用 LangChain 的 OpenAIEmbeddings 封装，但实际接的是智谱兼容接口。
    你可以把它理解成：
    - LangChain 负责统一调用方式
    - 智谱负责真正生成 embedding 向量

    默认模型使用智谱官方 embedding 模型：
    - embedding-3: 支持自定义维度，默认更适合作为当前项目的下一阶段
    """
    load_dotenv()

    api_key = os.getenv("ZAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少 Embedding API Key。请先在 .env 中配置 ZAI_API_KEY。"
        )

    model_kwargs: dict[str, str | int] = {
        "model": os.getenv("EMBEDDING_MODEL", "embedding-3"),
        "api_key": api_key,
    }

    base_url = (
        os.getenv("ZAI_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://open.bigmodel.cn/api/paas/v4/"
    ).strip()
    if base_url:
        model_kwargs["base_url"] = base_url

    # embedding-3 支持自定义向量维度。
    # 维度越大，通常语义表达越强，但存储和计算成本也更高。
    # 对这个学习项目来说，1024 是一个比较均衡的默认值。
    dimensions = os.getenv("EMBEDDING_DIMENSIONS", "1024").strip()
    if dimensions:
        model_kwargs["dimensions"] = int(dimensions)

    LOGGER.info(
        "构建 Embedding 模型 model=%s dimensions=%s base_url=%s",
        model_kwargs["model"],
        model_kwargs.get("dimensions"),
        model_kwargs.get("base_url"),
    )
    return OpenAIEmbeddings(**model_kwargs)


def build_vector_store() -> InMemoryVectorStore:
    """
    构建一个内存向量库。

    整个向量检索流程在这里真正串起来：
    1. 读取原始文档
    2. 文本切分
    3. 调用 Embeddings 模型把每个分片转成向量
    4. 把向量和原始文本一起放进 InMemoryVectorStore

    为什么选 InMemoryVectorStore：
    - LangChain 官方直接支持
    - 代码量小
    - 特别适合本地 demo 和学习

    它的局限也很明确：
    - 数据只存在内存里，进程结束就没了
    - 不适合大规模数据
    """
    documents = load_knowledge_documents()
    if not documents:
        raise RuntimeError(
            "当前知识库为空。请先在 data 目录中放入 .txt 或 .md 文件。"
        )

    chunked_documents = split_documents_into_chunks(documents)
    LOGGER.info(
        "开始构建内存向量库 documents=%s chunked_documents=%s",
        len(documents),
        len(chunked_documents),
    )
    embeddings = build_embeddings_model()

    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunked_documents)
    LOGGER.info("内存向量库构建完成")
    return vector_store


def get_vector_store() -> InMemoryVectorStore:
    """
    获取当前可用的向量库，并在需要时自动重建。

    这是整个模块里很实用的一层封装：
    - 第一次检索时：自动构建向量库
    - 后续检索时：如果知识文件没变，就直接复用
    - 如果知识文件变了：自动重建向量库
    """
    global _VECTOR_STORE_CACHE
    global _VECTOR_STORE_SIGNATURE

    current_signature = get_knowledge_signature()
    if (
        _VECTOR_STORE_CACHE is not None
        and _VECTOR_STORE_SIGNATURE == current_signature
    ):
        LOGGER.info("命中向量库缓存，直接复用已有索引")
        return _VECTOR_STORE_CACHE

    LOGGER.info("未命中向量库缓存，开始重建索引")
    _VECTOR_STORE_CACHE = build_vector_store()
    _VECTOR_STORE_SIGNATURE = current_signature
    return _VECTOR_STORE_CACHE


def clear_vector_store_cache() -> None:
    """
    清空当前进程内的向量库缓存。

    这个函数主要用于教学和调试：
    - 如果你想强制重新向量化一次
    - 或者后面想做“手动重建索引”工具
    都可以复用它
    """
    global _VECTOR_STORE_CACHE
    global _VECTOR_STORE_SIGNATURE

    _VECTOR_STORE_CACHE = None
    _VECTOR_STORE_SIGNATURE = None
    LOGGER.info("已清空向量库缓存")


def search_knowledge(
    query: str,
    max_results: int = 3,
) -> list[Document]:
    """
    在本地知识库中做语义检索，返回最相关的文档分片。

    和上一阶段最大的区别在于：
    - 之前：按关键词重叠做“字面匹配”
    - 现在：先把 query 向量化，再在向量空间里找语义最接近的分片

    这就是更接近真实 RAG 项目的检索方式。
    """
    query = query.strip()
    if not query:
        return []

    LOGGER.info("开始相似度检索 query=%r max_results=%s", query, max_results)
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=max_results)
    LOGGER.info("相似度检索完成 hits=%s", len(results))
    for index, item in enumerate(results, start=1):
        LOGGER.info(
            "命中结果 #%s source=%s chunk_index=%s chars=%s preview=%r",
            index,
            item.metadata.get("source"),
            item.metadata.get("chunk_index"),
            len(item.page_content),
            item.page_content[:120],
        )
    return results


def format_search_results(results: list[Document]) -> str:
    """
    把检索结果格式化成适合 agent 使用的文本。

    注意这里返回的仍然不是“最终答案”，而是“证据片段”。
    模型看到这些片段后，还需要再做最后一步：
    - 理解用户问题
    - 参考片段内容
    - 组织成适合用户阅读的回答

    这里也是“向量命中后如何传给 agent”的关键节点：
    1. similarity_search(...) 先返回一组 Document 分片
    2. format_search_results(...) 把这些 Document 整理成一段纯文本
    3. search_local_knowledge() 把这段文本作为工具返回值交给 LangChain
    4. LangChain 会把这份工具返回值包装成 ToolMessage，放回 messages
    5. 模型读取这条 ToolMessage 后，再生成最终 AIMessage
    """
    if not results:
        return "没有检索到相关知识片段。"

    lines = ["已从本地向量知识库检索到以下相关片段："]
    for item in results:
        source = item.metadata.get("source", "未知来源")
        chunk_index = item.metadata.get("chunk_index", "?")
        lines.extend(
            [
                "",
                f"[来源] {source} (chunk #{chunk_index})",
                "[内容]",
                item.page_content,
            ]
        )

    return "\n".join(lines)
