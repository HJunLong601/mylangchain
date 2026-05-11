from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# 这里单独定义 RAG 使用的数据目录和可支持的文件类型。
# 当前版本是“持久化向量检索版 RAG”：
# - 文档来源：data 目录下的 .txt / .md 文件
# - 文本切分：手写的固定窗口切分
# - 向量模型：通过 OpenAI 兼容接口接入智谱 embedding 模型
# - 向量库：Chroma，本地持久化到磁盘目录
#
# 和上一版 InMemoryVectorStore 相比，这一版最大的变化是：
# - 向量索引会落盘
# - 下次启动进程可以复用已有索引
# - 更接近真实项目里的“索引构建”和“索引加载”流程
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_EXTENSIONS = {".txt", ".md"}
DEFAULT_VECTOR_DIR = ROOT_DIR / ".rag_chroma"
DEFAULT_COLLECTION_NAME = "mylangchain_knowledge"
SIGNATURE_FILE_NAME = "knowledge_signature.json"
DEFAULT_MAX_DISTANCE = 1.2
DEFAULT_RERANK_ENABLED = True


@dataclass(frozen=True)
class RagSearchResult:
    """
    表示一次向量检索命中的结果。

    为什么不再只返回 Document：
    - Document 只包含正文和 metadata，看不到“为什么它被认为相关”
    - RAG 调优时，分数非常关键；没有分数，就只能凭感觉猜检索质量
    - 把分数和是否通过阈值一起返回，后面拼 Prompt 时就能只采用可靠证据

    字段说明：
    - document: LangChain 的文档分片，里面有 page_content 和 metadata
    - distance: Chroma 返回的距离分数，通常越小越相关
    - relevance_score: 为了方便阅读换算出的相关度分数，越接近 1 越相关
    - passed_threshold: 当前结果是否通过我们设置的最大距离阈值
    - retrieval_rank: 向量库原始召回顺序，数字越小代表越靠前
    - keyword_score: 教学版 rerank 的关键词命中分数
    - length_penalty: 教学版 rerank 的长度惩罚，避免过短或过长 chunk 盲目靠前
    - rerank_score: 教学版 rerank 最终分数，越大越应该优先放进 Prompt
    """
    document: Document
    distance: float
    relevance_score: float
    passed_threshold: bool
    retrieval_rank: int = 0
    keyword_score: float = 0
    length_penalty: float = 0
    rerank_score: float = 0


# 下面这几个模块级变量用于做“懒加载缓存”。
# 含义是：第一次真的用到 RAG 检索时，我们才去加载或构建 Chroma 向量库；
# 后续同一进程内如果 data 目录内容没变，就直接复用这个 Chroma 实例。
_VECTOR_STORE_CACHE: Chroma | None = None
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


def get_vector_store_dir() -> Path:
    """
    获取 Chroma 持久化目录。

    默认目录是项目根目录下的 .rag_chroma。
    也可以通过环境变量 RAG_VECTOR_DIR 自定义，例如：
    RAG_VECTOR_DIR=.rag_chroma

    这里做一次路径解析，后续清理和写入都基于绝对路径，避免误删其他目录。
    """
    load_dotenv()

    configured_dir = os.getenv("RAG_VECTOR_DIR", "").strip()
    if configured_dir:
        path = Path(configured_dir)
        if not path.is_absolute():
            path = ROOT_DIR / path
    else:
        path = DEFAULT_VECTOR_DIR

    return path.resolve()


def get_collection_name() -> str:
    """获取 Chroma collection 名称。"""
    load_dotenv()
    return os.getenv("RAG_COLLECTION_NAME", DEFAULT_COLLECTION_NAME).strip() or DEFAULT_COLLECTION_NAME


def get_max_distance_threshold() -> float:
    """
    获取 RAG 检索的最大距离阈值。

    Chroma 的 similarity_search_with_score(...) 会返回 distance：
    - distance 越小，说明 query 和 chunk 越接近
    - distance 越大，说明相关性越弱

    所以这里的阈值不是“分数越高越好”，而是“距离不能超过多少”。
    默认值 1.2 是一个学习项目里的宽松起点，不是生产最佳值。
    真实项目里应该通过日志观察不同问题的命中距离，再逐步调整。
    """
    load_dotenv()

    raw_threshold = os.getenv("RAG_MAX_DISTANCE", str(DEFAULT_MAX_DISTANCE)).strip()
    try:
        threshold = float(raw_threshold)
    except ValueError as exc:
        raise ValueError(
            f"RAG_MAX_DISTANCE 必须是数字，当前值为: {raw_threshold}"
        ) from exc

    if threshold <= 0:
        raise ValueError("RAG_MAX_DISTANCE 必须大于 0。")

    return threshold


def get_rerank_enabled() -> bool:
    """
    读取是否开启教学版 Rerank。

    生产项目里 Rerank 往往会接专门的重排序模型。
    这里先用规则版实现，是为了让你先看懂这一层在 RAG 链路里的位置：
    - Retrieval: 负责多召回一些候选
    - Rerank: 负责重新排序，挑出最值得进入 Prompt 的证据
    """
    load_dotenv()
    raw_value = os.getenv(
        "RAG_RERANK_ENABLED",
        str(DEFAULT_RERANK_ENABLED),
    ).strip().lower()

    if raw_value in {"true", "1", "yes", "on"}:
        return True
    if raw_value in {"false", "0", "no", "off"}:
        return False

    raise ValueError(f"RAG_RERANK_ENABLED 必须是布尔值，当前值为: {raw_value}")


def distance_to_relevance_score(distance: float) -> float:
    """
    把 Chroma distance 转成更适合人读的相关度分数。

    注意：不同向量库、不同距离算法的原始分数含义并不完全一样。
    Chroma 这里给我们的是 distance，学习时可以先记住：
    - distance: 越小越好
    - relevance_score: 越大越好

    这里做一个简单换算：1 / (1 + distance)。
    它不会改变排序，只是把“距离越小越好”转换成“相关度越高越好”，
    方便在日志和 Prompt 里观察。
    """
    return 1 / (1 + max(distance, 0))


def tokenize_for_rerank(text: str) -> list[str]:
    """
    为教学版 Rerank 做一个很轻量的分词。

    这里不是严格的中文分词器，而是一个足够透明的教学实现：
    - 英文和数字按单词抽取
    - 中文按连续字符片段抽取
    - 对较长中文片段额外生成 2 字符窗口，方便匹配“微调”“检索”等短词

    真实项目里可以换成 jieba、HanLP，或者直接使用 reranker 模型。
    当前版本故意保持简单，让你能从日志里看清楚分数是怎么来的。
    """
    raw_tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower())
    tokens: list[str] = []

    for token in raw_tokens:
        tokens.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 2:
            tokens.extend(
                token[index:index + 2]
                for index in range(0, len(token) - 1)
            )

    return [
        token
        for token in tokens
        if token.strip()
    ]


def calculate_keyword_score(query: str, text: str) -> float:
    """
    计算 query 和 chunk 的关键词重合分数。

    这是教学版 Rerank 的一个特征：
    - 如果 query 里的关键词在 chunk 里出现越多，分数越高
    - 分数范围控制在 0 到 1，方便和 relevance_score 混合

    这个分数不能替代向量相似度，只是补充“字面匹配”的信号。
    """
    query_tokens = set(tokenize_for_rerank(query))
    if not query_tokens:
        return 0

    text_tokens = set(tokenize_for_rerank(text))
    if not text_tokens:
        return 0

    matched_tokens = query_tokens & text_tokens
    return len(matched_tokens) / len(query_tokens)


def calculate_length_penalty(text: str) -> float:
    """
    计算 chunk 长度惩罚。

    一个很短的 chunk 可能信息不足，一个过长的 chunk 又容易浪费上下文窗口。
    当前项目 chunk_size 默认是 400，所以这里把 200 到 600 字符视作比较舒服的范围：
    - 在范围内：不惩罚
    - 太短或太长：给一个小惩罚

    注意这是教学规则，不是数学真理。真实项目要结合数据调参。
    """
    length = len(text)
    if 200 <= length <= 600:
        return 0
    if length < 200:
        return min((200 - length) / 200, 1) * 0.12
    return min((length - 600) / 600, 1) * 0.12


def calculate_rerank_score(query: str, result: RagSearchResult) -> tuple[float, float, float]:
    """
    计算教学版 Rerank 的最终分数。

    当前规则：
    - relevance_score 权重 70%：保留向量检索的语义排序能力
    - keyword_score 权重 30%：补充关键词命中信号
    - length_penalty：轻微惩罚过短或过长的 chunk

    这个函数返回三个值：
    - rerank_score: 最终排序分数
    - keyword_score: 关键词命中分数
    - length_penalty: 长度惩罚
    """
    keyword_score = calculate_keyword_score(
        query,
        result.document.page_content,
    )
    length_penalty = calculate_length_penalty(result.document.page_content)
    rerank_score = (
        result.relevance_score * 0.7
        + keyword_score * 0.3
        - length_penalty
    )
    return rerank_score, keyword_score, length_penalty


def get_signature_file_path(vector_store_dir: Path) -> Path:
    """获取用于保存知识库签名的文件路径。"""
    return vector_store_dir / SIGNATURE_FILE_NAME


def read_persisted_signature(
    vector_store_dir: Path,
) -> tuple[tuple[str, int], ...] | None:
    """
    读取上一次构建索引时保存的知识库签名。

    如果签名文件存在且和当前 data 目录一致，说明磁盘上的 Chroma 索引仍然可复用。
    如果签名不同，就说明知识库文件变了，需要重建索引。
    """
    signature_path = get_signature_file_path(vector_store_dir)
    if not signature_path.exists():
        LOGGER.info("未发现持久化签名文件: %s", signature_path)
        return None

    try:
        raw_signature = json.loads(signature_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.info("持久化签名文件解析失败，将重建索引: %s", signature_path)
        return None

    signature = tuple(
        (str(item[0]), int(item[1]))
        for item in raw_signature
    )
    LOGGER.info("读取到持久化知识库签名: %s", signature)
    return signature


def write_persisted_signature(
    vector_store_dir: Path,
    signature: tuple[tuple[str, int], ...],
) -> None:
    """
    保存当前知识库签名。

    Chroma 会负责保存向量数据；这个签名文件是我们自己额外保存的，
    用来判断下次启动时是否需要重建索引。
    """
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    signature_path = get_signature_file_path(vector_store_dir)
    signature_path.write_text(
        json.dumps(list(signature), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("已写入知识库签名: %s", signature_path)


def reset_vector_store_dir(vector_store_dir: Path) -> None:
    """
    清空 Chroma 持久化目录。

    只有当知识库文件发生变化时才会调用这里。
    为了避免误删，必须确认目标目录在当前项目根目录下面。
    """
    resolved_dir = vector_store_dir.resolve()
    resolved_root = ROOT_DIR.resolve()

    if resolved_dir == resolved_root or resolved_root not in resolved_dir.parents:
        raise RuntimeError(
            f"拒绝清空非项目目录下的向量库路径: {resolved_dir}"
        )

    if resolved_dir.exists():
        LOGGER.info("清空旧的 Chroma 持久化目录: %s", resolved_dir)
        shutil.rmtree(resolved_dir)


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


def open_persisted_vector_store(
    embeddings: OpenAIEmbeddings,
    vector_store_dir: Path,
) -> Chroma:
    """
    打开已经存在的 Chroma 向量库。

    这里不会重新向量化文档，只是把磁盘上的 collection 加载进来，
    所以速度比重建索引快很多。
    """
    LOGGER.info(
        "加载已有 Chroma 向量库 collection=%s dir=%s",
        get_collection_name(),
        vector_store_dir,
    )
    return Chroma(
        collection_name=get_collection_name(),
        embedding_function=embeddings,
        persist_directory=str(vector_store_dir),
    )


def build_vector_store(
    *,
    current_signature: tuple[tuple[str, int], ...],
    vector_store_dir: Path,
    embeddings: OpenAIEmbeddings,
) -> Chroma:
    """
    构建一个可持久化的 Chroma 向量库。

    整个向量检索流程在这里真正串起来：
    1. 读取原始文档
    2. 文本切分
    3. 调用 Embeddings 模型把每个分片转成向量
    4. 把向量和原始文本一起写入 Chroma
    5. 保存知识库签名，方便下次启动复用索引

    为什么选 Chroma：
    - 本地启动简单
    - 支持 persist_directory 落盘
    - 很适合作为从 demo 走向生产前的过渡方案

    它仍然不是最终生产形态：
    - 大规模、多租户、高并发场景通常会考虑 Qdrant、Milvus、pgvector 等组件
    """
    documents = load_knowledge_documents()
    if not documents:
        raise RuntimeError(
            "当前知识库为空。请先在 data 目录中放入 .txt 或 .md 文件。"
        )

    chunked_documents = split_documents_into_chunks(documents)
    LOGGER.info(
        "开始构建 Chroma 持久化向量库 documents=%s chunked_documents=%s dir=%s collection=%s",
        len(documents),
        len(chunked_documents),
        vector_store_dir,
        get_collection_name(),
    )

    reset_vector_store_dir(vector_store_dir)
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=get_collection_name(),
        embedding_function=embeddings,
        persist_directory=str(vector_store_dir),
    )
    vector_store.add_documents(chunked_documents)
    write_persisted_signature(vector_store_dir, current_signature)
    LOGGER.info("Chroma 持久化向量库构建完成")
    return vector_store


def get_vector_store() -> Chroma:
    """
    获取当前可用的向量库，并在需要时自动重建。

    这是整个模块里很实用的一层封装：
    - 同一进程内：优先复用内存里的 Chroma 实例
    - 重启进程后：如果知识库签名没变，就加载磁盘上的 Chroma 索引
    - 如果知识文件变了：清空旧索引并重建
    """
    global _VECTOR_STORE_CACHE
    global _VECTOR_STORE_SIGNATURE

    current_signature = get_knowledge_signature()
    vector_store_dir = get_vector_store_dir()

    if (
        _VECTOR_STORE_CACHE is not None
        and _VECTOR_STORE_SIGNATURE == current_signature
    ):
        LOGGER.info("命中向量库缓存，直接复用已有索引")
        return _VECTOR_STORE_CACHE

    embeddings = build_embeddings_model()
    persisted_signature = read_persisted_signature(vector_store_dir)

    if persisted_signature == current_signature:
        LOGGER.info("知识库签名未变化，复用磁盘上的 Chroma 索引")
        _VECTOR_STORE_CACHE = open_persisted_vector_store(
            embeddings=embeddings,
            vector_store_dir=vector_store_dir,
        )
    else:
        LOGGER.info("知识库签名发生变化，开始重建 Chroma 索引")
        _VECTOR_STORE_CACHE = build_vector_store(
            current_signature=current_signature,
            vector_store_dir=vector_store_dir,
            embeddings=embeddings,
        )

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
    LOGGER.info("已清空当前进程内的向量库缓存")


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
    scored_results = search_knowledge_with_scores(
        query=query,
        max_results=max_results,
    )
    return [
        result.document
        for result in scored_results
        if result.passed_threshold
    ]


def search_knowledge_with_scores(
    query: str,
    max_results: int = 3,
) -> list[RagSearchResult]:
    """
    在本地知识库中做带分数的语义检索。

    这是这一阶段新增的关键方法。
    和 search_knowledge(...) 相比，它不只是返回 Document，还会返回：
    - Chroma 原始 distance
    - 换算后的 relevance_score
    - 是否通过最大距离阈值

    为什么要这么做：
    - RAG 生产落地时，不能“只要搜到就塞给模型”
    - 如果检索结果本身不相关，模型会被错误上下文带偏
    - 阈值过滤可以让系统在证据不足时诚实兜底，而不是硬编答案
    """
    query = query.strip()
    if not query:
        return []

    if max_results <= 0:
        raise ValueError("max_results 必须大于 0。")

    max_distance = get_max_distance_threshold()
    LOGGER.info(
        "开始带分数的相似度检索 query=%r max_results=%s max_distance=%s",
        query,
        max_results,
        max_distance,
    )
    vector_store = get_vector_store()
    raw_results = vector_store.similarity_search_with_score(query, k=max_results)

    results: list[RagSearchResult] = []
    for retrieval_rank, (document, distance) in enumerate(raw_results, start=1):
        distance = float(distance)
        results.append(
            RagSearchResult(
                document=document,
                distance=distance,
                relevance_score=distance_to_relevance_score(distance),
                passed_threshold=distance <= max_distance,
                retrieval_rank=retrieval_rank,
            )
        )

    accepted_count = sum(
        1
        for result in results
        if result.passed_threshold
    )
    LOGGER.info(
        "带分数的相似度检索完成 raw_hits=%s accepted_hits=%s rejected_hits=%s",
        len(results),
        accepted_count,
        len(results) - accepted_count,
    )
    for index, result in enumerate(results, start=1):
        item = result.document
        LOGGER.info(
            (
                "命中结果 #%s source=%s chunk_index=%s distance=%.6f "
                "relevance_score=%.6f passed_threshold=%s chars=%s preview=%r"
            ),
            index,
            item.metadata.get("source"),
            item.metadata.get("chunk_index"),
            result.distance,
            result.relevance_score,
            result.passed_threshold,
            len(item.page_content),
            item.page_content[:120],
        )
    return results


def rerank_search_results(
    *,
    query: str,
    results: list[RagSearchResult],
    max_results: int,
) -> list[RagSearchResult]:
    """
    对向量召回结果做教学版 Rerank，并返回最终进入 Prompt 的 Top N。

    这一步模拟生产 RAG 里常见的“两阶段检索”：
    1. 向量库先召回更多候选，比如 top 8 / top 20
    2. Rerank 再根据更细的相关性判断重新排序
    3. 最后只把排序最靠前的少数 chunk 塞进 Prompt

    当前实现是规则版，不依赖额外模型：
    - relevance_score: 来自向量检索，表示语义相似度
    - keyword_score: query 和 chunk 的关键词重合
    - length_penalty: chunk 太短或太长时轻微扣分

    为什么只对通过阈值的结果排序：
    - 阈值过滤负责“证据够不够可靠”
    - Rerank 负责“可靠证据里谁更应该靠前”
    - 低相关结果即使字面命中，也不应该重新混进 Prompt
    """
    if max_results <= 0:
        raise ValueError("max_results 必须大于 0。")

    if not results:
        LOGGER.info("Rerank: 没有候选结果，跳过重排")
        return []

    LOGGER.info("Rerank: 向量召回原始顺序如下")
    for index, result in enumerate(results, start=1):
        item = result.document
        LOGGER.info(
            (
                "Rerank before #%s retrieval_rank=%s source=%s chunk_index=%s "
                "distance=%.6f relevance_score=%.6f passed_threshold=%s preview=%r"
            ),
            index,
            result.retrieval_rank,
            item.metadata.get("source"),
            item.metadata.get("chunk_index"),
            result.distance,
            result.relevance_score,
            result.passed_threshold,
            item.page_content[:80],
        )

    accepted_results = [
        result
        for result in results
        if result.passed_threshold
    ]
    if not accepted_results:
        LOGGER.info("Rerank: 没有候选通过阈值，返回空结果")
        return []

    if not get_rerank_enabled():
        LOGGER.info("Rerank 已关闭，按向量召回顺序取前 %s 条", max_results)
        return accepted_results[:max_results]

    reranked_results: list[RagSearchResult] = []
    for result in accepted_results:
        rerank_score, keyword_score, length_penalty = calculate_rerank_score(
            query,
            result,
        )
        reranked_results.append(
            RagSearchResult(
                document=result.document,
                distance=result.distance,
                relevance_score=result.relevance_score,
                passed_threshold=result.passed_threshold,
                retrieval_rank=result.retrieval_rank,
                keyword_score=keyword_score,
                length_penalty=length_penalty,
                rerank_score=rerank_score,
            )
        )

    reranked_results.sort(
        key=lambda result: (
            result.rerank_score,
            result.relevance_score,
            -result.retrieval_rank,
        ),
        reverse=True,
    )

    LOGGER.info("Rerank: 重排后顺序如下")
    for index, result in enumerate(reranked_results, start=1):
        item = result.document
        LOGGER.info(
            (
                "Rerank after #%s retrieval_rank=%s source=%s chunk_index=%s "
                "rerank_score=%.6f relevance_score=%.6f keyword_score=%.6f "
                "length_penalty=%.6f distance=%.6f preview=%r"
            ),
            index,
            result.retrieval_rank,
            item.metadata.get("source"),
            item.metadata.get("chunk_index"),
            result.rerank_score,
            result.relevance_score,
            result.keyword_score,
            result.length_penalty,
            result.distance,
            item.page_content[:80],
        )

    final_results = reranked_results[:max_results]
    LOGGER.info(
        "Rerank: 最终进入 Prompt 的结果数=%s max_results=%s",
        len(final_results),
        max_results,
    )
    return final_results


def format_search_results(results: list[Document]) -> str:
    """
    把检索结果格式化成适合模型阅读的文本。

    注意这里返回的仍然不是“最终答案”，而是“证据片段”。
    模型看到这些片段后，还需要再做最后一步：
    - 理解用户问题
    - 参考片段内容
    - 组织成适合用户阅读的回答

    当前默认走“直接 Prompt 版 RAG”，所以链路是：
    1. similarity_search(...) 先返回一组 Document 分片
    2. format_search_results(...) 把这些 Document 整理成一段纯文本
    3. main.build_direct_rag_prompt(...) 把这段文本拼进本轮用户 Prompt
    4. 模型直接从 HumanMessage 内容里读取这些参考资料
    5. 模型基于“参考资料 + 用户问题”生成最终 AIMessage

    对比上一版工具调用模式：
    - 工具版：检索结果通过 ToolMessage 传给模型
    - Prompt 版：检索结果直接拼进用户 Prompt 传给模型
    两者最后都会进入模型上下文，区别主要是“谁负责触发检索”和“消息结构长什么样”。
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


def format_scored_search_results(results: list[RagSearchResult]) -> str:
    """
    把带分数的检索结果格式化成适合直接拼进 Prompt 的文本。

    和 format_search_results(...) 的区别：
    - 这里只展示通过阈值的结果
    - 每个 chunk 会带上 distance / relevance_score / rerank_score
    - 如果没有结果通过阈值，会返回明确的“证据不足”提示

    这一步是 RAG 质量控制的核心：
    检索系统不是把所有命中都交给模型，而是先筛掉不可靠证据。
    """
    accepted_results = [
        result
        for result in results
        if result.passed_threshold
    ]

    if not accepted_results:
        return (
            "没有检索到通过相关性阈值的知识片段。\n"
            "这表示当前知识库没有足够可靠的证据支持回答这个问题。"
        )

    lines = ["已从本地向量知识库检索到以下通过阈值的相关片段："]
    for result in accepted_results:
        item = result.document
        source = item.metadata.get("source", "未知来源")
        chunk_index = item.metadata.get("chunk_index", "?")
        lines.extend(
            [
                "",
                f"[来源] {source} (chunk #{chunk_index})",
                f"[retrieval_rank] {result.retrieval_rank}，向量库原始召回排名",
                f"[distance] {result.distance:.6f}，越小越相关",
                f"[relevance_score] {result.relevance_score:.6f}，越接近 1 越相关",
                f"[keyword_score] {result.keyword_score:.6f}，关键词命中越高越相关",
                f"[length_penalty] {result.length_penalty:.6f}，长度惩罚越小越好",
                f"[rerank_score] {result.rerank_score:.6f}，重排最终分数越高越优先",
                "[内容]",
                item.page_content,
            ]
        )

    return "\n".join(lines)
