from __future__ import annotations

"""
RAGAS 评估和召回检查入口。

这个脚本复用 app.main 里的真实 RAG 链路，而不是另写一套演示代码：

1. 从 JSONL 评估集中读取问题、参考答案和可选的参考 chunk id。
2. 调用当前项目真实的 Query Rewrite、Chroma 检索、Rerank、Prompt 拼接和模型回答。
3. 生成两类输出：
   - RAGAS 评估结果：回答忠实度、上下文召回、事实正确性等。
   - 召回报告：每个问题实际召回了哪些 chunk、是否命中 reference_context_ids。

常见用法：
- python evals/run_ragas_eval.py --mode all
- python evals/run_ragas_eval.py --mode ragas --limit 1
- python evals/run_ragas_eval.py --mode retrieval
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# 直接运行 evals/run_ragas_eval.py 时，Python 默认只会把 evals/ 放进 sys.path。
# 为了稳定 import app.main，这里手动把项目根目录加入模块搜索路径。
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import (
    build_agent,
    build_direct_rag_payload,
    build_model,
    get_context_ids_from_rag_results,
    get_contexts_from_rag_results,
    get_sources_from_rag_results,
    run_rag_once,
)


DEFAULT_DATASET_PATH = ROOT_DIR / "evals" / "ragas_dataset.jsonl"
DEFAULT_RAGAS_OUTPUT_PATH = ROOT_DIR / "evals" / "ragas_results.csv"
DEFAULT_RETRIEVAL_OUTPUT_PATH = ROOT_DIR / "evals" / "retrieval_results.csv"
DEFAULT_RAGAS_TIMEOUT_SECONDS = 300
DEFAULT_RAGAS_MAX_WORKERS = 4
DEFAULT_MIN_ID_RECALL = 0.8
DEFAULT_MIN_ID_PRECISION = 0.5


def resolve_project_path(value: str, fallback: Path) -> Path:
    """把 .env 或命令行中的相对路径解析到项目根目录下。"""
    if not value.strip():
        return fallback

    path = Path(value.strip())
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def env_int(name: str, default: int) -> int:
    """读取正整数环境变量，用于控制 RAGAS 超时和并发。"""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(raw_value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数，当前值为: {raw_value}") from exc

    if value <= 0:
        raise ValueError(f"{name} 必须大于 0。")
    return value


def env_float(name: str, default: float) -> float:
    """读取 0 到 1 之间的小数环境变量，用于控制召回告警阈值。"""
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = float(raw_value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字，当前值为: {raw_value}") from exc

    if value < 0 or value > 1:
        raise ValueError(f"{name} 必须在 0 到 1 之间。")
    return value


def normalize_string_list(value: Any, *, field_name: str, line_number: int) -> list[str]:
    """
    把 JSONL 里的列表字段规范化成 list[str]。

    召回评估会用到 reference_context_ids：
    - 如果字段不存在，返回空列表
    - 如果字段存在，必须是字符串数组
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"第 {line_number} 行 {field_name} 必须是字符串数组。")

    items: list[str] = []
    for item in value:
        item_text = str(item).strip()
        if item_text:
            items.append(item_text)
    return items


def load_eval_items(path: Path) -> list[dict[str, Any]]:
    """
    读取 JSONL 评估集。

    必填字段：
    - user_input: 用户问题
    - reference: 人工参考答案

    可选字段：
    - reference_context_ids: 人工标注的正确 chunk id，例如 ["langchain_rag.md#chunk-1"]

    reference_context_ids 用来做精确召回检查：
    - id_recall = 命中的参考 chunk 数 / 参考 chunk 总数
    - id_precision = 命中的参考 chunk 数 / 实际召回 chunk 总数
    """
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            user_input = str(item.get("user_input", "")).strip()
            reference = str(item.get("reference", "")).strip()
            if not user_input or not reference:
                raise ValueError(
                    f"{path} 第 {line_number} 行必须包含 user_input 和 reference。"
                )

            items.append(
                {
                    "user_input": user_input,
                    "reference": reference,
                    "reference_context_ids": normalize_string_list(
                        item.get("reference_context_ids"),
                        field_name="reference_context_ids",
                        line_number=line_number,
                    ),
                }
            )

    if not items:
        raise ValueError(f"评估集为空: {path}")
    return items


def calculate_id_retrieval_scores(
    *,
    retrieved_context_ids: list[str],
    reference_context_ids: list[str],
) -> dict[str, Any]:
    """
    计算不依赖 LLM 的 ID 级召回/精度。

    这个分数非常适合调检索参数：
    - 它便宜，不需要模型裁判
    - 它稳定，同一批输入不会因为模型输出波动而变化
    - 它要求你在评估集里标注 reference_context_ids
    """
    retrieved_set = set(retrieved_context_ids)
    reference_set = set(reference_context_ids)
    hit_context_ids = sorted(retrieved_set & reference_set)

    id_recall = (
        len(hit_context_ids) / len(reference_set)
        if reference_set
        else None
    )
    id_precision = (
        len(hit_context_ids) / len(retrieved_set)
        if retrieved_set
        else None
    )
    return {
        "hit_context_ids": hit_context_ids,
        "id_recall": id_recall,
        "id_precision": id_precision,
    }


def result_scores_for_report(results: list) -> list[dict[str, Any]]:
    """把检索结果的分数压成适合写入 CSV 的结构。"""
    scores: list[dict[str, Any]] = []
    for result in results:
        context_id = f"{result.document.metadata.get('source')}#chunk-{result.document.metadata.get('chunk_index')}"
        scores.append(
            {
                "context_id": context_id,
                "retrieval_rank": result.retrieval_rank,
                "distance": result.distance,
                "relevance_score": result.relevance_score,
                "keyword_score": result.keyword_score,
                "rerank_score": result.rerank_score,
                "passed_threshold": result.passed_threshold,
            }
        )
    return scores


def build_retrieval_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    只运行召回链路，不生成最终回答。

    这个模式适合快速调参：
    - RAG_MAX_DISTANCE
    - RAG_RETRIEVAL_CANDIDATES
    - RAG_RERANK_ENABLED

    它仍然会使用 Query Rewrite，所以 retrieval_query 和真实主流程一致。
    """
    rewrite_model = build_model()
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        question = item["user_input"]
        print(f"[RETRIEVAL] ({index}/{len(items)}) 检查召回: {question}")
        payload = build_direct_rag_payload(
            question,
            conversation_messages=[],
            rewrite_model=rewrite_model,
        )
        retrieved_context_ids = get_context_ids_from_rag_results(payload.results)
        retrieved_contexts = get_contexts_from_rag_results(payload.results)
        sources = get_sources_from_rag_results(payload.results)
        id_scores = calculate_id_retrieval_scores(
            retrieved_context_ids=retrieved_context_ids,
            reference_context_ids=item["reference_context_ids"],
        )

        rows.append(
            {
                "user_input": question,
                # retrieval 模式不生成最终回答，但 RAGAS EvaluationDataset
                # 更习惯看到 response 字段；这里留空即可。
                "response": "",
                "reference": item["reference"],
                "retrieval_query": payload.retrieval_query,
                "retrieved_contexts": retrieved_contexts,
                "retrieved_context_ids": retrieved_context_ids,
                "reference_context_ids": item["reference_context_ids"],
                "sources": sources,
                "hit_context_ids": id_scores["hit_context_ids"],
                "id_recall": id_scores["id_recall"],
                "id_precision": id_scores["id_precision"],
                "retrieval_scores": result_scores_for_report(payload.results),
            }
        )

    return rows


def build_retrieval_warning(row: dict[str, Any]) -> str:
    """
    根据召回分数生成可读告警。

    这个字段是给人看的，不是给 RAGAS 的：
    - 空字符串表示召回检查通过或样本没有可检查的参考 chunk id
    - 非空字符串表示需要回头看检索参数、切分策略或评估集标注
    """
    warnings_text: list[str] = []
    reference_context_ids = row.get("reference_context_ids") or []
    retrieved_context_ids = row.get("retrieved_context_ids") or []

    if not retrieved_context_ids:
        warnings_text.append("没有召回任何通过阈值并进入 Prompt 的 chunk")

    if not reference_context_ids:
        warnings_text.append("样本未标注 reference_context_ids，无法做 ID 级召回判断")
        return "；".join(warnings_text)

    min_recall = env_float("RAGAS_MIN_ID_RECALL", DEFAULT_MIN_ID_RECALL)
    min_precision = env_float("RAGAS_MIN_ID_PRECISION", DEFAULT_MIN_ID_PRECISION)

    id_recall = row.get("ragas_id_based_context_recall", row.get("id_recall"))
    id_precision = row.get("ragas_id_based_context_precision", row.get("id_precision"))

    if id_recall is not None and id_recall != "" and float(id_recall) < min_recall:
        warnings_text.append(
            f"召回率过低: {float(id_recall):.3f} < {min_recall:.3f}"
        )
    if id_precision is not None and id_precision != "" and float(id_precision) < min_precision:
        warnings_text.append(
            f"召回精度过低: {float(id_precision):.3f} < {min_precision:.3f}"
        )

    missed_context_ids = sorted(set(reference_context_ids) - set(retrieved_context_ids))
    if missed_context_ids:
        warnings_text.append(
            "漏召回参考 chunk: " + ", ".join(missed_context_ids)
        )

    return "；".join(warnings_text)


def add_retrieval_warnings(rows: list[dict[str, Any]]) -> None:
    """给每一行召回报告补充告警字段。"""
    for row in rows:
        warning = build_retrieval_warning(row)
        row["has_retrieval_warning"] = bool(warning)
        row["retrieval_warning"] = warning


def add_ragas_id_metrics_to_retrieval_rows(rows: list[dict[str, Any]]) -> None:
    """
    在 retrieval 模式下也运行 RAGAS 的 ID-based 召回指标。

    这一步不需要 LLM，所以比完整 RAGAS 评估便宜很多。
    它会在 CSV 里新增：
    - ragas_id_based_context_recall
    - ragas_id_based_context_precision

    如果评估集没有 reference_context_ids，就跳过 RAGAS ID 指标，
    但仍然会生成“缺少标注”的 warning。
    """
    if not has_reference_context_ids(rows):
        add_retrieval_warnings(rows)
        return

    ensure_ragas_available()
    from ragas import EvaluationDataset, evaluate

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

    evaluation_dataset = EvaluationDataset.from_list(rows)
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            IDBasedContextRecall(),
            IDBasedContextPrecision(),
        ],
        show_progress=False,
    )
    records = result.to_pandas().to_dict("records")
    for row, record in zip(rows, records):
        row["ragas_id_based_context_recall"] = record.get("id_based_context_recall")
        row["ragas_id_based_context_precision"] = record.get("id_based_context_precision")

    add_retrieval_warnings(rows)


def build_ragas_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    运行完整 RAG 链路，并整理成 RAGAS 需要的数据结构。

    RAGAS 常用字段：
    - user_input: 用户问题
    - response: RAG 系统生成的回答
    - retrieved_contexts: 最终进入 Prompt 的上下文片段
    - reference: 人工参考答案

    这里额外加入 retrieved_context_ids / reference_context_ids，
    这样同一份数据也能跑 IDBasedContextRecall / IDBasedContextPrecision。
    """
    agent = build_agent()
    rewrite_model = build_model()
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        question = item["user_input"]
        print(f"[RAGAS] ({index}/{len(items)}) 运行 RAG: {question}")
        rag_answer = run_rag_once(
            question,
            agent=agent,
            rewrite_model=rewrite_model,
        )
        id_scores = calculate_id_retrieval_scores(
            retrieved_context_ids=rag_answer.context_ids,
            reference_context_ids=item["reference_context_ids"],
        )
        rows.append(
            {
                "user_input": question,
                "response": rag_answer.answer,
                "retrieved_contexts": rag_answer.contexts,
                "retrieved_context_ids": rag_answer.context_ids,
                "reference_context_ids": item["reference_context_ids"],
                "reference": item["reference"],
                "retrieval_query": rag_answer.retrieval_query,
                "sources": rag_answer.sources,
                "hit_context_ids": id_scores["hit_context_ids"],
                "id_recall": id_scores["id_recall"],
                "id_precision": id_scores["id_precision"],
            }
        )

    return rows


def has_reference_context_ids(rows: list[dict[str, Any]]) -> bool:
    """判断当前数据是否足够运行 ID-based 召回指标。"""
    return any(row.get("reference_context_ids") for row in rows)


def ensure_ragas_available() -> None:
    """在调用模型和 RAGAS 评分前确认依赖已经安装。"""
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "缺少 RAGAS 依赖。请先运行: pip install -r requirements.txt"
        ) from exc


def build_ragas_metrics(*, include_id_metrics: bool) -> list:
    """
    构建 RAGAS 指标列表。

    这里仍然使用 ragas.metrics 旧导入路径，是因为 RAGAS 0.4.3 的新 collections
    API 要求 InstructorLLM，而当前项目已经统一使用 LangChain ChatOpenAI。
    旧指标类在 0.4.3 仍可用，并且可以通过 evaluate(..., llm=...) 接收 LangChain wrapper。
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from ragas.metrics import (
            FactualCorrectness,
            Faithfulness,
            IDBasedContextPrecision,
            IDBasedContextRecall,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )

    metrics = [
        Faithfulness(),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
        FactualCorrectness(language="chinese"),
    ]
    if include_id_metrics:
        metrics.extend([
            IDBasedContextRecall(),
            IDBasedContextPrecision(),
        ])
    return metrics


def evaluate_with_ragas(rows: list[dict[str, Any]]):
    """调用 RAGAS 计算回答质量和召回质量。"""
    ensure_ragas_available()
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    evaluator_llm = LangchainLLMWrapper(build_model())
    evaluation_dataset = EvaluationDataset.from_list(rows)
    return evaluate(
        dataset=evaluation_dataset,
        metrics=build_ragas_metrics(
            include_id_metrics=has_reference_context_ids(rows),
        ),
        llm=evaluator_llm,
        run_config=RunConfig(
            timeout=env_int("RAGAS_TIMEOUT_SECONDS", DEFAULT_RAGAS_TIMEOUT_SECONDS),
            max_workers=env_int("RAGAS_MAX_WORKERS", DEFAULT_RAGAS_MAX_WORKERS),
        ),
    )


def json_cell(value: Any) -> str:
    """把列表/字典字段序列化成 CSV 单元格里的 JSON 字符串。"""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def save_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """保存普通行数据为 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json_cell(row.get(key))
                for key in fieldnames
            })
    print(f"[EVAL] CSV 已保存: {output_path}")


def save_ragas_result(result, output_path: Path) -> None:
    """保存 RAGAS 结果为 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_pandas().to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[RAGAS] 结果已保存: {output_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数，并允许 .env 提供默认路径。"""
    load_dotenv()
    default_dataset = resolve_project_path(
        os.getenv("RAGAS_EVAL_DATASET", ""),
        DEFAULT_DATASET_PATH,
    )
    default_ragas_output = resolve_project_path(
        os.getenv("RAGAS_EVAL_OUTPUT", ""),
        DEFAULT_RAGAS_OUTPUT_PATH,
    )
    default_retrieval_output = resolve_project_path(
        os.getenv("RAGAS_RETRIEVAL_OUTPUT", ""),
        DEFAULT_RETRIEVAL_OUTPUT_PATH,
    )

    parser = argparse.ArgumentParser(description="运行当前项目的 RAGAS 评估与召回检查。")
    parser.add_argument(
        "--mode",
        choices=["all", "ragas", "retrieval"],
        default="all",
        help="all 同时跑 RAGAS 和召回报告；ragas 只跑 RAGAS；retrieval 只跑召回。",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="JSONL 评估集路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_ragas_output,
        help="RAGAS CSV 结果输出路径。",
    )
    parser.add_argument(
        "--retrieval-output",
        type=Path,
        default=default_retrieval_output,
        help="召回报告 CSV 输出路径。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只评估前 N 条，0 表示全部评估。",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主入口。"""
    load_dotenv()
    args = parse_args()
    items = load_eval_items(args.dataset)
    if args.limit > 0:
        items = items[:args.limit]

    if args.mode == "retrieval":
        retrieval_rows = build_retrieval_rows(items)
        add_ragas_id_metrics_to_retrieval_rows(retrieval_rows)
        save_rows_csv(retrieval_rows, args.retrieval_output)
        return

    ensure_ragas_available()
    ragas_rows = build_ragas_rows(items)
    if args.mode == "all":
        add_retrieval_warnings(ragas_rows)
        save_rows_csv(ragas_rows, args.retrieval_output)

    result = evaluate_with_ragas(ragas_rows)
    print(result)
    save_ragas_result(result, args.output)


if __name__ == "__main__":
    main()
