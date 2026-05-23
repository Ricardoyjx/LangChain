"""金融 RAG 系统主入口。

基于 RAGPipeline 完成文档摄入、索引构建、智能问答全流程。

用法:
    python main.py                              # 交互式问答
    python main.py --query "美的营收多少"        # 单次查询
    python main.py --ingest-only                 # 仅摄入建索引
    python main.py --prompt analysis             # 指定提示词模板
"""

# ── numpy/torch 兼容补丁 ───────────────────────────────────
# PyTorch C++ 层引用了 numpy._globals._signature_descriptor，
# 该属性在 NumPy 1.26+ 中已移除。此处补充定义以抑制告警。
import numpy._globals

if not hasattr(numpy._globals, "_signature_descriptor"):

    class _signature_descriptor:
        pass

    numpy._globals._signature_descriptor = _signature_descriptor
# ────────────────────────────────────────────────────────────

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional

try:
    from requests.exceptions import ConnectionError as RequestsConnectionError
except ImportError:
    # requests not installed; ConnectionError will be caught via builtin
    RequestsConnectionError = ConnectionError

from src.evaluator import FinancialEvaluator
from src.generation.prompt_template import list_templates, format_prompt
from src.pipeline import RAGPipeline, setup_logging

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "vector_db"

DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_PROMPT = "rag"
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------


def init_pipeline(force_reindex: bool = False) -> RAGPipeline:
    """初始化 RAG 管线：优先从磁盘加载已有索引，否则摄入文档后建索引。

    Args:
        force_reindex: 为 True 时强制重新摄入。

    Returns:
        配置好的 RAGPipeline 实例。
    """
    pipeline = RAGPipeline(top_k=DEFAULT_TOP_K)

    if INDEX_DIR.exists() and not force_reindex:
        print(f"发现已有索引: {INDEX_DIR}")
        print("跳过摄入阶段，直接加载索引...")
        print("（如需重新构建，请删除 data/vector_db/ 目录或使用 --force-reindex）")
        pipeline._ingest.load_index(str(INDEX_DIR))
        from src.pipeline import QueryPipeline

        pipeline._query = QueryPipeline(
            vector_store=pipeline._ingest.vector_store,
            bm25_retriever=pipeline._ingest.bm25_retriever,
            llm=pipeline._ingest.llm,
            top_k=DEFAULT_TOP_K,
            rerank=True,
        )
    else:
        if force_reindex:
            print("强制重新摄入文档...")
        else:
            print("未发现已有索引，开始摄入文档...")

        pdf_files = sorted(RAW_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"错误: {RAW_DIR} 目录下未找到 PDF 文件。")
            print(f"请将金融文档放入 {RAW_DIR}/ 目录后重试。")
            sys.exit(1)

        file_paths = [str(f) for f in pdf_files]
        print(f"发现 {len(file_paths)} 个文档:")
        for fp in file_paths:
            size_mb = Path(fp).stat().st_size / 1024 / 1024
            print(f"  - {Path(fp).name} ({size_mb:.1f} MB)")

        result = pipeline.ingest_multiple(file_paths)
        print(
            f"摄入完成：共 {result['chunk_count']} 个文本块，{len(result['tables'])} 个表格"
        )

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        pipeline._ingest.save_index(str(INDEX_DIR))
        print(f"索引已保存至: {INDEX_DIR}/")

    return pipeline


# ---------------------------------------------------------------------------
# 交互式问答
# ---------------------------------------------------------------------------


def print_banner(pipeline: RAGPipeline, prompt_name: str):
    """打印欢迎信息和帮助提示。"""
    chunk_count = pipeline._ingest.chunk_count
    doc_count = len(pipeline._ingest._metadata.get("source_files", []))

    print()
    print("=" * 60)
    print("  金融 RAG 智能问答系统")
    print("=" * 60)
    print(f"  文档: {doc_count} 个 | 文本块: {chunk_count} 个")
    print(f"  模板: {prompt_name}")
    print(f"  LLM:     {DEFAULT_MODEL}")
    print(f"  嵌入:    nomic-embed-text")
    print("-" * 60)
    print("  直接输入问题开始问答")
    _print_help()
    print("=" * 60)
    print()


def _print_help():
    """打印可用命令列表。"""
    print("  命令:")
    print("    /help       显示帮助")
    print("    /prompt     切换提示词模板")
    print("    /history    显示对话历史")
    print("    /eval       评估上一条回答")
    print("    /quit       退出")
    print()


def run_interactive(pipeline: RAGPipeline, initial_prompt: str = DEFAULT_PROMPT):
    """交互式问答主循环。

    Args:
        pipeline: 已初始化的 RAGPipeline 实例。
        initial_prompt: 初始提示词模板名称。
    """
    prompt_name = initial_prompt
    chat_history: List[dict] = []
    last_answer = ""
    last_contexts: List[str] = []
    evaluator = FinancialEvaluator()

    # 获取可用模板列表（排除带 history 的变体）
    all_templates = [
        t["name"] for t in list_templates() if t["name"] != "rag_with_history"
    ]

    print_banner(pipeline, prompt_name)

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not user_input:
            continue

        # ---- 命令处理 ----
        if user_input.startswith("/"):
            cmd = user_input[1:].lower()

            if cmd in ("quit", "exit", "q"):
                print("再见！")
                break

            elif cmd == "help":
                _print_help()

            elif cmd == "prompt":
                _print_prompt_list(all_templates)
                try:
                    choice = input("选择模板 (输入名称或编号): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    continue

                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(all_templates):
                        prompt_name = all_templates[idx]
                    else:
                        print(f"编号无效，请输入 1-{len(all_templates)}")
                        continue
                elif choice in all_templates:
                    prompt_name = choice
                else:
                    print(f"未知模板: {choice}")
                    continue

                print(f"已切换至模板: {prompt_name}")
                # 重置对话历史（切换模板意味着新上下文）
                print("（对话历史已重置）")

            elif cmd == "history":
                if not chat_history:
                    print("暂无对话记录。")
                else:
                    print(f"对话历史 (共 {len(chat_history) // 2} 轮):")
                    for i, msg in enumerate(chat_history):
                        role = "你" if msg["role"] == "user" else "助手"
                        preview = msg["content"][:100]
                        print(
                            f"  [{role}] {preview}{'...' if len(msg['content']) > 100 else ''}"
                        )
                    print()

            elif cmd == "eval":
                if not last_answer:
                    print("尚无回答可供评估。")
                    continue
                if not last_contexts:
                    print("无检索上下文，仅评估数字准确率（需提供标准答案）。")
                    try:
                        gt = input("请输入标准答案: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        continue
                    if gt:
                        result = evaluator.evaluate("", last_answer, ground_truth=gt)
                        _print_eval_result(result)
                else:
                    print("评估上一条回答...")
                    try:
                        gt = input("标准答案 (可选，直接回车跳过): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        continue
                    result = evaluator.evaluate(
                        query=(
                            chat_history[-2]["content"]
                            if len(chat_history) >= 2
                            else ""
                        ),
                        answer=last_answer,
                        ground_truth=gt if gt else None,
                        contexts=last_contexts,
                    )
                    _print_eval_result(result)

            else:
                print(f"未知命令: {cmd}")
                _print_help()

            continue

        # ---- 问答 ----
        print("助手: ", end="", flush=True)

        full_response = ""
        try:
            for chunk in pipeline.stream(
                user_input,
                prompt_name=prompt_name,
                chat_history=chat_history if chat_history else None,
            ):
                print(chunk, end="", flush=True)
                full_response += chunk
            print()
        except (RequestsConnectionError, ConnectionResetError) as e:
            print(f"\n[错误] Ollama 连接断开: {e}")
            print("请确认 Ollama 服务正在运行: ollama serve")
            continue
        except Exception as e:
            print(f"\n[错误] 生成回答时出错: {e}")
            continue

        # 更新对话历史与缓存
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": full_response})
        last_answer = full_response

        # 获取检索上下文用于后续评估
        try:
            retrieved = pipeline._query._hybrid_retrieve(user_input)
            last_contexts = [d.page_content for d in retrieved]
        except Exception:
            last_contexts = []


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _print_prompt_list(templates: List[str]):
    """打印可用的提示词模板列表。"""
    print("可用提示词模板:")
    descriptions = {
        "rag": "RAG 问答（默认）",
        "cite": "带引用溯源的 RAG 回答",
        "analysis": "金融分析（观点→数据→风险）",
        "extract": "金融信息提取",
        "classify": "查询意图分类",
        "verify": "数值一致性校验",
    }
    for i, name in enumerate(templates, 1):
        desc = descriptions.get(name, "")
        marker = " *" if name == DEFAULT_PROMPT else ""
        print(f"  {i}. {name:15s} – {desc}{marker}")


def _print_eval_result(result):
    """打印评估结果。"""
    from src.evaluator import EvaluationResult

    print(f"  数字准确率: {result.accuracy:.1%}")
    print(f"  忠实度:     {result.faithfulness:.1%}")
    print(f"  综合评分:   {result.overall:.1%}")
    for key, val in result.details.items():
        print(f"  [{key}] {val}")


# ---------------------------------------------------------------------------
# 单次查询模式
# ---------------------------------------------------------------------------


def run_single_query(pipeline: RAGPipeline, query: str, prompt_name: str):
    """执行单次查询并打印结果。"""
    print(f"问题: {query}")
    print(f"模板: {prompt_name}")
    print(f"{'=' * 50}")
    print()

    try:
        answer = pipeline.query(query, prompt_name=prompt_name)
        print(f"回答:\n{answer}")
    except (RequestsConnectionError, ConnectionResetError) as e:
        print(f"[错误] 查询时 Ollama 连接失败: {e}")
        print("请确认 Ollama 服务正在运行: ollama serve")
    except Exception as e:
        print(f"[错误] 查询失败: {e}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="金融 RAG 智能问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python main.py                     # 交互模式\n"
            '  python main.py --query "营收多少"  # 单次查询\n'
            "  python main.py --ingest-only        # 仅建索引\n"
            "  python main.py --force-reindex      # 强制重建\n"
        ),
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="单次查询模式：直接回答问题后退出",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=DEFAULT_PROMPT,
        choices=["rag", "cite", "analysis", "extract", "classify", "verify"],
        help=f"提示词模板（默认: {DEFAULT_PROMPT}）",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="仅执行文档摄入与索引构建，退出",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="忽略已有索引，强制重新摄入",
    )
    return parser.parse_args()


# ===================================================================
# 程序入口
# ===================================================================


def main():
    setup_logging()

    args = parse_args()

    # 检查数据目录
    if not RAW_DIR.exists():
        print(f"[错误] 目录不存在: {RAW_DIR}/")
        print(f"请先创建目录并将金融研报/年报等 PDF 文件放入其中:")
        print(f"  mkdir -p {RAW_DIR}/")
        sys.exit(1)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[错误] 在 {RAW_DIR}/ 下未找到 PDF 文档。")
        print(f"请将 PDF 文件放入 {RAW_DIR}/ 目录后重试。")
        print(f"  cp /path/to/report.pdf {RAW_DIR}/")
        sys.exit(1)

    # 初始化管线（摄入或加载索引）
    try:
        pipeline = init_pipeline(force_reindex=args.force_reindex)
    except (RequestsConnectionError, ConnectionResetError) as e:
        print(f"[错误] Ollama 服务连接失败: {e}")
        print()
        print("可能的原因和解决方法:")
        print("  1. Ollama 服务未启动   -> 执行: ollama serve")
        print("  2. 端口不对             -> 检查 http://localhost:11434 是否可访问")
        print("  3. 模型未拉取           -> 执行: ollama pull qwen3.5:9b")
        print("  4. 如需远程 Ollama     -> 修改 pipeline.py 中的 _DEFAULT_OLLAMA_URL")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[错误] 文件未找到: {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"[错误] 缺少依赖库: {e}")
        print("请安装所需包:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except ValueError as e:
        print(f"[错误] 配置或参数错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 管线初始化发生未知异常: {e}")
        print()
        print("详细堆栈:")
        traceback.print_exc()
        sys.exit(1)

    # 仅摄入模式
    if args.ingest_only:
        print("摄入完成，索引已保存。")
        return

    # 单次查询模式
    if args.query:
        run_single_query(pipeline, args.query, args.prompt)
        return

    # 交互模式
    run_interactive(pipeline, initial_prompt=args.prompt)


if __name__ == "__main__":
    main()
