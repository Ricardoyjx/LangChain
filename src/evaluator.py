"""金融 RAG 评估模块：衡量答案的准确性与可靠性。

核心指标:
    - accuracy: 金融数字准确率（回答中的数值与标准答案的偏差）
    - faithfulness: 忠实度（回答是否基于检索到的上下文，而非幻觉）
    - overall: 综合评分

用法示例:
    evaluator = FinancialEvaluator()
    result = evaluator.evaluate(
        query="美的2024年营收是多少？",
        answer="约3720亿元",
        ground_truth="美的2024年营收为3721亿元",
    )
    print(result)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvaluationResult:
    """评估结果。"""
    accuracy: float = 0.0          # 数字准确率 0~1
    faithfulness: float = 0.0      # 忠实度 0~1
    overall: float = 0.0           # 综合评分
    details: Dict[str, str] = field(default_factory=dict)  # 详细说明


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> List[float]:
    """从文本中提取所有数值（支持亿/万单位）。"""
    # 匹配数字 + 可选单位
    pattern = r"(\d+(?:\.\d+)?)\s*(亿|万|千|百)?"
    matches = re.findall(pattern, text)
    results = []
    for num, unit in matches:
        val = float(num)
        if unit == "亿":
            val *= 1e8
        elif unit == "万":
            val *= 1e4
        elif unit == "千":
            val *= 1000
        elif unit == "百":
            val *= 100
        results.append(val)
    return results


# ---------------------------------------------------------------------------
# 评估器
# ---------------------------------------------------------------------------

class FinancialEvaluator:
    """金融 RAG 答案评估器。"""

    def __init__(self, tolerance: float = 0.05):
        """
        Args:
            tolerance: 数字匹配的容忍偏差比例（默认 5%）。
                       例如 tolerance=0.05 表示偏差在 5% 以内算正确。
        """
        self.tolerance = tolerance

    def evaluate(
        self,
        query: str,
        answer: str,
        ground_truth: Optional[str] = None,
        contexts: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """对问答结果进行评估。

        Args:
            query: 用户问题。
            answer: 模型生成的回答。
            ground_truth: 标准答案（可选，用于计算 accuracy）。
            contexts: 检索到的上下文片段列表（可选，用于计算 faithfulness）。

        Returns:
            EvaluationResult 包含各项评分。
        """
        result = EvaluationResult()

        # 1. 数字准确率评估
        if ground_truth:
            result.accuracy = self._eval_accuracy(answer, ground_truth)
            result.details["accuracy"] = (
                f"数字准确率: {result.accuracy:.1%}"
            )

        # 2. 忠实度评估
        if contexts:
            result.faithfulness = self._eval_faithfulness(answer, contexts)
            result.details["faithfulness"] = (
                f"忠实度: {result.faithfulness:.1%}"
            )

        # 3. 综合评分（加权平均）
        scores = []
        weights = []
        if ground_truth:
            scores.append(result.accuracy)
            weights.append(0.6)
        if contexts:
            scores.append(result.faithfulness)
            weights.append(0.4)

        if scores:
            result.overall = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

        return result

    def _eval_accuracy(self, answer: str, ground_truth: str) -> float:
        """基于数字偏差评估准确率。"""
        ans_nums = _extract_numbers(answer)
        gt_nums = _extract_numbers(ground_truth)

        if not gt_nums:
            return 1.0  # 标准答案不含数字，跳过评估

        if not ans_nums:
            return 0.0  # 标准答案有数字但回答没有

        correct = 0
        for gt in gt_nums:
            for ans in ans_nums:
                if abs(ans - gt) / max(abs(gt), 1) <= self.tolerance:
                    correct += 1
                    break

        return min(correct / len(gt_nums), 1.0)

    def _eval_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """基于回答与上下文的文本重叠评估忠实度。

        简单实现：检查回答中的关键词是否出现在上下文中。
        更精确的实现应使用 LLM 判断或 NLI 模型。
        """
        combined = " ".join(contexts).lower()
        # 提取 2~4 个中文词的短语作为关键证据
        phrases = re.findall(r"[\u4e00-\u9fff]{2,}", answer)

        if not phrases:
            return 1.0

        supported = sum(1 for p in phrases if p in combined)
        return supported / len(phrases)
