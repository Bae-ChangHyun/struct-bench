"""DeepJSONEval 범용 채점 함수.

Ground truth JSON과 추출 결과를 JSON Schema 기반으로 재귀적으로 비교.
"""

from __future__ import annotations

from typing import Any


def score_result(
    extracted: dict | None,
    ground_truth: dict,
    schema: dict,
) -> dict:
    """100점 만점 기반 채점.

    Args:
        extracted: LLM이 추출한 dict (None이면 0점)
        ground_truth: 정답 dict
        schema: JSON Schema dict (필드 구조 파악용)

    Returns:
        {"total": float, "max": float, "pct": float, "field_scores": dict}
    """
    if not extracted:
        total_fields = _count_fields(ground_truth)
        return {"total": 0, "max": total_fields, "pct": 0.0, "field_scores": {}}

    field_scores: dict[str, float] = {}
    total, max_total = _score_recursive(
        extracted, ground_truth, schema, "", field_scores
    )

    if max_total == 0:
        pct = 100.0 if total == 0 else 0.0
    else:
        pct = round(total / max_total * 100, 1)

    return {
        "total": round(total, 2),
        "max": round(max_total, 2),
        "pct": pct,
        "field_scores": field_scores,
    }


def _score_recursive(
    extracted: Any,
    ground_truth: Any,
    schema: dict,
    path: str,
    field_scores: dict[str, float],
) -> tuple[float, float]:
    """재귀적으로 필드별 점수 계산. (점수, 최대점수) 반환."""
    typ = schema.get("type", _infer_type(ground_truth))

    if typ == "object":
        return _score_object(extracted, ground_truth, schema, path, field_scores)
    elif typ == "array":
        return _score_array(extracted, ground_truth, schema, path, field_scores)
    else:
        return _score_leaf(extracted, ground_truth, typ, path, field_scores)


def _score_object(
    extracted: Any,
    ground_truth: Any,
    schema: dict,
    path: str,
    field_scores: dict[str, float],
) -> tuple[float, float]:
    """object 타입 비교."""
    if not isinstance(ground_truth, dict):
        return (0, 0)

    ext = extracted if isinstance(extracted, dict) else {}
    properties = schema.get("properties", {})
    total = 0.0
    max_total = 0.0

    for key, gt_val in ground_truth.items():
        child_path = f"{path}.{key}" if path else key
        child_schema = properties.get(key, {})
        ext_val = ext.get(key)
        sc, mx = _score_recursive(ext_val, gt_val, child_schema, child_path, field_scores)
        total += sc
        max_total += mx

    return total, max_total


def _score_array(
    extracted: Any,
    ground_truth: Any,
    schema: dict,
    path: str,
    field_scores: dict[str, float],
) -> tuple[float, float]:
    """array 타입 비교. 순서 기반 매칭."""
    if not isinstance(ground_truth, list) or len(ground_truth) == 0:
        return (0, 0)

    ext_list = extracted if isinstance(extracted, list) else []
    items_schema = schema.get("items", {})
    total = 0.0
    max_total = 0.0

    for i, gt_item in enumerate(ground_truth):
        child_path = f"{path}[{i}]"
        ext_item = ext_list[i] if i < len(ext_list) else None

        if isinstance(gt_item, dict):
            sc, mx = _score_recursive(
                ext_item, gt_item, items_schema, child_path, field_scores
            )
        else:
            sc, mx = _score_leaf(
                ext_item, gt_item,
                items_schema.get("type", _infer_type(gt_item)),
                child_path, field_scores,
            )
        total += sc
        max_total += mx

    return total, max_total


def _score_leaf(
    extracted: Any,
    ground_truth: Any,
    typ: str,
    path: str,
    field_scores: dict[str, float],
) -> tuple[float, float]:
    """리프 필드 비교. 각 리프는 1점 만점."""
    max_score = 1.0

    if extracted is None:
        field_scores[path] = 0.0
        return 0.0, max_score

    score = 0.0

    if typ in ("number", "integer"):
        score = _compare_number(extracted, ground_truth)
    elif typ == "boolean":
        score = 1.0 if bool(extracted) == bool(ground_truth) else 0.0
    elif typ == "string":
        score = _compare_string(extracted, ground_truth)
    else:
        # fallback: string comparison
        score = _compare_string(str(extracted), str(ground_truth))

    field_scores[path] = round(score, 3)
    return score, max_score


def _compare_number(extracted: Any, ground_truth: Any) -> float:
    """숫자 비교. exact match 또는 +-5% tolerance."""
    try:
        ext_val = float(extracted)
        gt_val = float(ground_truth)
    except (TypeError, ValueError):
        return 0.0

    if gt_val == ext_val:
        return 1.0

    if gt_val == 0:
        return 0.0

    rel_error = abs(ext_val - gt_val) / abs(gt_val)
    if rel_error <= 0.05:
        return 0.8
    elif rel_error <= 0.15:
        return 0.4
    return 0.0


def _compare_string(extracted: Any, ground_truth: Any) -> float:
    """문자열 비교. exact match 또는 keyword overlap."""
    ext_str = str(extracted).strip().lower()
    gt_str = str(ground_truth).strip().lower()

    if not gt_str:
        return 1.0 if not ext_str else 0.5

    if ext_str == gt_str:
        return 1.0

    # keyword overlap (단어 단위)
    gt_words = set(gt_str.split())
    ext_words = set(ext_str.split())

    if not gt_words:
        return 1.0

    overlap = gt_words & ext_words
    if not overlap:
        # substring check
        if gt_str in ext_str or ext_str in gt_str:
            return 0.7
        return 0.0

    return len(overlap) / len(gt_words)


def _infer_type(value: Any) -> str:
    """값에서 타입 추론."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _count_fields(data: Any) -> int:
    """dict/list 내 리프 필드 수 세기."""
    if isinstance(data, dict):
        count = 0
        for v in data.values():
            count += _count_fields(v)
        return max(count, 1)
    elif isinstance(data, list):
        count = 0
        for item in data:
            count += _count_fields(item)
        return max(count, 1)
    else:
        return 1
