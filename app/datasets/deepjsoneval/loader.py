"""DeepJSONEval 데이터셋 로더.

JSONL 파일에서 샘플을 로드하고 category/depth로 필터링한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.datasets.deepjsoneval.downloader import ensure_dataset


def load_samples(
    max_samples: int = 50,
    categories: list[str] | None = None,
    min_depth: int | None = None,
    max_depth: int | None = None,
    per_category: int | None = None,
) -> list[dict]:
    """데이터셋 로드 및 필터링.

    Args:
        max_samples: 최대 반환 샘플 수
        categories: 필터링할 카테고리 목록 (None이면 전체)
        min_depth: 최소 depth 필터
        max_depth: 최대 depth 필터
        per_category: 카테고리당 최대 샘플 수 (설정 시 균등 샘플링)

    Returns:
        list of dict: {id, text, schema, ground_truth, category, true_depth}
    """
    jsonl_path = ensure_dataset()
    samples = []
    category_counts: dict[str, int] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            category = row.get("category", "unknown")
            depth = row.get("true_depth", 0)

            # 필터링
            if categories and category not in categories:
                continue
            if min_depth is not None and depth < min_depth:
                continue
            if max_depth is not None and depth > max_depth:
                continue
            if per_category is not None:
                cnt = category_counts.get(category, 0)
                if cnt >= per_category:
                    continue
                category_counts[category] = cnt + 1

            schema_raw = row.get("schema", "{}")
            schema_dict = json.loads(schema_raw) if isinstance(schema_raw, str) else schema_raw
            gt_raw = row.get("json", "{}")
            gt_dict = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw

            samples.append({
                "id": f"djeval_{idx:04d}",
                "text": row.get("text", ""),
                "schema": schema_dict,
                "ground_truth": gt_dict,
                "category": category,
                "true_depth": depth,
            })

            if len(samples) >= max_samples:
                break

    return samples
