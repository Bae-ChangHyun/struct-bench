"""
전체 벤치마크: 3가지 스키마+프롬프트 조합 × 모든 프레임워크 × 모드

조합:
  A) description 있는 스키마 + 필드 설명 없는 프롬프트  (description만 의존)
  B) description 없는 스키마 + 필드 설명 없는 프롬프트  (아무 설명 없음)
  C) description 없는 스키마 + 필드 설명 있는 프롬프트  (프롬프트로만 설명)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.frameworks.registry import FrameworkRegistry
from app.schemas import get_schema
from app.prompts.loader import load_prompt

import app.frameworks  # noqa: F401

# ── 설정 ──
BASE_URL = "http://118.38.20.101:8001/v1"
MODEL = "openai/gpt-oss-120b"
API_KEY = "dummy"
SAMPLE_RESUME = Path(__file__).parent / "sample_resume.md"

# 3가지 조합 정의
COMBINATIONS = [
    {
        "id": "A_desc_schema",
        "label": "Schema(desc) + Prompt(no fields)",
        "schema": "Resume",           # description 있음
        "prompt": "extract_resume_minimal",  # 필드 설명 없음
    },
    {
        "id": "B_no_desc_no_prompt",
        "label": "Schema(no desc) + Prompt(no fields)",
        "schema": "ResumeNoDesc",      # description 없음
        "prompt": "extract_resume_minimal",  # 필드 설명 없음
    },
    {
        "id": "C_no_desc_rich_prompt",
        "label": "Schema(no desc) + Prompt(rich fields)",
        "schema": "ResumeNoDesc",      # description 없음
        "prompt": "extract_resume_vllm",     # 필드 설명 있음
    },
]

# 정답 (복잡한 이력서 기준)
EXPECTED = {
    "name_keywords": ["박지영", "Jiyoung"],
    "email": "jiyoung.park92@gmail.com",
    "phone_keyword": "9876-5432",
    "min_skills": 8,
    "work_count": 4,
    "companies": ["인텔리전스랩", "IntelligenceLab", "네이버", "Naver", "카카오브레인", "Kakao Brain", "데이터릭스", "Datarix"],
    "edu_count": 2,
    "institutions": ["KAIST", "한국과학기술원", "연세대", "Yonsei"],
    "degrees": ["석사", "Master", "학사", "Bachelor"],
}


def score_result(data: dict | None) -> dict:
    if not data:
        return {"total": 0, "max": 100, "pct": 0.0, "details": "No data"}

    scores = {}
    max_score = 0

    # 1. 이름 (10점)
    max_score += 10
    name = data.get("name", "")
    scores["name"] = 10 if any(k in name for k in EXPECTED["name_keywords"]) else 0

    # 2. 이메일 (10점)
    max_score += 10
    scores["email"] = 10 if data.get("email") == EXPECTED["email"] else 0

    # 3. 전화번호 (5점)
    max_score += 5
    phone = data.get("phone", "") or ""
    scores["phone"] = 5 if EXPECTED["phone_keyword"] in phone else 0

    # 4. 요약 (5점) - 비어있지 않으면 OK
    max_score += 5
    summary = data.get("summary", "")
    scores["summary"] = 5 if len(summary) > 20 else 0

    # 5. 스킬 (10점)
    max_score += 10
    skills = data.get("skills", [])
    if len(skills) >= 15:
        scores["skills"] = 10
    elif len(skills) >= EXPECTED["min_skills"]:
        scores["skills"] = 7
    elif len(skills) >= 3:
        scores["skills"] = 3
    else:
        scores["skills"] = 0

    # 6. 경력 수 (10점)
    max_score += 10
    work = data.get("work_experience", [])
    if len(work) == EXPECTED["work_count"]:
        scores["work_count"] = 10
    elif len(work) >= 3:
        scores["work_count"] = 7
    elif len(work) >= 2:
        scores["work_count"] = 4
    else:
        scores["work_count"] = 0

    # 7. 회사명 매칭 (15점)
    max_score += 15
    company_text = " ".join(w.get("company", "") for w in work)
    # 4개 회사, 각 회사당 한국어 or 영어 매칭
    company_matches = 0
    company_pairs = [
        ["인텔리전스랩", "IntelligenceLab"],
        ["네이버", "Naver"],
        ["카카오브레인", "Kakao Brain", "카카오"],
        ["데이터릭스", "Datarix"],
    ]
    for pair in company_pairs:
        if any(k.lower() in company_text.lower() for k in pair):
            company_matches += 1
    scores["companies"] = int(company_matches / 4 * 15)

    # 8. 경력 설명 품질 (5점)
    max_score += 5
    desc_lens = [len(w.get("description", "")) for w in work]
    non_empty_descs = sum(1 for d in desc_lens if d > 10)
    scores["work_desc"] = 5 if non_empty_descs >= 3 else int(non_empty_descs / 3 * 5)

    # 9. 날짜 형식 YYYY-MM (10점)
    max_score += 10
    date_ok = 0
    date_total = 0
    for w in work:
        sd = w.get("start_date", "") or ""
        if sd:
            date_total += 1
            if len(sd) >= 7 and "-" in sd[:8]:
                date_ok += 1
    scores["dates"] = 10 if date_total > 0 and date_ok == date_total else int(date_ok / max(date_total, 1) * 10)

    # 10. 학력 수 (5점)
    max_score += 5
    edu = data.get("education", [])
    scores["edu_count"] = 5 if len(edu) == EXPECTED["edu_count"] else (3 if len(edu) >= 1 else 0)

    # 11. 학교명 매칭 (10점)
    max_score += 10
    inst_text = " ".join(e.get("institution", "") for e in edu)
    inst_pairs = [["KAIST", "한국과학기술원"], ["연세대", "Yonsei"]]
    inst_matches = sum(1 for pair in inst_pairs if any(k in inst_text for k in pair))
    scores["institutions"] = int(inst_matches / 2 * 10)

    # 12. 학위 매칭 (5점)
    max_score += 5
    degree_text = " ".join(e.get("degree", "") for e in edu)
    degree_pairs = [["석사", "Master"], ["학사", "Bachelor"]]
    deg_matches = sum(1 for pair in degree_pairs if any(k.lower() in degree_text.lower() for k in pair))
    scores["degrees"] = int(deg_matches / 2 * 5)

    total = sum(scores.values())
    return {"total": total, "max": max_score, "pct": round(total / max_score * 100, 1), "details": scores}


async def run_single(fw: str, mode: str, schema_name: str, prompt_name: str, text: str) -> dict:
    adapter_cls = FrameworkRegistry.get(fw)
    adapter = adapter_cls(model=MODEL, base_url=BASE_URL, api_key=API_KEY, mode=mode)
    schema_class = get_schema(schema_name)
    prompt = load_prompt(prompt_name)

    result = await adapter.run(text=text, schema_class=schema_class, system_prompt=prompt.system_prompt)

    if result.success and result.data:
        score = score_result(result.data)
        return {
            "success": True,
            "latency_ms": round(result.latency_ms, 1),
            "score_pct": score["pct"],
            "score_details": score["details"],
            "error": None,
        }

    return {
        "success": False,
        "latency_ms": round(result.latency_ms, 1),
        "score_pct": 0,
        "score_details": None,
        "error": (result.error or "")[:200],
    }


async def main():
    text = SAMPLE_RESUME.read_text()

    # 프레임워크 × 모드 조합
    fw_modes: list[tuple[str, str]] = []
    for fw_name in FrameworkRegistry.list_names():
        adapter_cls = FrameworkRegistry.get(fw_name)
        for mode in adapter_cls.supported_modes:
            if mode == "default" and len(adapter_cls.supported_modes) > 1:
                continue
            fw_modes.append((fw_name, mode))

    total_tests = len(fw_modes) * len(COMBINATIONS)

    print(f"\n{'='*80}")
    print(f" Structured Output Benchmark — Description & Prompt Impact")
    print(f" Model: {MODEL} | Server: {BASE_URL}")
    print(f" Frameworks × Modes: {len(fw_modes)} | Combinations: {len(COMBINATIONS)}")
    print(f" Total test cases: {total_tests}")
    print(f"{'='*80}\n")

    all_results = []
    test_num = 0

    for combo in COMBINATIONS:
        print(f"\n--- {combo['label']} (schema={combo['schema']}, prompt={combo['prompt']}) ---")
        for fw, mode in fw_modes:
            test_num += 1
            label = f"{fw}/{mode}"
            print(f"[{test_num}/{total_tests}] {label:35s}", end=" ", flush=True)
            try:
                r = await run_single(fw, mode, combo["schema"], combo["prompt"], text)
                r["framework"] = fw
                r["mode"] = mode
                r["combination"] = combo["id"]
                r["combo_label"] = combo["label"]
                all_results.append(r)
                if r["success"]:
                    print(f"OK  {r['latency_ms']:>8.0f}ms  score: {r['score_pct']:>5.1f}%")
                else:
                    print(f"FAIL {r['latency_ms']:>7.0f}ms  {r['error'][:50]}")
            except Exception as e:
                print(f"ERROR {str(e)[:50]}")
                all_results.append({
                    "framework": fw, "mode": mode, "combination": combo["id"],
                    "combo_label": combo["label"], "success": False,
                    "latency_ms": 0, "score_pct": 0, "error": str(e)[:200],
                })

    # ── 결과 요약 테이블 ──
    print(f"\n{'='*80}")
    print(f" RESULTS SUMMARY")
    print(f"{'='*80}")

    # 조합별 그룹
    for combo in COMBINATIONS:
        cid = combo["id"]
        subset = [r for r in all_results if r["combination"] == cid]
        ok_subset = [r for r in subset if r["success"]]

        print(f"\n[{combo['label']}]")
        print(f"  {'Framework':<20} {'Mode':<18} {'Score':>8} {'Latency':>10}")
        print(f"  {'-'*20} {'-'*18} {'-'*8} {'-'*10}")

        for r in sorted(ok_subset, key=lambda x: -x["score_pct"]):
            print(f"  {r['framework']:<20} {r['mode']:<18} {r['score_pct']:>7.1f}% {r['latency_ms']:>9.0f}ms")

        failed = [r for r in subset if not r["success"]]
        for r in failed:
            print(f"  {r['framework']:<20} {r['mode']:<18} {'FAIL':>8} {r['latency_ms']:>9.0f}ms")

        if ok_subset:
            avg = sum(r["score_pct"] for r in ok_subset) / len(ok_subset)
            print(f"  {'AVG':>38} {avg:>7.1f}%")

    # ── 조합별 평균 비교 ──
    print(f"\n{'='*80}")
    print(f" COMBINATION COMPARISON (avg score of successful runs)")
    print(f"{'='*80}")
    for combo in COMBINATIONS:
        cid = combo["id"]
        ok = [r for r in all_results if r["combination"] == cid and r["success"]]
        avg = sum(r["score_pct"] for r in ok) / len(ok) if ok else 0
        cnt = len(ok)
        total = len([r for r in all_results if r["combination"] == cid])
        print(f"  {combo['label']:<50} avg={avg:>5.1f}%  ({cnt}/{total} OK)")

    # JSON 저장
    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
