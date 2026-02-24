"""
Career 스키마 벤치마크: 3가지 스키마+프롬프트 조합 × 모든 프레임워크 × 모드

조합:
  A) description 있는 스키마(MainInfo) + 필드 설명 없는 프롬프트
  B) description 없는 스키마(MainInfoNoDesc) + 필드 설명 없는 프롬프트
  C) description 없는 스키마(MainInfoNoDesc) + 필드 설명 있는 프롬프트
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
SAMPLE_RESUME = Path(__file__).parent / "sample_career_resume.md"

# 3가지 조합 정의
COMBINATIONS = [
    {
        "id": "A_desc_schema",
        "label": "Schema(desc) + Prompt(minimal)",
        "schema": "MainInfo",
        "prompt": "extract_career_minimal",
    },
    {
        "id": "B_no_desc_no_prompt",
        "label": "Schema(no desc) + Prompt(minimal)",
        "schema": "MainInfoNoDesc",
        "prompt": "extract_career_minimal",
    },
    {
        "id": "C_no_desc_rich_prompt",
        "label": "Schema(no desc) + Prompt(rich)",
        "schema": "MainInfoNoDesc",
        "prompt": "extract_career_rich",
    },
]


# ── Ground Truth ──
GROUND_TRUTH = {
    # 경력 3건 (비공개 1건 포함)
    "career_count": 3,
    "companies": [
        {"name_keywords": ["스마트솔루션즈", "SmartSolutions"], "is_private": False,
         "position": "수석연구원", "department": "AI연구소", "currently_employed": True,
         "start": "2021-09", "employment_type": "정규직", "location_keyword": "강남",
         "salary_keyword": "8,500"},
        {"name_keywords": ["네이버", "Naver"], "is_private": False,
         "position_keywords": ["시니어", "Senior"], "department_keywords": ["검색", "Search"],
         "start": "2018-03", "end": "2021-08", "employment_type": "정규직",
         "reason_keyword": "커리어"},
        {"is_private": True, "position_keywords": ["주니어", "Junior"],
         "start": "2016-07", "end": "2018-02", "employment_type": "계약직",
         "reason_keyword": "계약"},
    ],

    # 활동/경험 3건
    "activity_count": 3,
    "activities": [
        {"type": "인턴", "name_keyword": "삼성SDS", "org_keyword": "삼성"},
        {"type_keywords": ["대외활동", "교육프로그램", "부트캠프"],
         "name_keyword": "머신러닝", "org_keyword": "Google"},
        {"type": "봉사", "name_keyword": "코딩", "org_keyword": "코드클럽"},
    ],

    # 해외경험 2건
    "overseas_count": 2,
    "overseas": [
        {"type": "교환학생", "country_keywords": ["미국", "USA", "US"],
         "detail_keyword": "Berkeley"},
        {"type": "어학연수", "country_keywords": ["호주", "Australia"],
         "detail_keyword": "UNSW"},
    ],

    # 어학 5건 (회화 2 + 공인시험 3)
    "language_count": 5,
    "conversation_skills": [
        {"language": "영어", "level_keyword": "비즈니스"},
        {"language_keyword": "일본어", "level_keyword": "일상"},
    ],
    "test_skills": [
        {"test": "TOEIC", "score": "935", "date": "2023-03"},
        {"test_keywords": ["JLPT", "N2"], "date": "2022-07"},
        {"test": "OPIc", "score_keyword": "AL", "date": "2023-05"},
    ],

    # 자격증 4건
    "certificate_count": 4,
    "certificates": [
        {"name": "정보처리기사", "issuer_keyword": "산업인력공단", "date": "2016-05"},
        {"name_keyword": "AWS", "issuer_keyword": "Amazon", "date": "2022-11"},
        {"name_keyword": "Tensorflow", "issuer_keyword": "Google", "date": "2021-03"},
        {"name_keyword": "SQLD", "date": "2017-09"},
    ],

    # 수상 4건
    "award_count": 4,
    "awards": [
        {"name_keyword": "해커톤", "org_keyword": "네이버", "date": "2020-11"},
        {"name_keyword": "인턴", "org_keyword": "삼성", "date": "2016-06"},
        {"name_keyword": "ACM", "date": "2015-11"},
        {"name_keywords": ["캐글", "Kaggle"], "date": "2021-08"},
    ],

    # 병역
    "military": {
        "status": "군필",
        "branch": "육군",
        "rank": "병장",
        "start": "2011-03",
        "end": "2013-01",
        "is_veteran": False,
        "is_disabled": False,
    },

    # SNS
    "sns_count": 3,
    "sns_keywords": ["github", "linkedin", "tistory"],
}


def score_result(data: dict | None) -> dict:
    """Ground truth 기반 채점 (총 100점)"""
    if not data:
        return {"total": 0, "max": 100, "pct": 0.0, "details": "No data"}

    scores = {}
    max_score = 0

    # ─── 1. 경력 (30점) ───
    careers = data.get("careers", [])

    # 1-1. 경력 수 (5점)
    max_score += 5
    if len(careers) == GROUND_TRUTH["career_count"]:
        scores["career_count"] = 5
    elif len(careers) >= 2:
        scores["career_count"] = 3
    else:
        scores["career_count"] = 0

    # 1-2. 회사명 매칭 (5점)
    max_score += 5
    company_text = " ".join(c.get("company_name", "") or "" for c in careers)
    gt_companies = GROUND_TRUTH["companies"]
    company_matches = 0
    for gt in gt_companies:
        if gt.get("is_private"):
            # 비공개 회사: is_company_private 가 True인 경력이 있는지
            if any(c.get("is_company_private") is True for c in careers):
                company_matches += 1
        else:
            if any(k.lower() in company_text.lower() for k in gt["name_keywords"]):
                company_matches += 1
    scores["companies"] = int(company_matches / 3 * 5)

    # 1-3. 재직중 감지 (3점)
    max_score += 3
    currently_employed = any(c.get("is_currently_employed") is True for c in careers)
    scores["currently_employed"] = 3 if currently_employed else 0

    # 1-4. 날짜 정확도 (5점)
    max_score += 5
    date_matches = 0
    date_total = 0
    for gt in gt_companies:
        date_total += 1
        career_dates = " ".join(
            (c.get("start_date", "") or "") + " " + (c.get("end_date", "") or "")
            for c in careers
        )
        if gt["start"] in career_dates:
            date_matches += 1
    scores["career_dates"] = int(date_matches / max(date_total, 1) * 5)

    # 1-5. 업무 상세 (5점)
    max_score += 5
    work_details = [c.get("work_details", "") or "" for c in careers]
    non_empty = sum(1 for d in work_details if len(d) > 20)
    scores["work_details"] = 5 if non_empty >= 3 else int(non_empty / 3 * 5)

    # 1-6. 고용형태/근무지역/연봉/이직사유 (7점)
    max_score += 7
    meta_score = 0
    # 고용형태 매칭
    emp_types = " ".join(c.get("employment_type", "") or "" for c in careers)
    if "정규직" in emp_types:
        meta_score += 2
    # 근무지역
    locations = " ".join(c.get("work_location", "") or "" for c in careers)
    if "강남" in locations or "서울" in locations:
        meta_score += 1
    # 연봉
    salaries = " ".join(c.get("annual_salary", "") or "" for c in careers)
    if "8" in salaries or "8500" in salaries or "8,500" in salaries:
        meta_score += 2
    # 이직사유
    reasons = " ".join(c.get("reason_for_leaving", "") or "" for c in careers)
    if "커리어" in reasons or "성장" in reasons:
        meta_score += 1
    if "계약" in reasons:
        meta_score += 1
    scores["career_meta"] = meta_score

    # ─── 2. 활동/경험 (10점) ───
    activities = data.get("activity_experiences", [])
    max_score += 10
    act_score = 0

    # 활동 수
    if len(activities) == GROUND_TRUTH["activity_count"]:
        act_score += 3
    elif len(activities) >= 2:
        act_score += 1

    # 활동 내용 매칭
    act_text = " ".join(
        (a.get("activity_type", "") or "") + " " +
        (a.get("activity_name", "") or "") + " " +
        (a.get("organization", "") or "") + " " +
        (a.get("details", "") or "")
        for a in activities
    )
    for gt in GROUND_TRUTH["activities"]:
        kw = gt.get("name_keyword", "")
        if kw and kw.lower() in act_text.lower():
            act_score += 2
        elif gt.get("org_keyword", "").lower() in act_text.lower():
            act_score += 1
    scores["activities"] = min(act_score, 10)

    # ─── 3. 해외경험 (8점) ───
    overseas = data.get("overseas_experiences", [])
    max_score += 8
    ovs_score = 0

    if len(overseas) == GROUND_TRUTH["overseas_count"]:
        ovs_score += 2
    elif len(overseas) >= 1:
        ovs_score += 1

    ovs_text = " ".join(
        (o.get("experience_type", "") or "") + " " +
        (o.get("country", "") or "") + " " +
        (o.get("details", "") or "")
        for o in overseas
    )
    for gt in GROUND_TRUTH["overseas"]:
        if gt["type"] in ovs_text:
            ovs_score += 1
        if any(k in ovs_text for k in gt["country_keywords"]):
            ovs_score += 1
        if gt["detail_keyword"] in ovs_text:
            ovs_score += 1
    scores["overseas"] = min(ovs_score, 8)

    # ─── 4. 어학능력 (10점) ───
    lang_skills = data.get("language_skills", [])
    max_score += 10
    lang_score = 0

    if len(lang_skills) >= GROUND_TRUTH["language_count"]:
        lang_score += 2
    elif len(lang_skills) >= 3:
        lang_score += 1

    lang_text = " ".join(
        (ls.get("assessment_type", "") or "") + " " +
        (ls.get("language", "") or "") + " " +
        (ls.get("test_name", "") or "") + " " +
        (ls.get("test_score", "") or "") + " " +
        (ls.get("proficiency_level", "") or "")
        for ls in lang_skills
    )
    # 회화 매칭
    for gt in GROUND_TRUTH["conversation_skills"]:
        kw = gt.get("language", gt.get("language_keyword", ""))
        if kw in lang_text:
            lang_score += 1
    # 공인시험 매칭
    for gt in GROUND_TRUTH["test_skills"]:
        test_kw = gt.get("test", "")
        if not test_kw:
            test_kw = gt.get("test_keywords", [""])[0]
        if test_kw.lower() in lang_text.lower():
            lang_score += 1
            # 점수도 있으면 추가
            score_kw = gt.get("score", gt.get("score_keyword", ""))
            if score_kw and score_kw in lang_text:
                lang_score += 0.5
    scores["languages"] = min(int(lang_score), 10)

    # ─── 5. 자격증 (10점) ───
    certs = data.get("certificates", [])
    max_score += 10
    cert_score = 0

    if len(certs) == GROUND_TRUTH["certificate_count"]:
        cert_score += 3
    elif len(certs) >= 3:
        cert_score += 2
    elif len(certs) >= 1:
        cert_score += 1

    cert_text = " ".join(
        (c.get("certificate_name", "") or "") + " " +
        (c.get("issuing_authority", "") or "") + " " +
        (c.get("acquisition_date", "") or "")
        for c in certs
    )
    for gt in GROUND_TRUTH["certificates"]:
        kw = gt.get("name", gt.get("name_keyword", ""))
        if kw and kw.lower() in cert_text.lower():
            cert_score += 1
            if gt.get("date", "") in cert_text:
                cert_score += 0.5
    scores["certificates"] = min(int(cert_score), 10)

    # ─── 6. 수상경력 (8점) ───
    awards = data.get("award_experiences", [])
    max_score += 8
    award_score = 0

    if len(awards) == GROUND_TRUTH["award_count"]:
        award_score += 2
    elif len(awards) >= 2:
        award_score += 1

    award_text = " ".join(
        (a.get("award_name", "") or "") + " " +
        (a.get("organizer", "") or "") + " " +
        (a.get("award_date", "") or "") + " " +
        (a.get("details", "") or "")
        for a in awards
    )
    for gt in GROUND_TRUTH["awards"]:
        kws = gt.get("name_keywords", [gt.get("name_keyword", "")])
        if any(k.lower() in award_text.lower() for k in kws if k):
            award_score += 1
            if gt.get("date", "") in award_text:
                award_score += 0.5
    scores["awards"] = min(int(award_score), 8)

    # ─── 7. 병역사항 (10점) ───
    mil = data.get("employment_military_info")
    max_score += 10
    mil_score = 0

    if mil and isinstance(mil, dict):
        gt_mil = GROUND_TRUTH["military"]
        if mil.get("military_status") == gt_mil["status"]:
            mil_score += 2
        if mil.get("military_branch") == gt_mil["branch"]:
            mil_score += 2
        if mil.get("rank") == gt_mil["rank"]:
            mil_score += 2
        if mil.get("service_start_date") and gt_mil["start"] in str(mil.get("service_start_date", "")):
            mil_score += 1
        if mil.get("service_end_date") and gt_mil["end"] in str(mil.get("service_end_date", "")):
            mil_score += 1
        if mil.get("is_veteran_target") is False:
            mil_score += 1
        if mil.get("is_disabled") is False:
            mil_score += 1
    scores["military"] = mil_score

    # ─── 8. SNS (4점) ───
    sns = data.get("sns")
    max_score += 4
    sns_score = 0

    if sns and isinstance(sns, dict):
        links = sns.get("sns_links", [])
        if isinstance(links, list):
            link_text = " ".join(links).lower()
            for kw in GROUND_TRUTH["sns_keywords"]:
                if kw in link_text:
                    sns_score += 1
            if len(links) >= GROUND_TRUTH["sns_count"]:
                sns_score += 1
    scores["sns"] = min(sns_score, 4)

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
    print(f" Career Schema Benchmark — Description & Prompt Impact")
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

    # ── 카테고리별 상세 비교 ──
    print(f"\n{'='*80}")
    print(f" CATEGORY BREAKDOWN (avg by combination)")
    print(f"{'='*80}")
    categories = ["career_count", "companies", "currently_employed", "career_dates",
                   "work_details", "career_meta", "activities", "overseas",
                   "languages", "certificates", "awards", "military", "sns"]
    header = f"  {'Category':<22}"
    for combo in COMBINATIONS:
        header += f" {combo['id'][:15]:>15}"
    print(header)
    print(f"  {'-'*22}" + f" {'-'*15}" * len(COMBINATIONS))

    for cat in categories:
        row = f"  {cat:<22}"
        for combo in COMBINATIONS:
            cid = combo["id"]
            ok = [r for r in all_results if r["combination"] == cid and r["success"] and r.get("score_details")]
            if ok:
                avg_val = sum(r["score_details"].get(cat, 0) for r in ok) / len(ok)
                row += f" {avg_val:>14.1f}"
            else:
                row += f" {'N/A':>15}"
        print(row)

    # JSON 저장
    output_path = Path(__file__).parent / "career_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
