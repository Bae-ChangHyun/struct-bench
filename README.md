# struct-bench

> LLM에서 구조화된 출력(Structured Output)을 만들기 위한 프레임워크는 Instructor, LangChain, Marvin, PydanticAI 등 다양하게 존재한다.
> **struct-bench**는 이들을 동일한 조건에서 비교 테스트하는 벤치마크 도구이다.

Pydantic 스키마와 프롬프트를 정의하고, 여러 프레임워크에 동일한 입력을 넣어 결과 품질을 Ground Truth 기반으로 정량 비교한다. FastAPI 서버로도 개별 프레임워크를 즉시 테스트할 수 있다.

### 지원 프레임워크

| 프레임워크 | 모드 | 구조화 방식 |
|-----------|------|-----------|
| **Instructor** | tools, json_schema | Tool Calling / JSON Schema |
| **OpenAI Native** | default | JSON Schema (response_format) |
| **LangChain** | json_schema, function_calling | JSON Schema / Tool Calling |
| **Marvin** | default | Tool Calling |
| **PydanticAI** | default | Tool Calling |
| **Mirascope** | default | Tool Calling |
| **Guardrails** | default | litellm 경유 |

---

## 목차

- [왜 struct-bench인가?](#왜-struct-bench인가)
- [실험 배경](#실험-배경)
- [실험 설계](#실험-설계)
  - [테스트 환경](#테스트-환경)
  - [스키마 구조](#스키마-구조)
  - [3가지 실험 조합](#3가지-실험-조합)
  - [프롬프트 설계](#프롬프트-설계)
- [프레임워크별 동작 원리](#프레임워크별-동작-원리)
- [벤치마크 결과](#벤치마크-결과)
  - [종합 결과 매트릭스](#종합-결과-매트릭스)
  - [채점 기준](#채점-기준)
- [결과 분석](#결과-분석)
  - [조합 C (Rich Prompt): 프레임워크 무관하게 수렴](#1-조합-c-rich-prompt--모든-프레임워크가-96로-수렴)
  - [Tool Calling과 Description 전달](#2-tool-calling이-description을-전달하는-프레임워크만-ab에서-높은-점수)
  - [JSON Schema 방식의 한계](#3-json-schema-방식-xgrammar은-ab에서-중간-성능)
  - [저성능 프레임워크 분석](#4-일부-프레임워크는-ab에서-매우-낮은-성능)
  - [Description 필드 효과](#5-description-필드-효과-a-vs-b--미미함)
  - [Literal 타입 문제](#6-literal-타입-문제)
- [결론](#결론)
- [프로젝트 구조](#프로젝트-구조)
- [실행 방법](#실행-방법)

---

## 왜 struct-bench인가?

LLM에서 Pydantic 모델 형태의 구조화된 출력을 얻기 위한 프레임워크가 많아졌다. Instructor, LangChain, Marvin, PydanticAI, Mirascope, Guardrails 등 각각 다른 방식으로 동일한 문제를 풀고 있다. 하지만 **같은 모델, 같은 스키마, 같은 프롬프트를 넣었을 때 과연 결과가 동일한가?**

struct-bench는 이 질문에 답하기 위한 도구이다:

- **동일 조건 비교**: 같은 입력(텍스트, 스키마, 프롬프트)으로 여러 프레임워크를 실행하고 결과를 비교
- **정량 평가**: Ground Truth 기반 100점 만점 채점으로 프레임워크 간 성능 차이를 수치화
- **API 서버**: FastAPI 기반으로 개별 프레임워크를 즉시 테스트 가능
- **확장 가능**: 새 프레임워크는 `BaseFrameworkAdapter`를 상속하고 `@register` 데코레이터만 붙이면 추가

### 핵심 질문

1. 프레임워크마다 Pydantic 스키마의 `Field(description=...)` 을 LLM에 전달하는 방식이 다른가?
2. 프롬프트에 필드 설명을 명시하는 것과 스키마의 description에 의존하는 것 중 무엇이 더 효과적인가?
3. 프레임워크 선택이 최종 결과 품질에 얼마나 영향을 미치는가?

---

## 실험 배경

### 구조화 출력의 3가지 접근법

LLM에서 구조화된 출력을 얻는 방식은 크게 3가지로 나뉜다:

| 방식 | 동작 원리 | 대표 라이브러리 | 출력 보장 |
|------|----------|---------------|----------|
| **Prompting** | 프롬프트에 원하는 형식을 설명하고 LLM이 따르도록 유도 | spacy-llm | 보장 안됨 |
| **Function Calling (Tool Calling)** | Pydantic 스키마를 tool definition에 넣어 전달. LLM이 함수 인자 형태로 응답 | Instructor, Marvin, Mirascope | 거의 보장 |
| **Constrained Token Sampling** | 문법(CFG)으로 제약을 정의하고, 해당 제약을 만족하는 토큰만 샘플링 | Outlines, Guidance, **xgrammar** | 완전 보장 |

각 프레임워크는 이 방식 중 하나 이상을 사용한다. Instructor는 Function Calling과 JSON Schema 모드를 모두 지원하고, OpenAI Native SDK는 `response_format`으로 JSON Schema를 전달하여 xgrammar가 토큰을 제약한다. LangChain 역시 두 방식을 모두 제공한다.

### struct-bench가 주목하는 차이: Description 전달 여부

vLLM은 Constrained Token Sampling(xgrammar)과 Tool Calling 모두를 지원한다. 문제는 이 두 방식이 Pydantic 스키마의 `Field(description=...)` 을 처리하는 방식이 다르다는 것이다:

| 방식 | vLLM에서의 동작 | Description 전달 |
|------|---------------|-----------------|
| **JSON Schema (response_format)** | xgrammar가 `type`, `properties`, `required` 등 **구조적 제약만** 사용 | **description 무시** |
| **Tool Calling** | tool definition에 description 포함하여 LLM에 전달 | **description 전달됨** |

OpenAI SDK는 Pydantic 스키마를 JSON Schema로 변환하여 `response_format`으로 전달하며, 프롬프트에 스키마를 자동 주입하지 않는다. 따라서 vLLM의 xgrammar가 description을 무시하는 환경에서는, JSON Schema 방식을 사용하는 프레임워크에서 스키마의 description이 LLM에 도달하지 않는다. 이 차이가 프레임워크 간 성능 차이를 만드는 핵심 원인이다.

---

## 실험 설계

### 테스트 환경

| 항목 | 값 |
|------|-----|
| vLLM 서버 | `http://118.38.20.101:8001/v1` |
| 모델 | `openai/gpt-oss-120b` |
| 이력서 수 | 10건 (한국 채용 이력서) |
| 프레임워크/모드 수 | 9개 |
| 실험 조합 | 3가지 |
| **총 테스트 건수** | **270건** (10 x 9 x 3) |

### 스키마 구조

`MainInfo` 스키마는 한국 채용 이력서의 복잡한 구조를 반영한 Pydantic 모델이다. 총 8개 섹션, 약 60개 이상의 필드로 구성되어 있으며 `Literal` 타입 제약이 포함된 까다로운 스키마이다.

```
MainInfo
 |-- careers[]                    # 경력 (기업명, 입퇴사일, 재직중여부, 부서, 직책, 담당업무, 연봉, 고용형태 등)
 |-- activity_experiences[]       # 활동/경험 (인턴, 대외활동, 봉사, 부트캠프 등)
 |-- overseas_experiences[]       # 해외경험 (Literal["어학연수","교환학생","워킹홀리데이","유학"])
 |-- language_skills[]            # 어학능력 (Literal["회화능력","공인시험"], 언어, 수준, 시험점수 등)
 |-- certificates[]               # 자격증 (자격명, 발행처, 발행일)
 |-- award_experiences[]          # 수상 (수상명, 주최기관, 수상일)
 |-- employment_military_info     # 취업우대/병역 (Literal["군필","미필","면제","해당없음"], 군별, 계급 등)
 |-- sns                          # 포트폴리오/SNS 링크 목록
```

두 가지 버전의 스키마를 준비하였다:

- **`MainInfo`** (career.py): 모든 필드에 `Field(description="...")` 포함
- **`MainInfoNoDesc`** (career_no_desc.py): description 없이 타입과 기본값만 정의

### 3가지 실험 조합

| 조합 | 스키마 | 프롬프트 | 실험 의도 |
|------|--------|---------|----------|
| **A: Schema(desc) + Prompt(minimal)** | `MainInfo` - description 포함 | 필드 설명 없는 최소 프롬프트 | description 필드에만 의존했을 때의 성능 측정 |
| **B: Schema(no desc) + Prompt(minimal)** | `MainInfoNoDesc` - description 없음 | 필드 설명 없는 최소 프롬프트 | 아무 설명도 없을 때의 기저(baseline) 성능 측정 |
| **C: Schema(no desc) + Prompt(rich)** | `MainInfoNoDesc` - description 없음 | 모든 필드 설명이 포함된 상세 프롬프트 | 프롬프트로만 설명을 제공했을 때의 성능 측정 |

### 프롬프트 설계

**Minimal Prompt** (`extract_career_minimal.yaml`):
```yaml
system_prompt: |
  You are a data extraction assistant.
  Extract structured information from the given resume text.
  Use YYYY-MM format for dates.
  If a field is not present, use null or empty default.
```

**Rich Prompt** (`extract_career_rich.yaml`):
각 섹션과 필드에 대한 상세한 설명을 포함. 예를 들어 `assessment_type`이 `"회화능력"` 또는 `"공인시험"` 중 하나여야 함을 명시하고, 각 필드가 어떤 상황에서 사용되는지까지 기술한 프롬프트이다. 약 80줄 분량의 상세 가이드를 포함한다.

---

## 프레임워크별 동작 원리

### 1. Instructor (tools / json_schema 모드)

```python
client = instructor.from_provider("ollama/model", base_url=...)
result = client.chat.completions.create(
    response_model=schema_class,
    messages=[...],
)
```

`instructor.from_provider()`는 Tool Calling 방식으로 Pydantic 스키마를 tool definition에 넣어 전달한다. description 필드가 tool definition에 포함되므로 LLM이 필드의 의미를 파악할 수 있다. 자동 retry 및 validation이 내장되어 있다. `json_schema` 모드에서는 `response_format` 기반으로 동작한다.

### 2. OpenAI Native

```python
client.chat.completions.parse(
    model=model,
    messages=[...],
    response_format=schema_class,
)
```

OpenAI SDK의 `parse()` 메서드를 사용하며, JSON Schema를 `response_format`으로 전달한다. vLLM의 xgrammar가 구조를 강제하지만, **description 필드를 무시**하고 `type`, `properties`, `required` 등 구조적 제약만 사용한다.

### 3. LangChain (json_schema / function_calling 모드)

```python
llm = ChatOpenAI(model=model, base_url=...)
structured_llm = llm.with_structured_output(schema_class, method="json_schema")
result = await structured_llm.ainvoke(messages)
```

`json_schema` 모드에서는 `response_format` 기반으로 동작하고, `function_calling` 모드에서는 Tool Calling 방식으로 동작한다.

### 4. Marvin

```python
provider = OpenAIProvider(base_url=..., api_key=...)
model = OpenAIModel(model_name, provider=provider)
agent = marvin.Agent(model=model, instructions=system_prompt)
result = agent.run(text, result_type=schema_class)
```

pydantic_ai의 `OpenAIModel`/`OpenAIProvider`로 모델을 주입한 뒤 `marvin.Agent`를 통해 추출한다. 내부적으로 Tool Calling 방식을 사용하므로 description이 전달된다.

### 5. PydanticAI

```python
model = OpenAIModel(model_name, provider=OpenAIProvider(base_url=...))
agent = Agent(model, system_prompt=prompt, output_type=schema_class)
result = await agent.run(text)
```

`Agent`에 `output_type`으로 스키마를 전달하며, Tool Calling 방식으로 동작한다.

### 6. Mirascope

```python
register_provider("ollama", scope="ollama/", base_url=...)

@call("ollama/model", format=schema_class)
def do_extract(resume_text, sys_prompt):
    return f"{sys_prompt}\n\n{resume_text}"
```

`mirascope.llm.call` 데코레이터와 `format=schema_class`를 사용한다. ollama provider로 등록하여 vLLM에 연결하며, Tool Calling 방식으로 동작한다.

### 7. Guardrails

```python
guard = Guard.for_pydantic(output_class=schema_class)
result = guard(
    model="hosted_vllm/model",
    api_base=base_url,
    messages=[...],
)
```

내부적으로 litellm을 사용하며, `hosted_vllm/` provider로 vLLM에 연결한다.

---

## 벤치마크 결과

### 종합 결과 매트릭스

```
  Framework/Mode                       A_desc     B_nodesc       C_rich    Overall
  ------------------------------ ------------ ------------ ------------ ----------
  instructor/tools                     94.6%       94.9%       95.5%     95.0%
  instructor/json_schema               94.9%       94.7%       96.0%     95.2%
  openai/default                       82.5%       85.2%       96.0%     87.9%
  langchain/json_schema             84.0%(1F)       81.1%       96.0%     87.1%
  langchain/function_calling         ALL FAIL     ALL FAIL       96.0%     96.0%
  marvin/default                       93.5%       94.4%       95.8%     94.6%
  pydantic_ai/default                  38.7%    40.5%(2F)       96.0%     59.7%
  mirascope/default                 35.0%(5F)    34.2%(5F)       96.0%     65.3%
  guardrails/default                 7.8%(6F)     6.0%(7F)       96.0%     59.4%

  COMBINATION AVG                   73.6%(22F)    76.0%(24F)    95.9%(0F)
```

> `(NF)` = 10건 중 N건 실패 (파싱 에러 또는 validation 실패). `ALL FAIL` = 10건 모두 실패.

### 채점 기준

Ground Truth 기반 100점 만점 채점 방식을 사용하였다. 12개 카테고리로 나뉘며, 각 항목은 키워드 매칭, 개수 일치, 정확도 등을 종합적으로 평가한다.

| 카테고리 | 배점 | 평가 내용 |
|---------|------|----------|
| 경력 수 | 10점 | 경력 항목 수 일치 여부 |
| 회사명/비공개 매칭 | 10점 | 회사명 키워드 매칭, 비공개 감지 |
| 재직중 감지 | 5점 | `is_currently_employed` 정확도 |
| 날짜 정확도 | 10점 | 입퇴사 날짜 매칭 |
| 업무상세 품질 | 10점 | `work_details` 필드 충실도 |
| 활동/경험 | 10점 | 활동 수 + 키워드 매칭 |
| 해외경험 | 5점 | 해외경험 수 + 키워드 매칭 |
| 어학능력 | 10점 | 어학 항목 수 + 시험/점수 키워드 |
| 자격증 | 10점 | 자격증 수 + 명칭 키워드 |
| 수상 | 5점 | 수상 수 + 키워드 매칭 |
| 병역 | 10점 | 병역상태, 군별, 계급 정확도 |
| SNS | 5점 | SNS 링크 수 + URL 키워드 |

---

## 결과 분석

### 1. 조합 C (Rich Prompt) : 모든 프레임워크가 ~96%로 수렴

가장 두드러진 결과는, 프롬프트에 필드 설명을 상세히 넣은 조합 C에서 **모든 프레임워크가 95.5~96.0%로 수렴**했다는 점이다.

- Guardrails: 6~8% -> **96%** (12배 이상 향상)
- Mirascope: 34~35% -> **96%** (약 3배 향상)
- PydanticAI: 38~40% -> **96%** (약 2.5배 향상)
- LangChain function_calling: ALL FAIL -> **96%** (실패에서 성공으로)

프롬프트에 필드 설명을 명시하면 프레임워크 간 성능 차이가 사실상 사라진다. 이는 LLM이 "무엇을 추출해야 하는가"에 대한 정보를 프롬프트에서 직접 얻을 수 있기 때문이다.

### 2. Tool Calling이 Description을 전달하는 프레임워크만 A/B에서 높은 점수

프롬프트에 설명이 없는 조합 A/B에서도 높은 점수를 유지한 프레임워크들이 있다:

- **Instructor** (94.6~95.5%): `from_provider`가 tool definition에 description을 포함하여 전달
- **Marvin** (93.5~94.4%): pydantic_ai 기반 Tool Calling으로 description 전달

이 프레임워크들은 스키마의 description만으로도 높은 성능을 보였다. 공통점은 **Tool Calling 방식으로 description을 LLM에 직접 전달**한다는 것이다.

### 3. JSON Schema 방식 (xgrammar)은 A/B에서 중간 성능

- **OpenAI Native**: 82.5~85.2%
- **LangChain json_schema**: 81.1~84.0%

이 프레임워크들은 `response_format`으로 JSON Schema를 전달하지만, vLLM의 xgrammar가 description을 무시하기 때문에 구조만 강제되고 의미 정보가 부족하다. 그 결과 A/B에서 중간 수준의 성능에 머문다.

### 4. 일부 프레임워크는 A/B에서 매우 낮은 성능

- **PydanticAI** (38~40%): Tool Calling 방식이지만, NoDesc 스키마에서 description이 없으면 tool definition에도 설명이 포함되지 않아 낮은 성능
- **Mirascope** (34~35%): Literal 타입 validation 실패가 빈번하게 발생 (10건 중 5건 실패)
- **Guardrails** (6~8%): litellm을 경유하면서 description 전달이 불안정하고, 10건 중 6~7건이 실패

### 5. Description 필드 효과 (A vs B) : 미미함

| 조합 | 전체 평균 | 실패 수 |
|------|----------|---------|
| A (desc + minimal) | 73.6% | 22건 실패 |
| B (no desc + minimal) | 76.0% | 24건 실패 |

A와 B의 차이는 2.4%p로 거의 없다. 이는 두 가지를 의미한다:

1. JSON Schema 방식의 프레임워크에서는 description이 xgrammar에 의해 무시되므로 차이가 없음
2. Tool Calling 방식의 프레임워크에서는 A의 description이 전달되지만, B에서도 필드명만으로 어느 정도 추론이 가능하여 큰 차이가 나지 않음

### 6. Literal 타입 문제

스키마에 포함된 `Literal` 타입 제약이 실패의 주요 원인이었다:

```python
experience_type: Optional[Literal["어학연수", "교환학생", "워킹홀리데이", "유학"]]
assessment_type: Optional[Literal["회화능력", "공인시험"]]
military_status: Optional[Literal["군필", "미필", "면제", "해당없음"]]
```

LLM이 정확한 Literal 값 대신 유사한 값을 생성하는 경우가 빈번하였다:
- "회화" (O: "회화능력")
- "공인시효" (O: "공인시험")
- "공개시험" (O: "공인시험")

이로 인해 Pydantic validation error가 발생하며, `langchain/function_calling`이 조합 A/B에서 **전부 실패**한 주요 원인이기도 하다.

---

## 결론

```
핵심 발견:
  1. 프롬프트 엔지니어링 >> 프레임워크 선택
  2. Rich Prompt 사용 시 모든 프레임워크가 ~96%로 동일한 성능
  3. 스키마 description에만 의존하면 Tool Calling 계열에서만 효과 있음
  4. vLLM의 xgrammar는 JSON Schema의 description을 완전히 무시
```

**vLLM 환경에서 Structured Output 품질을 높이려면 프롬프트에 필드 설명을 명시하는 것이 가장 확실하고 효과적인 방법이다.** 이 경우 어떤 프레임워크를 사용하든 결과는 동일하다 (~96%).

스키마의 `Field(description=...)` 에만 의존하는 전략은, Tool Calling 방식으로 description을 LLM에 직접 전달하는 프레임워크(Instructor, Marvin)에서만 유효하다. JSON Schema 방식(OpenAI native, LangChain json_schema)에서는 xgrammar가 description을 무시하므로 효과가 없다.

결론적으로, **프레임워크의 선택보다 프롬프트 엔지니어링이 성능에 훨씬 더 큰 영향을 미친다.**

---

## 프로젝트 구조

```
structured_output_benchmark/
|-- app/
|   |-- main.py                          # FastAPI 앱 엔트리포인트
|   |-- config.py                        # 환경 설정 (pydantic-settings)
|   |-- api/
|   |   |-- router.py                    # API 엔드포인트 (/api/extract 등)
|   |   |-- models.py                    # 요청/응답 Pydantic 모델
|   |-- frameworks/
|   |   |-- base.py                      # BaseFrameworkAdapter 추상 클래스
|   |   |-- registry.py                  # 프레임워크 레지스트리 (데코레이터 패턴)
|   |   |-- instructor_fw.py             # Instructor 어댑터 (tools/json_schema 등)
|   |   |-- openai_native.py             # OpenAI Native 어댑터 (response_format)
|   |   |-- langchain_fw.py              # LangChain 어댑터 (json_schema/function_calling)
|   |   |-- marvin_fw.py                 # Marvin 어댑터 (pydantic_ai 기반)
|   |   |-- pydantic_ai_fw.py            # PydanticAI 어댑터
|   |   |-- mirascope_fw.py              # Mirascope 어댑터 (ollama provider)
|   |   |-- guardrails_fw.py             # Guardrails 어댑터 (litellm 경유)
|   |-- schemas/
|   |   |-- career.py                    # MainInfo - description 포함 스키마
|   |   |-- career_no_desc.py            # MainInfoNoDesc - description 없는 스키마
|   |   |-- resume.py                    # Resume 스키마 (참고용)
|   |-- prompts/
|       |-- loader.py                    # YAML 프롬프트 로더
|       |-- templates/
|           |-- extract_career_minimal.yaml   # 최소 프롬프트 (조합 A, B)
|           |-- extract_career_rich.yaml      # 상세 프롬프트 (조합 C)
|-- tests/
|   |-- run_multi_resume_benchmark.py    # 메인 벤치마크 실행 스크립트 (270건)
|   |-- run_career_benchmark.py          # 단일 이력서 벤치마크
|   |-- resumes/
|       |-- resume_01.md ~ resume_10.md  # 테스트 이력서 10건
|       |-- ground_truths.py             # Ground Truth 정의
|-- pyproject.toml                       # 프로젝트 의존성 (uv)
```

---

## 실행 방법

### 사전 요구사항

- Python 3.12 이상
- [uv](https://docs.astral.sh/uv/) 패키지 매니저
- vLLM 서버 접근 가능

### 설치

```bash
uv sync
```

### 벤치마크 실행

```bash
cd tests
uv run run_multi_resume_benchmark.py
```

결과는 `tests/multi_resume_benchmark_results.json`에 저장된다.

### FastAPI 서버 실행

```bash
uv run uvicorn app.main:app --reload
```

개별 프레임워크를 API로 테스트할 수 있다:

```bash
curl -X POST http://localhost:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{
    "framework": "instructor",
    "mode": "tools",
    "markdown": "이력서 텍스트...",
    "schema_name": "MainInfo",
    "prompt_name": "extract_career_rich",
    "model": "openai/gpt-oss-120b",
    "base_url": "http://118.38.20.101:8001/v1"
  }'
```

### 지원 프레임워크 조회

```bash
curl http://localhost:8000/api/frameworks
```

---

## 의존성

| 패키지 | 용도 |
|--------|------|
| `instructor` | Tool Calling / JSON Schema 기반 structured output |
| `openai` | OpenAI Native SDK (response_format) |
| `langchain-openai` | LangChain structured output |
| `marvin` | Marvin AI agent framework |
| `pydantic-ai` | PydanticAI agent framework |
| `mirascope` | Mirascope LLM call framework |
| `guardrails-ai` | Guardrails validation framework |
| `fastapi` / `uvicorn` | API 서버 |
| `pydantic` / `pydantic-settings` | 스키마 정의 및 설정 관리 |

---

## 참고 자료

- [The best library for structured LLM output](https://simmering.dev/blog/structured_output/) — Paul Simmering. 10개 라이브러리를 Prompting / Function Calling / Constrained Token Sampling 3가지 방식으로 분류하여 비교 분석
- [llm-structured-output-benchmarks](https://github.com/stephenleo/llm-structured-output-benchmarks) — Stephen Leo. Instructor, Mirascope, LangChain 등 10개+ 프레임워크를 분류/NER/합성 데이터 태스크로 벤치마크
- [JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench) — Guidance AI. 10K개 실제 JSON Schema로 constrained decoding 엔진(xgrammar, Outlines 등) 평가

---

## 라이선스

MIT License
