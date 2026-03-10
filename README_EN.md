# struct-output-bench
![unnamed](https://github.com/user-attachments/assets/2a616ee9-8f02-4ae7-840a-2ccfcaf11cd5)

**English** | [한국어](README.md)

> Multiple frameworks exist for generating structured output from LLMs — Instructor, LangChain, Marvin, PydanticAI, and more.
> **struct-output-bench** benchmarks them under **identical conditions** (same model, same schema, same prompt, temperature=0).

Define a Pydantic schema and prompt, feed the same input to multiple frameworks, and quantitatively compare output quality against Ground Truth. A FastAPI server is also included for interactive per-framework testing.

### Supported Frameworks

| Framework | Modes | Docs |
|-----------|-------|------|
| **[Instructor](https://python.useinstructor.com/)** | tools, tools_strict (Tool Calling) / json, json_schema, md_json (JSON Schema) | [docs](https://python.useinstructor.com/modes-comparison/) |
| **[OpenAI Native](https://platform.openai.com/)** | default (JSON Schema) / tool_calling (Tool Calling) / json_object (JSON Object) | [docs](https://platform.openai.com/docs/guides/structured-outputs) |
| **[LangChain](https://python.langchain.com/)** | json_schema (JSON Schema) / function_calling (Tool Calling) / json_mode (JSON Object) | [docs](https://python.langchain.com/docs/concepts/structured_outputs/) |
| **[Marvin](https://askmarvin.ai/)** | cast, extract (Tool Calling) | [docs](https://askmarvin.ai/) |
| **[PydanticAI](https://ai.pydantic.dev/)** | tool (Tool Calling) / json (JSON Schema) / text (Text Parsing) | [docs](https://ai.pydantic.dev/output/) |
| **[Mirascope](https://mirascope.com/)** | tool (Tool Calling) / json (JSON Mode) / strict (JSON Schema) | [docs](https://mirascope.com/docs/mirascope/learn/response_models) |
| **[Guardrails](https://www.guardrailsai.com/)** | default (JSON Schema via litellm) | [docs](https://www.guardrailsai.com/docs/how_to_guides/structured_data_with_guardrails) |
| **[LlamaIndex](https://docs.llamaindex.ai/)** | text (Text Completion) / function_calling (Tool Calling) | [docs](https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/) |

---

## Table of Contents

- [Motivation](#motivation)
- [Background](#background)
- [Experiment Design](#experiment-design)
- [Scoring](#scoring)
- [How Each Framework Works](#how-each-framework-works)
- [Benchmark Results](#benchmark-results)
- [Analysis](#analysis)
- [Conclusion](#conclusion)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [References](#references)

---

## Motivation

The number of frameworks for obtaining structured output from LLMs in the form of Pydantic models has grown rapidly — Instructor, LangChain, Marvin, PydanticAI, Mirascope, Guardrails, LlamaIndex, each solving the same problem in different ways. But **do they actually produce the same results given the same model, same schema, and same prompt?**

struct-output-bench is a tool built to answer this question:

- **Apples-to-apples comparison**: Run multiple frameworks with identical inputs (text, schema, prompt) and compare results
- **Quantitative evaluation**: Ground Truth-based scoring out of 100 to quantify performance differences across frameworks
- **API server**: Instantly test individual frameworks via a FastAPI-based server
- **Extensible**: Add a new framework by inheriting `BaseFrameworkAdapter` and applying the `@register` decorator

### Key Questions

1. Do frameworks differ in how they pass Pydantic schema `Field(description=...)` to the LLM?
2. Is it more effective to provide field descriptions in the prompt or to rely on schema descriptions?
3. How much does framework choice affect final output quality?

---

## Background

### 3 Approaches to Structured Output

There are three main approaches to obtaining structured output from LLMs:

| Approach | How It Works | Representative Libraries | Output Guarantee |
|----------|-------------|--------------------------|-----------------|
| **Prompting** | Describe the desired format in the prompt and hope the LLM follows it | spacy-llm | Not guaranteed |
| **Function Calling (Tool Calling)** | Pass a Pydantic schema as a tool definition; LLM responds with function arguments | Instructor, Marvin, Mirascope | Nearly guaranteed |
| **Constrained Token Sampling** | Define constraints via grammar (CFG) and sample only tokens satisfying those constraints | Outlines, Guidance, **xgrammar** | Fully guaranteed |

Each framework uses one or more of these approaches. Instructor supports both Function Calling and JSON Schema modes. The OpenAI Native SDK passes JSON Schema via `response_format`, letting xgrammar constrain tokens. LangChain also supports both approaches.

### The Problem: vLLM Ignores Schema Descriptions

This project began when we carefully wrote `Field(description=...)` in Pydantic schemas for structured output extraction in a vLLM environment, only to find that **the descriptions had absolutely no effect on the results**. Despite using the same model, results varied across frameworks. Investigation revealed the root cause:

| Approach | Behavior in vLLM | Description Delivery |
|----------|------------------|---------------------|
| **JSON Schema (response_format)** | xgrammar uses only structural constraints (`type`, `properties`, `required`, etc.) | **description ignored** |
| **Tool Calling** | description included in tool definition and passed to the LLM | **description delivered** |

vLLM's xgrammar completely ignores the description field in JSON Schema. The OpenAI SDK passes the Pydantic schema as `response_format` but does not automatically inject the schema into the prompt. Therefore, in frameworks using the JSON Schema approach, descriptions never reach the LLM no matter how carefully they are written. This difference was the key driver of performance variation across frameworks.

---

## Experiment Design

Two versions of the same Pydantic schema were prepared:

- **Schema with descriptions**: All fields include `Field(description="...")`
- **Schema without descriptions**: Same structure, but with types and defaults only — no descriptions

Using 100 samples with Ground Truth from the **DeepJSONEval** dataset (multilingual, deep-nested JSON extraction benchmark), we ran 8 frameworks (22 modes) × 3 combinations = **6,600 tests** in total.

### 4 Test Combinations

| Combination | Schema | Prompt | Experimental Intent |
|-------------|--------|--------|-------------------|
| **A: Schema(desc) + Prompt(minimal)** | with description | minimal prompt without field explanations | Measure performance when relying solely on schema descriptions |
| **B: Schema(no desc) + Prompt(minimal)** | no description | minimal prompt without field explanations | Measure baseline performance with no guidance at all |
| **C: Schema(no desc) + Prompt(rich)** | no description | detailed prompt with all field explanations | Measure performance when guidance is provided only via the prompt |
| **D: Schema(desc) + Prompt(rich)** | with description | detailed prompt with all field explanations | Measure upper bound when both schema descriptions and prompt are provided |

### Prompt Design

**Minimal Prompt**: Contains only basic instructions ("Extract structured information from the given text.")

**Rich Prompt**: Includes detailed descriptions of each section and field — permitted values for Literal types, field purposes, date formats, etc. Approximately 80 lines of guidance.

---

## Scoring

A Ground Truth-based scoring system out of 100 is used. All leaf fields of the extracted result are compared 1:1 with the GT, producing a per-field score (0.0–1.0), then averaged into a percentage.

### Scoring Pipeline

```
Predicted + Ground Truth + JSON Schema
  → flatten_to_pairs()  : Recursively extract leaf field pairs
  → Hungarian Matching   : Order-independent optimal matching for arrays
  → compare_leaf()       : Apply type-based metrics
  → Average → Percentage(%)
```

### Leaf Comparison Metrics

| Type | Metric | Description |
|------|--------|-------------|
| `string` | **NED** (1 - Normalized Edit Distance) | Levenshtein-based string similarity. 1.0 = perfect match |
| `number` / `integer` | **Relative Error** | Relative error ≤ 5% → 1.0, exceeding → 0.0. Absolute error when GT=0 |
| `boolean` | **Exact Match** | Match = 1.0, mismatch = 0.0 |
| `null` | **Exact Match** | Both null = 1.0, only one null = 0.0 |

### Array Matching

Since array element order may differ from the GT, the **Hungarian algorithm** is used for optimal matching. A similarity matrix is constructed between each GT element and each predicted element, and the matching that maximizes total similarity is selected.

---

## How Each Framework Works

<details>
<summary><b>Instructor</b> — Tool Calling / JSON Schema with built-in retry and validation</summary>

```python
client = instructor.from_openai(
    AsyncOpenAI(base_url=..., api_key=...),
    mode=instructor.Mode.TOOLS,  # or TOOLS_STRICT, JSON, JSON_SCHEMA, MD_JSON
)
result = await client.chat.completions.create(
    model=model,
    response_model=schema_class,
    max_retries=0,
    messages=[...],
)
```

Pass an `AsyncOpenAI` client to `instructor.from_openai()` to connect to OpenAI-compatible servers like vLLM. The `mode` parameter selects Tool Calling (TOOLS, TOOLS_STRICT) or JSON Schema (JSON, JSON_SCHEMA, MD_JSON). In Tool Calling mode, description fields are included in the tool definition, allowing the LLM to understand field semantics. Nested Pydantic model `$ref` references are automatically inlined.

</details>

<details>
<summary><b>OpenAI Native</b> — 3 modes (default / tool_calling / json_object)</summary>

```python
# default: response_format + parse()
client.chat.completions.parse(
    model=model, messages=[...], response_format=schema_class,
)

# tool_calling: tools + tool_choice
client.chat.completions.create(
    model=model, messages=[...],
    tools=[{"type": "function", "function": {"name": "...", "parameters": schema}}],
    tool_choice={"type": "function", "function": {"name": "..."}},
)

# json_object: response_format={"type": "json_object"} + schema in prompt
client.chat.completions.create(
    model=model, messages=[...], response_format={"type": "json_object"},
)
```

Supports 3 structured output modes via AsyncOpenAI. **default** mode passes the Pydantic schema via `response_format`, letting xgrammar enforce structure. **tool_calling** mode registers the schema as a function tool, directly passing descriptions to the LLM. **json_object** mode includes JSON Schema in the system prompt and forces JSON response via `response_format={"type": "json_object"}`.

</details>

<details>
<summary><b>LangChain</b> — json_schema / function_calling / json_mode</summary>

```python
llm = ChatOpenAI(model=model, base_url=...)
structured_llm = llm.with_structured_output(schema_class, method="json_schema")  # or "function_calling", "json_mode"
result = await structured_llm.ainvoke(messages)
```

`json_schema` mode uses `response_format`-based operation, `function_calling` mode uses Tool Calling, and `json_mode` uses `response_format={"type": "json_object"}` with schema injection into the prompt.

</details>

<details>
<summary><b>Marvin</b> — pydantic_ai-based cast_async / extract_async with Tool Calling</summary>

```python
provider = OpenAIProvider(base_url=..., api_key=...)
model = OpenAIModel(model_name, provider=provider)
agent = marvin.Agent(model=model, instructions=system_prompt)

# cast mode: convert text to target type (single object)
result = await marvin.cast_async(data=text, target=schema_class, instructions=system_prompt, agent=agent)

# extract mode: extract entities from text (list)
results = await marvin.extract_async(data=text, target=schema_class, instructions=system_prompt, agent=agent)
```

Marvin 3.x is rewritten on top of pydantic-ai. `cast_async` converts text to a target type, while `extract_async` extracts multiple entities as a list. Custom endpoints are configured via `OpenAIModel`/`OpenAIProvider`, and system prompts are passed through Agent `instructions`. Internally uses Tool Calling, so descriptions are delivered.

</details>

<details>
<summary><b>PydanticAI</b> — Agent + output_type with 4 modes (default/tool/json/text)</summary>

```python
model = OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=...))
# default/json: NativeOutput (JSON Schema-based)
agent = Agent(model, system_prompt=prompt, output_type=NativeOutput(schema_class))
# tool: ToolOutput (Tool Calling-based)
agent = Agent(model, system_prompt=prompt, output_type=ToolOutput(schema_class))
# text: plain text response + JSON parsing
agent = Agent(model, system_prompt=prompt, output_type=str)
result = await agent.run(text)
```

Using pydantic-ai v1.0.5, models are configured via `OpenAIChatModel`/`OpenAIProvider`. **default/json** modes use `NativeOutput` for JSON Schema-based structured output, **tool** mode uses `ToolOutput` for Tool Calling, and **text** mode receives plain text and parses it as JSON.

</details>

<details>
<summary><b>Mirascope</b> — openai provider registration with @call decorator (tool / json / strict)</summary>

```python
llm.register_provider("openai", scope="openai/model:completions", base_url=..., api_key=...)
fmt = llm.format(schema_class, mode="tool")  # or "json", "strict"

@llm.call("openai/model:completions", format=fmt)
async def do_extract(text: str, sys_prompt: str) -> str:
    return f"{sys_prompt}\n\n{text}"
```

Uses `mirascope.llm.call` decorator with `format=llm.format(schema_class, mode=...)`. Connects to vLLM via the openai provider, with the `:completions` suffix to force the Chat Completions API. `tool` mode uses Tool Calling, `json` uses JSON Mode, and `strict` uses Strict JSON Schema.

</details>

<details>
<summary><b>Guardrails</b> — AsyncGuard via litellm with hosted_vllm provider</summary>

```python
guard = AsyncGuard.for_pydantic(output_class=schema_class)
result = await guard(
    model="hosted_vllm/model",
    api_base=base_url,
    api_key=api_key,
    num_reasks=0,
    messages=[...],
)
```

Uses litellm internally and connects to vLLM via the `hosted_vllm/` provider. `AsyncGuard` provides native async support, and `num_reasks=0` ensures a single call without retries for benchmark fairness.

</details>

<details>
<summary><b>LlamaIndex</b> — FunctionCallingProgram / LLMTextCompletionProgram</summary>

```python
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.program import FunctionCallingProgram, LLMTextCompletionProgram

llm = OpenAILike(model=model, api_base=base_url, api_key=api_key,
                 is_chat_model=True, is_function_calling_model=True)

# function_calling mode: Tool Calling-based
program = FunctionCallingProgram.from_defaults(output_cls=schema_class, llm=llm, prompt=prompt_tpl)

# text mode: LLM generates JSON as text, parsed by Pydantic
program = LLMTextCompletionProgram.from_defaults(output_cls=schema_class, llm=llm, prompt=prompt_tpl)

result = await program.acall(system_prompt=prompt, text=text)
```

`FunctionCallingProgram` converts the Pydantic schema into a Function Calling tool definition. `LLMTextCompletionProgram` instructs the LLM to generate JSON as text and parses it with Pydantic. `OpenAILike` connects to OpenAI-compatible servers like vLLM.

</details>

---

## Benchmark Results

> **Model**: `openai/gpt-oss-120b` (vLLM, temperature=0) | **Dataset**: DeepJSONEval 100 samples | **Total**: 6,600 tests (A/B/C)

> [!IMPORTANT]
> These results are measured with a **specific model (`gpt-oss-120b`) on a specific serving environment (vLLM)**. Results may differ significantly with different models.
> - **Model capability matters most.** Models with robust Tool Calling support (GPT-4o, Claude, etc.) may show much lower failure rates in Tool Calling modes.
> - **Behavior varies by serving engine.** vLLM's xgrammar ignores JSON Schema descriptions, but the OpenAI API or other engines may behave differently.
> - This benchmark is designed to identify **relative characteristic differences** between frameworks, not to generalize absolute performance numbers.

### Result Matrix

| Framework / Mode | A_desc | B_nodesc | C_rich | D_both | Overall |
|-----------------|--------|----------|--------|--------|---------|
| **instructor**/tools | 91.8% (86F) | 91.1% (87F) | 93.6% (17F) | | 93.1% |
| **instructor**/tools_strict | 90.8% (92F) | 86.7% (90F) | 94.3% (16F) | | 93.2% |
| **instructor**/json | 93.5% (2F) | 93.0% (3F) | 93.7% (3F) | | **93.4%** |
| **instructor**/json_schema | 80.6% (52F) | 75.4% (49F) | 93.5% (3F) | | 85.6% |
| **instructor**/md_json | 93.8% (3F) | 93.0% (4F) | 93.8% (3F) | | **93.5%** |
| **openai**/default | 89.1% | 86.9% | 93.5% | | 89.8% |
| **openai**/tool_calling | 91.6% (73F) | 95.5% (73F) | 93.4% (11F) | | 93.5% |
| **openai**/json_object | 94.0% (4F) | 93.3% (2F) | 93.8% (4F) | | **93.7%** |
| **langchain**/json_schema | 86.9% | 86.5% | 92.5% | | 88.6% |
| **langchain**/function_calling | 96.9% (92F) | 87.4% (85F) | 94.2% (12F) | | 93.5% |
| **langchain**/json_mode | 94.1% (2F) | 93.1% (3F) | 94.0% (3F) | | **93.8%** |
| **marvin**/cast | 93.6% (3F) | 93.1% (7F) | 93.9% (1F) | | **93.5%** |
| **marvin**/extract | 93.8% (2F) | 93.7% (2F) | 93.7% (2F) | | **93.7%** |
| **pydantic_ai**/tool | 57.1% (99F) | ALL FAIL | 94.0% (38F) | | 93.4% |
| **pydantic_ai**/json | 77.8% (48F) | 74.8% (49F) | 92.8% | | 84.4% |
| **pydantic_ai**/text | 94.5% (2F) | 93.0% (2F) | 93.9% (2F) | | **93.8%** |
| **mirascope**/tool | 93.0% (11F) | 93.3% (9F) | 93.0% (4F) | | **93.1%** |
| **mirascope**/json | 94.7% (9F) | 93.4% (7F) | 93.6% (14F) | | **93.9%** |
| **mirascope**/strict | 78.3% (45F) | 80.3% (46F) | 93.3% | | 86.0% |
| **guardrails**/default | 72.9% (6F) | 72.3% (7F) | 93.1% | | 79.7% |
| **llamaindex**/text | 94.3% (2F) | 93.4% (3F) | 94.0% (2F) | | **93.9%** |
| **llamaindex**/function_calling | ALL FAIL | ALL FAIL | 95.9% (55F) | | 95.9% |

> `(NF)` = N failures out of 100 (parsing errors, missing tool calls, etc.). `ALL FAIL` = all 100 failed. Scores are calculated from successful samples only.

---

## Analysis

### 1. Rich Prompt → All frameworks converge to ~93%

With combination C (detailed field descriptions in the prompt), **most frameworks converge to 92–95%**, and failure rates drop dramatically.

- guardrails: 72% → **93.1%** (0 failures)
- pydantic_ai/json: 74–78% → **92.8%** (0 failures)
- mirascope/strict: 78–80% → **93.3%** (0 failures)
- llamaindex/function_calling: ALL FAIL → **95.9%** (from total failure to success)

When field descriptions are explicitly provided in the prompt, performance differences across frameworks effectively disappear.

### 2. Stable Modes vs Unstable Modes

**Stable (failure rate < 5%)**: Modes that include the schema directly in the prompt or operate JSON-based
- instructor/json, md_json, langchain/json_mode, openai/json_object, pydantic_ai/text, llamaindex/text, marvin/*, mirascope/tool

**Unstable (failure rate > 30%)**: Tool Calling modes where the LLM fails to generate tool calls
- instructor/tools, tools_strict, openai/tool_calling, langchain/function_calling, pydantic_ai/tool, llamaindex/function_calling

In the vLLM environment, Tool Calling shows unstable behavior even with forced `tool_choice` — the model may return JSON in content instead of a tool call, or generate multiple tool calls.

### 3. Limitations of JSON Schema (xgrammar) Mode

- **openai/default**: 86.9–89.1% (0 failures — stable but lower scores)
- **langchain/json_schema**: 86.5–86.9% (0 failures — same pattern)
- **instructor/json_schema**: 75.4–80.6% (high failure rate)
- **pydantic_ai/json**: 74.8–77.8% (high failure rate)

When JSON Schema is passed via `response_format`, xgrammar **enforces structure only** and ignores descriptions. openai/default and langchain/json_schema are stable with no failures but score lower, while instructor/json_schema and pydantic_ai/json suffer from high failure rates.

### 4. Schema Description Effect (A vs B) → Negligible

The score difference between A (desc) and B (nodesc) is mostly 1–3 percentage points. In JSON Schema mode, xgrammar ignores descriptions, and in Tool Calling mode, field names alone provide sufficient inference capability.

### 5. Tool Calling Failure Analysis

Tool Calling failures in vLLM fall into two main categories:

1. **No tool call generated**: Even with `tool_choice` forcing a specific function, the model returns JSON in content instead of a tool call (frequent in instructor/tools_strict, pydantic_ai/tool)
2. **Multiple tool calls**: The model generates multiple tool calls for a single schema extraction, causing the framework to reject the response (observed in instructor)

These are **limitations of the model's tool calling capability**, not framework code issues.

---

## Conclusion

```
Key Findings:
  1. Prompt engineering >> Framework choice
  2. With Rich Prompt, most frameworks converge to ~93%
  3. JSON-based modes (json, md_json, json_mode, json_object, text) are more stable than Tool Calling
  4. vLLM's xgrammar completely ignores JSON Schema descriptions
  5. Tool Calling shows high failure rates in vLLM (missing tool calls, multiple tool calls)
```

**To improve Structured Output quality in a vLLM environment:**
1. **Include field descriptions in the prompt** — results converge to ~93% regardless of framework
2. **Prefer JSON-based modes** — significantly lower failure rates compared to Tool Calling
3. **Use Tool Calling with caution** — models frequently fail to generate stable tool calls

In conclusion, **prompt engineering has a far greater impact on performance than framework choice**, and JSON-based modes are more stable than Tool Calling in a vLLM environment.

---

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- OpenAI API-compatible server (vLLM, etc.)

### Install

```bash
# Full install (all frameworks + server + dashboard)
uv sync --extra dev

# Core + specific frameworks only
uv sync --extra instructor --extra openai

# Core only (no frameworks)
uv sync
```

<details>
<summary>Optional dependency groups</summary>

| Group | Packages |
|-------|----------|
| `instructor` | instructor |
| `langchain` | langchain-openai |
| `llamaindex` | llama-index-llms-openai-like, llama-index-program-openai |
| `marvin` | marvin |
| `mirascope` | mirascope[openai] |
| `guardrails` | guardrails-ai |
| `pydantic-ai` | pydantic-ai |
| `all` | All frameworks above |
| `server` | fastapi, uvicorn |
| `dashboard` | streamlit, matplotlib, pandas |
| `datasets` | pymupdf |
| `dev` | all + server + dashboard + datasets + python-dotenv |

</details>

### Configuration

```bash
cp .env.example .env
# Set BASE_URL, MODEL, API_KEY in .env
```

### Run Benchmark

```bash
# Full benchmark (all frameworks × all combinations)
uv run python run_benchmark.py --dataset deepjsoneval

# Specific frameworks only
uv run python run_benchmark.py --dataset deepjsoneval --frameworks instructor/tools openai/default

# Specific combinations only
uv run python run_benchmark.py --dataset deepjsoneval --combos A_desc C_rich D_both

# Limit sample count
uv run python run_benchmark.py --dataset deepjsoneval --max-samples 10

# Resume a previous run (skip completed frameworks)
uv run python run_benchmark.py --resume results/deepjsoneval_20260304_163132

# Specify server settings directly
uv run python run_benchmark.py --dataset deepjsoneval --base-url http://localhost:8001/v1 --model my-model
```

### Run API Server

```bash
uv run uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{
    "framework": "instructor",
    "mode": "tools",
    "markdown": "...",
    "schema_name": "SchemaName",
    "prompt_name": "prompt_name",
    "model": "your-model",
    "base_url": "http://your-server/v1"
  }'
```

### Dashboard

```bash
uv run streamlit run dashboard.py
```

---

## Project Structure

```
struct-output-bench/
├── run_benchmark.py          # CLI entry point
├── dashboard.py              # Streamlit dashboard
├── app/
│   ├── benchmark/
│   │   ├── config.py         # Combinations (A/B/C/D), framework/mode definitions
│   │   ├── runner.py         # Benchmark execution engine
│   │   └── datasets.py       # Dataset adapter registry
│   ├── frameworks/           # Framework adapters (8)
│   │   ├── base.py           # BaseFrameworkAdapter
│   │   ├── registry.py       # @FrameworkRegistry.register decorator
│   │   ├── instructor_fw.py
│   │   ├── openai_native.py
│   │   ├── langchain_fw.py
│   │   ├── marvin_fw.py
│   │   ├── pydantic_ai_fw.py
│   │   ├── mirascope_fw.py
│   │   ├── guardrails_fw.py
│   │   └── llamaindex_fw.py
│   ├── scoring/              # Unified scoring system
│   │   ├── scorer.py         # score_result() entry point
│   │   ├── matcher.py        # Recursive flatten + Hungarian matching
│   │   ├── metrics.py        # NED, numeric comparison, boolean comparison
│   │   ├── hungarian.py      # Hungarian algorithm
│   │   └── schema_utils.py   # JSON Schema utilities
│   ├── datasets/             # Per-dataset loaders / schema generators
│   ├── schemas/              # Pydantic schema definitions
│   ├── prompts/              # Prompt templates
│   └── api/                  # FastAPI routers
├── results/                  # Benchmark results (gitignored)
└── pyproject.toml
```

### Adding a New Framework

Inherit `BaseFrameworkAdapter` and apply the `@FrameworkRegistry.register` decorator for automatic registration.

```python
from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

@FrameworkRegistry.register("my_framework")
class MyAdapter(BaseFrameworkAdapter):
    name = "my_framework"
    supported_modes = ["default"]

    async def extract(self, text, schema_class, system_prompt) -> ExtractionResult:
        # implementation
        return ExtractionResult(success=True, data={...})
```

---

## References

### Articles & Tools
- [The best library for structured LLM output](https://simmering.dev/blog/structured_output/) — Paul Simmering
- [llm-structured-output-benchmarks](https://github.com/stephenleo/llm-structured-output-benchmarks) — Stephen Leo

### Benchmark Datasets
- [JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench) — 10K real-world JSON Schemas for constrained decoding evaluation
- [ExtractBench](https://arxiv.org/abs/2602.12247) — PDF-to-JSON structured extraction, 35 docs + JSON Schema + human-annotated GT (12,867 fields)
- [DeepJSONEval](https://arxiv.org/abs/2509.25922) — Multilingual deep-nested JSON extraction benchmark with schema + input + GT (2,100 instances)

---

## License

MIT License
