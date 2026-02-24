from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    framework: str = Field(description="사용할 프레임워크 이름")
    markdown: str = Field(description="추출할 마크다운 텍스트")
    schema_name: str = Field(description="사용할 스키마 이름 (예: Resume)")
    prompt_name: str = Field(description="사용할 프롬프트 템플릿 이름")
    model: str | None = Field(default=None, description="LLM 모델명 (기본값: 환경변수)")
    base_url: str | None = Field(default=None, description="OpenAI API base URL (기본값: 환경변수)")
    api_key: str | None = Field(default=None, description="API 키 (기본값: 환경변수)")
    mode: str = Field(default="default", description="프레임워크 모드 (예: instructor의 json_schema, tools 등)")


class ExtractionResponse(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float = 0.0
    framework: str = ""
    model: str = ""
    mode: str = ""
