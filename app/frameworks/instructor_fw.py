from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import instructor
from openai import AsyncOpenAI

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel

_MODE_MAP = {
    "default": instructor.Mode.TOOLS,
    "tools": instructor.Mode.TOOLS,
    "tools_strict": instructor.Mode.TOOLS_STRICT,
    "json": instructor.Mode.JSON,
    "json_schema": instructor.Mode.JSON_SCHEMA,
    "md_json": instructor.Mode.MD_JSON,
}


@FrameworkRegistry.register("instructor")
class InstructorAdapter(BaseFrameworkAdapter):
    name = "instructor"
    supported_modes = list(_MODE_MAP.keys())

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        mode = _MODE_MAP.get(self.mode, instructor.Mode.TOOLS)

        # vLLM용: from_provider("ollama/model") 패턴 사용
        # from_provider는 sync client를 반환하므로 to_thread 사용
        if self.base_url and self.base_url != "https://api.openai.com/v1":
            client = instructor.from_provider(
                f"ollama/{self.model}",
                base_url=self.base_url,
            )

            def _call():
                return client.chat.completions.create(
                    response_model=schema_class,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                )

            result = await asyncio.to_thread(_call)
        else:
            # OpenAI 직접 연결 시 기존 from_openai + mode 사용
            client = instructor.from_openai(
                AsyncOpenAI(base_url=self.base_url, api_key=self.api_key),
                mode=mode,
            )

            result = await client.chat.completions.create(
                model=self.model,
                response_model=schema_class,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )

        return ExtractionResult(
            success=True,
            data=result.model_dump(),
        )
