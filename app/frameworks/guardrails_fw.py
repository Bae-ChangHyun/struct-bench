from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from guardrails import Guard

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel


@FrameworkRegistry.register("guardrails")
class GuardrailsAdapter(BaseFrameworkAdapter):
    name = "guardrails"

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        guard = Guard.for_pydantic(output_class=schema_class)

        # guardrails는 내부적으로 litellm 사용
        # vLLM 등 커스텀 서버: hosted_vllm/ provider 사용 (모델명 그대로 전달)
        model_name = f"hosted_vllm/{self.model}"
        api_key = self.api_key or "dummy"
        api_base = self.base_url

        def _call():
            return guard(
                model=model_name,
                api_base=api_base,
                api_key=api_key,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )

        result = await asyncio.to_thread(_call)

        if result.validated_output:
            data = (
                result.validated_output.model_dump()
                if hasattr(result.validated_output, "model_dump")
                else dict(result.validated_output)
            )
            return ExtractionResult(success=True, data=data)

        return ExtractionResult(
            success=False,
            error=str(result.error) if result.error else "Validation failed",
        )
