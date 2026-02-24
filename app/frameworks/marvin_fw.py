from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import marvin
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel


@FrameworkRegistry.register("marvin")
class MarvinAdapter(BaseFrameworkAdapter):
    name = "marvin"

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        # pydantic_ai의 OpenAIModel/OpenAIProvider로 모델 직접 주입
        provider = OpenAIProvider(
            base_url=self.base_url,
            api_key=self.api_key or "dummy",
        )
        model = OpenAIModel(self.model, provider=provider)

        agent = marvin.Agent(
            model=model,
            instructions=system_prompt,
        )

        def _call():
            return agent.run(text, result_type=schema_class)

        result = await asyncio.to_thread(_call)

        return ExtractionResult(
            success=True,
            data=result.model_dump(),
        )
