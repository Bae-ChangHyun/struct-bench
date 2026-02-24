from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel


@FrameworkRegistry.register("pydantic_ai")
class PydanticAIAdapter(BaseFrameworkAdapter):
    name = "pydantic_ai"

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        model = OpenAIModel(
            self.model,
            provider=OpenAIProvider(
                base_url=self.base_url,
                api_key=self.api_key,
            ),
        )
        agent = Agent(
            model,
            system_prompt=system_prompt,
            output_type=schema_class,
        )
        result = await agent.run(text)

        return ExtractionResult(
            success=True,
            data=result.output.model_dump(),
        )
