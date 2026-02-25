from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from llama_index.llms.openai_like import OpenAILike
from llama_index.program.openai import OpenAIPydanticProgram

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel


@FrameworkRegistry.register("llamaindex")
class LlamaIndexAdapter(BaseFrameworkAdapter):
    name = "llamaindex"

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        llm = OpenAILike(
            model=self.model,
            api_base=self.base_url,
            api_key=self.api_key,
            is_chat_model=True,
            is_function_calling_model=True,
        )
        program = OpenAIPydanticProgram.from_defaults(
            output_cls=schema_class,
            prompt_template_str="{system_prompt}\n\n{text}",
            llm=llm,
        )
        result = await asyncio.to_thread(
            program, system_prompt=system_prompt, text=text
        )

        return ExtractionResult(
            success=True,
            data=result.model_dump(),
        )
