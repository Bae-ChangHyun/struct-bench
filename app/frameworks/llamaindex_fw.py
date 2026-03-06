from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.program import FunctionCallingProgram, LLMTextCompletionProgram
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai_like import OpenAILike

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel

_PROMPT_TPL = PromptTemplate("{system_prompt}\n\n{text}")


@FrameworkRegistry.register("llamaindex")
class LlamaIndexAdapter(BaseFrameworkAdapter):
    name = "llamaindex"
    supported_modes = ["default", "text", "function_calling"]

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        if self.mode == "function_calling":
            return await self._extract_function_calling(text, schema_class, system_prompt)
        return await self._extract_text(text, schema_class, system_prompt)

    async def _extract_text(
        self, text: str, schema_class: type[BaseModel], system_prompt: str,
    ) -> ExtractionResult:
        """LLMTextCompletionProgram (JSON text mode) — Pydantic v2 직접 사용."""
        llm = OpenAILike(
            model=self.model,
            api_base=self.base_url,
            api_key=self.api_key,
            is_chat_model=True,
        )

        program = LLMTextCompletionProgram.from_defaults(
            output_cls=schema_class,
            llm=llm,
            prompt=_PROMPT_TPL,
        )
        result = await program.acall(system_prompt=system_prompt, text=text)

        data = result.model_dump()
        return ExtractionResult(success=True, data=data)

    async def _extract_function_calling(
        self, text: str, schema_class: type[BaseModel], system_prompt: str,
    ) -> ExtractionResult:
        """FunctionCallingProgram — LlamaIndex 네이티브 Function Calling."""
        llm = OpenAILike(
            model=self.model,
            api_base=self.base_url,
            api_key=self.api_key,
            is_chat_model=True,
            is_function_calling_model=True,
        )
        program = FunctionCallingProgram.from_defaults(
            output_cls=schema_class,
            llm=llm,
            prompt=_PROMPT_TPL,
        )
        result = await program.acall(system_prompt=system_prompt, text=text)
        data = result.model_dump()
        return ExtractionResult(success=True, data=data)
