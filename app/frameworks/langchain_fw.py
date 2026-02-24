from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel

_MODE_MAP = {
    "default": "json_schema",
    "json_schema": "json_schema",
    "function_calling": "function_calling",
    "json_mode": "json_mode",
}


@FrameworkRegistry.register("langchain")
class LangChainAdapter(BaseFrameworkAdapter):
    name = "langchain"
    supported_modes = list(_MODE_MAP.keys())

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        method = _MODE_MAP.get(self.mode, "json_schema")
        llm = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        structured_llm = llm.with_structured_output(schema_class, method=method)

        result = await structured_llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
        )

        return ExtractionResult(
            success=True,
            data=result.model_dump(),
        )
