from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from mirascope.llm import call, register_provider

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel


@FrameworkRegistry.register("mirascope")
class MirascopeAdapter(BaseFrameworkAdapter):
    name = "mirascope"

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        # vLLM 등 커스텀 base_url: ollama provider로 등록
        register_provider(
            "ollama",
            scope="ollama/",
            base_url=self.base_url,
        )

        # ollama/model_name 형식으로 호출 (vLLM은 model 이름 그대로 전달)
        model_id = self.model
        if not model_id.startswith("ollama/"):
            model_id = f"ollama/{model_id}"

        @call(model_id, format=schema_class)
        def do_extract(resume_text: str, sys_prompt: str) -> str:
            return f"{sys_prompt}\n\n{resume_text}"

        response = await asyncio.to_thread(do_extract, text, system_prompt)
        parsed = response.parse()

        return ExtractionResult(
            success=True,
            data=parsed.model_dump(),
        )
