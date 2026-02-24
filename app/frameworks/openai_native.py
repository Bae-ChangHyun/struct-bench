from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from openai import OpenAI

from app.frameworks.base import BaseFrameworkAdapter, ExtractionResult
from app.frameworks.registry import FrameworkRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel


@FrameworkRegistry.register("openai")
class OpenAINativeAdapter(BaseFrameworkAdapter):
    name = "openai"

    async def extract(
        self,
        text: str,
        schema_class: type[BaseModel],
        system_prompt: str,
    ) -> ExtractionResult:
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        def _call():
            return client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                response_format=schema_class,
            )

        completion = await asyncio.to_thread(_call)
        message = completion.choices[0].message

        if message.parsed:
            return ExtractionResult(
                success=True,
                data=message.parsed.model_dump(),
            )
        return ExtractionResult(
            success=False,
            error=message.refusal or "No parsed response",
        )
