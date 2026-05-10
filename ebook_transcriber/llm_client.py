from __future__ import annotations

import os

from openai import OpenAI


class LLMConfigError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        api_key = api_key or os.environ.get("API_KEY")
        base_url = base_url or os.environ.get("API_BASE_URL")
        if not api_key:
            raise LLMConfigError("API_KEY is not set")
        if not base_url:
            raise LLMConfigError("API_BASE_URL is not set")
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def vision_chat(self, prompt: str, image_b64: str, mime_type: str = "image/jpeg") -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                        },
                    ],
                }
            ],
            temperature=0,
        )
        content = response.choices[0].message.content
        return content or ""
