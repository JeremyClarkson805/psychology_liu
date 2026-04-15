import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ── temperature 必须固定为 1 的模型 ─────────────────────────────
TEMPERATURE_ONE_MODELS = {
    "kimi-k2.5",
}

# ── 哪些模型有 thinking 模式，需要关闭 ──────────────────────────
# 如果服务商不支持该参数会报错，就把模型名从这里移除即可
THINKING_MODELS = {
    # Qwen3 系列：通过 enable_thinking 关闭
    "qwen3.5-397b-a17b": {"enable_thinking": False},
    # Gemini thinking 系列：通过 thinking_config 关闭
    "gemini-3.1-pro-preview-thinking": {"thinking_config": {"thinking_budget": 0}},
}


class LLMConfig(BaseModel):
    api_key: str = Field(
        default_factory=lambda: os.getenv("LLM_API_KEY", ""),
        description="API Key for the LLM service",
    )
    base_url: Optional[str] = Field(
        default="https://api.gpt.ge/v1/", description="Base URL for the API"
    )
    default_headers: Optional[Dict[str, str]] = Field(
        default_factory=lambda: {"x-foo": "true"},
        description="Default headers for the OpenAI client",
    )
    model_names: List[str] = Field(
        default=[
            "qwen3.5-397b-a17b",
            "deepseek-v3.2",
            # "llama-4-scout-17b-16e-instruct",
            "doubao-seed-1-8-251228",
            "kimi-k2.5",
            "gemini-3.1-pro-preview-thinking",
            "gpt-5.4-2026-03-05",
        ],
        description="模型列表，按顺序调用",
    )
    temperature: float = Field(default=0.5, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens")

    def get_extra_body(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        返回该模型需要附加的 extra_body 参数。
        没有特殊配置的模型返回 None，client.py 里不传该参数。
        """
        return THINKING_MODELS.get(model_name, None)

    def get_temperature(self, model_name: str) -> float:
        """
        返回该模型应使用的 temperature。
        kimi 等模型只允许 temperature=1，在此统一处理。
        """
        if model_name in TEMPERATURE_ONE_MODELS:
            return 1
        return self.temperature