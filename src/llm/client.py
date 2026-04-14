from typing import Dict, Optional, List
from openai import OpenAI
from config.llm_config import LLMConfig
import time
import re

MAX_RETRIES = 3   # 每次请求最多重试次数


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            default_headers=self.config.default_headers
        )

    def _do_stream(
        self,
        model_name: str,
        messages: list,
        temperature: float,
        extra_body: Optional[dict],
        verbose: bool,
    ) -> str:
        """
        执行一次 streaming 请求，返回去掉 think 块的纯文本。
        失败时抛出异常，由外层重试逻辑处理。
        """
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
            **({"extra_body": extra_body} if extra_body else {}),
        )

        full_text = ""
        in_think_block = False

        if verbose:
            print("[ASSISTANT] ", end="", flush=True)

        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                token = delta.content
                full_text += token

                if "<think>" in token:
                    in_think_block = True
                if "</think>" in token:
                    in_think_block = False
                    continue

                if verbose and not in_think_block:
                    print(token, end="", flush=True)

        if verbose:
            print(f"\n{'─'*60}\n", flush=True)

        # 剔除 <think>...</think> 块
        clean = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
        return clean

    def chat(
        self,
        prompt: str,
        system_prompt: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = True,
        verbose: bool = True,
        model_name: Optional[str] = None,   # 指定单个模型；None 则遍历所有
    ) -> Dict[str, str]:
        """
        发送请求，支持 streaming、自动重试、temperature 自动修正。

        Returns:
            { model_name: full_response_text }
        """
        targets = [model_name] if model_name else self.config.model_names
        results = {}

        for name in targets:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})

            # ── 打印发送内容 ──────────────────────────────
            if verbose:
                print(f"\n{'─'*60}", flush=True)
                print(f"📤 发送给 [{name}]", flush=True)
                print(f"{'─'*60}", flush=True)
                print(f"[USER]\n{prompt}", flush=True)
                print(f"{'─'*60}", flush=True)
                print("📥 模型回复中...", flush=True)

            extra_body = self.config.get_extra_body(name)
            current_temp = self.config.get_temperature(name)
            last_err = None

            for attempt in range(1, MAX_RETRIES + 1):
                if attempt > 1:
                    print(f"  [retry {attempt}/{MAX_RETRIES}] temperature={current_temp} ...", flush=True)

                try:
                    if stream:
                        text = self._do_stream(name, messages, current_temp, extra_body, verbose)
                    else:
                        resp = self.client.chat.completions.create(
                            model=name,
                            messages=messages,
                            temperature=current_temp,
                            max_tokens=self.config.max_tokens,
                            **({"extra_body": extra_body} if extra_body else {}),
                        )
                        text = resp.choices[0].message.content if resp.choices else "Error: Empty response"
                        if verbose:
                            print(f"[ASSISTANT]\n{text}", flush=True)
                            print(f"{'─'*60}\n", flush=True)

                    results[name] = text
                    last_err = None
                    break  # 成功，退出重试

                except Exception as e:
                    last_err = e
                    err_msg = str(e)
                    print(f"\n  ⚠ 第{attempt}次请求失败: {err_msg}", flush=True)

                    # temperature 限制错误 → 自动改为 1 重试
                    if "invalid temperature" in err_msg or ("temperature" in err_msg.lower() and attempt == 1):
                        print("  → 检测到 temperature 限制，自动改为 1 重试", flush=True)
                        current_temp = 1
                    elif attempt < MAX_RETRIES:
                        wait = 2 * attempt
                        print(f"  → {wait}s 后重试 ...", flush=True)
                        time.sleep(wait)

            if last_err is not None:
                print(f"\n❌ [{name}] 重试 {MAX_RETRIES} 次后仍失败: {last_err}", flush=True)
                results[name] = f"Error: {last_err}"

        return results