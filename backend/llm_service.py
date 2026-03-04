"""LLM 统一调用层 — 基于 litellm 支持多种后端

用法:
    llm = LLMService(cfg.llm)
    reply = await llm.chat("请分析这步棋为什么不好 …")
"""

import logging
import os
from typing import List, Optional

import litellm

os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")

from config import LLMConfig

logger = logging.getLogger(__name__)

# litellm 自身日志过于冗长，降级处理
litellm.suppress_debug_info = True


class LLMService:
    """统一的 LLM 调用封装，通过 litellm 支持 OpenAI / Claude / DeepSeek / Ollama 等。"""

    SYSTEM_PROMPT = (
        "你是一位专业的围棋 AI 解说员。"
        "你会结合 KataGo 的数值分析结果，用清晰易懂的中文向棋手解释局面和着手的优劣。"
        "请使用围棋术语，但同时照顾业余棋手的理解水平。"
    )

    def __init__(self, cfg: LLMConfig):
        self._cfg = cfg
        self._model = cfg.model
        self._api_key = cfg.api_key or None
        self._api_base = cfg.base_url or None

    async def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """发送一轮对话，返回助手回复文本。

        Args:
            user_message: 用户消息（通常是组装好的分析 prompt）
            system_prompt: 系统提示词，None 则用默认围棋解说角色
            history: 额外的历史消息列表 [{"role":..., "content":...}, ...]
            temperature: 温度
            max_tokens: 最大生成 token 数

        Returns:
            助手回复的纯文本
        """
        messages = [{"role": "system", "content": system_prompt or self.SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        logger.info("调用 LLM: model=%s, messages=%d条", self._model, len(messages))

        kwargs = dict(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content
        logger.debug("LLM 回复: %s", content[:100])
        return content


# ── 独立测试入口 ──────────────────────────────────────────────

async def _test():
    """简单测试: 发送一段围棋分析上下文给 LLM。"""
    from config import load_config
    cfg = load_config()

    llm = LLMService(cfg.llm)

    prompt = (
        "以下是 KataGo 对黑棋第 42 手的分析:\n"
        "- 实战着手: D4\n"
        "- 胜率变化: 从 55.2% 跌至 48.1% (跌幅 7.1%)\n"
        "- KataGo 推荐: Q10 (胜率 56.3%), R14 (胜率 55.8%)\n"
        "- 推荐变化: Q10 → W:R14 → B:O3 → W:P16 → B:C6\n\n"
        "请简要分析这步棋的问题，并解释推荐着手的思路。"
    )

    print("发送 prompt 给 LLM …\n")
    reply = await llm.chat(prompt)
    print("LLM 回复:\n")
    print(reply)


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    asyncio.run(_test())
