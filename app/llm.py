"""模型客户端：LLM（生成）+ Embedding（向量化），统一走 OpenAI 兼容协议。

改造后的四个要点：

1. **客户端复用。** 以前每次提问都 `OpenAI(...)` 新建一个实例，等于每次丢掉连接池，
   高并发下光 TLS 握手就是一笔开销。现在按 (Key, base_url) 缓存。

2. **Token 计量。** 流式请求默认拿不到 usage，必须显式传
   `stream_options={"include_usage": True}`；服务商不支持时退回本地估算，
   并在结果里标记 `estimated=True` —— 宁可标注是估算，也不要给假精确的数字。

3. **向量化失败不再静默降级。** 以前 embedding 接口挂了会悄悄换成本地哈希向量，
   但本地哈希和真实 embedding **不在同一个向量空间**，混进索引后检索会整体错乱，
   而且没有任何报错。现在直接抛错：宁可这次上传失败，也不要污染整个知识库。

4. **embedding 指纹。** 每个片段都记录 embed 时用的模型（mode:model），
   换模型后能明确报出来"索引是旧模型建的，请重建"，而不是继续检索出错乱的答案。
"""

from __future__ import annotations

import logging
import time
from typing import Iterator, Sequence

from .config import Settings
from .embedding_local import embed_local

logger = logging.getLogger(__name__)


class LLMNotConfigured(Exception):
    """没有可用的 API Key。"""


class EmbeddingUnavailable(Exception):
    """向量化服务不可用（Key 无效 / 限流 / 网络故障 / 模型名错误）。"""


def estimate_tokens(text: str) -> int:
    """没有 usage 时的兜底估算。

    中文大约 1 字 ≈ 1 token，英文大约 4 字符 ≈ 1 token。
    只用它来"大致知道花了多少钱"，所以宁可粗一点，也不要假装精确。
    """
    text = text or ""
    if not text:
        return 0
    han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - han
    return han + max(1, other // 4) if other else han


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._clients: dict[tuple[str, str], object] = {}

    # ------------------------------------------------------------------ 内部
    def _client(self, api_key: str, base_url: str):
        if not api_key:
            raise LLMNotConfigured("未配置 API Key，请在 .env 里填写 LLM_API_KEY")
        cache_key = (api_key, base_url)
        client = self._clients.get(cache_key)
        if client is not None:
            return client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMNotConfigured("未安装 openai：pip install openai") from exc
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.s.llm_timeout)
        self._clients[cache_key] = client
        return client

    @property
    def llm_ready(self) -> bool:
        return self.s.llm_ready

    # ------------------------------------------------------------------ 生成
    def chat(self, messages: Sequence[dict], stream: bool = False, usage_sink: dict | None = None):
        """生成回答。stream=True 时返回迭代器；usage_sink 会被填上 token 用量。"""
        client = self._client(self.s.llm_api_key, self.s.llm_base_url)
        prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
        kwargs: dict = {
            "model": self.s.llm_model,
            "messages": list(messages),
            "temperature": self.s.llm_temperature,
            "stream": stream,
        }
        if stream:
            # 关键参数：不加它，流式响应里根本没有 usage 字段
            kwargs["stream_options"] = {"include_usage": True}
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # 个别服务商不认 stream_options，去掉再试一次（兼容性优先）
            if stream and "stream_options" in kwargs:
                logger.warning("服务商不支持 stream_options，已去掉重试：%s", exc)
                kwargs.pop("stream_options")
                response = client.chat.completions.create(**kwargs)
            else:
                raise

        if stream:
            return self._iter_stream(response, usage_sink, prompt_text)
        return self._text_and_usage(response, usage_sink, prompt_text)

    @staticmethod
    def _text_and_usage(response, usage_sink: dict | None, prompt_text: str = "") -> str:
        text = ""
        try:
            text = response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            text = ""
        if usage_sink is not None:
            usage_sink.update(LLMClient._usage_of(getattr(response, "usage", None), prompt_text, text))
        return text

    def _iter_stream(self, response, usage_sink: dict | None, prompt_text: str = "") -> Iterator[str]:
        parts: list[str] = []
        usage = None
        for chunk in response:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage:
                usage = chunk_usage
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                parts.append(text)
                yield text
        if usage_sink is not None:
            usage_sink.update(self._usage_of(usage, prompt_text, "".join(parts)))

    @staticmethod
    def _usage_of(usage, prompt_text: str, completion_text: str) -> dict:
        def _get(source, name: str) -> int:
            if source is None:
                return 0
            if isinstance(source, dict):
                return int(source.get(name, 0) or 0)
            return int(getattr(source, name, 0) or 0)

        prompt = _get(usage, "prompt_tokens")
        completion = _get(usage, "completion_tokens")
        if prompt or completion:
            return {"prompt_tokens": prompt, "completion_tokens": completion, "estimated": False}
        return {
            "prompt_tokens": estimate_tokens(prompt_text),
            "completion_tokens": estimate_tokens(completion_text),
            "estimated": True,
        }

    # ---------------------------------------------------------------- 向量化
    def embedding_key(self) -> str:
        """挑出 embedding 要用的 Key。

        只有在 embedding 和对话用的是**同一家服务商**（base_url 同源）时，
        才允许复用 LLM_API_KEY。否则会出现"拿 DeepSeek 的 Key 去调通义的接口"
        —— 结果必然是 401，而且报错信息很难看出原因。
        """
        if self.s.embedding_api_key:
            return self.s.embedding_api_key
        from urllib.parse import urlparse

        if urlparse(self.s.embedding_base_url).hostname == urlparse(self.s.llm_base_url).hostname:
            return self.s.llm_api_key
        return ""

    @property
    def signature(self) -> str:
        """embedding 指纹：写进每个片段的 metadata，用来发现"换过模型"。"""
        if self.s.embedding_mode == "api":
            return f"api:{self.s.embedding_model}"
        return f"local:hash-{self.s.local_embedding_dim}d"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.s.embedding_mode == "api":
            api_key = self.embedding_key()
            if api_key:
                return self._embed_api(list(texts), api_key)
            logger.warning(
                "EMBEDDING_MODE=api 但没有可用的 EMBEDDING_API_KEY"
                "（embedding 与对话不同源时不能复用 LLM_API_KEY），已回退本地向量化"
            )
        return [embed_local(text, self.s.local_embedding_dim) for text in texts]

    def _embed_api(self, texts: list[str], api_key: str) -> list[list[float]]:
        """批量向量化（带重试 + 进度日志）。

        为什么失败就抛异常、而不是回退本地：
        本地哈希向量和真实 embedding 不在同一个向量空间，混进同一个索引后
        检索结果会整体错乱，而且**不报错**。这种"静默质量下降"比一次明确的失败危险得多。
        """
        client = self._client(api_key, self.s.embedding_base_url)
        vectors: list[list[float]] = []
        batch_size = max(1, self.s.embedding_batch_size)
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = [(t[:2000] or " ") for t in texts[start : start + batch_size]]
            response = self._embed_batch_with_retry(client, batch, texts[start])
            vectors.extend(item.embedding for item in response.data)
            done = min(start + batch_size, total)
            if start == 0 or done == total or (start // batch_size) % 20 == 0:
                logger.info("向量化进度：%d/%d", done, total)
        return vectors

    def _embed_batch_with_retry(self, client, batch: list[str], sample_text: str, attempts: int = 3):
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return client.embeddings.create(model=self.s.embedding_model, input=batch)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == attempts - 1:
                    break
                wait = 2 ** attempt
                logger.warning("向量化第 %d 次失败（%s），%d 秒后重试", attempt + 1, exc, wait)
                time.sleep(wait)
        raise EmbeddingUnavailable(
            f"向量化失败：{type(last_error).__name__}: {last_error}\n"
            f"  模型 = {self.s.embedding_model}；地址 = {self.s.embedding_base_url}\n"
            "  排查：401 → Key 不对；404 → 模型名写错；403 → Key 没勾选该模型；"
            "网络不通 → 检查出口代理。\n"
            "  （特意不自动回退本地向量化：两种向量不在同一空间，混用会静默污染整个索引）"
        ) from last_error

    # ------------------------------------------------------------------ 描述
    def describe(self) -> dict:
        embedding_model = (
            self.s.embedding_model
            if self.s.embedding_mode == "api" and self.embedding_key()
            else f"local-hash-{self.s.local_embedding_dim}d"
        )
        return {
            "llm_ready": self.llm_ready,
            "llm_model": self.s.llm_model,
            "embedding_mode": self.s.embedding_mode,
            "embedding_model": embedding_model,
            "embedding_signature": self.signature,
        }
