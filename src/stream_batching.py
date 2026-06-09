"""
Stream Batching (Token Buffer) — Tầng Tạo sinh
Gom lô Token đầu ra (vd: 50ms) trước khi yield qua SSE.
Tối ưu latency perceived bởi người dùng.
"""
import time
import json
import logging
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


class StreamBatcher:
    """
    Token Buffer for SSE streaming.
    Accumulates tokens for a configurable time window before yielding batches.
    This reduces the number of SSE events while keeping perceived latency low.
    """

    def __init__(self, buffer_ms: int = 50):
        self.buffer_ms = buffer_ms
        self.buffer_s = buffer_ms / 1000.0

    def batch(self, token_stream: Iterator[str]) -> Iterator[str]:
        """
        Buffer tokens from the stream and yield in batches.

        Args:
            token_stream: Raw token iterator from LLM

        Yields:
            Batched text chunks
        """
        buffer = []
        last_yield_time = time.time()

        for token in token_stream:
            buffer.append(token)
            now = time.time()

            if now - last_yield_time >= self.buffer_s:
                batch_text = "".join(buffer)
                buffer.clear()
                last_yield_time = now
                if batch_text:
                    yield batch_text

        # Flush remaining buffer
        if buffer:
            yield "".join(buffer)

    def batch_as_sse(self, token_stream: Iterator[str]) -> Iterator[str]:
        """
        Buffer tokens and format as SSE data events.

        Yields:
            SSE-formatted strings: 'data: {"text": "..."}\n\n'
        """
        for batch_text in self.batch(token_stream):
            yield f"data: {json.dumps({'text': batch_text}, ensure_ascii=False)}\n\n"

        # Send completion signal
        yield f"data: {json.dumps({'done': True})}\n\n"


def get_stream_batcher() -> StreamBatcher:
    from src.config import settings
    return StreamBatcher(buffer_ms=settings.stream_buffer_ms)
