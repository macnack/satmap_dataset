from __future__ import annotations

import asyncio
from dataclasses import dataclass
import random
from typing import Any
import logging

import httpx

logger = logging.getLogger("satmap_dataset.http")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 8
    backoff_seconds: float = 1.0
    jitter_seconds: float = 0.6
    retry_for_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)


async def request_with_retry(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 20.0,
    retry_policy: RetryPolicy | None = None,
    client: httpx.AsyncClient | None = None,
) -> httpx.Response:
    policy = retry_policy or RetryPolicy()
    owns_client = client is None
    active_client = client or httpx.AsyncClient(timeout=timeout)

    def _sleep_seconds(attempt: int) -> float:
        base = policy.backoff_seconds * (2 ** (attempt - 1))
        jitter = random.uniform(0.0, policy.jitter_seconds)
        return base + jitter

    try:
        for attempt in range(1, policy.max_attempts + 1):
            try:
                response = await active_client.request(method=method, url=url, params=params)
                if (
                    response.status_code in policy.retry_for_statuses
                    and attempt < policy.max_attempts
                ):
                    logger.warning(
                        "HTTP retryable status=%s attempt=%s/%s url=%s",
                        response.status_code,
                        attempt,
                        policy.max_attempts,
                        url,
                    )
                    await asyncio.sleep(_sleep_seconds(attempt))
                    continue

                response.raise_for_status()
                return response
            except httpx.HTTPError:
                if attempt >= policy.max_attempts:
                    logger.error("HTTP failed after %s attempts url=%s", attempt, url)
                    raise
                logger.warning("HTTP exception, retrying attempt=%s/%s url=%s", attempt, policy.max_attempts, url)
                await asyncio.sleep(_sleep_seconds(attempt))
    finally:
        if owns_client:
            await active_client.aclose()

    raise RuntimeError("request_with_retry exhausted attempts without a terminal result")
