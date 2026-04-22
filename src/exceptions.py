"""Shared exception types.

Re-exports the OpenAI SDK's ``APITimeoutError`` so prod code can catch it
via ``src.exceptions.APITimeoutError`` without importing the SDK directly.
The SDK raises this exception when its internal retry budget exhausts on
timeouts, which is the signal helper callers use to fall back to a soft
summary (``topic.shortened``) rather than crashing the crawl.
"""

from openai import APITimeoutError

__all__ = ["APITimeoutError"]
