"""Keep secrets out of the service's logs.

Requests carry presigned URLs (`image_urls`, or https:// URIs whose query string is a signature),
and webhooks may carry tokens. Nothing the service logs is a request body; what it does log
(error messages, webhook targets) goes through these first."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

REDACTED = "<redacted>"
_URL = re.compile(r"(?P<base>[a-zA-Z][a-zA-Z0-9+.-]*://[^\s?#'\"<>]*)(?P<rest>[?#][^\s'\"<>]*)?")
_SECRET_KEYS = {"image_urls", "authorization", "api_key", "webhook"}


def redact_text(text: str) -> str:
    """Strip the query string and fragment from every URL in text, and any userinfo."""
    def sub(m: re.Match) -> str:
        base = m.group("base")
        if "@" in base.split("://", 1)[1].split("/", 1)[0]:          # scheme://user:pass@host
            scheme, rest = base.split("://", 1)
            host, _, path = rest.partition("/")
            base = f"{scheme}://{REDACTED}@{host.rsplit('@', 1)[1]}" + (f"/{path}" if path else "")
        rest = m.group("rest") or ""
        tail = rest[len(rest.rstrip(".,:;)]}")):] if rest else ""     # sentence punctuation stays
        return base + ("?" + REDACTED + tail if rest else "")
    return _URL.sub(sub, str(text))


def redact_url(url: str) -> str:
    """scheme://host/path of a URL (for logging a webhook target)."""
    try:
        p = urlsplit(url)
        return f"{p.scheme}://{p.hostname or ''}{p.path}"
    except ValueError:
        return REDACTED


def redact(obj: Any) -> Any:
    """A copy of a JSON value safe to log: secret-named keys hidden, URLs stripped."""
    if isinstance(obj, dict):
        return {k: (REDACTED if str(k).lower() in _SECRET_KEYS else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj
