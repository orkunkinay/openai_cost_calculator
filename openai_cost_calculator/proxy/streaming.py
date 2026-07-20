"""Incremental content decoding for streaming accounting.

When a streamed upstream response is content-encoded (for example gzip), the
raw bytes forwarded to the client are compressed.  Local accounting observes a
*decompressed copy* of that stream so it can parse usage, while the bytes sent
to the client remain exactly as received.  Only gzip, deflate, and identity are
supported; any other encoding is reported as unsupported so the request is
marked unavailable rather than silently costed as zero.
"""

from __future__ import annotations

import zlib
from typing import Optional


class IncrementalDecoder:
    """Feed compressed chunks, get decompressed bytes for accounting only."""

    def __init__(self, content_encoding: Optional[str]) -> None:
        self.encoding = (content_encoding or "identity").strip().lower()
        self.supported = True
        self._decompressor: Optional["zlib._Decompress"] = None
        self._deflate_retry = False
        if self.encoding in ("gzip", "x-gzip"):
            self._decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif self.encoding == "deflate":
            self._decompressor = zlib.decompressobj()
            self._deflate_retry = True
        elif self.encoding in ("", "identity"):
            self._decompressor = None
        else:
            self.supported = False

    def decompress(self, chunk: bytes) -> bytes:
        if not self.supported or not chunk:
            return b""
        if self._decompressor is None:
            return chunk
        try:
            return self._decompressor.decompress(chunk)
        except zlib.error:
            # Some servers send raw (headerless) deflate; retry once with the
            # raw-deflate window before giving up.
            if self._deflate_retry:
                self._deflate_retry = False
                self._decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
                try:
                    return self._decompressor.decompress(chunk)
                except zlib.error:
                    pass
            self.supported = False
            return b""

    def flush(self) -> bytes:
        if self._decompressor is None or not self.supported:
            return b""
        try:
            return self._decompressor.flush()
        except zlib.error:
            return b""


__all__ = ["IncrementalDecoder"]
