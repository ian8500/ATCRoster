"""Small response-compression boundary for dynamic application pages."""

from __future__ import annotations

import gzip

from flask import Flask, Response, request


COMPRESSIBLE_TYPES = (
    "application/javascript",
    "application/json",
    "application/xml",
    "image/svg+xml",
    "text/",
)


def register_response_compression(app: Flask, *, minimum_size: int = 1024) -> None:
    """Gzip suitable responses when the browser advertises support."""

    def compress_response(response: Response) -> Response:
        accepted = request.headers.get("Accept-Encoding", "").lower()
        content_type = response.headers.get("Content-Type", "").lower()
        if (
            "gzip" not in accepted
            or response.status_code < 200
            or response.status_code >= 300
            or response.direct_passthrough
            or response.is_streamed
            or response.headers.get("Content-Encoding")
            or content_type.startswith("text/event-stream")
            or request.headers.get("Range")
            or not any(content_type.startswith(item) for item in COMPRESSIBLE_TYPES)
        ):
            return response
        payload = response.get_data()
        if len(payload) < minimum_size:
            return response
        compressed = gzip.compress(payload, compresslevel=5)
        if len(compressed) >= len(payload):
            return response
        response.set_data(compressed)
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Content-Length"] = str(len(compressed))
        response.headers.add("Vary", "Accept-Encoding")
        return response

    compress_response.__name__ = "_compress_response"
    app.after_request(compress_response)
