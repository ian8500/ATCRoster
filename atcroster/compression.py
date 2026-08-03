"""Conservative response compression without affecting streamed kiosk feeds."""

from __future__ import annotations

import gzip
from typing import Any


COMPRESSIBLE_MIMETYPES = frozenset({
    "text/html", "text/css", "text/javascript", "application/javascript",
    "application/json", "application/manifest+json", "image/svg+xml",
})


def register_response_compression(app: Any, minimum_size: int = 1024) -> None:
    @app.after_request
    def compress_response(response):
        from flask import request

        response.vary.add("Accept-Encoding")
        accepts_gzip = request.accept_encodings["gzip"] > 0
        if (
            not accepts_gzip
            or response.direct_passthrough
            or response.is_streamed
            or response.status_code < 200
            or response.status_code in {204, 304}
            or response.headers.get("Content-Encoding")
            or response.mimetype not in COMPRESSIBLE_MIMETYPES
        ):
            return response
        payload = response.get_data()
        if len(payload) < minimum_size:
            return response
        response.set_data(gzip.compress(payload, compresslevel=5))
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Content-Length"] = str(len(response.get_data()))
        response.headers.pop("ETag", None)
        return response

    return None
