#!/usr/bin/env python3
"""Repeatable, dependency-free HTTP measurements for ATC Roster surfaces.

Run against a local or staging instance.  Supply an authenticated session cookie
to measure protected pages without placing credentials in source control.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import statistics
import time
from html.parser import HTMLParser
from http.cookiejar import Cookie
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener


class ElementCounter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.count += 1


def make_cookie(name: str, value: str, host: str) -> Cookie:
    return Cookie(0, name, value, None, False, host, False, False, "/", True,
                  False, None, True, None, None, {}, False)


def measure(url: str, repetitions: int, cookie: str | None) -> dict[str, object]:
    from http.cookiejar import CookieJar
    from urllib.parse import urlparse

    jar = CookieJar()
    parsed = urlparse(url)
    if cookie:
        name, value = cookie.split("=", 1)
        jar.set_cookie(make_cookie(name, value, parsed.hostname or "localhost"))
    opener = build_opener(HTTPCookieProcessor(jar))
    timings: list[float] = []
    body = b""
    status = 0
    for _ in range(repetitions):
        started = time.perf_counter()
        try:
            with opener.open(Request(url, headers={"Accept-Encoding": "identity"}), timeout=30) as response:
                body = response.read()
                status = response.status
        except HTTPError as error:
            body, status = error.read(), error.code
        timings.append((time.perf_counter() - started) * 1000)
    counter = ElementCounter()
    counter.feed(body.decode("utf-8", errors="replace"))
    return {"url": url, "status": status, "samples": repetitions,
            "response_ms_median": round(statistics.median(timings), 2),
            "response_ms_min": round(min(timings), 2), "html_bytes": len(body),
            "gzip_bytes": len(gzip.compress(body)), "dom_elements": counter.count}


def limit_failures(
    measurements: list[dict[str, object]],
    *,
    max_median_ms: float | None,
    max_html_bytes: int | None,
    max_dom_elements: int | None,
) -> list[str]:
    """Return human-readable breaches without concealing the measurements."""
    failures: list[str] = []
    for measurement in measurements:
        url = str(measurement["url"])
        if (
            max_median_ms is not None
            and float(measurement["response_ms_median"]) > max_median_ms
        ):
            failures.append(f"{url}: median response time exceeds {max_median_ms}ms")
        if max_html_bytes is not None and int(measurement["html_bytes"]) > max_html_bytes:
            failures.append(f"{url}: HTML payload exceeds {max_html_bytes} bytes")
        if (
            max_dom_elements is not None
            and int(measurement["dom_elements"]) > max_dom_elements
        ):
            failures.append(f"{url}: DOM elements exceed {max_dom_elements}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="+", help="Absolute URLs to measure")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cookie", help="Authenticated cookie as NAME=VALUE")
    parser.add_argument("--max-median-ms", type=float)
    parser.add_argument("--max-html-bytes", type=int)
    parser.add_argument("--max-dom-elements", type=int)
    args = parser.parse_args()
    measurements = [measure(url, args.repetitions, args.cookie) for url in args.urls]
    print(json.dumps(measurements, indent=2))
    failures = limit_failures(
        measurements,
        max_median_ms=args.max_median_ms,
        max_html_bytes=args.max_html_bytes,
        max_dom_elements=args.max_dom_elements,
    )
    if failures:
        print("Performance budget failed:", *failures, sep="\n- ", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
