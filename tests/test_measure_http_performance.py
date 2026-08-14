from scripts.measure_http_performance import limit_failures


def test_performance_limits_report_each_measurable_breach():
    failures = limit_failures(
        [
            {
                "url": "https://example.test/roster/2026-08",
                "response_ms_median": 800.0,
                "html_bytes": 120_000,
                "dom_elements": 2_000,
            }
        ],
        max_median_ms=500.0,
        max_html_bytes=100_000,
        max_dom_elements=1_500,
    )

    assert failures == [
        "https://example.test/roster/2026-08: median response time exceeds 500.0ms",
        "https://example.test/roster/2026-08: HTML payload exceeds 100000 bytes",
        "https://example.test/roster/2026-08: DOM elements exceed 1500",
    ]


def test_performance_limits_allow_measurements_within_budget():
    assert not limit_failures(
        [
            {
                "url": "https://example.test/roster/2026-08",
                "response_ms_median": 120.0,
                "html_bytes": 12_000,
                "dom_elements": 400,
            }
        ],
        max_median_ms=500.0,
        max_html_bytes=100_000,
        max_dom_elements=1_500,
    )
