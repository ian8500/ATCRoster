from datetime import date

from atcroster.roster import impacts
from atcroster.roster.impacts import invalidate_impact_months


def test_cache_invalidation_failure_is_visible_but_does_not_block_an_impact(
    monkeypatch,
):
    class FailingCache:
        @staticmethod
        def delete_memoized(*_args):
            raise ConnectionError("cache unavailable")

    warnings = []
    monkeypatch.setattr(
        impacts.logger, "warning", lambda *args: warnings.append(args)
    )

    invalidate_impact_months(
        7,
        date(2026, 8, 12),
        date(2026, 9, 1),
        cache=FailingCache(),
        cached_loader=lambda *_args: None,
        add_months=lambda year, month, _offset: (
            (year + 1, 1) if month == 12 else (year, month + 1)
        ),
    )

    assert [entry[:4] for entry in warnings] == [
        (
            "roster_cache_invalidation_failed unit_id=%s year=%s month=%s error=%s",
            7,
            2026,
            8,
        ),
        (
            "roster_cache_invalidation_failed unit_id=%s year=%s month=%s error=%s",
            7,
            2026,
            9,
        ),
    ]
    assert [str(entry[4]) for entry in warnings] == [
        "cache unavailable",
        "cache unavailable",
    ]
