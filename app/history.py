"""Date-range and bucketing helpers for the History page."""

from datetime import timedelta
import math

# (key, label, window length in days; None means since the first detection)
RANGES = (
    ('7d', '7 days', 7),
    ('30d', '30 days', 30),
    ('1y', 'Year', 365),
    ('all', 'All time', None),
)
DEFAULT_RANGE = 'all'
MAX_AXIS_LABELS = 8


def resolve_range(key, today, earliest):
    """Return (key, start, end) dates; start never precedes the first detection."""
    windows = {range_key: days for range_key, _label, days in RANGES}
    if key not in windows:
        key = DEFAULT_RANGE

    start = earliest or today
    if windows[key]:
        start = max(start, today - timedelta(days=windows[key] - 1))

    return key, min(start, today), today


def pick_granularity(start, end):
    span_days = (end - start).days + 1
    if span_days <= 31:
        return 'day'
    if span_days <= 182:
        return 'week'
    if span_days <= 1100:
        return 'month'
    return 'year'


def _bucket_start(day, granularity):
    if granularity == 'week':
        return day - timedelta(days=day.weekday())
    if granularity == 'month':
        return day.replace(day=1)
    if granularity == 'year':
        return day.replace(month=1, day=1)
    return day


def _next_bucket(bucket, granularity):
    if granularity == 'week':
        return bucket + timedelta(days=7)
    if granularity == 'month':
        return (bucket.replace(day=28) + timedelta(days=4)).replace(day=1)
    if granularity == 'year':
        return bucket.replace(year=bucket.year + 1)
    return bucket + timedelta(days=1)


def _describe(bucket, granularity):
    """Return (axis label, full description) for one bucket."""
    day_month = f'{bucket.day} {bucket:%b}'
    if granularity == 'week':
        return day_month, f'Week of {day_month} {bucket.year}'
    if granularity == 'month':
        return f'{bucket:%b} ’{bucket:%y}', f'{bucket:%B %Y}'
    if granularity == 'year':
        return str(bucket.year), str(bucket.year)
    return day_month, f'{bucket:%A} {day_month} {bucket.year}'


def build_trend(daily_counts, start, end):
    """Zero-filled buckets from start to end, sized for a bar chart.

    daily_counts maps a date to its detection count.
    """
    granularity = pick_granularity(start, end)

    totals = {}
    bucket = _bucket_start(start, granularity)
    while bucket <= end:
        totals[bucket] = 0
        bucket = _next_bucket(bucket, granularity)

    for day, total in daily_counts.items():
        if start <= day <= end:
            totals[_bucket_start(day, granularity)] += total

    peak = max(totals.values(), default=0)
    label_every = max(1, math.ceil(len(totals) / MAX_AXIS_LABELS))
    # Phones fit about half as many labels; keep a subset of the wide ones.
    labelled = math.ceil(len(totals) / label_every)
    narrow_every = label_every * (2 if labelled > MAX_AXIS_LABELS // 2 else 1)

    buckets = []
    for index, (bucket, total) in enumerate(totals.items()):
        label, description = _describe(bucket, granularity)
        buckets.append({
            'date': bucket.isoformat(),
            'label': label,
            'description': description,
            'total': total,
            'percent': round(total / peak * 100, 1) if peak else 0,
            'show_label': index % label_every == 0,
            'label_wide_only': index % narrow_every != 0,
        })

    return {
        'granularity': granularity,
        'peak': peak,
        'buckets': buckets,
    }
