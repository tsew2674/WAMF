"""Tests for the History page: range helpers, queries and the route."""
from datetime import date, datetime
import re
import sqlite3

import pytest

from app import history

TODAY = date(2026, 11, 14)


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 14, 12, 0, 0)


def _make_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_time TEXT NOT NULL,
            detection_index INTEGER,
            score REAL,
            display_name TEXT,
            category_name TEXT,
            frigate_event TEXT,
            camera_name TEXT,
            wamf_snapshot_path TEXT,
            wamf_clip_path TEXT
        )
    """)
    conn.executemany(
        """
        INSERT INTO detections
            (detection_time, detection_index, score, display_name,
             category_name, frigate_event, camera_name)
        VALUES (?, -1, 0.9, ?, 'bird', ?, ?)
        """,
        [
            (when, name, f'evt-{index}', camera)
            for index, (when, name, camera) in enumerate(rows)
        ],
    )
    conn.commit()
    conn.close()


ROWS = [
    # An old robin, outside every window except "all".
    ('2025-03-02 07:00:00', 'Turdus migratorius', 'bird_feeder'),
    ('2026-10-20 08:00:00', 'Turdus migratorius', 'bird_feeder'),
    ('2026-11-10 09:00:00', 'Cyanocitta cristata', 'bird_feeder'),
    ('2026-11-10 09:30:00', 'Cyanocitta cristata', 'native_garden'),
    ('2026-11-10 10:00:00', 'Squirrel', 'native_garden'),
    ('2026-11-14 06:00:00', 'Cyanocitta cristata', ''),
    ('2026-11-14 23:59:59', 'Turdus migratorius', None),
]


@pytest.fixture()
def history_queries(tmp_path, tmp_dbs, patched_queries, monkeypatch):
    db = str(tmp_path / 'history.db')
    _make_db(db, ROWS)
    monkeypatch.setattr(patched_queries, 'DBPATH', db)
    return patched_queries


# ---------------------------------------------------------------------------
# Range and bucketing helpers
# ---------------------------------------------------------------------------

def test_resolve_range_unknown_key_falls_back_to_all_time():
    key, start, end = history.resolve_range('bogus', TODAY, date(2026, 9, 1))
    assert (key, start, end) == ('all', date(2026, 9, 1), TODAY)


def test_resolve_range_window_is_inclusive_of_today():
    _key, start, end = history.resolve_range('7d', TODAY, date(2020, 1, 1))
    assert (start, end) == (date(2026, 11, 8), TODAY)


def test_resolve_range_never_starts_before_first_detection():
    _key, start, _end = history.resolve_range('1y', TODAY, date(2026, 9, 21))
    assert start == date(2026, 9, 21)


def test_resolve_range_without_detections_is_just_today():
    _key, start, end = history.resolve_range('30d', TODAY, None)
    assert (start, end) == (TODAY, TODAY)


@pytest.mark.parametrize('span_days, expected', [
    (1, 'day'), (31, 'day'), (32, 'week'), (182, 'week'),
    (183, 'month'), (1100, 'month'), (1101, 'year'),
])
def test_pick_granularity_thresholds(span_days, expected):
    start = date.fromordinal(TODAY.toordinal() - span_days + 1)
    assert history.pick_granularity(start, TODAY) == expected


def test_build_trend_zero_fills_days_without_detections():
    trend = history.build_trend(
        {date(2026, 11, 10): 3, date(2026, 11, 14): 6},
        date(2026, 11, 8),
        TODAY,
    )
    assert trend['granularity'] == 'day'
    assert [bucket['total'] for bucket in trend['buckets']] == [0, 0, 3, 0, 0, 0, 6]
    assert trend['peak'] == 6
    assert trend['buckets'][2]['percent'] == 50.0
    assert trend['buckets'][-1]['description'] == 'Saturday 14 Nov 2026'


def test_build_trend_groups_weeks_from_monday():
    # 2026-09-21 is a Monday; a 55-day span buckets by week.
    start = date(2026, 9, 21)
    trend = history.build_trend(
        {date(2026, 9, 21): 1, date(2026, 9, 27): 2, date(2026, 9, 28): 4},
        start,
        TODAY,
    )
    assert trend['granularity'] == 'week'
    assert [bucket['total'] for bucket in trend['buckets'][:2]] == [3, 4]
    assert trend['buckets'][0]['description'] == 'Week of 21 Sep 2026'
    assert sum(bucket['total'] for bucket in trend['buckets']) == 7


def test_build_trend_months_and_years_cover_the_whole_span():
    months = history.build_trend({date(2026, 1, 31): 2}, date(2025, 12, 15), TODAY)
    assert months['granularity'] == 'month'
    assert [bucket['date'] for bucket in months['buckets'][:3]] == [
        '2025-12-01', '2026-01-01', '2026-02-01',
    ]
    assert months['buckets'][1]['total'] == 2
    assert months['buckets'][1]['label'] == 'Jan ’26'

    years = history.build_trend({date(2021, 6, 1): 5}, date(2020, 2, 29), TODAY)
    assert years['granularity'] == 'year'
    assert [bucket['label'] for bucket in years['buckets']] == [
        '2020', '2021', '2022', '2023', '2024', '2025', '2026',
    ]


def test_build_trend_thins_axis_labels():
    trend = history.build_trend({}, date(2026, 10, 16), TODAY)
    labelled = [bucket for bucket in trend['buckets'] if bucket['show_label']]
    assert len(trend['buckets']) == 30
    assert 0 < len(labelled) <= history.MAX_AXIS_LABELS
    narrow = [bucket for bucket in labelled if not bucket['label_wide_only']]
    assert 0 < len(narrow) <= history.MAX_AXIS_LABELS // 2
    assert trend['peak'] == 0
    assert all(bucket['percent'] == 0 for bucket in trend['buckets'])


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def test_daily_counts_include_both_ends_of_the_range(history_queries):
    counts = history_queries.get_history_daily_counts(date(2026, 11, 10), TODAY)
    assert counts == {date(2026, 11, 10): 3, TODAY: 2}


def test_daily_counts_exclude_days_outside_the_range(history_queries):
    counts = history_queries.get_history_daily_counts(
        date(2026, 11, 11), date(2026, 11, 13)
    )
    assert counts == {}


def test_species_are_ranked_with_all_time_first_and_last_seen(history_queries):
    species = history_queries.get_history_species(date(2026, 11, 8), TODAY)

    assert [(item['common_name'], item['total']) for item in species] == [
        ('Blue Jay', 3),
        ('American Robin', 1),
        ('Squirrel', 1),
    ]
    robin = species[1]
    # Only one robin falls inside the window, but it was first seen in 2025.
    assert robin['first_seen'] == '2025-03-02 07:00:00'
    assert robin['last_seen'] == '2026-11-14 23:59:59'
    # A tracked non-bird label has no taxonomy row and keeps its own name.
    assert species[2]['scientific_name'] == 'Squirrel'


def test_cameras_group_missing_names_as_unknown(history_queries):
    cameras = history_queries.get_history_cameras(date(2025, 1, 1), TODAY)

    assert cameras == [
        {'camera': 'bird_feeder', 'total': 3, 'species_count': 2},
        {'camera': 'Unknown', 'total': 2, 'species_count': 2},
        {'camera': 'native_garden', 'total': 2, 'species_count': 2},
    ]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@pytest.fixture()
def history_client(flask_client, history_queries, monkeypatch):
    import routes.public
    monkeypatch.setattr(routes.public, 'datetime', _FixedDatetime)
    return flask_client


def test_history_defaults_to_all_time(history_client):
    response = history_client.get('/history')
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert '2 Mar 2025 – 14 Nov 2026' in page
    assert 'Detections per month' in page
    assert re.search(r'aria-current="page"\s*>All time<', page)
    # 7 detections over 623 days, on 4 of them.
    assert 'Detections on 4 of 623 days' in page
    assert '<span>First seen 2 Mar 2025</span>' in page
    assert '<span>Last seen 14 Nov 2026</span>' in page


def test_history_range_limits_counts_but_not_first_seen(history_client):
    page = history_client.get('/history?range=7d').get_data(as_text=True)

    assert '8 Nov 2026 – 14 Nov 2026 (7 days)' in page
    assert 'Detections per day' in page
    assert 'Tuesday 10 Nov 2026: 3 detections' in page
    assert 'href="/daily_summary/2026-11-10"' in page
    assert 'First seen 2 Mar 2025' in page
    assert '/detections/by_scientific_name/Squirrel/2026-11-10' in page
    assert '40.0% of detections · 2 species' in page


def test_history_unknown_range_falls_back(history_client):
    response = history_client.get('/history?range=../../etc')
    assert response.status_code == 200
    assert 'Detections per month' in response.get_data(as_text=True)


def test_history_empty_database(flask_client, tmp_path, patched_queries, monkeypatch):
    db = str(tmp_path / 'empty.db')
    _make_db(db, [])
    monkeypatch.setattr(patched_queries, 'DBPATH', db)

    response = flask_client.get('/history')
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'No detections in this range yet.' in page
    assert 'Species Leaderboard' not in page


def test_history_is_linked_from_the_navbar(flask_client):
    page = flask_client.get('/').get_data(as_text=True)
    assert 'href="/history"' in page
