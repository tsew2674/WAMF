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
    conn.execute("""
        CREATE TABLE species_info (
            scientific_name TEXT PRIMARY KEY,
            common_name TEXT,
            description TEXT,
            wikipedia_url TEXT,
            ebird_url TEXT,
            inaturalist_url TEXT,
            gbif_url TEXT,
            last_updated TEXT,
            thumbnail_url TEXT
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


def test_species_are_ranked_with_first_and_last_seen_in_the_range(history_queries):
    species = history_queries.get_history_species(date(2026, 11, 8), TODAY)

    assert [(item['common_name'], item['total']) for item in species] == [
        ('Blue Jay', 3),
        ('American Robin', 1),
        ('Squirrel', 1),
    ]
    # The robin was first seen in 2025, but only once inside this window.
    robin = species[1]
    assert robin['first_seen'] == '2026-11-14 23:59:59'
    assert robin['last_seen'] == '2026-11-14 23:59:59'
    jay = species[0]
    assert jay['first_seen'] == '2026-11-10 09:00:00'
    assert jay['last_seen'] == '2026-11-14 06:00:00'
    # A tracked non-bird label has no taxonomy row and keeps its own name.
    assert species[2]['scientific_name'] == 'Squirrel'


def test_species_dates_span_everything_for_the_all_time_range(history_queries):
    species = history_queries.get_history_species(date(2025, 3, 2), TODAY)
    robin = next(item for item in species if item['common_name'] == 'American Robin')

    assert robin['total'] == 3
    assert robin['first_seen'] == '2025-03-02 07:00:00'
    assert robin['last_seen'] == '2026-11-14 23:59:59'


def test_records_for_a_species_are_newest_first_and_paged(history_queries):
    first = history_queries.get_records_for_scientific_name('Cyanocitta cristata', 1, 2)
    second = history_queries.get_records_for_scientific_name('Cyanocitta cristata', 2, 2)

    assert [record['detection_time'] for record in first] == [
        '2026-11-14 06:00:00',
        '2026-11-10 09:30:00',
    ]
    assert [record['detection_time'] for record in second] == ['2026-11-10 09:00:00']
    assert first[0]['common_name'] == 'Blue Jay'


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
    import webui
    monkeypatch.setattr(routes.public, 'datetime', _FixedDatetime)
    # The species page would otherwise queue a Wikipedia lookup.
    monkeypatch.setattr(webui, 'queue_metadata_refresh', lambda name: None)
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


def test_history_range_limits_counts_and_dates(history_client):
    page = history_client.get('/history?range=7d').get_data(as_text=True)

    assert '8 Nov 2026 – 14 Nov 2026 (7 days)' in page
    assert 'Detections per day' in page
    assert 'Tuesday 10 Nov 2026: 3 detections' in page
    assert 'href="/daily_summary/2026-11-10"' in page
    # The robin's 2025 sighting is outside the last 7 days.
    assert '2 Mar 2025' not in page
    assert '<span>First seen 10 Nov 2026</span>' in page
    assert 'href="/detections/by_scientific_name/Squirrel"' in page
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


# ---------------------------------------------------------------------------
# All-time species page
# ---------------------------------------------------------------------------

def test_species_page_without_a_date_shows_all_time_stats(history_client):
    response = history_client.get('/detections/by_scientific_name/Cyanocitta%20cristata')
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert re.search(r'Total Detections</h3>\s*<div class="wamf-summary-total">3</div>', page)
    assert 'Detections on' not in page
    # First and last seen across every day, not one.
    assert re.search(r'First Seen</h3>.*?2026-11-10\s*<br />\s*09:00', page, re.S)
    assert re.search(r'Last Seen</h3>.*?2026-11-14\s*<br />\s*06:00', page, re.S)
    # Two of the three sightings were in the 09:00 hour.
    assert re.search(r'Peak Hour</h3>\s*<div class="wamf-summary-total">09:00</div>', page)
    # Each detection shows its date, since they span several days.
    assert re.search(r'Seen 2026-11-10\s+at 09:30', page)
    assert 'View all-time stats' not in page


def test_species_page_all_time_works_for_non_bird_labels(history_client):
    page = history_client.get('/detections/by_scientific_name/Squirrel').get_data(as_text=True)

    assert re.search(r'Total Detections</h3>\s*<div class="wamf-summary-total">1</div>', page)
    assert re.search(r'Seen 2026-11-10\s+at 10:00', page)


def test_species_page_without_a_date_is_404_for_an_unknown_species(history_client):
    assert history_client.get('/detections/by_scientific_name/Dodo').status_code == 404


def test_dated_species_page_keeps_daily_stats_and_links_to_all_time(history_client):
    page = history_client.get(
        '/detections/by_scientific_name/Cyanocitta%20cristata/2026-11-10'
    ).get_data(as_text=True)

    assert re.search(
        r'Detections on 2026-11-10</h3>\s*<div class="wamf-summary-total">2</div>', page
    )
    assert 'href="/detections/by_scientific_name/Cyanocitta%20cristata"' in page
    assert 'View all-time stats' in page
    assert 'Seen 2026-11-10' not in page


def test_species_page_all_time_pagination_stays_all_time(
    flask_client, tmp_path, patched_queries, monkeypatch
):
    import webui
    db = str(tmp_path / 'many.db')
    _make_db(db, [
        (f'2026-11-{day:02d} 08:00:00', 'Squirrel', 'feeder')
        for day in range(1, 31)
    ])
    monkeypatch.setattr(patched_queries, 'DBPATH', db)
    monkeypatch.setattr(webui, 'queue_metadata_refresh', lambda name: None)

    first = flask_client.get('/detections/by_scientific_name/Squirrel').get_data(as_text=True)
    second = flask_client.get(
        '/detections/by_scientific_name/Squirrel?page=2'
    ).get_data(as_text=True)

    assert 'Page 1 of 2 (30 detections)' in first
    assert 'href="/detections/by_scientific_name/Squirrel?page=2"' in first
    assert first.count('% confidence') == 25
    assert 'Page 2 of 2 (30 detections)' in second
    assert second.count('% confidence') == 5
    assert 'href="/detections/by_scientific_name/Squirrel?page=1"' in second
