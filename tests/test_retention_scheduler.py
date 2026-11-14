"""Tests for the built-in daily retention schedule."""
from datetime import datetime, time, timedelta, timezone
import sqlite3
import threading
from zoneinfo import ZoneInfo

import pytest

import retention
from app import retention_scheduler as scheduler
from app import system_events

NEW_YORK = ZoneInfo('America/New_York')


def _utc(*parts):
    return datetime(*parts, tzinfo=timezone.utc)


def _config(**schedule):
    return {'retention': {'schedule': schedule}}


@pytest.fixture()
def events(monkeypatch):
    logged = []
    monkeypatch.setattr(
        scheduler,
        'log_system_event',
        lambda severity, event_type, message: logged.append((severity, message)),
    )
    return logged


# ---------------------------------------------------------------------------
# parse_schedule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('config', [
    None,
    {},
    {'retention': {}},
    {'retention': {'schedule': None}},
    _config(enabled=False, time='not checked while disabled'),
])
def test_schedule_is_off_unless_enabled(config):
    assert scheduler.parse_schedule(config) is None


def test_schedule_defaults_to_three_am_utc():
    daily_time, tzinfo = scheduler.parse_schedule(_config(enabled=True))
    assert daily_time == time(3, 0)
    assert tzinfo.key == 'UTC'


def test_schedule_reads_time_and_timezone():
    daily_time, tzinfo = scheduler.parse_schedule(
        _config(enabled=True, time='04:15', timezone='America/New_York')
    )
    assert daily_time == time(4, 15)
    assert tzinfo.key == 'America/New_York'


def test_unquoted_yaml_time_is_understood():
    # PyYAML reads an unquoted 13:30 as the base-60 integer 810.
    daily_time, _tzinfo = scheduler.parse_schedule(_config(enabled=True, time=810))
    assert daily_time == time(13, 30)


@pytest.mark.parametrize('schedule, field', [
    ({'enabled': 'yes'}, 'retention.schedule.enabled'),
    ({'enabled': True, 'time': '3am'}, 'retention.schedule.time'),
    ({'enabled': True, 'time': '24:00'}, 'retention.schedule.time'),
    ({'enabled': True, 'time': '03:60'}, 'retention.schedule.time'),
    ({'enabled': True, 'time': 1440}, 'retention.schedule.time'),
    ({'enabled': True, 'time': True}, 'retention.schedule.time'),
    ({'enabled': True, 'timezone': 'Mars/Olympus'}, 'retention.schedule.timezone'),
    ({'enabled': True, 'timezone': ''}, 'retention.schedule.timezone'),
    ({'enabled': True, 'timezone': 5}, 'retention.schedule.timezone'),
])
def test_invalid_schedule_names_the_bad_field(schedule, field):
    with pytest.raises(ValueError, match=field):
        scheduler.parse_schedule({'retention': {'schedule': schedule}})


def test_schedule_must_be_a_mapping():
    with pytest.raises(ValueError, match='retention.schedule must be'):
        scheduler.parse_schedule({'retention': {'schedule': ['03:00']}})


# ---------------------------------------------------------------------------
# next_run
# ---------------------------------------------------------------------------

def test_next_run_is_later_today_when_the_time_has_not_passed():
    # 02:59 in New York (EST, UTC-5).
    assert scheduler.next_run(
        _utc(2026, 11, 14, 7, 59), time(3, 0), NEW_YORK
    ) == _utc(2026, 11, 14, 8, 0)


@pytest.mark.parametrize('now', [
    _utc(2026, 11, 14, 8, 0),
    _utc(2026, 11, 14, 8, 0, 1),
    _utc(2026, 11, 15, 4, 0),
])
def test_next_run_is_tomorrow_once_the_time_has_passed(now):
    assert scheduler.next_run(now, time(3, 0), NEW_YORK) == _utc(2026, 11, 15, 8, 0)


def test_next_run_follows_the_configured_zone_not_utc():
    # 23:30 UTC on the 14th is already 08:30 on the 15th in Tokyo.
    assert scheduler.next_run(
        _utc(2026, 11, 14, 23, 30), time(9, 0), ZoneInfo('Asia/Tokyo')
    ) == _utc(2026, 11, 15, 0, 0)


def test_next_run_in_the_spring_forward_gap_still_happens_that_night():
    # 02:30 does not exist on 8 March 2026 in New York; it runs an hour late.
    assert scheduler.next_run(
        _utc(2026, 3, 8, 6, 0), time(2, 30), NEW_YORK
    ) == _utc(2026, 3, 8, 7, 30)


def test_next_run_in_the_fall_back_hour_happens_once():
    # 01:30 happens twice on 1 November 2026 in New York.
    first = scheduler.next_run(_utc(2026, 11, 1, 4, 0), time(1, 30), NEW_YORK)
    assert first == _utc(2026, 11, 1, 5, 30)
    assert scheduler.next_run(
        first + timedelta(seconds=1), time(1, 30), NEW_YORK
    ) == _utc(2026, 11, 2, 6, 30)


# ---------------------------------------------------------------------------
# scheduler_loop
# ---------------------------------------------------------------------------

class _FakeStop:
    """An Event whose wait() advances a fake clock instead of sleeping."""

    def __init__(self, start):
        self.now = start
        self.stopped = False
        self.waits = []

    def is_set(self):
        return self.stopped

    def wait(self, timeout):
        self.waits.append(timeout)
        self.now += timedelta(seconds=timeout)


def test_loop_runs_once_a_day_at_the_scheduled_time(monkeypatch):
    stop = _FakeStop(_utc(2026, 11, 14, 12, 0))
    runs = []

    def run():
        runs.append(stop.now)
        stop.stopped = len(runs) == 3

    monkeypatch.setattr(scheduler, '_now', lambda: stop.now)
    monkeypatch.setattr(scheduler, 'run_scheduled_retention', run)

    scheduler.scheduler_loop(time(3, 0), NEW_YORK, stop)

    assert runs == [
        _utc(2026, 11, 15, 8, 0),
        _utc(2026, 11, 16, 8, 0),
        _utc(2026, 11, 17, 8, 0),
    ]
    assert max(stop.waits) <= scheduler._MAX_WAIT_SECONDS


def test_loop_stops_without_running_when_asked(monkeypatch):
    stop = _FakeStop(_utc(2026, 11, 14, 12, 0))
    stop.wait = lambda timeout: setattr(stop, 'stopped', True)
    runs = []

    monkeypatch.setattr(scheduler, '_now', lambda: stop.now)
    monkeypatch.setattr(scheduler, 'run_scheduled_retention', lambda: runs.append(1))

    scheduler.scheduler_loop(time(3, 0), NEW_YORK, stop)

    assert runs == []


def test_a_scheduled_run_does_one_full_retention_pass(monkeypatch, events):
    calls = []
    monkeypatch.setattr(retention, 'run_retention', lambda: calls.append('run'))

    scheduler.run_scheduled_retention()

    assert calls == ['run']
    assert events == []


def test_a_failed_run_is_reported_and_does_not_raise(monkeypatch, events):
    def explode():
        raise sqlite3.OperationalError('database is locked')

    monkeypatch.setattr(retention, 'run_retention', explode)

    scheduler.run_scheduled_retention()

    assert len(events) == 1
    assert events[0][0] == 'ERROR'
    assert 'Scheduled retention run failed' in events[0][1]


# ---------------------------------------------------------------------------
# start_retention_scheduler
# ---------------------------------------------------------------------------

@pytest.fixture()
def parked_loop(monkeypatch):
    """Replace the loop with one that idles until the test ends."""
    release = threading.Event()
    calls = []

    def loop(daily_time, tzinfo):
        calls.append((daily_time, tzinfo.key))
        release.wait(5)

    monkeypatch.setattr(scheduler, 'scheduler_loop', loop)
    monkeypatch.setattr(scheduler, '_thread', None)
    yield calls
    release.set()


def test_start_does_nothing_when_the_schedule_is_disabled(parked_loop, events):
    assert scheduler.start_retention_scheduler(_config(enabled=False)) is None
    assert parked_loop == []
    assert events == []


def test_start_reports_an_invalid_schedule_instead_of_crashing(parked_loop, events):
    assert scheduler.start_retention_scheduler(
        _config(enabled=True, time='late')
    ) is None
    assert parked_loop == []
    assert events[0][0] == 'ERROR'
    assert 'retention.schedule.time' in events[0][1]


def test_start_launches_one_background_thread(parked_loop, events):
    config = _config(enabled=True, time='03:00', timezone='America/New_York')

    thread = scheduler.start_retention_scheduler(config)
    again = scheduler.start_retention_scheduler(config)
    thread.join(0.2)

    assert thread.daemon
    assert again is thread
    assert parked_loop == [(time(3, 0), 'America/New_York')]
    assert events[0] == (
        'INFO',
        'Retention scheduled daily at 03:00 (America/New_York)',
    )


# ---------------------------------------------------------------------------
# run_retention: what a scheduled run actually does
# ---------------------------------------------------------------------------

def test_run_keeps_snapshots_and_removes_expired_clips(tmp_path, monkeypatch):
    snapshots = tmp_path / 'snapshots'
    clips = tmp_path / 'clips'
    snapshots.mkdir()
    clips.mkdir()
    db = str(tmp_path / 'speciesid.db')

    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_time TEXT NOT NULL,
            display_name TEXT,
            wamf_snapshot_path TEXT,
            wamf_clip_path TEXT
        );
        CREATE TABLE system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            severity TEXT NOT NULL,
            event_type TEXT NOT NULL,
            message TEXT NOT NULL
        );
        CREATE TABLE retention_status (
            last_run TEXT,
            rows_scanned INTEGER,
            orphan_count INTEGER,
            missing_count INTEGER
        );
    """)
    now = datetime.now()
    for name, age_days in (('old', 120), ('recent', 5)):
        (snapshots / f'{name}.jpg').write_bytes(b'jpg')
        (clips / f'{name}.mp4').write_bytes(b'mp4')
        conn.execute(
            "INSERT INTO detections "
            "(detection_time, display_name, wamf_snapshot_path, wamf_clip_path) "
            "VALUES (?, 'Squirrel', ?, ?)",
            (
                (now - timedelta(days=age_days)).strftime('%Y-%m-%d %H:%M:%S'),
                str(snapshots / f'{name}.jpg'),
                str(clips / f'{name}.mp4'),
            ),
        )
    conn.commit()
    conn.close()
    (clips / 'leftover.mp4').write_bytes(b'mp4')

    config = {'retention': {
        'enabled': True,
        'snapshots_days': 36500,
        'clips_days': 30,
        'delete_media': True,
        'orphan_scan_enabled': True,
        'delete_orphaned_media': True,
    }}
    monkeypatch.setattr(retention, 'load_config', lambda: config)
    monkeypatch.setattr(retention, 'DB_PATH', db)
    monkeypatch.setattr(retention, 'get_snapshots_path', lambda: snapshots)
    monkeypatch.setattr(retention, 'get_clips_path', lambda: clips)
    monkeypatch.setattr(system_events, 'DB_PATH', db)

    retention.run_retention()

    assert sorted(path.name for path in snapshots.iterdir()) == ['old.jpg', 'recent.jpg']
    assert sorted(path.name for path in clips.iterdir()) == ['recent.mp4']

    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT wamf_snapshot_path IS NOT NULL, wamf_clip_path IS NOT NULL "
        "FROM detections ORDER BY id"
    ).fetchall()
    last_run = conn.execute("SELECT COUNT(*) FROM retention_status").fetchone()[0]
    conn.close()

    # Both detections survive; only the old one loses its clip.
    assert rows == [(1, 0), (1, 1)]
    assert last_run == 1
