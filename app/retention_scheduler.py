"""Optional daily in-process retention run, configured by retention.schedule."""

from datetime import datetime, time, timedelta, timezone
import logging
import re
import threading
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.logging_utils import safe_exception_details
from app.system_events import log_system_event

logger = logging.getLogger(__name__)

DEFAULT_TIME = '03:00'
DEFAULT_TIMEZONE = 'UTC'
_TIME_PATTERN = re.compile(r'([0-9]{2}):([0-9]{2})')
# Re-check the clock at least this often so a wall-clock jump cannot strand a run.
_MAX_WAIT_SECONDS = 300

_thread = None
_start_lock = threading.Lock()


def _now():
    return datetime.now(timezone.utc)


def _parse_time(value):
    # Unquoted YAML such as `time: 13:30` arrives as base-60 integer 810.
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 1440:
        return time(*divmod(value, 60))

    match = _TIME_PATTERN.fullmatch(value) if isinstance(value, str) else None
    if match and int(match.group(1)) <= 23 and int(match.group(2)) <= 59:
        return time(int(match.group(1)), int(match.group(2)))

    raise ValueError('retention.schedule.time must be a 24-hour "HH:MM" value')


def parse_schedule(config):
    """Return (daily_time, tzinfo) if the schedule is enabled, otherwise None."""
    schedule = ((config or {}).get('retention') or {}).get('schedule') or {}
    if not isinstance(schedule, dict):
        raise ValueError('retention.schedule must be a YAML mapping')

    enabled = schedule.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('retention.schedule.enabled must be true or false')
    if not enabled:
        return None

    daily_time = _parse_time(schedule.get('time', DEFAULT_TIME))

    timezone_name = schedule.get('timezone', DEFAULT_TIMEZONE)
    try:
        tzinfo = ZoneInfo(timezone_name)
    except (TypeError, ValueError, OSError, ZoneInfoNotFoundError):
        raise ValueError(
            'retention.schedule.timezone must be an IANA name such as '
            'America/New_York'
        ) from None

    return daily_time, tzinfo


def next_run(now, daily_time, tzinfo):
    """The first scheduled moment strictly after now."""

    def scheduled_on(day):
        # Returned in UTC: same-zone datetimes compare by wall clock, which is
        # wrong across a daylight-saving change.
        return datetime.combine(day, daily_time, tzinfo=tzinfo).astimezone(
            timezone.utc
        )

    today = now.astimezone(tzinfo).date()
    candidate = scheduled_on(today)
    if candidate <= now:
        candidate = scheduled_on(today + timedelta(days=1))
    return candidate


def run_scheduled_retention():
    """Run retention once; a failure must not end the scheduler thread."""
    try:
        # Imported here so WAMF's startup never depends on the retention script.
        import retention

        retention.run_retention()
    except Exception as error:
        details = safe_exception_details(error)
        logger.error('Scheduled retention run failed: %s', details)
        log_system_event(
            'ERROR',
            'RETENTION',
            f'Scheduled retention run failed: {details}',
        )


def scheduler_loop(daily_time, tzinfo, stop_event=None):
    stop_event = stop_event or threading.Event()

    while not stop_event.is_set():
        due = next_run(_now(), daily_time, tzinfo)

        while not stop_event.is_set():
            remaining = (due - _now()).total_seconds()
            if remaining <= 0:
                break
            stop_event.wait(min(remaining, _MAX_WAIT_SECONDS))

        if stop_event.is_set():
            return

        run_scheduled_retention()


def start_retention_scheduler(config):
    """Start the daily retention thread when retention.schedule is enabled."""
    global _thread

    try:
        schedule = parse_schedule(config)
    except ValueError as error:
        logger.error('Retention schedule not started: %s', error)
        log_system_event(
            'ERROR',
            'RETENTION',
            f'Retention schedule not started: {error}',
        )
        return None

    if schedule is None:
        return None

    daily_time, tzinfo = schedule

    with _start_lock:
        if _thread is not None and _thread.is_alive():
            return _thread

        _thread = threading.Thread(
            target=scheduler_loop,
            args=(daily_time, tzinfo),
            name='wamf-retention-scheduler',
            daemon=True,
        )
        _thread.start()

    message = f'Retention scheduled daily at {daily_time:%H:%M} ({tzinfo.key})'
    logger.info(message)
    log_system_event('INFO', 'RETENTION', message)
    return _thread
