"""
Tests for tracked_labels: non-bird animal types (squirrel, cat, rabbit, etc.)
that an upstream source (e.g. a Home Assistant automation) has already
identified. WAMF records these directly using the Frigate/HA-supplied label,
without running its bird-only species classifier against them.
"""
import json
import os
import sqlite3
import sys
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault('WHOSATMYFEEDER_CONFIG', 'config/config.yml.example')

for _mod in [
    "numpy",
    "tflite_support",
    "tflite_support.task",
    "tflite_support.task.core",
    "tflite_support.task.processor",
    "tflite_support.task.vision",
    "PIL", "PIL.Image", "PIL.ImageOps",
    "paho", "paho.mqtt", "paho.mqtt.client",
]:
    sys.modules.setdefault(_mod, MagicMock())

import speciesid  # noqa: E402


def _make_det_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_time TEXT NOT NULL,
            detection_index INTEGER,
            score REAL,
            display_name TEXT,
            category_name TEXT,
            frigate_event TEXT UNIQUE,
            camera_name TEXT,
            wamf_snapshot_path TEXT,
            wamf_clip_path TEXT
        )
    """)
    conn.commit()
    conn.close()


@pytest.fixture()
def fresh_db(tmp_path):
    db = str(tmp_path / "fresh.db")
    _make_det_db(db)
    return db


def _run_on_message(fresh_db, label, event_id='evt-tracked-001', tracked_labels=('squirrel',),
                    camera='birdcam', retained=False, topic='frigate/events', event_type='new',
                    deliveries=1, snapshot_path='/media/snapshots/evt.jpg', clip_path=None):
    speciesid.config = {
        'frigate': {
            'camera': [camera],
            'frigate_url': 'http://localhost:5000',
            'tracked_labels': (
                tracked_labels if isinstance(tracked_labels, str) else list(tracked_labels)
            ),
        },
        'classification': {'threshold': 0.7},
    }

    msg = MagicMock()
    msg.retain = retained
    msg.topic = topic
    msg.payload = json.dumps({
        'type': event_type,
        'after': {
            'camera': camera,
            'label': label,
            'id': event_id,
            'start_time': 1700000000.0,
        }
    })

    client = MagicMock()

    with patch.object(speciesid, 'DBPATH', fresh_db), \
         patch('speciesid.classify') as mock_classify, \
         patch('speciesid.archive_snapshot', return_value=snapshot_path), \
         patch('speciesid.archive_clip', return_value=clip_path), \
         patch('speciesid.publish_new_species') as mock_publish:
        for _ in range(deliveries):
            speciesid.on_message(client, None, msg)

    return client, mock_classify, mock_publish


def test_tracked_label_recorded_without_classification(fresh_db):
    """A tracked non-bird label is stored directly; WAMF's classifier never runs."""
    client, mock_classify, mock_publish = _run_on_message(fresh_db, label='squirrel')

    mock_classify.assert_not_called()

    conn = sqlite3.connect(fresh_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT display_name, category_name, score, detection_index "
        "FROM detections WHERE frigate_event = 'evt-tracked-001'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == 'Squirrel'
    assert row[1] == 'external_label'
    assert row[2] == 1.0
    assert row[3] == -1
    mock_publish.assert_called_once()


def test_tracked_label_is_case_insensitive(fresh_db):
    """Configured tracked_labels match regardless of case in either direction."""
    _run_on_message(fresh_db, label='SQUIRREL', tracked_labels=('Squirrel',))

    conn = sqlite3.connect(fresh_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM detections WHERE frigate_event = 'evt-tracked-001'")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 1


def test_untracked_label_ignored(fresh_db):
    """A label that is neither 'bird' nor in tracked_labels is ignored entirely."""
    _run_on_message(fresh_db, label='raccoon', tracked_labels=('squirrel',))

    conn = sqlite3.connect(fresh_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM detections")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 0


def test_wildcard_tracks_any_non_bird_label(fresh_db):
    """tracked_labels: "*" accepts any label without a fixed list."""
    _run_on_message(fresh_db, label='opossum', tracked_labels='*')

    conn = sqlite3.connect(fresh_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT display_name FROM detections WHERE frigate_event = 'evt-tracked-001'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == 'Opossum'


def test_wildcard_does_not_match_empty_label(fresh_db):
    """An empty/missing label is never tracked, even under the wildcard."""
    _run_on_message(fresh_db, label='', tracked_labels='*')

    conn = sqlite3.connect(fresh_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM detections")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 0


def test_tracked_label_dedupes_on_frigate_event(fresh_db):
    """Replaying the same event id does not create a duplicate row."""
    _run_on_message(fresh_db, label='squirrel', deliveries=2)

    conn = sqlite3.connect(fresh_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM detections WHERE frigate_event = 'evt-tracked-001'")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 1


def test_tracked_label_skipped_when_snapshot_missing(fresh_db):
    """No snapshot available (archive_snapshot returns None) → no DB write."""
    _run_on_message(fresh_db, label='squirrel', snapshot_path=None)

    conn = sqlite3.connect(fresh_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM detections")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 0


def test_bird_label_unaffected_by_tracked_labels_config(fresh_db):
    """The 'bird' label still goes through WAMF's own classifier, not the tracked-label path."""
    fake_category = MagicMock()
    fake_category.index = 42
    fake_category.score = 0.9
    fake_category.display_name = 'Turdus migratorius'
    fake_category.category_name = 'bird'

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'fakejpegdata'
    mock_image = MagicMock()
    mock_image.size = (100, 100)

    speciesid.config = {
        'frigate': {
            'camera': ['birdcam'],
            'frigate_url': 'http://localhost:5000',
            'tracked_labels': ['squirrel'],
        },
        'classification': {'threshold': 0.7},
    }

    msg = MagicMock()
    msg.retain = False
    msg.topic = 'frigate/events'
    msg.payload = json.dumps({
        'type': 'new',
        'after': {
            'camera': 'birdcam',
            'label': 'bird',
            'id': 'evt-bird-001',
            'start_time': 1700000000.0,
        }
    })

    client = MagicMock()

    with patch.object(speciesid, 'DBPATH', fresh_db), \
         patch('speciesid.requests.get', return_value=mock_response), \
         patch('speciesid.Image') as mock_Image, \
         patch('speciesid.classify', return_value=[fake_category]) as mock_classify, \
         patch('speciesid.get_common_name', return_value='American Robin'), \
         patch('speciesid.archive_snapshot', return_value='/media/snapshots/evt.jpg'), \
         patch('speciesid.archive_clip', return_value=None), \
         patch('speciesid.set_sublabel'), \
         patch('speciesid.publish_new_species'):
        mock_Image.open.return_value = mock_image
        speciesid.on_message(client, None, msg)

    mock_classify.assert_called_once()

    conn = sqlite3.connect(fresh_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT category_name FROM detections WHERE frigate_event = 'evt-bird-001'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] != 'external_label'
