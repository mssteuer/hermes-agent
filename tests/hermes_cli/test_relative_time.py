import time

from hermes_cli.main import _relative_time


def test_relative_time_accepts_iso_timestamps():
    ts = time.time() - 7200
    iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))

    assert _relative_time(iso) == "2h ago"
