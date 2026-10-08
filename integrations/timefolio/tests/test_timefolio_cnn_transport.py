import subprocess
from unittest.mock import patch

import pytest

from quant.timefolio_cnn_remote import probe_upload, upload_timeout


def test_upload_deadline_scales_for_large_frozen_dataset():
    assert upload_timeout(2*1024**3, 4*1024**2) == 1084
    assert upload_timeout(1024, 2*1024**2) == 300
    assert upload_timeout(5*1024**3, 1024**2) == 1800


def test_upload_rejects_slow_or_nonfinite_routes():
    for speed in [0, 100000, float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            upload_timeout(2*1024**3, speed)


def test_compressed_small_archive_can_use_a_slower_bounded_route():
    assert upload_timeout(128*1024**2, 512*1024, minimum_speed=256*1024) == 572
    with pytest.raises(ValueError):
        upload_timeout(128*1024**2, 128*1024, minimum_speed=256*1024)
    assert upload_timeout(448804166, 1523886.7, minimum_speed=256*1024,
                          reserve_for_minimum=True) == 1773


def test_probe_requires_byte_acknowledgement_and_rejects_slow_link():
    result = subprocess.CompletedProcess([], 0, b'8388608\n', b'')
    with patch('quant.timefolio_cnn_remote.subprocess.run', return_value=result) as run:
        with patch('quant.timefolio_cnn_remote.time.monotonic', side_effect=[0., 2.]):
            assert probe_upload(['ssh'])['bytes_per_second'] == 4*1024**2
        assert run.call_args.kwargs['timeout'] == 30
        assert len(run.call_args.kwargs['input']) == 8*1024**2
        with patch('quant.timefolio_cnn_remote.time.monotonic', side_effect=[0., 20.]):
            with pytest.raises(ValueError):
                probe_upload(['ssh'])
        result.stdout = b'1\n'
        with patch('quant.timefolio_cnn_remote.time.monotonic', side_effect=[0., 2.]):
            with pytest.raises(ValueError):
                probe_upload(['ssh'])
