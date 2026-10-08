import hashlib
from pathlib import Path
import tarfile

import pytest

from quant.timefolio_cnn_transfer import compress_archive, decode_archive, decoder_code


def fixture_archive(tmp_path):
    payload = tmp_path/'values.bin'
    payload.write_bytes(bytes(range(256))*100)
    archive = tmp_path/'input.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(payload, arcname='dataset/values.bin')
    packed = compress_archive(archive)
    params = dict(expected_bytes=archive.stat().st_size,
                  expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  wire_sha256=hashlib.sha256(packed.read_bytes()).hexdigest())
    return payload, archive, packed, params


def test_compression_preserves_every_input_byte(tmp_path):
    payload, archive, packed, params = fixture_archive(tmp_path)
    result = decode_archive(packed, tmp_path/'restored', **params)
    assert result['exact_input_bytes_verified']
    assert (tmp_path/'restored/dataset/values.bin').read_bytes() == payload.read_bytes()
    assert packed.with_suffix('.decoded.tar').read_bytes() == archive.read_bytes()
    assert packed.stat().st_size < archive.stat().st_size
    compile(decoder_code(**params), '<remote_decoder>', 'exec')


def test_changed_input_and_oversized_frame_cannot_be_extracted(tmp_path):
    _, _, packed, params = fixture_archive(tmp_path)
    destination = tmp_path/'forbidden'
    for change in [dict(wire_sha256='0'*64), dict(expected_sha256='0'*64),
                   dict(expected_bytes=params['expected_bytes']+1), dict(expected_bytes=5*1024**3)]:
        with pytest.raises(ValueError):
            decode_archive(packed, destination, **(params|change))
        assert not destination.exists()
