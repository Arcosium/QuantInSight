"""Lossless archive transport; original input bytes must survive exactly."""
from pathlib import Path
import inspect
import re
import subprocess


def compress_archive(archive):
    archive = Path(archive)
    output = archive.with_suffix(archive.suffix+'.zst')
    if output.exists() or not 0 < archive.stat().st_size <= 4*1024**3:
        raise ValueError('A fresh archive of at most4GiB is required')
    subprocess.run(['zstd', '-1', '--single-thread', '--check', '-q', str(archive),
                    '-o', str(output)], check=True, timeout=1800)
    return output


def decode_archive(source, destination, *, expected_bytes, expected_sha256, wire_sha256):
    """Runs only on the disposable GPU; requires its existing system libzstd.

    The4GiB limit bounds decoder allocation on a GPU host with at least16GiB RAM.
    Both wire and original archive hashes are checked before opening the tar.
    """
    import ctypes
    import ctypes.util
    import hashlib
    from pathlib import Path
    import tarfile

    source, destination = Path(source), Path(destination)
    if not isinstance(expected_bytes, int) or not 0 < expected_bytes <= 4*1024**3:
        raise ValueError('Archive exceeds the registered memory bound')
    if source.stat().st_size > 4*1024**3+16*1024**2:
        raise ValueError('Compressed archive is too large')
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != wire_sha256:
        raise ValueError('Compressed transfer hash mismatch')
    library = ctypes.util.find_library('zstd')
    if not library:
        raise RuntimeError('The disposable GPU has no system libzstd')
    zstd = ctypes.CDLL(library)
    zstd.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    zstd.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
    src = ctypes.c_char_p(data)
    if zstd.ZSTD_getFrameContentSize(src, len(data)) != expected_bytes:
        raise ValueError('Unexpected or unknown decoded frame size')
    zstd.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                   ctypes.c_void_p, ctypes.c_size_t]
    zstd.ZSTD_decompress.restype = ctypes.c_size_t
    zstd.ZSTD_isError.argtypes = [ctypes.c_size_t]
    zstd.ZSTD_isError.restype = ctypes.c_uint
    decoded = ctypes.create_string_buffer(expected_bytes)
    count = zstd.ZSTD_decompress(decoded, expected_bytes, src, len(data))
    if zstd.ZSTD_isError(count) or count != expected_bytes:
        raise ValueError('Lossless decompression failed')
    view = memoryview(decoded).cast('B')
    if hashlib.sha256(view).hexdigest() != expected_sha256:
        raise ValueError('Original archive hash mismatch; training is forbidden')
    raw = source.with_suffix('.decoded.tar')
    if raw.exists():
        raise ValueError('Refusing to overwrite an existing decoded archive')
    with raw.open('xb') as stream:
        stream.write(view)
    del view, decoded, src, data
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(raw) as archive:
        archive.extractall(destination, filter='data')
    return dict(decoded_bytes=expected_bytes, original_sha256=expected_sha256,
                wire_sha256=wire_sha256, exact_input_bytes_verified=True)


def decoder_code(expected_bytes, expected_sha256, wire_sha256):
    if (not isinstance(expected_bytes, int) or not 0 < expected_bytes <= 4*1024**3
            or any(not re.fullmatch('[0-9a-f]{64}', s) for s in [expected_sha256, wire_sha256])):
        raise ValueError('Bounded bytes and exact SHA256 digests required')
    return inspect.getsource(decode_archive)+'\nimport json\nprint(json.dumps(decode_archive('+\
        repr('/workspace/cnn4y.input.tar.zst')+', '+repr('/workspace/cnn4y')+', expected_bytes='+\
        str(expected_bytes)+', expected_sha256='+repr(expected_sha256)+', wire_sha256='+\
        repr(wire_sha256)+')))\n'
