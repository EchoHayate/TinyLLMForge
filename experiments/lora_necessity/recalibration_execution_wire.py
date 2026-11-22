import hashlib
import re
import struct

from pilot_protocol import require as _require


MAGIC = b"LRV01"
HEADER = struct.Struct("!5sBQ")
WIRE_LIMIT = 8388608
FRAME_LIMITS = (4096, 8388608, 131072, 65536, 4096)
FRAME_NAMES = (
    "ready_raw", "backend_stdout_raw", "backend_stderr_raw", "diagnostic_raw", "done_raw",
)


def _read_frame(raw, offset, expected_channel):
    _require(len(raw) - offset >= HEADER.size, "short frame header")
    magic, channel, size = HEADER.unpack_from(raw, offset)
    _require(magic == MAGIC, "frame magic")
    _require(channel == expected_channel, "frame channel order")
    _require(size <= FRAME_LIMITS[channel - 1], "frame payload limit")
    start = offset + HEADER.size
    stop = start + size
    _require(stop <= len(raw), "short frame payload")
    return raw[start:stop], stop


def decode_frame(frame_raw, *, expected_channel):
    _require(type(expected_channel) is int and 1 <= expected_channel <= 5,
             "expected channel")
    _require(type(frame_raw) is bytes
             and len(frame_raw) <= HEADER.size + FRAME_LIMITS[expected_channel - 1],
             "frame bytes")
    payload, stop = _read_frame(frame_raw, 0, expected_channel)
    _require(stop == len(frame_raw), "trailing frame bytes")
    return payload


def decode_wire(wire_raw, *, expected_sha256):
    _require(type(wire_raw) is bytes and len(wire_raw) <= WIRE_LIMIT, "wire bytes")
    _require(type(expected_sha256) is str
             and re.fullmatch("[0-9a-f]{64}", expected_sha256) is not None,
             "expected wire digest")
    _require(hashlib.sha256(wire_raw).hexdigest() == expected_sha256, "wire digest")
    offset = 0
    decoded = {}
    for channel, name in enumerate(FRAME_NAMES, 1):
        payload, offset = _read_frame(wire_raw, offset, channel)
        decoded[name] = payload
    _require(offset == len(wire_raw), "trailing wire bytes")
    return decoded
