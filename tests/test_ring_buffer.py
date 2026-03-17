from __future__ import annotations

from apsgnn.buffer import TemporalRingBuffer


def test_ring_buffer_schedules_with_wraparound() -> None:
    buffer: TemporalRingBuffer[str] = TemporalRingBuffer(8)
    buffer.schedule(1, "a")
    buffer.schedule(8, "b")

    assert buffer.pop_current() == []
    buffer.advance()
    assert buffer.pop_current() == ["a"]

    for _ in range(6):
        buffer.advance()
        assert buffer.pop_current() == []

    buffer.advance()
    assert buffer.pop_current() == ["b"]
