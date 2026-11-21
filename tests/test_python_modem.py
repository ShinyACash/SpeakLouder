import unittest
import numpy as np
import air.air as modem

class FecTests(unittest.TestCase):
    def test_crc16_ccitt_standard_check_value(self) -> None:
        self.assertEqual(modem.crc16_ccitt(b"123456789"), 0x29B1)

    def test_reed_solomon_corrects_four_bytes(self) -> None:
        source = bytes(range(128))
        codeword = bytearray(modem.rs_encode(source))
        for index in range(4):
            position = (index * 37 + 11) % len(codeword)
            codeword[position] ^= 0x31 + index
        decoded = modem.rs_decode(bytes(codeword))
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded[0][:len(source)], source)
        self.assertEqual(decoded[1], 4)


class ModemTests(unittest.TestCase):
    def test_frame_round_trip_for_data_and_control_frames(self) -> None:
        for frame_type in (
            modem.DATA, modem.SYN, modem.SYN_ACK, modem.ACK, modem.NACK
        ):
            payload = b"python modem" if frame_type == modem.DATA else b""
            decoded = modem.decode(modem.modulate(payload, 0xA7, frame_type))
            self.assertIsNotNone(decoded)
            assert decoded is not None
            self.assertEqual(decoded.frame_type, frame_type)
            self.assertEqual(decoded.sequence, 0xA7)
            self.assertEqual(decoded.payload, payload)

    def test_noisy_multipath_frame_and_exact_sync_index(self) -> None:
        payload = b"\x00hello\xff"
        transmitted = modem.modulate(payload, 9)
        leading_silence = 137
        channel = np.zeros(leading_silence + len(transmitted) + 173,
                           dtype=np.float32)
        channel[leading_silence:leading_silence + len(transmitted)] += transmitted
        channel[leading_silence + 173:leading_silence + 173 + len(transmitted)] += (
            0.22 * transmitted
        )
        rng = np.random.default_rng(0x5EED)
        channel += rng.normal(0.0, 0.008, channel.size).astype(np.float32)
        decoded = modem.decode(channel)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded.payload, payload)
        self.assertEqual(decoded.sequence, 9)
        self.assertEqual(decoded.start_sample, leading_silence)

    def test_transport_stream_receiver_extracts_frame_from_chunks(self) -> None:
        signal = np.concatenate((
            np.zeros(2_401, dtype=np.float32),
            modem.modulate(b"from queued capture", sequence=3),
            np.zeros(4_800, dtype=np.float32),
        ))

        class ChunkAudio:
            def __init__(self) -> None:
                self.offset = 0

            def read(self, timeout: float) -> np.ndarray | None:
                del timeout
                if self.offset >= signal.size:
                    return None
                result = signal[self.offset:self.offset + 4_800]
                self.offset += result.size
                return result

        receiver = modem.Transport(ChunkAudio())  # type: ignore[arg-type]
        decoded = receiver.receive_frame(timeout=1.0)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded.payload, b"from queued capture")
        self.assertEqual(decoded.sequence, 3)
        self.assertEqual(decoded.start_sample, 2_401)

    def test_text_packetization_preserves_utf8_boundaries(self) -> None:
        text = "a" * 127 + "D" + "z"
        packets = modem._split_text(text)
        self.assertTrue(all(len(packet) <= modem.MAX_PAYLOAD for packet in packets))
        self.assertEqual(b"".join(packets).decode("utf-8"), text)
        self.assertTrue(all(packet.decode("utf-8") for packet in packets))

    def test_decode_rejects_heavy_low_frequency_room_noise_and_dc_offset(self) -> None:
        payload = b"long range packet over ambient noise"
        # quiet ultrasonic signal buried under loud room/fan hum
        transmitted = modem.modulate(payload, sequence=11) * 0.008
        leading = 600
        channel = np.zeros(leading + transmitted.size + 1200, dtype=np.float32)
        channel[leading:leading + transmitted.size] = transmitted
        t = np.arange(channel.size, dtype=np.float64) / modem.SAMPLE_RATE
        audible_interference = (
            0.08
            + 0.12 * np.sin(2 * np.pi * 60.0 * t)
            + 0.07 * np.sin(2 * np.pi * 400.0 * t)
            + 0.05 * np.cos(2 * np.pi * 1500.0 * t)
        ).astype(np.float32)
        channel += audible_interference
        decoded = modem.decode(channel)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded.payload, payload)
        self.assertEqual(decoded.sequence, 11)
        self.assertGreaterEqual(decoded.correlation, modem.CORRELATION_THRESHOLD)

    def test_decode_recovers_corrupted_header_via_reed_solomon(self) -> None:
        payload = b"header error recovery"
        wire = bytearray(modem.encode_frame(payload, sequence=5, frame_type=modem.DATA))
        # trash SYNC, frame_type, and 1 symbol of the length byte; RS should still save it
        wire[0] ^= 0x55
        wire[1] ^= 0x03
        wire[3] ^= 0x02
        preamble = modem.make_preamble()
        window = np.hanning(modem.SAMPLES_PER_SYMBOL)
        symbols = np.empty(len(wire) * 4 * modem.SAMPLES_PER_SYMBOL, dtype=np.float32)
        sample_index = np.arange(modem.SAMPLES_PER_SYMBOL, dtype=np.float64)
        offset = 0
        for byte in wire:
            for shift in (6, 4, 2, 0):
                symbol = (byte >> shift) & 3
                tone = 0.6 * window * np.sin(
                    2 * np.pi * modem.CARRIERS[symbol] * sample_index / modem.SAMPLE_RATE
                )
                symbols[offset:offset + modem.SAMPLES_PER_SYMBOL] = tone
                offset += modem.SAMPLES_PER_SYMBOL
        frame_signal = np.concatenate(
            (preamble, np.zeros(modem.GUARD_SAMPLES, np.float32), symbols)
        )
        decoded = modem.decode(frame_signal)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded.payload, payload)
        self.assertEqual(decoded.sequence, 5)
        self.assertEqual(decoded.corrected_bytes, 3)

    def test_handshake_listener_accepts_data_when_final_ack_dropped(self) -> None:
        frames = [
            modem.modulate(b"", sequence=0, frame_type=modem.SYN),
            modem.modulate(b"first packet after lost ack", sequence=0, frame_type=modem.DATA),
        ]
        signal = np.concatenate([
            np.zeros(1_000, dtype=np.float32),
            frames[0],
            np.zeros(4_800, dtype=np.float32),
            frames[1],
            np.zeros(4_800, dtype=np.float32),
        ])

        class MockDuplexAudio:
            def __init__(self) -> None:
                self.offset = 0
                self.played: list[np.ndarray] = []

            def read(self, timeout: float) -> np.ndarray | None:
                del timeout
                if self.offset >= signal.size:
                    return None
                chunk = signal[self.offset:self.offset + 4_800]
                self.offset += chunk.size
                return chunk

            def discard_capture(self) -> None:
                pass

            def suppress_capture(self, suppressed: bool) -> None:
                del suppressed

            def play(self, samples: np.ndarray) -> None:
                self.played.append(samples)

        audio = MockDuplexAudio()
        transport = modem.Transport(audio)  # type: ignore[arg-type]
        self.assertTrue(transport.handshake_listener())
        received = transport.receive_data(timeout=0.5)
        self.assertIsNotNone(received)
        assert received is not None
        self.assertEqual(received.payload, b"first packet after lost ack")


if __name__ == "__main__":
    unittest.main()
