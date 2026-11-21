#!/usr/bin/env python3 from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48_000
SAMPLES_PER_SYMBOL = 480
PREAMBLE_SAMPLES = 2_880
GUARD_SAMPLES = 720
PREAMBLE_START_HZ = 17_800.0
PREAMBLE_END_HZ = 20_200.0
CARRIERS = (18_200.0, 18_600.0, 19_000.0, 19_400.0)
MAX_PAYLOAD = 128
RS_PARITY = 8
SYNC = 0xD3
DATA = 1
SYN = 2
SYN_ACK = 3
ACK = 4
NACK = 5
FRAME_NAMES = {
    DATA: "DATA",
    SYN: "SYN",
    SYN_ACK: "SYN_ACK",
    ACK: "ACK",
    NACK: "NACK",
}
CORRELATION_THRESHOLD = 0.32
ACK_TIMEOUT = 3.5
MAX_RETRIES = 5
TURNAROUND_GUARD = 0.08
TX_LEVEL = 0.35

BAND_PRESETS: dict[str, tuple[float, float, tuple[float, float, float, float]]] = {
    "ultrasonic": (17_800.0, 20_200.0, (18_200.0, 18_600.0, 19_000.0, 19_400.0)),
    "near": (14_400.0, 16_800.0, (14_800.0, 15_200.0, 15_600.0, 16_000.0)),
    "audible": (1_600.0, 4_000.0, (2_000.0, 2_400.0, 2_800.0, 3_200.0)),
}
_FIR_CACHE: dict[tuple[float, float], np.ndarray] = {}


def set_band(band: str) -> None:
    global PREAMBLE_START_HZ, PREAMBLE_END_HZ, CARRIERS
    if band not in BAND_PRESETS:
        raise ValueError(f"unknown band preset: {band}")
    PREAMBLE_START_HZ, PREAMBLE_END_HZ, CARRIERS = BAND_PRESETS[band]


def _bandpass_kernel() -> np.ndarray:
    low_hz = max(300.0, min(PREAMBLE_START_HZ, min(CARRIERS)) - 600.0)
    high_hz = min(SAMPLE_RATE * 0.49, max(PREAMBLE_END_HZ, max(CARRIERS)) + 600.0)
    key = (low_hz, high_hz)
    cached = _FIR_CACHE.get(key)
    if cached is not None:
        return cached
    half = 64
    n = np.arange(-half, half + 1, dtype=np.float64)
    h_high = (2.0 * high_hz / SAMPLE_RATE) * np.sinc(2.0 * high_hz * n / SAMPLE_RATE)
    h_low = (2.0 * low_hz / SAMPLE_RATE) * np.sinc(2.0 * low_hz * n / SAMPLE_RATE)
    window = np.blackman(2 * half + 1)
    kernel = (h_high - h_low) * window
    center_hz = 0.5 * (low_hz + high_hz)
    response = np.abs(np.sum(kernel * np.exp(-2j * np.pi * center_hz * n / SAMPLE_RATE)))
    if response > 1e-12:
        kernel /= response
    _FIR_CACHE[key] = kernel
    return kernel


def bandpass_filter(samples: np.ndarray) -> np.ndarray:
    # Filter out fan hum and room noise before correlating, since laptop mics love 60 Hz more than 19 kHz.
    signal = np.asarray(samples, dtype=np.float64)
    if signal.size == 0:
        return signal
    kernel = _bandpass_kernel()
    half = (kernel.size - 1) // 2
    full = np.convolve(signal, kernel, mode="full")
    return full[half:half + signal.size]


@dataclass
class Decoded:
    frame_type: int
    sequence: int
    payload: bytes
    correlation: float
    start_sample: int
    snr_db: float
    corrected_bytes: int


def gf_mul(a: int, b: int) -> int:
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & 0x100:
            a ^= 0x11D
    return result


GF_EXP = [0] * 512
GF_LOG = [0] * 256
_value = 1
for _index in range(255):
    GF_EXP[_index] = _value
    GF_LOG[_value] = _index
    _value <<= 1
    if _value & 0x100:
        _value ^= 0x11D
for _index in range(255, len(GF_EXP)):
    GF_EXP[_index] = GF_EXP[_index - 255]


def gf_mul_table(a: int, b: int) -> int:
    if not a or not b:
        return 0
    return GF_EXP[GF_LOG[a] + GF_LOG[b]]


def gf_div(a: int, b: int) -> int:
    if not b:
        raise ValueError("division by zero in GF(2^8)")
    if not a:
        return 0
    return GF_EXP[(GF_LOG[a] - GF_LOG[b]) % 255]


def gf_pow(exponent: int) -> int:
    return GF_EXP[exponent % 255]


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def rs_encode(data: bytes, parity_bytes: int = RS_PARITY) -> bytes:
    if not 0 < parity_bytes or len(data) + parity_bytes > 255:
        raise ValueError("invalid Reed-Solomon codeword length")
    generator = [1]
    for i in range(parity_bytes):
        root = gf_pow(i)
        nxt = [0] * (len(generator) + 1)
        for j, coefficient in enumerate(generator):
            nxt[j] ^= coefficient
            nxt[j + 1] ^= gf_mul_table(coefficient, root)
        generator = nxt
    work = list(data) + [0] * parity_bytes
    for i in range(len(data)):
        coefficient = work[i]
        if coefficient:
            for j, value in enumerate(generator):
                work[i + j] ^= gf_mul_table(coefficient, value)
    return data + bytes(work[len(data):])


def _eval_ascending(poly: list[int], x: int) -> int:
    result = 0
    for coefficient in reversed(poly):
        result = gf_mul_table(result, x) ^ coefficient
    return result


def _solve_magnitudes(locations: list[int], syndromes: list[int]) -> list[int] | None:
    count = len(locations)
    matrix = [[0] * (count + 1) for _ in range(count)]
    for row in range(count):
        for column, location in enumerate(locations):
            matrix[row][column] = 1
            for _ in range(row):
                matrix[row][column] = gf_mul_table(matrix[row][column], location)
        matrix[row][count] = syndromes[row]

    for column in range(count):
        pivot = next((r for r in range(column, count) if matrix[r][column]), None)
        if pivot is None:
            return None
        matrix[pivot], matrix[column] = matrix[column], matrix[pivot]
        divisor = matrix[column][column]
        for j in range(column, count + 1):
            matrix[column][j] = gf_div(matrix[column][j], divisor)
        for row in range(count):
            if row == column or not matrix[row][column]:
                continue
            factor = matrix[row][column]
            for j in range(column, count + 1):
                matrix[row][j] ^= gf_mul_table(factor, matrix[column][j])
    return [matrix[i][count] for i in range(count)]


def rs_decode(codeword: bytes, parity_bytes: int = RS_PARITY) -> tuple[bytes, int] | None:
    if not 0 < parity_bytes < len(codeword) or len(codeword) > 255:
        return None
    corrected = bytearray(codeword)
    syndromes = []
    for i in range(parity_bytes):
        x = gf_pow(i)
        value = 0
        for byte in corrected:
            value = gf_mul_table(value, x) ^ byte
        syndromes.append(value)
    if not any(syndromes):
        return bytes(corrected), 0

    locator = [1]
    previous = [1]
    degree = 0
    shift = 1
    last_discrepancy = 1
    for n in range(parity_bytes):
        discrepancy = syndromes[n]
        for i in range(1, degree + 1):
            discrepancy ^= gf_mul_table(locator[i], syndromes[n - i])
        if not discrepancy:
            shift += 1
            continue
        saved = locator.copy()
        scale = gf_div(discrepancy, last_discrepancy)
        if len(locator) < len(previous) + shift:
            locator.extend([0] * (len(previous) + shift - len(locator)))
        for i, value in enumerate(previous):
            locator[i + shift] ^= gf_mul_table(scale, value)
        if 2 * degree <= n:
            degree = n + 1 - degree
            previous = saved
            last_discrepancy = discrepancy
            shift = 1
        else:
            shift += 1

    if not degree or degree > parity_bytes // 2 or len(locator) <= degree:
        return None
    positions: list[int] = []
    locations: list[int] = []
    for position in range(len(corrected)):
        exponent = len(corrected) - 1 - position
        location = gf_pow(exponent)
        if _eval_ascending(locator, gf_pow(-exponent)) == 0:
            positions.append(position)
            locations.append(location)
    if len(positions) != degree:
        return None
    magnitudes = _solve_magnitudes(locations, syndromes[:degree])
    if magnitudes is None:
        return None
    for position, magnitude in zip(positions, magnitudes):
        corrected[position] ^= magnitude
    if any(
        _eval_codeword(corrected, gf_pow(i)) != 0
        for i in range(parity_bytes)
    ):
        return None
    return bytes(corrected), len(positions)


def _eval_codeword(codeword: bytes | bytearray, x: int) -> int:
    result = 0
    for byte in codeword:
        result = gf_mul_table(result, x) ^ byte
    return result


def encode_frame(payload: bytes, sequence: int = 0, frame_type: int = DATA) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("payload exceeds 128 bytes")
    if frame_type not in (DATA, SYN, SYN_ACK, ACK, NACK):
        raise ValueError("unknown frame type")
    body = bytes((SYNC, frame_type, sequence & 0xFF, len(payload))) + payload
    crc = crc16_ccitt(body)
    return rs_encode(body + bytes((crc >> 8, crc & 0xFF)))


def make_preamble() -> np.ndarray:
    n = np.arange(PREAMBLE_SAMPLES, dtype=np.float64)
    duration = PREAMBLE_SAMPLES / SAMPLE_RATE
    sweep_rate = (PREAMBLE_END_HZ - PREAMBLE_START_HZ) / duration
    t = n / SAMPLE_RATE
    window = 0.5 - 0.5 * np.cos(2 * np.pi * n / (PREAMBLE_SAMPLES - 1))
    phase = 2 * np.pi * (PREAMBLE_START_HZ * t + 0.5 * sweep_rate * t * t)
    return (0.6 * window * np.sin(phase)).astype(np.float32)


def _make_preamble_quadrature() -> tuple[np.ndarray, np.ndarray]:
    n = np.arange(PREAMBLE_SAMPLES, dtype=np.float64)
    duration = PREAMBLE_SAMPLES / SAMPLE_RATE
    sweep_rate = (PREAMBLE_END_HZ - PREAMBLE_START_HZ) / duration
    t = n / SAMPLE_RATE
    window = 0.5 - 0.5 * np.cos(2 * np.pi * n / (PREAMBLE_SAMPLES - 1))
    phase = 2 * np.pi * (PREAMBLE_START_HZ * t + 0.5 * sweep_rate * t * t)
    return 0.6 * window * np.sin(phase), 0.6 * window * np.cos(phase)


def modulate(payload: bytes, sequence: int = 0, frame_type: int = DATA) -> np.ndarray:
    preamble = make_preamble()
    window = np.hanning(SAMPLES_PER_SYMBOL)
    wire = encode_frame(payload, sequence, frame_type)
    symbols = np.empty(len(wire) * 4 * SAMPLES_PER_SYMBOL, dtype=np.float32)
    sample_index = np.arange(SAMPLES_PER_SYMBOL, dtype=np.float64)
    offset = 0
    for byte in wire:
        for shift in (6, 4, 2, 0):
            symbol = (byte >> shift) & 3
            tone = 0.6 * window * np.sin(
                2 * np.pi * CARRIERS[symbol] * sample_index / SAMPLE_RATE
            )
            symbols[offset:offset + SAMPLES_PER_SYMBOL] = tone
            offset += SAMPLES_PER_SYMBOL
    return np.concatenate(
        (preamble, np.zeros(GUARD_SAMPLES, np.float32), symbols)
    )


def find_preamble(samples: np.ndarray) -> tuple[int, float] | None:
    ref_sin, ref_cos = _make_preamble_quadrature()
    raw = np.asarray(samples, dtype=np.float64)
    if raw.size < ref_sin.size:
        return None
    signal = bandpass_filter(raw)
    fft_size = 1 << (signal.size + ref_sin.size - 2).bit_length()
    signal_fft = np.fft.rfft(signal, fft_size)
    corr_sin = np.fft.irfft(
        signal_fft * np.fft.rfft(ref_sin[::-1], fft_size),
        fft_size,
    )[ref_sin.size - 1:signal.size]
    corr_cos = np.fft.irfft(
        signal_fft * np.fft.rfft(ref_cos[::-1], fft_size),
        fft_size,
    )[ref_cos.size - 1:signal.size]
    energy_prefix = np.concatenate(([0.0], np.cumsum(signal * signal)))
    energies = energy_prefix[ref_sin.size:] - energy_prefix[:-ref_sin.size]
    if not energies.size:
        return None
    max_energy = float(np.max(energies))
    if max_energy <= 1e-16:
        return 0, 0.0
    min_energy = max(1e-12, 0.02 * max_energy)
    reference_energy = float(np.dot(ref_sin, ref_sin))
    denom = np.sqrt(np.maximum(energies, min_energy) * reference_energy)
    real_scores = np.abs(corr_sin) / denom
    quad_scores = np.hypot(corr_sin, corr_cos) / denom
    env_index = int(np.argmax(quad_scores))
    lo = max(0, env_index - 2)
    hi = min(real_scores.size, env_index + 3)
    refined_index = lo + int(np.argmax(real_scores[lo:hi]))
    best_score = float(min(1.0, max(quad_scores[env_index], real_scores[refined_index])))
    return refined_index, best_score


_DEMOD_BASIS_CACHE: dict[
    tuple[float, ...], tuple[np.ndarray, np.ndarray]
] = {}


def _get_demod_basis() -> tuple[np.ndarray, np.ndarray]:
    key = tuple(CARRIERS)
    cached = _DEMOD_BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    n = np.arange(SAMPLES_PER_SYMBOL, dtype=np.float64)
    offsets = (-35.0, -17.5, 0.0, 17.5, 35.0)
    sine_basis = np.empty((len(CARRIERS), len(offsets), SAMPLES_PER_SYMBOL), dtype=np.float64)
    cos_basis = np.empty((len(CARRIERS), len(offsets), SAMPLES_PER_SYMBOL), dtype=np.float64)
    for c_idx, carrier in enumerate(CARRIERS):
        for o_idx, delta in enumerate(offsets):
            phase = 2 * np.pi * (carrier + delta) * n / SAMPLE_RATE
            sine_basis[c_idx, o_idx] = np.sin(phase)
            cos_basis[c_idx, o_idx] = np.cos(phase)
    _DEMOD_BASIS_CACHE[key] = (sine_basis, cos_basis)
    return sine_basis, cos_basis


def demodulate(
    samples: np.ndarray, byte_count: int, equalize: bool = False
) -> tuple[bytes, float]:
    signal = np.asarray(samples, dtype=np.float64)
    required = byte_count * 4 * SAMPLES_PER_SYMBOL
    if signal.size < required:
        raise ValueError("not enough samples to demodulate frame")
    if byte_count == 0:
        return b"", 0.0
    symbol_count = byte_count * 4
    window = np.hanning(SAMPLES_PER_SYMBOL)
    bursts = signal[:required].reshape(symbol_count, SAMPLES_PER_SYMBOL) * window
    sine_basis, cos_basis = _get_demod_basis()
    s_proj = np.einsum("sn,con->sco", bursts, sine_basis)
    c_proj = np.einsum("sn,con->sco", bursts, cos_basis)
    powers = np.max(s_proj * s_proj + c_proj * c_proj, axis=2)
    if equalize and symbol_count >= 16:
        # Consumer speakers roll off hard above 18.5 kHz, so give the upper carriers a fighting chance.
        carrier_levels = np.percentile(powers, 85, axis=0)
        max_level = float(np.max(carrier_levels))
        if max_level > 1e-15:
            gains = np.clip(max_level / np.maximum(carrier_levels, max_level / 16.0), 1.0, 16.0)
            powers = powers * gains[np.newaxis, :]
    ranking = np.argsort(powers, axis=1)
    best = ranking[:, -1]
    second = ranking[:, -2]
    row_idx = np.arange(symbol_count)
    snrs = 10.0 * np.log10((powers[row_idx, best] + 1e-15) / (powers[row_idx, second] + 1e-15))
    packed = (
        (best[0::4] << 6)
        | (best[1::4] << 4)
        | (best[2::4] << 2)
        | best[3::4]
    )
    return bytes(packed.astype(np.uint8)), float(np.mean(snrs))


def _candidate_payload_lengths(raw_length: int, avail_bytes: int) -> list[int]:
    max_possible = min(MAX_PAYLOAD, avail_bytes - 14)
    if max_possible < 0:
        return []
    candidates: list[int] = []

    def _add(val: int) -> None:
        if 0 <= val <= max_possible and val not in candidates:
            candidates.append(val)

    _add(raw_length)
    for shift in (0, 2, 4, 6):
        for sym in range(4):
            _add((raw_length & ~(3 << shift)) | (sym << shift))
    _add(0)
    _add(avail_bytes - 14)
    _add(MAX_PAYLOAD)
    return candidates


def _try_decode_at(
    filtered: np.ndarray,
    data_start: int,
    start_sample: int,
    correlation: float,
    equalize: bool = False,
) -> Decoded | None:
    if data_start < 0:
        return None
    avail_bytes = (len(filtered) - data_start) // (4 * SAMPLES_PER_SYMBOL)
    if avail_bytes < 14:
        return None
    header, _ = demodulate(
        filtered[data_start:data_start + 16 * SAMPLES_PER_SYMBOL],
        4,
        equalize=equalize,
    )
    lengths = _candidate_payload_lengths(header[3], avail_bytes)
    if not lengths:
        return None
    max_wire = 4 + max(lengths) + 2 + RS_PARITY
    full_wire, _ = demodulate(
        filtered[data_start:data_start + max_wire * 4 * SAMPLES_PER_SYMBOL],
        max_wire,
        equalize=equalize,
    )
    for length in lengths:
        wire_size = 4 + length + 2 + RS_PARITY
        wire = full_wire[:wire_size]
        decoded = rs_decode(wire)
        if decoded is None:
            continue
        corrected, corrected_count = decoded
        body_size = 4 + corrected[3]
        if (
            corrected[0] != SYNC
            or corrected[1] not in (DATA, SYN, SYN_ACK, ACK, NACK)
            or corrected[3] != length
            or corrected[3] > MAX_PAYLOAD
            or len(corrected) != body_size + 2 + RS_PARITY
        ):
            continue
        expected_crc = (corrected[body_size] << 8) | corrected[body_size + 1]
        if crc16_ccitt(corrected[:body_size]) != expected_crc:
            continue
        _, snr = demodulate(
            filtered[data_start:data_start + wire_size * 4 * SAMPLES_PER_SYMBOL],
            wire_size,
            equalize=equalize,
        )
        return Decoded(
            corrected[1],
            corrected[2],
            corrected[4:body_size],
            correlation,
            start_sample,
            snr,
            corrected_count,
        )
    return None


def decode(samples: np.ndarray) -> Decoded | None:
    found = find_preamble(samples)
    if found is None:
        return None
    start, correlation = found
    if correlation < CORRELATION_THRESHOLD:
        return None
    filtered = bandpass_filter(samples)
    base_data_start = start + PREAMBLE_SAMPLES + GUARD_SAMPLES
    offsets = (
        0, -24, 24, -48, 48, -72, 72, -120, 120,
        -180, 180, -240, 240, -360, 360, -480, 480,
    )
    for delta in offsets:
        for eq in (False, True):
            result = _try_decode_at(
                filtered, base_data_start + delta, start, correlation, equalize=eq
            )
            if result is not None:
                return result
    return None


def describe_capture(samples: np.ndarray) -> str:
    found = find_preamble(samples)
    if found is None:
        return "no correlation candidate"
    start, correlation = found
    filtered = bandpass_filter(samples)
    data_start = start + PREAMBLE_SAMPLES + GUARD_SAMPLES
    if data_start + 16 * SAMPLES_PER_SYMBOL > len(filtered):
        return f"chirp corr={correlation:.3f} at {start}, but capture ends before header"
    header, _ = demodulate(
        filtered[data_start:data_start + 16 * SAMPLES_PER_SYMBOL], 4
    )
    detail = (
        f"chirp corr={correlation:.3f} at {start}; "
        f"header={header.hex(' ')}"
    )
    result = decode(samples)
    if result is not None:
        return (
            detail + f"; full-frame symbol SNR={result.snr_db:.1f} dB, "
            f"RS corrected={result.corrected_bytes}, CRC=ok, "
            f"payload={result.payload!r}"
        )
    if header[0] != SYNC or header[3] > MAX_PAYLOAD:
        return detail + "; sync byte or length is wrong"
    wire_size = 4 + header[3] + 2 + RS_PARITY
    end = data_start + wire_size * 4 * SAMPLES_PER_SYMBOL
    if end > len(filtered):
        return detail + "; capture ends before complete frame"
    wire, snr = demodulate(filtered[data_start:end], wire_size)
    corrected = rs_decode(wire)
    if corrected is None:
        return detail + f"; full-frame symbol SNR={snr:.1f} dB, RS failed"
    codeword, count = corrected
    body_end = 4 + codeword[3]
    crc_ok = crc16_ccitt(codeword[:body_end]) == (
        (codeword[body_end] << 8) | codeword[body_end + 1]
    )
    return (
        detail + f"; full-frame symbol SNR={snr:.1f} dB, "
        f"RS corrected={count}, CRC={'ok' if crc_ok else 'bad'}, "
        f"payload={codeword[4:body_end]!r}"
    )


class AudioIO:
    """Blocking playback plus queued continuous mono capture."""

    def __init__(
        self,
        input_device: int | None = None,
        output_device: int | None = None,
        tx_level: float | None = None,
    ) -> None:
        default_input, default_output = sd.default.device
        self.input_device = default_input if input_device is None else input_device
        self.output_device = default_output if output_device is None else output_device
        self.tx_level = TX_LEVEL if tx_level is None else tx_level
        self._chunks: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.Stream | None = None
        self._in_stream: sd.InputStream | None = None
        self._out_stream: sd.OutputStream | None = None
        self._capture_suppressed = False
        self._playback_lock = threading.Lock()
        self._playback_done = threading.Event()
        self._playback_samples: np.ndarray | None = None
        self._playback_offset = 0

    def _fill_outdata(self, outdata: np.ndarray) -> None:
        outdata.fill(0.0)
        with self._playback_lock:
            if self._playback_samples is None:
                return
            available = self._playback_samples.size - self._playback_offset
            copied = min(available, outdata.shape[0])
            outdata[:copied, 0] = self._playback_samples[
                self._playback_offset:self._playback_offset + copied
            ]
            self._playback_offset += copied
            if self._playback_offset == self._playback_samples.size:
                self._playback_samples = None
                self._playback_offset = 0
                self._playback_done.set()

    def start(self) -> None:
        def duplex_callback(
            indata: np.ndarray,
            outdata: np.ndarray,
            frames: int,
            timing: object,
            status: sd.CallbackFlags,
        ) -> None:
            del frames, timing
            if status:
                print(f"[audio] stream status: {status}", flush=True)
            if not self._capture_suppressed:
                self._chunks.put(indata[:, 0].copy())
            self._fill_outdata(outdata)

        def in_callback(
            indata: np.ndarray,
            frames: int,
            timing: object,
            status: sd.CallbackFlags,
        ) -> None:
            del frames, timing
            if status:
                print(f"[audio-in] stream status: {status}", flush=True)
            if not self._capture_suppressed:
                self._chunks.put(indata[:, 0].copy())

        def out_callback(
            outdata: np.ndarray,
            frames: int,
            timing: object,
            status: sd.CallbackFlags,
        ) -> None:
            del frames, timing
            if status:
                print(f"[audio-out] stream status: {status}", flush=True)
            self._fill_outdata(outdata)

        try:
            self._stream = sd.Stream(
                device=(self.input_device, self.output_device),
                samplerate=SAMPLE_RATE,
                channels=(1, 1),
                dtype=("float32", "float32"),
                callback=duplex_callback,
                blocksize=4_800,
            )
            self._stream.start()
        except Exception:
            self._stream = None
            self._in_stream = sd.InputStream(
                device=self.input_device,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=in_callback,
                blocksize=4_800,
            )
            self._out_stream = sd.OutputStream(
                device=self.output_device,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=out_callback,
                blocksize=4_800,
            )
            self._in_stream.start()
            self._out_stream.start()

    def close(self) -> None:
        for stream in (self._stream, self._in_stream, self._out_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        self._stream = None
        self._in_stream = None
        self._out_stream = None

    def discard_capture(self) -> None:
        while True:
            try:
                self._chunks.get_nowait()
            except queue.Empty:
                return

    def suppress_capture(self, suppressed: bool) -> None:
        self._capture_suppressed = suppressed

    def read(self, timeout: float) -> np.ndarray | None:
        try:
            return self._chunks.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None

    def play(self, samples: np.ndarray) -> None:
        scaled = np.clip(
            np.asarray(samples, dtype=np.float32) * self.tx_level, -0.98, 0.98
        )
        drain_pad = np.zeros(9_600, dtype=np.float32)
        playback = np.concatenate((scaled, drain_pad))
        with self._playback_lock:
            if self._playback_samples is not None:
                raise RuntimeError("audio playback is already active")
            self._playback_done.clear()
            self._playback_samples = playback
            self._playback_offset = 0
        timeout = playback.size / SAMPLE_RATE + 2.5
        if not self._playback_done.wait(timeout):
            with self._playback_lock:
                self._playback_samples = None
                self._playback_offset = 0
            raise TimeoutError("audio playback timed out")


class Transport:
    """Half-duplex stop-and-wait modem transport over an AudioIO instance."""

    def __init__(self, audio: AudioIO, debug: bool = False) -> None:
        self.audio = audio
        self.debug = debug
        self.tx_sequence = 0
        self.rx_sequence = 0
        self._pending_data: Decoded | None = None

    def send_frame(self, frame_type: int, sequence: int, payload: bytes = b"") -> None:
        time.sleep(TURNAROUND_GUARD)
        self.audio.suppress_capture(True)
        self.audio.discard_capture()
        try:
            self.audio.play(modulate(payload, sequence, frame_type))
            time.sleep(TURNAROUND_GUARD)
        finally:
            self.audio.discard_capture()
            self.audio.suppress_capture(False)

    def receive_frame(self, timeout: float | None = None) -> Decoded | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        buffer = np.empty(0, dtype=np.float32)
        base_sample = 0
        candidate: np.ndarray | None = None
        candidate_start = 0
        expected_samples = 0
        header_samples = PREAMBLE_SAMPLES + GUARD_SAMPLES + 17 * SAMPLES_PER_SYMBOL
        min_frame_samples = (
            PREAMBLE_SAMPLES + GUARD_SAMPLES + (14 * 4 + 1) * SAMPLES_PER_SYMBOL
        )
        rolling_limit = PREAMBLE_SAMPLES + 9_600
        step_after_fail = SAMPLES_PER_SYMBOL
        tried_early = False

        while deadline is None or time.monotonic() < deadline:
            wait = 0.1 if deadline is None else min(0.1, max(0.0, deadline - time.monotonic()))
            chunk = self.audio.read(wait)
            if chunk is None:
                if candidate is not None and candidate.size >= min_frame_samples - SAMPLES_PER_SYMBOL:
                    decoded = decode(candidate)
                    if decoded is not None:
                        decoded.start_sample += candidate_start
                        return decoded
                continue

            if candidate is None:
                buffer = np.concatenate((buffer, chunk))
                if buffer.size < PREAMBLE_SAMPLES:
                    continue
                found = find_preamble(buffer)
                if found is None or found[1] < CORRELATION_THRESHOLD:
                    if buffer.size > rolling_limit:
                        discard = buffer.size - rolling_limit
                        buffer = buffer[discard:]
                        base_sample += discard
                    continue
                offset, score = found
                if (
                    offset + PREAMBLE_SAMPLES + 240 > buffer.size
                    and buffer.size < rolling_limit + 4_800
                ):
                    continue
                candidate_start = base_sample + offset
                candidate = buffer[offset:].copy()
                buffer = np.empty(0, dtype=np.float32)
                tried_early = False
                if self.debug:
                    print(
                        f"[sync] correlation={score:.3f} "
                        f"sample={candidate_start}",
                        flush=True,
                    )
            else:
                candidate = np.concatenate((candidate, chunk))

            if expected_samples == 0:
                if candidate.size < min_frame_samples:
                    continue
                # Any 0-byte control frame (SYN, SYN_ACK, ACK, NACK) is already complete here.
                decoded_ctrl = decode(candidate[:min_frame_samples])
                if decoded_ctrl is not None:
                    decoded_ctrl.start_sample += candidate_start
                    if self.debug:
                        fname = FRAME_NAMES.get(decoded_ctrl.frame_type, str(decoded_ctrl.frame_type))
                        print(
                            f"[frame] type={fname}({decoded_ctrl.frame_type}) "
                            f"sequence={decoded_ctrl.sequence} "
                            f"bytes={len(decoded_ctrl.payload)} "
                            f"snr={decoded_ctrl.snr_db:.1f}dB "
                            f"fec={decoded_ctrl.corrected_bytes}",
                            flush=True,
                        )
                    return decoded_ctrl

                header_start = PREAMBLE_SAMPLES + GUARD_SAMPLES
                filtered_head = bandpass_filter(candidate[:min_frame_samples])
                best_header: bytes | None = None
                best_score = -1
                for delta in (0, -24, 24, -48, 48, -120, 120, -240, 240):
                    h_pos = header_start + delta
                    if h_pos < 0 or h_pos + 16 * SAMPLES_PER_SYMBOL > filtered_head.size:
                        continue
                    for eq in (False, True):
                        hdr, _ = demodulate(
                            filtered_head[h_pos:h_pos + 16 * SAMPLES_PER_SYMBOL],
                            4,
                            equalize=eq,
                        )
                        score = (
                            (2 if hdr[0] == SYNC else 0)
                            + (2 if hdr[1] in (DATA, SYN, SYN_ACK, ACK, NACK) else 0)
                            + (1 if hdr[3] <= MAX_PAYLOAD else 0)
                        )
                        if score > best_score:
                            best_score = score
                            best_header = hdr
                            if score == 5:
                                break
                    if best_score == 5:
                        break

                assert best_header is not None
                header = best_header
                if self.debug:
                    ftype_name = FRAME_NAMES.get(header[1], f"0x{header[1]:02x}")
                    print(
                        "[header] "
                        + " ".join(f"{byte:02x}" for byte in header)
                        + f" (type={ftype_name}, seq={header[2]}, len={header[3]})",
                        flush=True,
                    )
                if best_score >= 3 and header[3] <= MAX_PAYLOAD:
                    wire_bytes = 4 + header[3] + 2 + RS_PARITY
                    expected_samples = (
                        PREAMBLE_SAMPLES + GUARD_SAMPLES
                        + (wire_bytes * 4 + 1) * SAMPLES_PER_SYMBOL
                    )
                else:
                    advance = min(step_after_fail, candidate.size)
                    buffer = candidate[advance:][-rolling_limit:]
                    base_sample = candidate_start + candidate.size - buffer.size
                    candidate = None
                    continue

            if candidate.size >= expected_samples:
                decoded = decode(candidate)
                if decoded is not None:
                    decoded.start_sample += candidate_start
                    if self.debug:
                        fname = FRAME_NAMES.get(decoded.frame_type, str(decoded.frame_type))
                        print(
                            f"[frame] type={fname}({decoded.frame_type}) "
                            f"sequence={decoded.sequence} "
                            f"bytes={len(decoded.payload)} "
                            f"snr={decoded.snr_db:.1f}dB "
                            f"fec={decoded.corrected_bytes}",
                            flush=True,
                        )
                    return decoded
                if not tried_early:
                    # Wait one more chunk in case multipath or a corrupted length dibit needs extra samples
                    tried_early = True
                    expected_samples = candidate.size + 4_800
                    continue
                if self.debug:
                    print("[frame] FEC/CRC/frame decode failed", flush=True)
                advance = min(step_after_fail, candidate.size)
                buffer = candidate[advance:][-rolling_limit:]
                base_sample = candidate_start + candidate.size - buffer.size
                candidate = None
                expected_samples = 0
                tried_early = False
        return None

    def handshake_initiator(self) -> bool:
        sequence = self.tx_sequence
        for attempt in range(1, MAX_RETRIES + 2):
            if self.debug:
                print(f"[handshake] sending SYN (attempt {attempt}/{MAX_RETRIES + 1})", flush=True)
            self.send_frame(SYN, sequence)
            reply = self.receive_frame(ACK_TIMEOUT)
            if reply and reply.frame_type == SYN_ACK and reply.sequence == sequence:
                self.send_frame(ACK, sequence)
                self.rx_sequence = 0
                return True
        return False

    def handshake_listener(self) -> bool:
        while True:
            request = self.receive_frame()
            if request is None:
                continue
            if request.frame_type == DATA:
                print(
                    f"[handshake] Received direct DATA frame (seq={request.sequence}, "
                    f"corr={request.correlation:.3f}, snr={request.snr_db:.1f}dB).",
                    flush=True,
                )
                self.rx_sequence = (request.sequence + 1) & 0xFF
                self.send_frame(ACK, request.sequence)
                self._pending_data = request
                return True
            if request.frame_type == SYN:
                print(
                    f"[handshake] SYN received (seq={request.sequence}, "
                    f"corr={request.correlation:.3f}, snr={request.snr_db:.1f}dB); "
                    "sending SYN_ACK...",
                    flush=True,
                )
                self.send_frame(SYN_ACK, request.sequence)
                reply = self.receive_frame(ACK_TIMEOUT * 2)
                if reply is None:
                    continue
                if reply.frame_type == ACK and reply.sequence == request.sequence:
                    self.rx_sequence = 0
                    return True
                if reply.frame_type == DATA:
                    self.rx_sequence = (reply.sequence + 1) & 0xFF
                    self.send_frame(ACK, reply.sequence)
                    self._pending_data = reply
                    return True
                if reply.frame_type == SYN:
                    self.send_frame(SYN_ACK, reply.sequence)
                    self.rx_sequence = 0
                    return True

    def send_reliable(self, payload: bytes) -> bool:
        if len(payload) > MAX_PAYLOAD:
            raise ValueError("payload exceeds 128 bytes")
        sequence = self.tx_sequence
        for _ in range(MAX_RETRIES + 1):
            self.send_frame(DATA, sequence, payload)
            reply = self.receive_frame(ACK_TIMEOUT)
            if reply and reply.sequence == sequence:
                if reply.frame_type == ACK:
                    self.tx_sequence = (sequence + 1) & 0xFF
                    return True
                if reply.frame_type == NACK:
                    continue
        return False

    def send_unreliable(self, payload: bytes, repeats: int = 1) -> None:
        if len(payload) > MAX_PAYLOAD:
            raise ValueError("payload exceeds 128 bytes")
        sequence = self.tx_sequence
        for _ in range(max(1, repeats)):
            self.send_frame(DATA, sequence, payload)
        self.tx_sequence = (sequence + 1) & 0xFF

    def receive_data(
        self, timeout: float | None = None, send_ack: bool = True
    ) -> Decoded | None:
        if self._pending_data is not None:
            pending = self._pending_data
            self._pending_data = None
            return pending
        deadline = None if timeout is None else time.monotonic() + timeout
        while deadline is None or time.monotonic() < deadline:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            frame = self.receive_frame(remaining)
            if frame is None:
                return None
            if frame.frame_type == SYN:
                if send_ack:
                    print(
                        f"\n[handshake] New SYN received (seq={frame.sequence}, "
                        f"corr={frame.correlation:.3f}, snr={frame.snr_db:.1f}dB); "
                        "sending SYN_ACK...",
                        flush=True,
                    )
                    self.send_frame(SYN_ACK, frame.sequence)
                self.rx_sequence = 0
                continue
            if frame.frame_type != DATA:
                continue
            if not send_ack:
                self.rx_sequence = (frame.sequence + 1) & 0xFF
                return frame
            if frame.sequence == self.rx_sequence:
                self.rx_sequence = (self.rx_sequence + 1) & 0xFF
                self.send_frame(ACK, frame.sequence)
                return frame
            if frame.sequence == ((self.rx_sequence - 1) & 0xFF):
                self.send_frame(ACK, frame.sequence)
            else:
                self.rx_sequence = (frame.sequence + 1) & 0xFF
                self.send_frame(ACK, frame.sequence)
                return frame
        return None


def simulate(payload: bytes) -> bool:
    transmitted = modulate(payload)
    rng = np.random.default_rng(0x5EED)
    channel = np.zeros(transmitted.size + 173, dtype=np.float32)
    channel[:transmitted.size] += transmitted
    channel[173:173 + transmitted.size] += 0.22 * transmitted
    channel += rng.normal(0.0, 0.008, channel.size).astype(np.float32)
    result = decode(channel)
    if result is None or result.payload != payload:
        print("Python synthetic loopback failed")
        return False
    print(
        f"Python synthetic loopback recovered {len(payload)} bytes; "
        f"correlation={result.correlation:.3f}, SNR={result.snr_db:.1f} dB"
    )
    return True


def _warn_if_headphone_output(output_device: int | None) -> None:
    # Transmitting ultrasonic packets into Bluetooth headphones rarely reaches the other laptop.
    try:
        default_in, default_out = sd.default.device
        out_idx = default_out if output_device is None else output_device
        if out_idx is None or out_idx < 0:
            return
        info = sd.query_devices(out_idx)
        name = str(info.get("name", ""))
        if any(tag in name.lower() for tag in ("headphone", "headset", "earphone", "buds")):
            all_devs = sd.query_devices()
            speakers = [
                f"{idx} ('{d.get('name')}')"
                for idx, d in enumerate(all_devs)
                if d.get("max_output_channels", 0) > 0
                and "speaker" in str(d.get("name", "")).lower()
                and "steam" not in str(d.get("name", "")).lower()
            ]
            hint = f" Try --output-device {speakers[0]}." if speakers else ""
            print(
                f"[notice] Output device {out_idx} is '{name}' (headphones/headset). "
                f"For over-the-air acoustic transmission, select a physical speaker.{hint}",
                flush=True,
            )
    except Exception:
        pass


def _carrier_snrs_db(samples: np.ndarray) -> dict[float, float]:
    filtered = bandpass_filter(samples)
    if filtered.size < 2048:
        return {c: 0.0 for c in CARRIERS}
    windowed = filtered * np.hanning(filtered.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(filtered.size, 1.0 / SAMPLE_RATE)
    low_hz = min(PREAMBLE_START_HZ, min(CARRIERS)) - 500.0
    high_hz = max(PREAMBLE_END_HZ, max(CARRIERS)) + 500.0
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    noise_floor = float(np.median(spectrum[band_mask])) if np.any(band_mask) else 1e-12
    result: dict[float, float] = {}
    for carrier in CARRIERS:
        mask = np.abs(freqs - carrier) <= 60.0
        peak = float(np.max(spectrum[mask])) if np.any(mask) else 0.0
        result[carrier] = float(20.0 * np.log10((peak + 1e-12) / (noise_floor + 1e-12)))
    return result


def hwtest(
    payload: bytes,
    output_device: int | None,
    input_device: int | None,
    tx_level: float | None = None,
) -> bool:
    _warn_if_headphone_output(output_device)
    tx = modulate(payload)
    lead = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    tail = np.zeros(int(1.25 * SAMPLE_RATE), dtype=np.float32)
    levels = (tx_level,) if tx_level is not None else (0.20, 0.35, 0.55)
    devices = None
    if output_device is not None or input_device is not None:
        default_input, default_output = sd.default.device
        devices = (
            default_input if input_device is None else input_device,
            default_output if output_device is None else output_device,
        )
    print(f"Duplex test at 48 kHz; devices={devices or sd.default.device}")
    print("Use a quiet room. Start with the mic 5-10 cm from built-in speakers "
          "or 10-20 cm from separate speakers.")
    for level in levels:
        scaled = np.clip(tx * level, -0.98, 0.98)
        playback = np.concatenate((lead, scaled, tail))
        sd.stop()
        time.sleep(0.15)
        recording = sd.playrec(
            playback,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=devices,
            blocking=True,
        ).reshape(-1)
        result = decode(recording)
        peak = float(np.max(np.abs(recording))) if recording.size else 0.0
        rms = float(np.sqrt(np.mean(recording.astype(np.float64) ** 2))) if recording.size else 0.0
        inband = bandpass_filter(recording) if recording.size else np.empty(0)
        inband_rms = float(np.sqrt(np.mean(inband ** 2))) if inband.size else 0.0
        if result is not None and result.payload == payload:
            print(
                f"PASS level={level:.2f} corr={result.correlation:.3f} "
                f"SNR={result.snr_db:.1f} dB corrected={result.corrected_bytes} "
                f"start={result.start_sample} peak={peak:.4f} "
                f"RMS={rms:.5f} (in-band RMS={inband_rms:.5f}); "
                "payload matches"
            )
            return True
        print(
            f"FAIL level={level:.2f} peak={peak:.4f} RMS={rms:.5f} "
            f"(in-band RMS={inband_rms:.5f}); {describe_capture(recording)}"
        )
    print("Python duplex link did not decode the frame.")
    return False


def rangetest(
    payload: bytes,
    output_device: int | None,
    input_device: int | None,
    tx_level: float = TX_LEVEL,
    count: int = 3,
    distance_label: str | None = None,
) -> int:
    """Measure packet reception, correlation, SNR, and carrier roll-off across distances."""
    _warn_if_headphone_output(output_device)
    lead = np.zeros(int(0.35 * SAMPLE_RATE), dtype=np.float32)
    tail = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    devices = None
    if output_device is not None or input_device is not None:
        default_input, default_output = sd.default.device
        devices = (
            default_input if input_device is None else input_device,
            default_output if output_device is None else output_device,
        )
    print(
        f"Range test at {SAMPLE_RATE} Hz (carriers={tuple(int(c) for c in CARRIERS)} Hz, "
        f"level={tx_level:.2f}); devices={devices or sd.default.device}"
    )

    def run_trial(label: str, baseline_start: int | None) -> tuple[int, int | None]:
        passed = 0
        corrs: list[float] = []
        snrs: list[float] = []
        fecs: list[int] = []
        starts: list[int] = []
        inband_rms_list: list[float] = []
        wide_rms_list: list[float] = []
        carrier_acc: dict[float, list[float]] = {c: [] for c in CARRIERS}

        print(f"\n--- Testing distance: {label} ({count} packet(s) at level={tx_level:.2f}) ---")
        for seq in range(1, count + 1):
            trial_payload = payload + f" #{seq}".encode("utf-8")
            if len(trial_payload) > MAX_PAYLOAD:
                trial_payload = payload[:MAX_PAYLOAD]
            tx = modulate(trial_payload, sequence=seq)
            scaled = np.clip(tx * tx_level, -0.98, 0.98)
            playback = np.concatenate((lead, scaled, tail))
            sd.stop()
            time.sleep(0.1)
            recording = sd.playrec(
                playback,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=devices,
                blocking=True,
            ).reshape(-1)
            wide_rms = float(np.sqrt(np.mean(recording.astype(np.float64) ** 2))) if recording.size else 0.0
            inband = bandpass_filter(recording) if recording.size else np.empty(0)
            inband_rms = float(np.sqrt(np.mean(inband ** 2))) if inband.size else 0.0
            wide_rms_list.append(wide_rms)
            inband_rms_list.append(inband_rms)
            for c_hz, c_snr in _carrier_snrs_db(recording).items():
                carrier_acc[c_hz].append(c_snr)

            result = decode(recording)
            if result is not None and result.payload == trial_payload:
                passed += 1
                corrs.append(result.correlation)
                snrs.append(result.snr_db)
                fecs.append(result.corrected_bytes)
                starts.append(result.start_sample)
                delta_str = ""
                if baseline_start is not None:
                    delta_samples = result.start_sample - baseline_start
                    est_delta_cm = delta_samples * (34300.0 / SAMPLE_RATE)
                    delta_str = f" delay_delta={delta_samples:+d} samp (~{est_delta_cm:+.0f} cm)"
                print(
                    f"  [{seq}/{count}] PASS corr={result.correlation:.3f} "
                    f"symbol_SNR={result.snr_db:.1f}dB fec={result.corrected_bytes} "
                    f"inband_RMS={inband_rms:.5f} wide_RMS={wide_rms:.5f}{delta_str}"
                )
            else:
                print(
                    f"  [{seq}/{count}] FAIL inband_RMS={inband_rms:.5f} "
                    f"wide_RMS={wide_rms:.5f} | {describe_capture(recording)}"
                )

        pdr = 100.0 * passed / max(1, count)
        carrier_summary = ", ".join(
            f"{int(c)}Hz:{np.mean(vals):.1f}dB" for c, vals in carrier_acc.items() if vals
        )
        new_baseline = baseline_start
        if starts and new_baseline is None:
            new_baseline = int(np.median(starts))
        if passed:
            print(
                f"Summary [{label}]: {passed}/{count} ({pdr:.0f}%) packets decoded | "
                f"avg corr={np.mean(corrs):.3f} (min {np.min(corrs):.3f}) | "
                f"avg symbol SNR={np.mean(snrs):.1f} dB | max FEC={np.max(fecs)}/4 | "
                f"carriers=[{carrier_summary}]"
            )
        else:
            print(
                f"Summary [{label}]: 0/{count} (0%) decoded | "
                f"carriers=[{carrier_summary}] | "
                "Tip: increase --level (e.g. 0.6-0.9), raise speaker volume, or try --band near"
            )
        return passed, new_baseline

    if distance_label is not None:
        passed, _ = run_trial(distance_label, None)
        return 0 if passed > 0 else 1

    print(
        "\nInteractive Range Test: place the microphone at each distance and press Enter.\n"
        "Type a distance label (e.g. '25cm', '50cm', '1m', '2m') or 'q' to finish."
    )
    baseline: int | None = None
    step_default = ["25cm", "50cm", "100cm", "150cm", "200cm", "300cm"]
    step_idx = 0
    any_passed = False
    while True:
        suggested = step_default[min(step_idx, len(step_default) - 1)]
        try:
            raw = input(f"\nEnter distance label [default={suggested}, 'q'=quit]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nRange test finished.")
            break
        if raw.lower() in ("q", "quit", "exit"):
            break
        label = raw if raw else suggested
        step_idx += 1
        passed, baseline = run_trial(label, baseline)
        if passed > 0:
            any_passed = True
    return 0 if any_passed else 1


def _show_received(frame: Decoded) -> None:
    try:
        message = frame.payload.decode("utf-8")
        print(
            f"\n[rx seq={frame.sequence} corr={frame.correlation:.3f} "
            f"snr={frame.snr_db:.1f}dB fec={frame.corrected_bytes}] {message}",
            flush=True,
        )
    except UnicodeDecodeError:
        print(
            f"\n[rx seq={frame.sequence} corr={frame.correlation:.3f} "
            f"snr={frame.snr_db:.1f}dB fec={frame.corrected_bytes}] "
            f"hex:{frame.payload.hex()}",
            flush=True,
        )


def _audio_transport(
    input_device: int | None,
    output_device: int | None,
    debug: bool,
    tx_level: float | None = None,
) -> tuple[AudioIO, Transport]:
    _warn_if_headphone_output(output_device)
    audio = AudioIO(input_device, output_device, tx_level=tx_level)
    audio.start()
    print(
        f"Audio ready: input device {audio.input_device}, "
        f"output device {audio.output_device}, {SAMPLE_RATE} Hz mono "
        f"(carriers={tuple(int(c) for c in CARRIERS)} Hz, level={audio.tx_level:.2f})."
    )
    return audio, Transport(audio, debug)


def _split_text(text: str) -> list[bytes]:
    packets: list[bytes] = []
    current = bytearray()
    for character in text:
        encoded = character.encode("utf-8")
        if len(current) + len(encoded) > MAX_PAYLOAD:
            packets.append(bytes(current))
            current.clear()
        current.extend(encoded)
    if current or not packets:
        packets.append(bytes(current))
    return packets


def run_tx(
    payload: bytes,
    input_device: int | None,
    output_device: int | None,
    debug: bool = False,
    text: str | None = None,
    no_ack: bool = False,
    tx_level: float | None = None,
    repeats: int = 1,
) -> int:
    audio, transport = _audio_transport(input_device, output_device, debug, tx_level)
    try:
        packets = (
            _split_text(text)
            if text is not None
            else [
                payload[i:i + MAX_PAYLOAD]
                for i in range(0, len(payload), MAX_PAYLOAD)
            ]
        )
        if no_ack:
            print(
                f"Broadcasting {len(packets)} packet(s) in one-way mode "
                f"(--no-ack, repeats={repeats})..."
            )
            for index, packet in enumerate(packets, start=1):
                transport.send_unreliable(packet, repeats=repeats)
                print(f"TX {index}/{len(packets)} ({len(packet)} byte(s))")
            return 0

        print("Starting SYN handshake (use --no-ack for one-way broadcast if reverse audio link is unavailable)...")
        if not transport.handshake_initiator():
            print(
                "Handshake failed after retries. "
                "Tip: verify both devices' speakers/mics, raise --level, or pass --no-ack for one-way transmission."
            )
            return 1
        for index, packet in enumerate(packets, start=1):
            if not transport.send_reliable(packet):
                print(f"Packet {index}/{len(packets)} was not acknowledged.")
                return 1
            print(f"ACK {index}/{len(packets)} ({len(packet)} byte(s))")
        return 0
    finally:
        audio.close()


def run_rx(
    input_device: int | None,
    output_device: int | None,
    debug: bool = False,
    no_ack: bool = False,
    tx_level: float | None = None,
) -> int:
    audio, transport = _audio_transport(input_device, output_device, debug, tx_level)
    try:
        if no_ack:
            print("Listening in passive one-way mode (--no-ack). Press Ctrl+C to stop.")
        else:
            print("Listening for SYN or DATA frames... Press Ctrl+C to stop.")
            if not transport.handshake_listener():
                print("Handshake failed.")
                return 1
            print("Connected; receiving DATA frames. Press Ctrl+C to stop.")
        while True:
            frame = transport.receive_data(send_ack=not no_ack)
            if frame is not None:
                _show_received(frame)
    except KeyboardInterrupt:
        print("\nReceiver stopped.")
        return 0
    finally:
        audio.close()


def run_chat(
    listener: bool,
    input_device: int | None,
    output_device: int | None,
    debug: bool = False,
    tx_level: float | None = None,
) -> int:
    audio, transport = _audio_transport(input_device, output_device, debug, tx_level)
    try:
        if listener:
            print("Waiting for chat initiator...")
            connected = transport.handshake_listener()
        else:
            print("Starting chat handshake...")
            connected = transport.handshake_initiator()
        if not connected:
            print("Chat handshake failed.")
            return 1
        print("Connected. Turns alternate; enter /quit to exit.")

        while True:
            if listener:
                incoming = transport.receive_data()
                if incoming is None:
                    continue
                _show_received(incoming)
                if incoming.payload == b"/quit":
                    return 0

            try:
                text = input("reply> " if listener else "you> ")
            except EOFError:
                text = "/quit"
            if text == "/quit":
                if not transport.send_reliable(b"/quit"):
                    print("Peer did not acknowledge /quit.")
                    return 1
                return 0

            payload = text.encode("utf-8")
            if len(payload) > MAX_PAYLOAD:
                print(f"Message is {len(payload)} UTF-8 bytes; limit is {MAX_PAYLOAD}.")
                continue
            if not transport.send_reliable(payload):
                print("Peer did not acknowledge the message.")
                return 1

            if not listener:
                incoming = transport.receive_data()
                if incoming is None:
                    continue
                _show_received(incoming)
                if incoming.payload == b"/quit":
                    return 0
    except KeyboardInterrupt:
        print("\nChat stopped.")
        return 0
    finally:
        audio.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sim", metavar="TEXT", help="run deterministic synthetic loopback")
    group.add_argument("--hwtest", action="store_true", help="play and record a real frame")
    group.add_argument("--rangetest", action="store_true", help="measure packet reception across distances")
    group.add_argument("--devices", action="store_true", help="list sounddevice indices")
    group.add_argument("--tx", metavar="TEXT", help="send text to a listening modem")
    group.add_argument("--tx-file", metavar="PATH", help="send a binary file")
    group.add_argument("--rx", action="store_true", help="listen for reliable packets")
    group.add_argument("--chat", action="store_true", help="interactive half-duplex chat")
    parser.add_argument("--output-device", type=int, help="sounddevice output device index")
    parser.add_argument("--input-device", type=int, help="sounddevice input device index")
    parser.add_argument("--text", default="LIVE-AIR-GAP", help="hardware/range test payload")
    parser.add_argument("--listen", action="store_true", help="wait for chat initiator")
    parser.add_argument("--verbose", action="store_true", help="print sync/frame diagnostics")
    parser.add_argument(
        "--level",
        type=float,
        default=None,
        help=f"transmit amplitude scale 0.05..1.0 (default {TX_LEVEL})",
    )
    parser.add_argument(
        "--band",
        choices=tuple(BAND_PRESETS.keys()),
        default="ultrasonic",
        help="frequency band preset: ultrasonic (18.2-19.4kHz), near (14.8-16.0kHz), audible (2.0-3.2kHz)",
    )
    parser.add_argument(
        "--no-ack",
        action="store_true",
        help="one-way broadcast/receive without requiring reverse-link SYN/ACK handshake",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="number of times to transmit each packet in --no-ack mode (default 1)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="packets per distance trial in --rangetest (default 3)",
    )
    parser.add_argument(
        "--distance",
        metavar="LABEL",
        default=None,
        help="run a single non-interactive --rangetest trial for this distance label (e.g. '50cm')",
    )
    args = parser.parse_args()
    set_band(args.band)
    if args.devices:
        print(sd.query_devices())
        return 0
    if args.sim is not None:
        return 0 if simulate(args.sim.encode("utf-8")) else 1
    if args.hwtest:
        return 0 if hwtest(
            args.text.encode("utf-8"), args.output_device, args.input_device, args.level
        ) else 1
    if args.rangetest:
        return rangetest(
            args.text.encode("utf-8"),
            args.output_device,
            args.input_device,
            tx_level=TX_LEVEL if args.level is None else args.level,
            count=args.count,
            distance_label=args.distance,
        )
    if args.tx is not None:
        return run_tx(
            args.tx.encode("utf-8"),
            args.input_device,
            args.output_device,
            args.verbose,
            args.tx,
            no_ack=args.no_ack,
            tx_level=args.level,
            repeats=args.repeats,
        )
    if args.tx_file is not None:
        with open(args.tx_file, "rb") as source:
            payload = source.read()
        return run_tx(
            payload,
            args.input_device,
            args.output_device,
            args.verbose,
            no_ack=args.no_ack,
            tx_level=args.level,
            repeats=args.repeats,
        )
    if args.rx:
        return run_rx(
            args.input_device,
            args.output_device,
            args.verbose,
            no_ack=args.no_ack,
            tx_level=args.level,
        )
    if args.chat:
        return run_chat(
            args.listen, args.input_device, args.output_device, args.verbose, args.level
        )
    parser.error("select an operation")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
