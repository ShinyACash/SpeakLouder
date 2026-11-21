import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48000
DURATION = 2.0
TEST_FREQS = [18000, 19000, 20000]
AMPLITUDE = 0.25  # loud enough to register, quiet enough not to cook the tweeters

def main():
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    
    # 50ms fade-in/out so the speaker doesn't click
    tx_signal = sum(np.sin(2 * np.pi * f * t) for f in TEST_FREQS) / len(TEST_FREQS)
    fade_samples = int(0.05 * SAMPLE_RATE)
    envelope = np.ones_like(tx_signal)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    tx_signal = (tx_signal * envelope * AMPLITUDE).astype(np.float32)

    print(f"Transmitting {TEST_FREQS} Hz at {SAMPLE_RATE} Hz sample rate...")
    rx_signal = sd.playrec(tx_signal, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    rx_signal = rx_signal.flatten()

    # drop first/last 0.2s of driver startup/tail latency
    trim = int(0.2 * SAMPLE_RATE)
    rx_trimmed = rx_signal[trim:-trim]

    fft_vals = np.abs(np.fft.rfft(rx_trimmed * np.hanning(len(rx_trimmed))))
    freqs = np.fft.rfftfreq(len(rx_trimmed), 1.0 / SAMPLE_RATE)

    # ultrasonic noise floor in 17.2 - 17.8 kHz
    noise_mask = (freqs >= 17200) & (freqs <= 17800)
    noise_floor = np.median(fft_vals[noise_mask])

    print("\n--- Hardware Frequency Response ---")
    for target in TEST_FREQS:
        idx = np.argmin(np.abs(freqs - target))
        # +/- 15 Hz window for minor clock drift
        peak = np.max(fft_vals[idx - 15 : idx + 15])
        snr_db = 20 * np.log10((peak + 1e-12) / (noise_floor + 1e-12))
        status = "PASS" if snr_db >= 15.0 else "WEAK / FILTERED"
        print(f"{target} Hz | SNR: {snr_db:6.2f} dB | {status}")

if __name__ == "__main__":
    main()