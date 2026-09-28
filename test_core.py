"""
Automated unit tests for morse_code.py and signal_proc.py.
Run with: python test_core.py
"""

import os
import tempfile
import numpy as np

import morse_code as mc
import signal_proc as sp


def test_text_roundtrip():
    """Verify converting text -> Morse -> text returns original text."""
    for text in ["HELLO WORLD", "SOS", "Python 3.12!", "A B C", "FAST CW 2026"]:
        morse = mc.text_to_morse(text)
        back = mc.morse_to_text(morse)
        assert back == text.upper(), f"Roundtrip failed for {text!r}: got {back!r}"
    print("[OK] text <-> morse roundtrip")


def test_signal_generation():
    """Verify Morse text produces non-empty audio within amplitude limits."""
    morse = mc.text_to_morse("SOS")
    sig = sp.morse_to_signal(morse, freq=700, wpm=20, sample_rate=8000, amplitude=0.8)
    assert len(sig) > 0
    assert np.max(np.abs(sig)) <= 0.8 + 1e-9
    print(f"[OK] signal generation ({len(sig)} samples, {len(sig)/8000:.2f}s)")
    return sig


def test_fft_recovers_frequency(sig):
    """Verify FFT accurately detects carrier pitch (~700 Hz)."""
    freq_est = sp.dominant_frequency(sig, 8000)
    assert abs(freq_est - 700) < 15, f"Expected ~700 Hz, got {freq_est} Hz"
    print(f"[OK] FFT dominant frequency ~= {freq_est:.1f} Hz")


def test_carrier_detection():
    """Verify PSD carrier pitch detection works in noisy conditions."""
    for f in [550, 750, 1000]:
        morse = mc.text_to_morse("TEST")
        sig = sp.morse_to_signal(morse, freq=f, wpm=18, sample_rate=16000)
        # Add background room noise and DC bias
        noisy = sig + 0.05 * np.random.randn(len(sig)) + 0.02
        detected = sp.detect_carrier_frequency(noisy, 16000)
        assert abs(detected - f) < 25, f"Expected ~{f} Hz, got {detected} Hz"
    print("[OK] carrier frequency detection across multiple frequencies")


def test_noise_filter_decode_pipeline(sig):
    """Verify the full pipeline: clean -> add noise -> band-pass filter -> decode."""
    noisy = sp.add_noise(sig, snr_db=6)
    filtered = sp.bandpass_filter(noisy, 700, 8000, bandwidth=250)
    morse_out = sp.decode_signal_to_morse(filtered, 8000)
    text_out = mc.morse_to_text(morse_out)
    assert text_out == "SOS", f"Expected SOS, got {text_out!r}"
    print(f"[OK] noise -> filter -> decode pipeline recovered: {text_out!r}")


def test_dash_only_and_dot_only_words():
    """Verify decoding words composed entirely of dashes or dots."""
    for word in ["M", "O", "T", "E", "HI", "NO"]:
        morse = mc.text_to_morse(word)
        sig = sp.morse_to_signal(morse, freq=750, wpm=20, sample_rate=16000)
        padded = np.concatenate([np.zeros(2000), sig, np.zeros(2000)])
        morse_decoded = sp.decode_signal_to_morse(padded, 16000, carrier_freq=750)
        text_decoded = mc.morse_to_text(morse_decoded)
        assert text_decoded == word, f"Expected {word}, got {text_decoded!r} (morse: {morse_decoded})"
    print("[OK] dash-only and single-pulse word decoding ('M', 'O', 'T', 'E', 'HI', 'NO')")


def test_realistic_mic_recording_decode():
    """Simulate room microphone audio: leading/trailing silence, 60Hz hum, and click noise."""
    text = "HELLO WORLD"
    morse = mc.text_to_morse(text)
    sig = sp.morse_to_signal(morse, freq=700, wpm=18, sample_rate=44100)
    lead = np.zeros(int(44100 * 1.0))
    trail = np.zeros(int(44100 * 1.0))
    audio = np.concatenate([lead, sig, trail])

    t = np.arange(len(audio)) / 44100.0
    noise = 0.04 * np.sin(2 * np.pi * 60 * t) + 0.02 * np.random.randn(len(audio)) + 0.015
    audio_noisy = audio + noise
    audio_noisy[3000:3300] += 0.5  # Transient click spike

    morse_out, details = sp.decode_signal_to_morse(audio_noisy, 44100, return_details=True)
    text_out = mc.morse_to_text(morse_out)
    assert text_out == text, f"Realistic mic decode failed! Expected {text!r}, got {text_out!r}"
    assert details['wpm'] > 12 and details['wpm'] < 25
    print(f"[OK] realistic mic recording decode recovered: {text_out!r} (est. {details['wpm']:.1f} WPM)")


def test_wav_file_roundtrip():
    """Verify audio waveform can be saved to WAV file and read back accurately."""
    sig = sp.morse_to_signal(mc.text_to_morse("WAV TEST"), freq=800, wpm=22, sample_rate=16000)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_path = f.name
    try:
        sp.save_wav(tmp_path, sig, 16000)
        loaded, sr = sp.load_wav(tmp_path)
        assert sr == 16000
        assert len(loaded) == len(sig)
        assert np.max(np.abs(loaded - sig)) < 1e-3
        print("[OK] WAV save and load roundtrip")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_invalid_morse_detection():
    """Verify validation helper catches invalid Morse characters."""
    assert mc.is_valid_morse(".... . .-.. .-.. --- / .-- --- .-. .-.. -..")
    assert not mc.is_valid_morse("HELLO")
    print("[OK] morse validity check")


def test_noise_tab_wav_export():
    """Verify Noise & Filter tab exports noisy and filtered waves to WAV files."""
    try:
        from app import MorseSignalLab
        app = MorseSignalLab()
    except Exception as e:
        print(f"[SKIP] Noise tab WAV export UI test (display unavailable: {e})")
        return

    noise_tab = app.noise_tab
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f1:
        tmp_noisy = f1.name
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f2:
        tmp_filtered = f2.name

    try:
        noise_tab.add_awgn()
        p1 = noise_tab.export_noisy_wav(filepath=tmp_noisy)
        assert p1 == tmp_noisy
        data_n, sr_n = sp.load_wav(tmp_noisy)
        assert len(data_n) > 0 and sr_n == 44100

        p2 = noise_tab.export_filtered_wav(filepath=tmp_filtered)
        assert p2 == tmp_filtered
        data_f, sr_f = sp.load_wav(tmp_filtered)
        assert len(data_f) > 0 and sr_f == 44100

        assert hasattr(noise_tab, 'export_btn')
        print("[OK] Noise tab noisy & filtered WAV export")
    finally:
        if os.path.exists(tmp_noisy):
            os.remove(tmp_noisy)
        if os.path.exists(tmp_filtered):
            os.remove(tmp_filtered)
        app.destroy()


if __name__ == '__main__':
    test_text_roundtrip()
    sig = test_signal_generation()
    test_fft_recovers_frequency(sig)
    test_carrier_detection()
    test_noise_filter_decode_pipeline(sig)
    test_dash_only_and_dot_only_words()
    test_realistic_mic_recording_decode()
    test_wav_file_roundtrip()
    test_invalid_morse_detection()
    test_noise_tab_wav_export()
    print("\nAll core & advanced DSP tests passed successfully.")