# DotDash — DSP & Systems (CSE-220)

**Project by:**
- **Rezoan Kabbo** (Roll: 2305138) — Part I: Generation & Signal Lab (Tabs 1 & 2)
- **Arefin Sowad** (Roll: 2305145) — Part II: Noise, Filtering & Decoder (Tabs 3 & 4)

An interactive Digital Signal Processing (DSP) desktop application built with Python and CustomTkinter. The project implements a complete Morse code communication pipeline: text-to-Morse translation, audio waveform synthesis, frequency-domain analysis (FFT & Spectrogram), channel noise injection, Butterworth band-pass filtering, and **microphone & recorded audio demodulation/decoding**.

---

## Key Features Across All 4 Tabs

### 1. Converter Tab (`morse_code.py` & UI)
- **Bidirectional Conversion**: Translate between Plaintext and International Morse Code (`.-`).
- **Live Timing Estimation**: Automatically computes standard PARIS timing units and transmission duration in seconds at target WPM.
- **Audio Preview**: Listen to synthesized Morse tones immediately using the `🔊 Listen` preview button.
- **Quick Presets & Clipboard**: 1-click test phrases (`HELLO WORLD`, `SOS`, `CQ CQ CQ`) and clipboard copy buttons.
- **Searchable Reference Table**: Complete International Morse Code lookup table with real-time letter filtering.

### 2. Signal Lab Tab (`signal_proc.py` & UI)
- **Sine-Tone Synthesis**: Generates clean audio waveforms with 5ms fade-in/fade-out envelopes to prevent click artifacts.
- **Adjustable Parameters**:
  - Carrier Frequency (200 – 1500 Hz, default 700 Hz)
  - Speed in WPM (5 – 40 WPM, default 20 WPM)
  - Volume (0 – 100%)
  - Sample Rate (8000, 16000, 22050, 44100 Hz)
- **Interactive Multi-View Visualizations**:
  - **Time-Domain Waveform**
  - **Frequency Spectrum (FFT)**: Displays tone carrier peak and harmonic structure.
  - **Time-Frequency Spectrogram**: Visual representation of Morse pulses over time.
- **Audio Playback & WAV Export**: Listen in real-time or export synthesized audio as standard 16-bit PCM `.wav` files.
- **Pipeline Routing**: 1-click transmission to Noise Lab or Mic/Audio Decoder.

### 3. Noise & Filter Tab (`signal_proc.py` & UI)
- **Channel Noise Simulation**:
  - **Additive White Gaussian Noise (AWGN)**
  - **60Hz Mains Hum + Harmonics** (realistic powerline noise)
  - **Acoustic Clicks & Pops** (ambient room noise)
  - Target SNR slider from -10 dB to +30 dB
- **Butterworth Band-pass Filtering**:
  - Zero-phase filtering (`scipy.signal.filtfilt`) centered around the carrier frequency.
  - Adjustable bandwidth (50 – 600 Hz).
- **Dual Visualizations**:
  - Time-Domain: Noisy vs. Filtered waveforms.
  - Frequency Spectrum (FFT): Overlaid spectra demonstrating out-of-band noise rejection.
- **Audio Audition & WAV Export**:
  - `💾 Export (Filtered)` / `💾 Export (Noisy)` button dynamically saves the active audition waveform to 16-bit PCM `.wav`.
  - `Route / Export ▾` menu options for 1-click export of Filtered or Noisy signals.
  - Right-click context menu on export button to quickly choose between Filtered, Noisy, or Clean waveforms.
- **Direct Demodulation & Accuracy**: Demodulates filtered signal and verifies recovery against the original transmission.

### 4. Microphone & Audio Decoder Tab (Star Feature)
- **Multi-Source Audio Input**:
  - **Live Microphone Recording**: Record directly from default mic with interactive countdown and progress bar.
  - **WAV Audio File Import**: Open and decode any pre-recorded audio file (`.wav`).
  - **Signal Lab Import**: 1-click import from Signal Lab or Noise Lab for rapid DSP verification.
  - **Quick Simulation Presets**: Built-in test signals with simulated room noise, silence, and acoustic transients.
- **Robust DSP Demodulation Pipeline**:
  - **Automatic Carrier Tone Detection**: Uses Welch Power Spectral Density (PSD) to locate the carrier peak in the Morse audio band (350 – 2500 Hz), completely ignoring DC offset, speech rumble, and high-frequency hiss.
  - **Bandpass Pre-filtering**: Isolates carrier tone before envelope detection.
  - **Hilbert Transform Envelope**: Extracts instantaneous analytic signal amplitude with moving-average smoothing.
  - **Otsu Adaptive Thresholding**: Automatically determines optimal bimodal threshold separating active Morse pulses from background room noise.
  - **Acoustic Glitch Debouncing**: Rejects transient acoustic clicks or sample dropouts (< 15 ms).
  - **Lead-in / Lead-out Silence Trimming**: Strips quiet gaps before transmission starts and after it ends.
  - **Smart Dot vs Dash Timing Calibration**: Accurately distinguishes dots ($1T$) and dashes ($3T$) even for signals with only dashes (e.g. `M`, `O`, `T`) or only dots (e.g. `E`, `S`, `H`) by analyzing intra-character gap durations.
- **Performance Metrics Cards**:
  - 🎯 Detected Carrier Frequency (Hz)
  - ⚡ Estimated Speed (WPM)
  - 📶 Signal Quality / SNR (dB)
  - ⏱ Detected Timing Unit (Dot duration in ms)
- **Audio Playback & WAV Export**:
  - `▶ Play Audio`: Listen back to what the microphone captured.
  - `💾 Save Recording (WAV)`: Export recorded mic audio to a `.wav` file.
- **Multi-View DSP Plots**:
  - **Full Demodulation**: Top plot shows Raw Audio and Bandpass Filtered Audio; Bottom plot shows Envelope curve, Otsu Threshold line, and active pulse shading.
  - **Waveform Only**: Clean view of the captured audio.
  - **Frequency Spectrum (FFT)**: Highlights carrier peak with a vertical reference line.

---

## How to Run

### Requirements
Install dependencies using the virtual environment or system Python:
```bash
pip install -r requirements.txt
```

### Running the Desktop Application
```bash
python app.py
```
*(or `./.venv/bin/python app.py`)*

### Running Core & DSP Sanity Tests
```bash
python test_core.py
```

---

## File Structure

- `app.py`: Clean application entry point (~140 lines) setting up the CustomTkinter root window, shared state, and tabs.
- `ui/`: Modular package containing all user interface tab views and styling:
  - `ui/theme.py`: Consistent color palettes for Light/Dark modes and Matplotlib axis styler.
  - `ui/converter_tab.py`: Tab 1 Plaintext ↔ Morse converter, PARIS timing, and reference table.
  - `ui/signal_tab.py`: Tab 2 Signal synthesis, Waveform, FFT Spectrum, Spectrogram, and WAV export.
  - `ui/noise_tab.py`: Tab 3 Channel noise simulation, Butterworth filter, and |H(f)| response curve.
  - `ui/mic_tab.py`: Tab 4 Live microphone capture, input device selection, WAV audio import, and demodulator.
- `morse_code.py`: International Morse code dictionary, text <-> morse conversions, and PARIS timing calculations.
- `signal_proc.py`: Digital signal processing engine (synthesis, sounddevice audio I/O, WAV I/O, FFT, Welch PSD carrier detection, noise models, Butterworth filtering, Hilbert envelope, Otsu thresholding, debouncing, and Morse decoding).
- `test_core.py`: Comprehensive test suite testing text roundtrip, synthesis, FFT, carrier detection, dash-only words, realistic mic noise decoding, WAV roundtrip, and validation.
- `requirements.txt`: Python package dependencies.

