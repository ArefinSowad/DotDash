"""
Tab 4: Microphone recording and WAV audio demodulation lab.
Records audio, detects carrier pitch, recovers Morse timing via adaptive thresholding, and decodes to text.
"""

import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import TYPE_CHECKING, Optional
import numpy as np

import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import morse_code as mc
import signal_proc as sp
from ui.theme import (
    get_plot_theme, style_axis,
    BTN_PRIMARY, BTN_PRIMARY_HOVER,
    BTN_PLAY, BTN_PLAY_HOVER,
    BTN_STOP, BTN_STOP_HOVER,
    BTN_SECONDARY, BTN_SECONDARY_HOVER, BTN_SECONDARY_TEXT,
    BTN_TRANSFER, BTN_TRANSFER_HOVER, BTN_TRANSFER_TEXT,
    BTN_HEIGHT_MD, BTN_HEIGHT_SM
)

if TYPE_CHECKING:
    from app import MorseSignalLab


class MicDecodeTab(ctk.CTkFrame):
    """Tab for recording live audio and demodulating Morse signals."""

    def __init__(self, parent: ctk.CTkBaseClass, app: "MorseSignalLab"):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self._is_decoding = False
        self._is_playing = False
        self.pack(fill='both', expand=True, padx=4, pady=2)

        # Build UI layout
        self._build_settings_card()
        self._build_action_bar()
        self._build_results_card()
        self._build_plot_section()

    # UI Layout Construction

    def _build_settings_card(self):
        """Card for hardware device selection and carrier pitch tuning."""
        card = ctk.CTkFrame(self, corner_radius=8)
        card.pack(fill='x', pady=(0, 4), padx=1)
        card.grid_columnconfigure((0, 1), weight=1)

        # Microphone input device and recording duration
        left_box = ctk.CTkFrame(card, fg_color="transparent")
        left_box.grid(row=0, column=0, sticky='nsew', padx=(8, 6), pady=4)
        left_box.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(left_box, text="Input Device:", font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=0, sticky='w', pady=1)
        input_devices = sp.get_input_devices()
        dev_names = [f"{idx}: {name[:24]}" for idx, name in input_devices] or ["Default Microphone"]
        self.dev_var = ctk.StringVar(value=dev_names[0])
        self.dev_menu = ctk.CTkOptionMenu(left_box, variable=self.dev_var, values=dev_names, height=20, font=ctk.CTkFont(size=10))
        self.dev_menu.grid(row=0, column=1, sticky='ew', padx=(6, 0), pady=1)

        ctk.CTkLabel(left_box, text="Rec Duration:", font=ctk.CTkFont(size=10)).grid(row=1, column=0, sticky='w', pady=1)
        self.dur_lbl = ctk.CTkLabel(left_box, text="5 s", width=36, font=ctk.CTkFont(size=10, weight="bold"))
        self.dur_lbl.grid(row=1, column=2, sticky='e', pady=1)
        self.duration_var = tk.DoubleVar(value=5.0)
        dur_slider = ctk.CTkSlider(
            left_box, from_=1, to=20, variable=self.duration_var, height=12,
            command=lambda v: self.dur_lbl.configure(text=f"{v:.0f} s")
        )
        dur_slider.grid(row=1, column=1, sticky='ew', padx=6, pady=1)

        self.source_var = ctk.StringVar(value="Source: None")
        ctk.CTkLabel(left_box, textvariable=self.source_var, font=ctk.CTkFont(size=9), text_color="gray").grid(
            row=2, column=0, columnspan=3, sticky='w', pady=(2, 0)
        )

        # Carrier frequency auto-detection or manual tuning
        right_box = ctk.CTkFrame(card, fg_color="transparent")
        right_box.grid(row=0, column=1, sticky='nsew', padx=(6, 8), pady=4)
        right_box.grid_columnconfigure(1, weight=1)

        mode_row = ctk.CTkFrame(right_box, fg_color="transparent")
        mode_row.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 2))
        ctk.CTkLabel(mode_row, text="Carrier Tuning:", font=ctk.CTkFont(size=10, weight="bold")).pack(side='left', padx=(0, 6))

        self.carrier_mode_var = ctk.StringVar(value="Auto-Detect (FFT Peak)")
        self.carrier_mode_seg = ctk.CTkSegmentedButton(
            mode_row,
            values=["Auto-Detect (FFT Peak)", "Manual Frequency"],
            variable=self.carrier_mode_var,
            command=self.on_carrier_mode_change,
            height=20,
            font=ctk.CTkFont(size=10)
        )
        self.carrier_mode_seg.pack(side='left', padx=(0, 6))

        ctk.CTkLabel(mode_row, text="Bandwidth:", font=ctk.CTkFont(size=10)).pack(side='left', padx=(2, 3))
        self.bw_var = tk.DoubleVar(value=200)
        self.bw_menu = ctk.CTkOptionMenu(
            mode_row, values=["150 Hz", "200 Hz", "250 Hz", "350 Hz"], width=78, height=20, font=ctk.CTkFont(size=10),
            command=lambda val: self.bw_var.set(float(val.replace(" Hz", "")))
        )
        self.bw_menu.set("200 Hz")
        self.bw_menu.pack(side='left')

        self.carrier_freq_var = tk.DoubleVar(value=700)
        ctk.CTkLabel(right_box, text="Manual Pitch:", font=ctk.CTkFont(size=10)).grid(row=1, column=0, sticky='w', pady=1)
        self.freq_slider_lbl = ctk.CTkLabel(right_box, text="700 Hz", font=ctk.CTkFont(size=10, weight="bold"), width=45)
        self.freq_slider_lbl.grid(row=1, column=2, sticky='e', pady=1)
        self.freq_slider = ctk.CTkSlider(
            right_box, from_=300, to=1500, variable=self.carrier_freq_var, height=12,
            command=lambda v: self.freq_slider_lbl.configure(text=f"{v:.0f} Hz")
        )
        self.freq_slider.grid(row=1, column=1, sticky='ew', padx=6, pady=1)
        self.freq_slider.configure(state="disabled")

    def _build_action_bar(self):
        """Action bar for microphone capture, file opening, demodulation, and routing."""
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill='x', pady=(0, 3))

        left_ops = ctk.CTkFrame(bar, fg_color="transparent")
        left_ops.pack(side='left')

        self.record_btn = ctk.CTkButton(
            left_ops,
            text="🎤 Record",
            width=90,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=BTN_PRIMARY,
            hover_color=BTN_PRIMARY_HOVER,
            command=self.start_mic_recording
        )
        self.record_btn.pack(side='left', padx=(0, 3))
        if not sp.SOUND_AVAILABLE:
            self.record_btn.configure(state="disabled")

        self.load_wav_btn = ctk.CTkButton(
            left_ops,
            text="📁 Open WAV",
            width=88,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11),
            fg_color=BTN_SECONDARY,
            hover_color=BTN_SECONDARY_HOVER,
            text_color=BTN_SECONDARY_TEXT,
            command=self.open_wav_file
        )
        self.load_wav_btn.pack(side='left', padx=(0, 3))

        self.preset_menu = ctk.CTkOptionMenu(
            left_ops,
            values=["Presets ▾", "From Signal Lab", "Test SOS (with noise)", "Test HELLO WORLD"],
            width=110,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11),
            command=self._on_preset_select
        )
        self.preset_menu.set("Presets ▾")
        self.preset_menu.pack(side='left', padx=(0, 5))

        self.demodulate_btn = ctk.CTkButton(
            left_ops,
            text="⚡ Demodulate",
            width=110,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=BTN_PLAY,
            hover_color=BTN_PLAY_HOVER,
            command=self.demodulate_and_decode
        )
        self.demodulate_btn.pack(side='left', padx=(0, 3))

        self.play_audio_btn = ctk.CTkButton(
            left_ops,
            text="▶ Play",
            width=60,
            height=BTN_HEIGHT_MD,
            fg_color=BTN_PLAY,
            hover_color=BTN_PLAY_HOVER,
            command=self.play_audio
        )
        self.play_audio_btn.pack(side='left', padx=(0, 3))

        self.stop_btn = ctk.CTkButton(
            left_ops,
            text="■ Stop",
            width=52,
            height=BTN_HEIGHT_MD,
            fg_color=BTN_STOP,
            hover_color=BTN_STOP_HOVER,
            command=self.stop_audio
        )
        self.stop_btn.pack(side='left', padx=(0, 4))

        self.route_menu = ctk.CTkOptionMenu(
            left_ops,
            values=["Route / Save ▾", "To Noise Lab →", "To Signal Lab →", "Send to Converter →", "💾 Save WAV..."],
            width=130,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11),
            command=self._on_route_select
        )
        self.route_menu.set("Route / Save ▾")
        self.route_menu.pack(side='left')

        # Status and progress display
        right_status = ctk.CTkFrame(bar, fg_color="transparent")
        right_status.pack(side='right')

        self.progress_bar = ctk.CTkProgressBar(right_status, width=80, height=6)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(side='right', padx=(3, 0))

        self.status_var = ctk.StringVar(value="Ready")
        self.status_lbl = ctk.CTkLabel(
            right_status,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.status_lbl.pack(side='right', padx=3)

    def _build_results_card(self):
        """Card displaying detected metrics (Carrier, WPM, SNR, Unit) and decoded outputs."""
        card = ctk.CTkFrame(self, corner_radius=8, fg_color=("gray92", "#14161f"))
        card.pack(fill='x', pady=(0, 4), padx=1)
        card.grid_columnconfigure((0, 1), weight=1)

        # Four metric tiles: carrier frequency, speed, SNR, dot unit duration
        metrics_strip = ctk.CTkFrame(card, fg_color="transparent")
        metrics_strip.grid(row=0, column=0, columnspan=2, sticky='ew', padx=8, pady=(4, 3))
        metrics_strip.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.metric_carrier = self._build_metric_tile(metrics_strip, "CARRIER TONE", "— Hz", 0)
        self.metric_wpm = self._build_metric_tile(metrics_strip, "SPEED (WPM)", "— WPM", 1)
        self.metric_snr = self._build_metric_tile(metrics_strip, "SIGNAL QUALITY", "— dB", 2)
        self.metric_unit = self._build_metric_tile(metrics_strip, "TIMING UNIT", "— ms", 3)

        # Decoded Morse and plain text boxes
        out_row = ctk.CTkFrame(card, fg_color="transparent")
        out_row.grid(row=1, column=0, columnspan=2, sticky='ew', padx=8, pady=(0, 6))
        out_row.grid_columnconfigure((0, 1), weight=1)

        col_left = ctk.CTkFrame(out_row, fg_color="transparent")
        col_left.grid(row=0, column=0, sticky='nsew', padx=(0, 4))
        m_hdr = ctk.CTkFrame(col_left, fg_color="transparent")
        m_hdr.pack(fill='x', pady=(0, 1))
        ctk.CTkLabel(m_hdr, text="Decoded Morse:", font=ctk.CTkFont(size=11, weight="bold")).pack(side='left')
        ctk.CTkButton(
            m_hdr, text="📋 Copy", width=55, height=20, font=ctk.CTkFont(size=10),
            fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER, text_color=BTN_SECONDARY_TEXT, corner_radius=5,
            command=lambda: self.app.copy_to_clipboard(self.morse_out.get('1.0', 'end').strip(), "Decoded Morse copied!")
        ).pack(side='right')
        self.morse_out = ctk.CTkTextbox(col_left, height=36, font=('Consolas', 11))
        self.morse_out.pack(fill='x')

        col_right = ctk.CTkFrame(out_row, fg_color="transparent")
        col_right.grid(row=0, column=1, sticky='nsew', padx=(4, 0))
        t_hdr = ctk.CTkFrame(col_right, fg_color="transparent")
        t_hdr.pack(fill='x', pady=(0, 1))
        ctk.CTkLabel(t_hdr, text="Decoded Plaintext:", font=ctk.CTkFont(size=11, weight="bold")).pack(side='left')
        ctk.CTkButton(
            t_hdr, text="📋 Copy", width=55, height=20, font=ctk.CTkFont(size=10),
            fg_color=BTN_SECONDARY, hover_color=BTN_SECONDARY_HOVER, text_color=BTN_SECONDARY_TEXT, corner_radius=5,
            command=lambda: self.app.copy_to_clipboard(self.text_out.get('1.0', 'end').strip(), "Decoded Plaintext copied!")
        ).pack(side='right')
        self.text_out = ctk.CTkTextbox(col_right, height=36, font=('Consolas', 12, 'bold'))
        self.text_out.pack(fill='x')

    def _build_metric_tile(self, parent: ctk.CTkFrame, title: str, default_val: str, col: int) -> ctk.CTkLabel:
        """Helper to create a single metric readout badge."""
        tile = ctk.CTkFrame(parent, fg_color="transparent")
        tile.grid(row=0, column=col, padx=4, pady=2, sticky='nsew')
        ctk.CTkLabel(tile, text=title, font=ctk.CTkFont(size=9, weight="bold"), text_color="gray").pack(anchor='w')
        val_lbl = ctk.CTkLabel(tile, text=default_val, font=ctk.CTkFont(size=12, weight="bold"))
        val_lbl.pack(anchor='w')
        return val_lbl

    def _build_plot_section(self):
        """Plot section for waveform, envelope with threshold, and FFT spectrum."""
        plot_ctrls = ctk.CTkFrame(self, fg_color="transparent")
        plot_ctrls.pack(fill='x', pady=(0, 2))

        self.plot_view_var = ctk.StringVar(value="Full Demodulation (Waveform + Envelope + Threshold)")
        ctk.CTkSegmentedButton(
            plot_ctrls,
            values=["Full Demodulation (Waveform + Envelope + Threshold)", "Waveform Only", "Frequency Spectrum (FFT)"],
            variable=self.plot_view_var,
            command=lambda _: self.redraw(),
            width=360,
            height=24,
            font=ctk.CTkFont(size=11)
        ).pack(side='right')

        self.figure = Figure(figsize=(6, 3.6), dpi=100)
        self.ax_top = self.figure.add_subplot(211)
        self.ax_bottom = self.figure.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, pady=(2, 2))
        self.redraw()

    # Dropdown Menu Callbacks

    def _on_preset_select(self, choice: str):
        """Load test audio presets for verification."""
        self.preset_menu.set("Presets ▾")
        if "Signal Lab" in choice:
            self.load_from_signal_lab()
        elif "Test SOS" in choice:
            self.run_simulation_test("SOS")
        elif "Test HELLO" in choice:
            self.run_simulation_test("HELLO WORLD")

    def _on_route_select(self, choice: str):
        """Route recorded audio to other tabs or export to WAV."""
        self.route_menu.set("Route / Save ▾")
        if "Noise Lab" in choice:
            self.send_to_noise_lab()
        elif "Signal Lab" in choice:
            self.send_to_signal_lab()
        elif "Converter" in choice:
            self.send_decoded_to_converter()
        elif "Save WAV" in choice:
            self.save_recording_wav()

    # Recording & Audio Capture

    def on_carrier_mode_change(self, mode: str):
        """Enable slider only when Manual Frequency mode is chosen."""
        if mode == "Manual Frequency":
            self.freq_slider.configure(state="normal")
        else:
            self.freq_slider.configure(state="disabled")

    def _get_selected_device_index(self) -> Optional[int]:
        """Extract device index from device dropdown text."""
        dev_str = self.dev_var.get()
        if ":" in dev_str:
            try:
                return int(dev_str.split(":")[0])
            except ValueError:
                return None
        return None

    def start_mic_recording(self):
        """Start recording audio from microphone in a background thread."""
        duration = self.duration_var.get()
        self.record_btn.configure(state="disabled")
        self.load_wav_btn.configure(state="disabled")
        self.play_audio_btn.configure(state="disabled")
        self.status_var.set(f"Recording... {duration:.0f}s left")
        self.progress_bar.set(0.0)

        device_idx = self._get_selected_device_index()
        thread = threading.Thread(target=self._record_worker, args=(duration, device_idx), daemon=True)
        thread.start()

    def _record_worker(self, duration: float, device_idx: Optional[int]):
        """Background thread updating recording countdown progress."""
        sr = 44100
        start_time = time.time()
        rec_thread = threading.Thread(target=self._rec_inner, args=(duration, sr, device_idx), daemon=True)
        rec_thread.start()

        while rec_thread.is_alive():
            elapsed = time.time() - start_time
            frac = min(elapsed / duration, 1.0)
            rem = max(0.0, duration - elapsed)
            self.app.run_on_ui_thread(lambda f=frac, r=rem: self._update_progress(f, r))
            time.sleep(0.05)

        rec_thread.join()

    def _rec_inner(self, duration: float, sr: int, device_idx: Optional[int]):
        """Capture raw audio samples from sounddevice."""
        try:
            audio = sp.record_audio(duration, sample_rate=sr, device=device_idx)
            self.app.run_on_ui_thread(lambda: self._on_record_finished(audio, sr))
        except Exception as e:
            self.app.run_on_ui_thread(lambda: self._on_record_error(str(e)))

    def _update_progress(self, frac: float, remaining: float):
        """Update progress bar and remaining countdown text."""
        self.progress_bar.set(frac)
        self.status_var.set(f"Recording... {remaining:.1f}s left")

    def _on_record_error(self, err_msg: str):
        """Handle microphone recording errors."""
        self.record_btn.configure(state="normal")
        self.load_wav_btn.configure(state="normal")
        self.play_audio_btn.configure(state="normal")
        self.progress_bar.set(0.0)
        self.status_var.set("Recording error.")
        messagebox.showerror("Recording Error", err_msg)

    def _on_record_finished(self, audio: np.ndarray, sr: int):
        """Finish recording and immediately begin demodulation."""
        self.record_btn.configure(state="normal")
        self.load_wav_btn.configure(state="normal")
        self.play_audio_btn.configure(state="normal")
        self.progress_bar.set(1.0)
        self.load_audio_data(audio, sr, source_name=f"Microphone Capture ({len(audio)/sr:.1f}s)")

    def open_wav_file(self):
        """Open a WAV audio file from disk."""
        path = filedialog.askopenfilename(
            filetypes=[("Audio Files (*.wav)", "*.wav"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            audio, sr = sp.load_wav(path)
            fname = os.path.basename(path)
            self.load_audio_data(audio, sr, source_name=f"File: {fname} ({len(audio)/sr:.1f}s)")
        except Exception as e:
            messagebox.showerror("File Load Error", f"Could not read audio file:\n{e}")

    def load_from_signal_lab(self):
        """Import the synthesized signal from Signal Lab."""
        sig = self.app.current_signal
        sr = self.app.current_sample_rate
        if len(sig) == 0:
            self.app.signal_tab.generate()
            sig = self.app.current_signal
            sr = self.app.current_sample_rate
        self.load_audio_data(sig, sr, source_name="Signal Lab (Direct Signal)")

    def run_simulation_test(self, text: str):
        """Create simulated microphone audio with silence and room noise for testing."""
        morse = mc.text_to_morse(text)
        sr = 44100
        sig = sp.morse_to_signal(morse, freq=750, wpm=18, sample_rate=sr, amplitude=0.7)
        lead = np.zeros(int(sr * 0.8))
        trail = np.zeros(int(sr * 0.8))
        audio = np.concatenate([lead, sig, trail])

        t = np.arange(len(audio)) / float(sr)
        noisy = audio + 0.04 * np.sin(2 * np.pi * 60 * t) + 0.02 * np.random.randn(len(audio)) + 0.015
        noisy[2000:2300] += 0.4
        self.load_audio_data(noisy, sr, source_name=f"Simulation: '{text}'")

    def load_audio_data(self, audio: np.ndarray, sr: int, source_name: str = "Audio"):
        """Store audio samples and trigger demodulation pipeline."""
        self.app.recorded_signal = audio
        self.app.recorded_sample_rate = sr
        self.source_var.set(f"Source: {source_name}")
        self.status_var.set("Demodulating audio...")
        self.demodulate_and_decode()

    def _set_decoding(self, busy: bool):
        """Disable buttons during audio demodulation."""
        self._is_decoding = busy
        state = "disabled" if busy else "normal"
        if hasattr(self, 'demodulate_btn'):
            self.demodulate_btn.configure(state=state)
        if hasattr(self, 'record_btn') and sp.SOUND_AVAILABLE:
            self.record_btn.configure(state=state)
        if hasattr(self, 'load_wav_btn'):
            self.load_wav_btn.configure(state=state)
        if hasattr(self, 'play_audio_btn'):
            self.play_audio_btn.configure(state=state)

    def demodulate_and_decode(self):
        """Demodulate audio: detect carrier, compute envelope, and decode Morse timing."""
        if getattr(self, '_is_decoding', False):
            return

        audio = self.app.recorded_signal
        sr = self.app.recorded_sample_rate or 44100
        if len(audio) == 0:
            self.app.show_status("No audio to decode — record or load audio first", level="warning")
            return

        carrier_mode = self.carrier_mode_var.get()
        carrier_freq = self.carrier_freq_var.get() if carrier_mode == "Manual Frequency" else None
        bw = self.bw_var.get()

        self._set_decoding(True)
        self.app.set_busy(True, "Demodulating and decoding audio...")
        self.status_var.set("Demodulating audio in background...")

        def _worker():
            try:
                morse, details = sp.decode_signal_to_morse(
                    audio,
                    sample_rate=sr,
                    carrier_freq=carrier_freq,
                    bandwidth=bw,
                    return_details=True
                )
                self.app.run_on_ui_thread(lambda: self._on_decode_complete(audio, sr, morse, details))
            except Exception as e:
                self.app.run_on_ui_thread(lambda: self._on_decode_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_decode_complete(self, audio: np.ndarray, sr: int, morse: str, details: dict):
        """Update metrics and text boxes with decoded Morse and plaintext."""
        self.app.recorded_details = details

        cf = details.get('carrier_freq', 700.0)
        wpm = details.get('wpm', 0.0)
        snr = details.get('snr_est', 0.0)
        unit = details.get('unit_duration', 0.0)

        self.carrier_freq_var.set(cf)
        if hasattr(self, 'freq_slider'):
            self.freq_slider.set(cf)
        self.freq_slider_lbl.configure(text=f"{cf:.0f} Hz")

        self.metric_carrier.configure(text=f"{cf:.1f} Hz")
        self.metric_wpm.configure(text=f"{wpm:.1f} WPM" if wpm > 0 else "— WPM")

        snr_quality = "High" if snr > 15 else ("Good" if snr > 8 else ("Fair" if snr > 2 else "Low"))
        self.metric_snr.configure(text=f"{snr:.1f} dB ({snr_quality})")
        self.metric_unit.configure(text=f"{unit*1000.0:.0f} ms" if unit > 0 else "— ms")

        text = mc.morse_to_text(morse) if morse else ""
        self.morse_out.delete('1.0', 'end')
        self.morse_out.insert('1.0', morse if morse else "(no recognizable Morse tone detected)")

        self.text_out.delete('1.0', 'end')
        self.text_out.insert('1.0', text if text else "(none)")

        dur_s = len(audio) / sr if sr else 0.0
        self.status_var.set(f"Decoded {dur_s:.2f}s • {cf:.0f} Hz, {wpm:.1f} WPM")
        self.redraw()
        self._set_decoding(False)
        self.app.set_busy(False)
        self.app.show_status(
            f"Decoded {dur_s:.1f}s audio: '{text}' ({cf:.0f} Hz, {wpm:.1f} WPM)" if text else f"Decoded {dur_s:.1f}s audio: (no Morse tone)",
            level="success" if text else "info",
            duration_ms=4500
        )

    def _on_decode_error(self, err_msg: str):
        """Handle demodulation failure."""
        self._set_decoding(False)
        self.app.set_busy(False)
        self.status_var.set("Demodulation error.")
        self.app.show_status(f"Demodulation error: {err_msg}", level="error")

    def play_audio(self):
        """Play current recorded or loaded audio."""
        audio = self.app.recorded_signal
        sr = self.app.recorded_sample_rate or 44100
        if len(audio) == 0:
            self.app.show_status("Nothing to play — record or load audio first", level="warning")
            return
        try:
            sp.play_signal(audio, sr)
            self._is_playing = True
            dur = len(audio) / float(sr)
            self.app.show_status(f"Playing recorded audio ({dur:.2f}s)...", level="info")
            self.after(int(dur * 1000) + 100, lambda: setattr(self, '_is_playing', False))
        except RuntimeError as e:
            self.app.show_status(f"Playback error: {e}", level="error")

    def stop_audio(self):
        """Stop audio playback."""
        sp.stop_playback()
        self._is_playing = False
        self.app.show_status("Audio playback stopped", level="info")

    def save_recording_wav(self):
        """Save recorded audio to a WAV file."""
        audio = self.app.recorded_signal
        sr = self.app.recorded_sample_rate or 44100
        if len(audio) == 0:
            self.app.show_status("Nothing to save — record or load audio first", level="warning")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV Audio", "*.wav")],
            initialfile="mic_recorded_audio.wav"
        )
        if not path:
            return
        try:
            sp.save_wav(path, audio, sr)
            fname = os.path.basename(path)
            self.app.show_status(f"Recorded audio saved to: {fname}", level="success", duration_ms=4500)
        except Exception as e:
            self.app.show_status(f"Save error: {e}", level="error")

    def send_to_noise_lab(self):
        """Route mic audio to Noise & Filter Lab (Tab 3)."""
        audio = self.app.recorded_signal
        sr = self.app.recorded_sample_rate or 44100
        if len(audio) == 0:
            self.app.show_status("No audio to send — record or load audio first", level="warning")
            return
        self.app.noise_tab.load_external_noisy_audio(audio, sr, source_name="Mic Recording")
        self.app.tabview.set("3. Noise & Filter")
        self.app.show_status("Mic recording sent to Noise Lab for filtering", level="success")

    def send_to_signal_lab(self):
        """Route mic audio to Signal Lab (Tab 2)."""
        audio = self.app.recorded_signal
        sr = self.app.recorded_sample_rate or 44100
        if len(audio) == 0:
            self.app.show_status("No audio to send — record or load audio first", level="warning")
            return
        self.app.signal_tab.import_from_mic()
        self.app.tabview.set("2. Signal Lab")
        self.app.show_status("Mic recording sent to Signal Lab", level="success")

    def send_decoded_to_converter(self):
        """Send decoded text to Converter (Tab 1)."""
        decoded_text = self.text_out.get('1.0', 'end').strip()
        decoded_morse = self.morse_out.get('1.0', 'end').strip()
        if decoded_text and decoded_text not in ("(none)", "(no recognizable Morse tone detected)"):
            self.app.converter_tab.load_text_or_morse(decoded_text, is_morse=False)
            self.app.tabview.set("1. Converter")
            self.app.show_status(f"Decoded text '{decoded_text}' sent to Converter", level="success")
        elif decoded_morse and decoded_morse != "(no recognizable Morse tone detected)":
            self.app.converter_tab.load_text_or_morse(decoded_morse, is_morse=True)
            self.app.tabview.set("1. Converter")
            self.app.show_status("Decoded Morse sent to Converter", level="success")
        else:
            self.app.show_status("No decoded output to send — demodulate audio first", level="warning")

    def redraw(self):
        """Draw waveform, envelope detection, or frequency spectrum plots."""
        colors = get_plot_theme()
        self.figure.patch.set_facecolor(colors['fig_bg'])
        self.ax_top.clear()
        self.ax_bottom.clear()

        audio = self.app.recorded_signal
        sr = self.app.recorded_sample_rate or 44100
        details = self.app.recorded_details

        if len(audio) == 0:
            self.ax_bottom.set_visible(False)
            self.ax_top.set_position([0.08, 0.15, 0.88, 0.75])
            style_axis(self.ax_top, colors, title="Captured Audio Waveform (Click 'Record Mic' or 'Open WAV')")
            self.canvas.draw()
            return

        view = self.plot_view_var.get()

        if "Full Demodulation" in view:
            # Dual view: raw waveform + filtered carrier on top; amplitude envelope + threshold on bottom
            self.ax_bottom.set_visible(True)
            self.ax_top.set_position([0.08, 0.55, 0.88, 0.38])
            self.ax_bottom.set_position([0.08, 0.12, 0.88, 0.35])

            step = max(1, len(audio) // 20000)
            t = np.arange(0, len(audio), step) / sr

            self.ax_top.plot(t, audio[::step], linewidth=0.5, color='gray', alpha=0.5, label='Raw Audio')
            filtered = details.get('filtered', None)
            if filtered is not None and len(filtered) == len(audio):
                self.ax_top.plot(t, filtered[::step], linewidth=0.7, color=colors['filtered'], label='Filtered Carrier')
            self.ax_top.legend(loc='upper right', fontsize=8)
            cf = details.get('carrier_freq', 700)
            style_axis(self.ax_top, colors, title=f"1. Captured Waveform & Filtered Carrier ({cf:.0f} Hz)", ylabel="Amp")

            env = details.get('envelope', None)
            thresh = details.get('threshold', 0)
            if env is not None and len(env) == len(audio):
                self.ax_bottom.plot(t, env[::step], linewidth=0.9, color=colors['env'], label='Envelope')
                if thresh > 0:
                    self.ax_bottom.axhline(thresh, color=colors['thresh'], linestyle='--', linewidth=1.2, label=f'Threshold ({thresh:.3f})')
                    self.ax_bottom.fill_between(
                        t, 0, env[::step], where=(env[::step] > thresh),
                        color=colors['filtered'], alpha=0.25, label='Tone ON'
                    )
                self.ax_bottom.legend(loc='upper right', fontsize=8)
            style_axis(self.ax_bottom, colors, title="2. Envelope & Adaptive Detection Threshold", xlabel="Time (seconds)", ylabel="Envelope")

        elif "Waveform Only" in view:
            # Single waveform view
            self.ax_bottom.set_visible(False)
            self.ax_top.set_position([0.08, 0.15, 0.88, 0.75])
            step = max(1, len(audio) // 25000)
            t = np.arange(0, len(audio), step) / sr
            self.ax_top.plot(t, audio[::step], linewidth=0.8, color=colors['mic'])
            style_axis(self.ax_top, colors, title="Microphone Captured Audio Waveform", xlabel="Time (seconds)", ylabel="Amplitude")

        else:
            # Frequency spectrum (FFT) view
            self.ax_bottom.set_visible(False)
            self.ax_top.set_position([0.08, 0.15, 0.88, 0.75])
            freqs, mag = sp.compute_fft(audio, sr)
            self.ax_top.plot(freqs, mag, linewidth=0.9, color=colors['fft'])
            cf = details.get('carrier_freq', 0)
            if cf > 0:
                self.ax_top.axvline(cf, color=colors['clean'], linestyle=':', linewidth=1.5, label=f'Carrier ({cf:.1f} Hz)')
                self.ax_top.legend(loc='upper right', fontsize=8)
            self.ax_top.set_xlim(0, min(3000, sr / 2.0))
            style_axis(self.ax_top, colors, title="Frequency Spectrum (FFT) of Captured Audio", xlabel="Frequency (Hz)", ylabel="Magnitude")

        self.canvas.draw()
