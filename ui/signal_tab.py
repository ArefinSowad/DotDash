"""
Tab 2: Audio Synthesis & Frequency Analysis.
Generates sine wave Morse audio, plots waveforms, computes FFT spectrum
and spectrograms, and exports audio to WAV.
"""

import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from typing import TYPE_CHECKING
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
    BTN_HEIGHT_MD, BTN_HEIGHT_SM
)

if TYPE_CHECKING:
    from app import MorseSignalLab

SAMPLE_RATE_OPTIONS = ["8000", "16000", "22050", "44100"]


class SignalLabTab(ctk.CTkFrame):
    """Tab for synthesizing Morse audio and visualizing waveforms and frequencies."""

    def __init__(self, parent: ctk.CTkBaseClass, app: "MorseSignalLab"):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.cbar = None
        self._is_generating = False
        self._is_playing = False
        self.pack(fill='both', expand=True, padx=6, pady=4)

        # Build UI layout
        self._build_parameters_card()
        self._build_action_toolbar()
        self._build_plot_canvas()

        # Generate default signal on open
        self.generate()

    # UI Layout Construction

    def _build_parameters_card(self):
        """Card containing Morse code input and synthesis tuning sliders."""
        card = ctk.CTkFrame(self, corner_radius=8)
        card.pack(fill='x', pady=(0, 6), padx=1)
        card.grid_columnconfigure((0, 1), weight=1)

        # Morse code text input and quick presets
        top_row = ctk.CTkFrame(card, fg_color="transparent")
        top_row.grid(row=0, column=0, columnspan=2, sticky='ew', padx=10, pady=(6, 4))

        ctk.CTkLabel(top_row, text="Morse Code Input:", font=ctk.CTkFont(size=12, weight="bold")).pack(side='left', padx=(0, 6))

        self.morse_var = ctk.StringVar(value=mc.text_to_morse("SOS"))
        self.morse_entry = ctk.CTkEntry(top_row, textvariable=self.morse_var, font=('Consolas', 12), height=26)
        self.morse_entry.pack(side='left', fill='x', expand=True, padx=(0, 8))

        for p in ["SOS", "HELLO", "CQ CQ"]:
            ctk.CTkButton(
                top_row,
                text=p,
                width=55,
                height=22,
                font=ctk.CTkFont(size=10, weight="bold"),
                fg_color=BTN_SECONDARY,
                hover_color=BTN_SECONDARY_HOVER,
                text_color=BTN_SECONDARY_TEXT,
                command=lambda text=p: self.set_morse_input(mc.text_to_morse(text))
            ).pack(side='left', padx=2)

        # Audio parameters: pitch, speed, volume, and sample rate
        left_col = ctk.CTkFrame(card, fg_color="transparent")
        left_col.grid(row=1, column=0, sticky='ew', padx=(10, 6), pady=(0, 6))
        left_col.grid_columnconfigure(1, weight=1)

        right_col = ctk.CTkFrame(card, fg_color="transparent")
        right_col.grid(row=1, column=1, sticky='ew', padx=(6, 10), pady=(0, 6))
        right_col.grid_columnconfigure(1, weight=1)

        self.freq_var = tk.DoubleVar(value=700)
        self.wpm_var = tk.DoubleVar(value=20)
        self.vol_var = tk.DoubleVar(value=70)
        self.sr_var = ctk.StringVar(value="44100")

        self._add_slider(left_col, "Carrier (Hz):", self.freq_var, 200, 1500, 0, unit=" Hz")
        self._add_slider(left_col, "Speed (WPM):", self.wpm_var, 5, 40, 1, unit=" WPM")

        self._add_slider(right_col, "Volume (%):", self.vol_var, 0, 100, 0, unit=" %")

        sr_row = ctk.CTkFrame(right_col, fg_color="transparent")
        sr_row.grid(row=1, column=0, columnspan=3, sticky='ew', pady=1)
        ctk.CTkLabel(sr_row, text="Sample Rate:", font=ctk.CTkFont(size=11)).pack(side='left', padx=(0, 8))
        self.sr_menu = ctk.CTkOptionMenu(
            sr_row, variable=self.sr_var, values=SAMPLE_RATE_OPTIONS, width=100, height=22, font=ctk.CTkFont(size=11)
        )
        self.sr_menu.pack(side='left')

    def _add_slider(self, parent, label, var, from_, to, row, unit=""):
        """Helper to create a slider with a dynamic label showing current value."""
        ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=11)).grid(row=row, column=0, sticky='w', pady=1)
        val_lbl = ctk.CTkLabel(parent, text=f"{var.get():.0f}{unit}", width=50, font=ctk.CTkFont(size=11, weight="bold"))
        val_lbl.grid(row=row, column=2, sticky='e', pady=1)

        def on_slide(val):
            val_lbl.configure(text=f"{val:.0f}{unit}")

        slider = ctk.CTkSlider(parent, from_=from_, to=to, variable=var, command=on_slide, height=14)
        slider.grid(row=row, column=1, sticky='ew', padx=6, pady=1)

    def _build_action_toolbar(self):
        """Action bar with generate, playback, file export, routing, and plot views."""
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill='x', pady=(0, 4))

        left_ops = ctk.CTkFrame(bar, fg_color="transparent")
        left_ops.pack(side='left')

        self.generate_btn = ctk.CTkButton(
            left_ops,
            text="⚡ Generate",
            width=105,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=BTN_PRIMARY,
            hover_color=BTN_PRIMARY_HOVER,
            command=self.generate
        )
        self.generate_btn.pack(side='left', padx=(0, 4))

        self.play_btn = ctk.CTkButton(
            left_ops,
            text="▶ Play",
            width=65,
            height=BTN_HEIGHT_MD,
            fg_color=BTN_PLAY,
            hover_color=BTN_PLAY_HOVER,
            command=self.play
        )
        self.play_btn.pack(side='left', padx=(0, 4))

        self.stop_btn = ctk.CTkButton(
            left_ops,
            text="■ Stop",
            width=55,
            height=BTN_HEIGHT_MD,
            fg_color=BTN_STOP,
            hover_color=BTN_STOP_HOVER,
            command=self.stop
        )
        self.stop_btn.pack(side='left', padx=(0, 4))

        self.export_btn = ctk.CTkButton(
            left_ops,
            text="💾 Save WAV...",
            width=105,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11),
            fg_color=BTN_SECONDARY,
            hover_color=BTN_SECONDARY_HOVER,
            text_color=BTN_SECONDARY_TEXT,
            command=self.export_wav
        )
        self.export_btn.pack(side='left', padx=(0, 8))

        self.route_menu = ctk.CTkOptionMenu(
            left_ops,
            values=["Route: Select ▾", "To Noise Lab →", "To Mic Decoder →", "📥 Import from Mic", "📥 Import from Noise Lab"],
            width=150,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11),
            command=self._on_route_select
        )
        self.route_menu.set("Route: Select ▾")
        self.route_menu.pack(side='left')

        # Toggle between Waveform, FFT Spectrum, and Spectrogram
        self.plot_mode = ctk.StringVar(value="Waveform")
        self.plot_seg = ctk.CTkSegmentedButton(
            bar,
            values=["Waveform", "FFT Spectrum", "Spectrogram"],
            variable=self.plot_mode,
            command=lambda _: self.redraw(),
            width=270,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11)
        )
        self.plot_seg.pack(side='right')

    def _build_plot_canvas(self):
        """Create embedded Matplotlib figure canvas."""
        self.figure = Figure(figsize=(6, 3.8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, pady=(2, 2))

    # Event Handlers & Synthesis Operations

    def _on_route_select(self, choice: str):
        """Handle routing signal to or from other tabs."""
        self.route_menu.set("Route: Select ▾")
        if "To Noise Lab" in choice:
            self.send_to_noise_lab()
        elif "To Mic Decoder" in choice:
            self.send_to_mic_decode()
        elif "Import from Mic" in choice:
            self.import_from_mic()
        elif "Import from Noise" in choice:
            self.import_from_noise_lab()

    def set_morse_input(self, morse_str: str):
        """Update Morse code input box and regenerate audio."""
        self.morse_var.set(morse_str)
        self.generate()

    def generate(self):
        """Synthesize sine wave audio in a background thread."""
        if self._is_generating:
            return

        morse = self.morse_var.get().strip()
        if not morse:
            self.app.show_status("Enter a Morse string first", level="warning")
            return

        freq = self.freq_var.get()
        wpm = self.wpm_var.get()
        amp = self.vol_var.get() / 100.0
        sr = int(self.sr_var.get())

        self._set_generating(True)
        self.app.set_busy(True, f"Synthesizing {wpm:.0f} WPM audio @ {freq:.0f} Hz...")

        def _worker():
            try:
                sig = sp.morse_to_signal(morse, freq=freq, wpm=wpm, sample_rate=sr, amplitude=amp)
                self.app.run_on_ui_thread(lambda: self._on_generation_complete(sig, sr, morse))
            except Exception as e:
                self.app.run_on_ui_thread(lambda: self._on_generation_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _set_generating(self, busy: bool):
        """Enable or disable generate and play buttons while synthesizing."""
        self._is_generating = busy
        state = "disabled" if busy else "normal"
        self.generate_btn.configure(state=state)
        self.play_btn.configure(state=state)

    def _on_generation_complete(self, sig: np.ndarray, sr: int, morse: str):
        """Store newly generated signal and refresh plots."""
        self.app.current_signal = sig
        self.app.current_sample_rate = sr
        self.app.current_morse = morse
        self.redraw()
        self._set_generating(False)
        self.app.set_busy(False)
        dur = len(sig) / float(sr)
        self.app.show_status(f"Signal generated: {len(sig):,} samples ({dur:.2f}s) @ {sr} Hz", level="success")

    def _on_generation_error(self, err_msg: str):
        """Handle signal generation failure."""
        self._set_generating(False)
        self.app.set_busy(False)
        self.app.show_status(f"Synthesis error: {err_msg}", level="error")

    def redraw(self):
        """Draw waveform, FFT frequency spectrum, or spectrogram according to current view."""
        colors = get_plot_theme()
        self.figure.patch.set_facecolor(colors['fig_bg'])
        self.ax.clear()
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None

        sig = self.app.current_signal
        sr = self.app.current_sample_rate
        mode = self.plot_mode.get()

        if len(sig) == 0:
            style_axis(self.ax, colors, title="No Signal Generated", xlabel="Time (s)", ylabel="Amp")
            self.canvas.draw()
            return

        if mode == "Waveform":
            # Downsample if signal is very long for snappy plotting
            step = max(1, len(sig) // 20000)
            t = np.arange(0, len(sig), step) / sr
            self.ax.plot(t, sig[::step], linewidth=0.7, color=colors['clean'])
            style_axis(
                self.ax, colors,
                title=f"Synthesized Waveform ({len(sig)/sr:.2f}s @ {self.freq_var.get():.0f} Hz)",
                xlabel="Time (seconds)",
                ylabel="Amplitude"
            )

        elif mode == "FFT Spectrum":
            # Frequency spectrum showing carrier tone peak
            freqs, mag = sp.compute_fft(sig, sr)
            self.ax.plot(freqs, mag, linewidth=0.9, color=colors['fft'], label='Spectrum')
            peak = self.freq_var.get()
            self.ax.axvline(peak, color=colors['thresh'], linestyle='--', linewidth=1.2, label=f"Carrier: {peak:.0f} Hz")
            self.ax.set_xlim(0, min(3000, sr / 2.0))
            self.ax.legend(loc='upper right', fontsize=8)
            style_axis(
                self.ax, colors,
                title="Frequency Spectrum (FFT)",
                xlabel="Frequency (Hz)",
                ylabel="Magnitude"
            )

        else:
            # Time-frequency spectrogram
            f, t, sxx = sp.compute_spectrogram(sig, sr)
            if len(f) > 0 and len(t) > 0:
                mesh = self.ax.pcolormesh(t, f, sxx, shading='gouraud', cmap='viridis')
                self.ax.set_ylim(0, min(3000, sr / 2.0))
                style_axis(
                    self.ax, colors,
                    title="Time-Frequency Spectrogram (dB)",
                    xlabel="Time (seconds)",
                    ylabel="Frequency (Hz)"
                )
                self.cbar = self.figure.colorbar(mesh, ax=self.ax, pad=0.02, aspect=18)
                self.cbar.set_label("Power (dB)", color=colors['text'], fontsize=8)
                self.cbar.ax.tick_params(colors=colors['text'], labelsize=7)
                for spine in self.cbar.ax.spines.values():
                    spine.set_color(colors['spine'])

        self.figure.tight_layout()
        self.canvas.draw()

    def play(self):
        """Play synthesized audio."""
        sig = self.app.current_signal
        if len(sig) == 0:
            self.app.show_status("Nothing to play — generate a signal first", level="warning")
            return
        try:
            sp.play_signal(sig, self.app.current_sample_rate)
            self._is_playing = True
            dur = len(sig) / float(self.app.current_sample_rate)
            self.app.show_status(f"Playing Morse audio ({dur:.2f}s)...", level="info")
            self.after(int(dur * 1000) + 100, lambda: setattr(self, '_is_playing', False))
        except RuntimeError as e:
            self.app.show_status(f"Playback error: {e}", level="error")

    def stop(self):
        """Stop current audio playback."""
        sp.stop_playback()
        self._is_playing = False
        self.app.show_status("Audio playback stopped", level="info")

    def export_wav(self):
        """Save synthesized audio waveform to a standard WAV file."""
        if len(self.app.current_signal) == 0:
            self.app.show_status("Nothing to export — generate a signal first", level="warning")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV Audio", "*.wav")],
            initialfile="synthesized_morse.wav"
        )
        if not path:
            return
        try:
            sp.save_wav(path, self.app.current_signal, self.app.current_sample_rate)
            self.app.show_status(f"WAV saved successfully: {Path(path).name}", level="success", duration_ms=4500)
        except Exception as e:
            self.app.show_status(f"Failed to save WAV: {e}", level="error")

    def send_to_noise_lab(self):
        """Send clean synthesized audio to Noise & Filter Lab (Tab 3)."""
        if len(self.app.current_signal) == 0:
            self.generate()
        self.app.noise_tab.load_clean_signal()
        self.app.tabview.set("3. Noise & Filter")
        self.app.show_status("Signal routed to Noise Lab", level="info")

    def send_to_mic_decode(self):
        """Send audio to Mic & Audio Decoder (Tab 4)."""
        if len(self.app.current_signal) == 0:
            self.generate()
        self.app.mic_tab.load_audio_data(
            self.app.current_signal,
            self.app.current_sample_rate,
            source_name="Signal Lab (Direct Synthesis)"
        )
        self.app.tabview.set("4. Mic & Audio Decode")
        self.app.show_status("Signal routed to Audio Decoder", level="info")

    def import_from_mic(self):
        """Load recorded audio from Mic Decoder into Signal Lab."""
        audio = self.app.recorded_signal
        sr = self.app.recorded_sample_rate or 44100
        if len(audio) == 0:
            self.app.show_status("No mic audio available — record in Mic tab first", level="warning")
            return
        self.app.current_signal = audio
        self.app.current_sample_rate = sr
        detected = sp.detect_carrier_frequency(audio, sr)
        self.freq_var.set(detected)
        self.redraw()
        self.app.show_status(f"Loaded mic audio ({len(audio):,} samples, carrier detected: {detected:.0f} Hz)", level="success")

    def import_from_noise_lab(self):
        """Load filtered or noisy signal from Noise Lab into Signal Lab."""
        sig = self.app.filtered_signal if len(self.app.filtered_signal) > 0 else self.app.noisy_signal
        if len(sig) == 0:
            self.app.show_status("No signal in Noise Lab — add noise or filter first", level="warning")
            return
        self.app.current_signal = sig
        self.redraw()
        self.app.show_status("Loaded audio from Noise Lab", level="success")
