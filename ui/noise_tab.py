"""
Tab 3: Noise simulation and Butterworth band-pass filter lab.
Injects noise (white noise, 60Hz hum, clicks) and filters it to recover clean Morse tones.
"""

import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog
from typing import TYPE_CHECKING, Dict, Optional
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
    BTN_WARN, BTN_WARN_HOVER,
    BTN_HEIGHT_MD, BTN_HEIGHT_SM
)

if TYPE_CHECKING:
    from app import MorseSignalLab


class NoiseFilterTab(ctk.CTkFrame):
    """Tab for experimenting with noise corruption and Butterworth filtering."""

    def __init__(self, parent: ctk.CTkBaseClass, app: "MorseSignalLab"):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self._is_filtering = False
        self._is_playing = False
        self._clean_ref: np.ndarray = np.array([])
        self._noise_counts: Dict[str, int] = {"awgn": 0, "hum": 0, "clicks": 0}
        self.pack(fill='both', expand=True, padx=4, pady=2)

        # Build UI layout
        self._build_control_card()
        self._build_action_bar()
        self._build_plot_section()

    # UI Layout Construction

    def _build_control_card(self):
        """Card containing noise injection controls and filter parameters."""
        card = ctk.CTkFrame(self, corner_radius=8)
        card.pack(fill='x', pady=(0, 4), padx=1)
        card.grid_columnconfigure((0, 1), weight=1)

        # Left Column: Noise injection (white noise, 60Hz hum, pops/clicks)
        left_box = ctk.CTkFrame(card, fg_color="transparent")
        left_box.grid(row=0, column=0, sticky='nsew', padx=(8, 6), pady=4)

        noise_hdr = ctk.CTkFrame(left_box, fg_color="transparent")
        noise_hdr.pack(fill='x', pady=(0, 3))

        ctk.CTkLabel(
            noise_hdr,
            text="Channel Impairments:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(side='left')

        self.noise_badge = ctk.CTkLabel(
            noise_hdr,
            text="● Clean",
            font=ctk.CTkFont(size=10),
            text_color="#10b981"
        )
        self.noise_badge.pack(side='right')

        chips_row = ctk.CTkFrame(left_box, fg_color="transparent")
        chips_row.pack(fill='x', pady=(0, 3))

        self.add_awgn_btn = ctk.CTkButton(
            chips_row,
            text="+ AWGN",
            width=72,
            height=BTN_HEIGHT_SM + 2,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=("#475569", "#334155"),
            hover_color=("#334155", "#1e293b"),
            command=self.add_awgn
        )
        self.add_awgn_btn.pack(side='left', padx=(0, 3))

        self.add_hum_btn = ctk.CTkButton(
            chips_row,
            text="+ 60Hz Hum",
            width=88,
            height=BTN_HEIGHT_SM + 2,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=BTN_WARN,
            hover_color=BTN_WARN_HOVER,
            command=self.add_hum
        )
        self.add_hum_btn.pack(side='left', padx=(0, 3))

        self.add_clicks_btn = ctk.CTkButton(
            chips_row,
            text="+ Clicks",
            width=72,
            height=BTN_HEIGHT_SM + 2,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=("#dc2626", "#b91c1c"),
            hover_color=("#b91c1c", "#991b1b"),
            command=self.add_clicks
        )
        self.add_clicks_btn.pack(side='left', padx=(0, 4))

        self.reset_noise_btn = ctk.CTkButton(
            chips_row,
            text="↺ Clear",
            width=60,
            height=BTN_HEIGHT_SM + 2,
            font=ctk.CTkFont(size=10),
            fg_color=BTN_SECONDARY,
            hover_color=BTN_SECONDARY_HOVER,
            text_color=BTN_SECONDARY_TEXT,
            command=self.reset_noise
        )
        self.reset_noise_btn.pack(side='left')

        # Sliders for noise intensity
        noise_grid = ctk.CTkFrame(left_box, fg_color="transparent")
        noise_grid.pack(fill='x')
        noise_grid.grid_columnconfigure(1, weight=1)

        self.snr_var = tk.DoubleVar(value=8)
        ctk.CTkLabel(noise_grid, text="AWGN Target SNR:", font=ctk.CTkFont(size=11)).grid(row=0, column=0, sticky='w', pady=1)
        self.snr_val_lbl = ctk.CTkLabel(noise_grid, text="8 dB", width=45, font=ctk.CTkFont(size=11, weight="bold"))
        self.snr_val_lbl.grid(row=0, column=2, sticky='e', pady=1)
        self.snr_slider = ctk.CTkSlider(
            noise_grid, from_=-10, to=30, variable=self.snr_var,
            command=lambda v: self.snr_val_lbl.configure(text=f"{v:.0f} dB"), height=14
        )
        self.snr_slider.grid(row=0, column=1, sticky='ew', padx=6, pady=1)

        self.hum_amp_var = tk.DoubleVar(value=0.35)
        ctk.CTkLabel(noise_grid, text="60Hz Hum Level:", font=ctk.CTkFont(size=11)).grid(row=1, column=0, sticky='w', pady=1)
        self.hum_val_lbl = ctk.CTkLabel(noise_grid, text="0.35", width=45, font=ctk.CTkFont(size=11, weight="bold"))
        self.hum_val_lbl.grid(row=1, column=2, sticky='e', pady=1)
        self.hum_slider = ctk.CTkSlider(
            noise_grid, from_=0.05, to=0.80, variable=self.hum_amp_var,
            command=lambda v: self.hum_val_lbl.configure(text=f"{v:.2f}"), height=14
        )
        self.hum_slider.grid(row=1, column=1, sticky='ew', padx=6, pady=1)

        self.clicks_count_var = tk.IntVar(value=5)
        ctk.CTkLabel(noise_grid, text="Click Spikes:", font=ctk.CTkFont(size=11)).grid(row=2, column=0, sticky='w', pady=1)
        self.clicks_val_lbl = ctk.CTkLabel(noise_grid, text="5 pops", width=45, font=ctk.CTkFont(size=11, weight="bold"))
        self.clicks_val_lbl.grid(row=2, column=2, sticky='e', pady=1)
        self.clicks_slider = ctk.CTkSlider(
            noise_grid, from_=1, to=15, number_of_steps=14, variable=self.clicks_count_var,
            command=lambda v: self.clicks_val_lbl.configure(text=f"{int(v)} pops"), height=14
        )
        self.clicks_slider.grid(row=2, column=1, sticky='ew', padx=6, pady=1)

        # Right Column: Butterworth band-pass filter settings
        right_box = ctk.CTkFrame(card, fg_color="transparent")
        right_box.grid(row=0, column=1, sticky='nsew', padx=(6, 8), pady=4)
        right_box.grid_columnconfigure(1, weight=1)

        filter_hdr = ctk.CTkFrame(right_box, fg_color="transparent")
        filter_hdr.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 3))

        ctk.CTkLabel(
            filter_hdr,
            text="Band-pass Filter:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(side='left')

        self.route_menu = ctk.CTkOptionMenu(
            filter_hdr,
            values=[
                "Route / Export ▾",
                "💾 Export Filtered WAV...",
                "💾 Export Noisy WAV...",
                "📥 Import Mic Audio",
                "Send to Converter →",
                "Send to Mic Decoder →"
            ],
            width=150,
            height=20,
            font=ctk.CTkFont(size=10),
            command=self._on_route_select
        )
        self.route_menu.set("Route / Export ▾")
        self.route_menu.pack(side='right')

        self.carrier_freq_var = tk.DoubleVar(value=700)
        ctk.CTkLabel(right_box, text="Carrier (Hz):", font=ctk.CTkFont(size=11)).grid(row=1, column=0, sticky='w', pady=1)
        self.carrier_val_lbl = ctk.CTkLabel(right_box, text="700 Hz", width=50, font=ctk.CTkFont(size=11, weight="bold"))
        self.carrier_val_lbl.grid(row=1, column=2, sticky='e', pady=1)
        self.carrier_slider = ctk.CTkSlider(
            right_box, from_=200, to=2000, variable=self.carrier_freq_var,
            command=lambda v: self.carrier_val_lbl.configure(text=f"{v:.0f} Hz"), height=14
        )
        self.carrier_slider.grid(row=1, column=1, sticky='ew', padx=6, pady=1)

        self.bw_var = tk.DoubleVar(value=200)
        ctk.CTkLabel(right_box, text="Bandwidth (Hz):", font=ctk.CTkFont(size=11)).grid(row=2, column=0, sticky='w', pady=1)
        self.bw_val_lbl = ctk.CTkLabel(right_box, text="200 Hz", width=50, font=ctk.CTkFont(size=11, weight="bold"))
        self.bw_val_lbl.grid(row=2, column=2, sticky='e', pady=1)
        self.bw_slider = ctk.CTkSlider(
            right_box, from_=50, to=600, variable=self.bw_var,
            command=lambda v: self.bw_val_lbl.configure(text=f"{v:.0f} Hz"), height=14
        )
        self.bw_slider.grid(row=2, column=1, sticky='ew', padx=6, pady=1)

        order_frame = ctk.CTkFrame(right_box, fg_color="transparent")
        order_frame.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(2, 0))
        ctk.CTkLabel(order_frame, text="Filter Order:", font=ctk.CTkFont(size=10)).pack(side='left', padx=(0, 6))
        self.order_var = ctk.StringVar(value="4th Order")
        ctk.CTkSegmentedButton(
            order_frame,
            values=["2nd Order", "4th Order", "6th Order"],
            variable=self.order_var,
            command=lambda _: self.apply_filter(),
            height=20,
            font=ctk.CTkFont(size=10)
        ).pack(side='left')

        # Status text inside the card
        self.status_lbl = ctk.CTkLabel(card, text="Ready.", font=ctk.CTkFont(size=10), text_color="gray")
        self.status_lbl.grid(row=1, column=0, columnspan=2, sticky='w', padx=8, pady=(0, 4))

    def _build_action_bar(self):
        """Action bar for applying filter, decoding recovered signal, and audio playback."""
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill='x', pady=(0, 3))

        left_box = ctk.CTkFrame(bar, fg_color="transparent")
        left_box.pack(side='left')

        self.apply_filter_btn = ctk.CTkButton(
            left_box,
            text="⚡ Apply Filter",
            width=110,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=BTN_PRIMARY,
            hover_color=BTN_PRIMARY_HOVER,
            command=self.apply_filter
        )
        self.apply_filter_btn.pack(side='left', padx=(0, 4))

        self.decode_btn = ctk.CTkButton(
            left_box,
            text="✓ Decode",
            width=85,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11),
            fg_color=BTN_PLAY,
            hover_color=BTN_PLAY_HOVER,
            command=self.decode_filtered
        )
        self.decode_btn.pack(side='left', padx=(0, 4))

        self.apply_to_input_btn = ctk.CTkButton(
            left_box,
            text="Apply to Input",
            width=100,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=10),
            fg_color=BTN_TRANSFER,
            hover_color=BTN_TRANSFER_HOVER,
            text_color=BTN_TRANSFER_TEXT,
            command=self.apply_filter_to_input
        )
        self.apply_to_input_btn.pack(side='left')

        # Audio playback source toggle (Clean, Noisy, Filtered)
        audition_box = ctk.CTkFrame(bar, fg_color="transparent")
        audition_box.pack(side='right')

        ctk.CTkLabel(
            audition_box,
            text="Audition:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray"
        ).pack(side='left', padx=(0, 6))

        self.audition_source_var = ctk.StringVar(value="Filtered")
        self.audition_seg = ctk.CTkSegmentedButton(
            audition_box,
            values=["Clean", "Noisy", "Filtered"],
            variable=self.audition_source_var,
            command=self._on_audition_source_change,
            width=190,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11)
        )
        self.audition_seg.pack(side='left', padx=(0, 6))

        self.play_btn = ctk.CTkButton(
            audition_box,
            text="▶ Play",
            width=65,
            height=BTN_HEIGHT_MD,
            fg_color=BTN_PLAY,
            hover_color=BTN_PLAY_HOVER,
            command=self._play_selected
        )
        self.play_btn.pack(side='left', padx=(0, 4))

        self.stop_btn = ctk.CTkButton(
            audition_box,
            text="■ Stop",
            width=55,
            height=BTN_HEIGHT_MD,
            fg_color=BTN_STOP,
            hover_color=BTN_STOP_HOVER,
            command=sp.stop_playback
        )
        self.stop_btn.pack(side='left', padx=(0, 4))

        self.export_btn = ctk.CTkButton(
            audition_box,
            text="💾 Export (Filtered)",
            width=130,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11),
            fg_color=BTN_SECONDARY,
            hover_color=BTN_SECONDARY_HOVER,
            text_color=BTN_SECONDARY_TEXT,
            command=self.export_wav
        )
        self.export_btn.pack(side='left')

        # Right-click context popup on export button for quick explicit selection
        self._export_menu_popup = tk.Menu(self, tearoff=0)
        self._export_menu_popup.add_command(label="Export Filtered Wave (.wav)", command=self.export_filtered_wav)
        self._export_menu_popup.add_command(label="Export Noisy Wave (.wav)", command=self.export_noisy_wav)
        self._export_menu_popup.add_command(label="Export Clean Wave (.wav)", command=lambda: self.export_wav(target="clean"))
        self.export_btn.bind("<Button-2>", self._show_export_popup)
        self.export_btn.bind("<Button-3>", self._show_export_popup)

    def _build_plot_section(self):
        """Dual-axis plot section showing input vs filtered signals."""
        plot_hdr = ctk.CTkFrame(self, fg_color="transparent")
        plot_hdr.pack(fill='x', pady=(0, 2))

        self.plot_type_var = ctk.StringVar(value="Time Domain (Waveforms)")
        ctk.CTkSegmentedButton(
            plot_hdr,
            values=["Time Domain (Waveforms)", "FFT & Filter Response |H(f)|"],
            variable=self.plot_type_var,
            command=lambda _: self.redraw(),
            width=320,
            height=24,
            font=ctk.CTkFont(size=11)
        ).pack(side='right')

        self.figure = Figure(figsize=(6, 3.8), dpi=100)
        self.ax_before = self.figure.add_subplot(211)
        self.ax_after = self.figure.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, pady=(2, 2))
        self.redraw()

    # Routing Menu Callback

    def _on_audition_source_change(self, value: str):
        """Update export button label when audition source changes."""
        if hasattr(self, 'export_btn'):
            self.export_btn.configure(text=f"💾 Export ({value})")

    def _show_export_popup(self, event):
        """Show context popup to select which wave to export."""
        try:
            self._export_menu_popup.tk_popup(event.x_root, event.y_root)
        finally:
            self._export_menu_popup.grab_release()

    def _on_route_select(self, choice: str):
        """Handle signal routing between tabs and direct WAV export."""
        self.route_menu.set("Route / Export ▾")
        if "Export Filtered" in choice:
            self.export_filtered_wav()
        elif "Export Noisy" in choice:
            self.export_noisy_wav()
        elif "Import Mic" in choice:
            self.import_from_mic()
        elif "Converter" in choice:
            self.send_decoded_to_converter()
        elif "Mic Decoder" in choice:
            self.send_to_mic_decode()

    # Clean Reference & Noise Helpers

    def _get_clean_signal(self) -> np.ndarray:
        """Get or initialize the reference clean Morse audio signal."""
        if len(self._clean_ref) == 0:
            if len(self.app.current_signal) == 0:
                morse = getattr(self.app, 'current_morse', '... --- ...') or '... --- ...'
                freq = self.carrier_freq_var.get() if hasattr(self, 'carrier_freq_var') else 700.0
                sr = self.app.current_sample_rate or 44100
                self.app.current_signal = sp.morse_to_signal(morse, freq=freq, wpm=20.0, sample_rate=sr, amplitude=0.7)
            self._clean_ref = self.app.current_signal.copy()
            if len(self.app.noisy_signal) == 0:
                self.app.noisy_signal = self._clean_ref.copy()
        return self._clean_ref

    def _get_carrier_freq(self) -> float:
        """Return selected carrier frequency in Hz."""
        return self.carrier_freq_var.get()

    def _set_carrier_freq(self, freq: float):
        """Update carrier frequency slider and label."""
        clamped = max(200, min(2000, freq))
        self.carrier_freq_var.set(clamped)
        if hasattr(self, 'carrier_slider'):
            self.carrier_slider.set(clamped)
        if hasattr(self, 'carrier_val_lbl'):
            self.carrier_val_lbl.configure(text=f"{clamped:.0f} Hz")

    def sync_bandwidth_to_carrier(self, carrier_freq: float):
        """Auto-adjust filter bandwidth proportionally to the carrier frequency."""
        target_bw = float(max(50, min(600, int(round(carrier_freq * 0.30 / 10.0) * 10))))
        self.bw_var.set(target_bw)
        if hasattr(self, 'bw_slider'):
            self.bw_slider.set(target_bw)
        if hasattr(self, 'bw_val_lbl'):
            self.bw_val_lbl.configure(text=f"{target_bw:.0f} Hz")

    def _update_noise_badge(self):
        """Update status badge showing which noise layers are currently applied."""
        parts = []
        if self._noise_counts.get("awgn", 0) > 0:
            parts.append(f"{self._noise_counts['awgn']}× AWGN")
        if self._noise_counts.get("hum", 0) > 0:
            parts.append(f"{self._noise_counts['hum']}× Hum")
        if self._noise_counts.get("clicks", 0) > 0:
            parts.append(f"{self._noise_counts['clicks']}× Clicks")

        clean = self._get_clean_signal()
        if parts:
            desc = ", ".join(parts)
            snr_est = sp.estimate_snr_db(clean, self.app.noisy_signal)
            snr_str = f"{snr_est:.1f} dB" if snr_est != float('inf') else "∞"
            self.noise_badge.configure(
                text=f"⚠ {desc} (SNR: {snr_str})",
                text_color="#f59e0b"
            )
        else:
            self.noise_badge.configure(
                text="● Clean",
                text_color="#10b981"
            )

    # Noise Injection

    def add_awgn(self):
        """Inject white Gaussian noise at target Signal-to-Noise Ratio (SNR)."""
        clean = self._get_clean_signal()
        if len(self.app.noisy_signal) == 0:
            self.app.noisy_signal = clean.copy()

        snr = self.snr_var.get()
        self.app.noisy_signal = sp.add_awgn(self.app.noisy_signal, snr_db=snr, ref_signal=clean)
        self._noise_counts["awgn"] = self._noise_counts.get("awgn", 0) + 1

        self._update_noise_badge()
        self.status_lbl.configure(text=f"Injected AWGN layer @ {snr:.0f} dB. Total AWGN: {self._noise_counts['awgn']}.")
        self.app.show_status(f"Added AWGN layer ({self._noise_counts['awgn']}x total). Ready to filter.", level="info")
        self.redraw()

    def add_hum(self):
        """Inject 60Hz power-line mains hum and harmonics."""
        clean = self._get_clean_signal()
        if len(self.app.noisy_signal) == 0:
            self.app.noisy_signal = clean.copy()

        sr = self.app.current_sample_rate or 44100
        amp = self.hum_amp_var.get()
        self.app.noisy_signal = sp.add_mains_hum(self.app.noisy_signal, sample_rate=sr, amplitude=amp)
        self._noise_counts["hum"] = self._noise_counts.get("hum", 0) + 1

        self._update_noise_badge()
        self.status_lbl.configure(text=f"Injected 60Hz Hum (amp {amp:.2f}). Total Hum: {self._noise_counts['hum']}.")
        self.app.show_status(f"Added 60Hz Mains Hum layer ({self._noise_counts['hum']}x total).", level="info")
        self.redraw()

    def add_clicks(self):
        """Inject random acoustic click/pop noise spikes."""
        clean = self._get_clean_signal()
        if len(self.app.noisy_signal) == 0:
            self.app.noisy_signal = clean.copy()

        count = int(self.clicks_count_var.get())
        self.app.noisy_signal = sp.add_acoustic_clicks(self.app.noisy_signal, num_clicks=count, amplitude=0.55)
        self._noise_counts["clicks"] = self._noise_counts.get("clicks", 0) + 1

        self._update_noise_badge()
        self.status_lbl.configure(text=f"Injected {count} acoustic clicks. Total Click bursts: {self._noise_counts['clicks']}.")
        self.app.show_status(f"Added {count} acoustic click transients ({self._noise_counts['clicks']}x total).", level="info")
        self.redraw()

    def reset_noise(self):
        """Reset signal back to original clean audio."""
        clean = self._get_clean_signal()
        self.app.noisy_signal = clean.copy()
        self.app.filtered_signal = clean.copy()
        self._noise_counts = {"awgn": 0, "hum": 0, "clicks": 0}

        self._update_noise_badge()
        self.status_lbl.configure(text="All noise cleared. Signal restored to clean.")
        self.app.show_status("All channel noise cleared. Signal restored to clean!", level="success")
        self.redraw()

    # Filtering Operations

    def _set_filtering(self, busy: bool):
        """Enable or disable buttons during filter calculation."""
        self._is_filtering = busy
        state = "disabled" if busy else "normal"
        self.apply_filter_btn.configure(state=state)
        self.add_awgn_btn.configure(state=state)
        self.add_hum_btn.configure(state=state)
        self.add_clicks_btn.configure(state=state)
        self.decode_btn.configure(state=state)
        if hasattr(self, 'export_btn'):
            self.export_btn.configure(state=state)

    def apply_filter(self):
        """Filter noisy audio with a Butterworth band-pass filter to isolate the Morse tone."""
        if self._is_filtering:
            return

        noisy = self.app.noisy_signal
        if len(noisy) == 0:
            clean = self._get_clean_signal()
            self.app.noisy_signal = clean.copy()
            noisy = self.app.noisy_signal

        freq = self._get_carrier_freq()
        bw = self.bw_var.get()
        sr = self.app.current_sample_rate or 44100
        order = int(self.order_var.get().split()[0].replace('nd', '').replace('th', ''))

        self._set_filtering(True)
        self.app.set_busy(True, "Applying Butterworth band-pass filter (noise vanishing)...")
        self.status_lbl.configure(text="Filtering signal in background...")

        def _worker():
            try:
                filtered = sp.bandpass_filter(noisy, freq, sr, bandwidth=bw, order=order, squelch=True)
                clean = self._get_clean_signal()
                snr_before = sp.estimate_snr_db(clean, noisy)
                snr_after = sp.estimate_snr_db(clean, filtered)
                self.app.run_on_ui_thread(lambda: self._on_filter_complete(filtered, freq, bw, order, snr_before, snr_after))
            except Exception as e:
                self.app.run_on_ui_thread(lambda: self._on_filter_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_filter_complete(self, filtered: np.ndarray, freq: float, bw: float, order: int, snr_before: float, snr_after: float):
        """Handle completed filtering and update SNR metrics."""
        self.app.filtered_signal = filtered
        snr_b_str = f"{snr_before:.1f} dB" if snr_before != float('inf') else "Clean"
        snr_a_str = f"{snr_after:.1f} dB" if snr_after != float('inf') else "Clean"
        self.status_lbl.configure(
            text=f"Filter applied! Noise vanished. Input SNR: {snr_b_str} → Recovered SNR: {snr_a_str} (±{bw/2:.0f} Hz @ {freq:.0f} Hz)."
        )
        self.redraw()
        self._set_filtering(False)
        self.app.set_busy(False)
        self.app.show_status(f"Filter applied! Noise vanished (Recovered SNR: {snr_a_str})", level="success")

    def _on_filter_error(self, err_msg: str):
        """Handle filter failure."""
        self._set_filtering(False)
        self.app.set_busy(False)
        self.status_lbl.configure(text=f"Filter error: {err_msg}")
        self.app.show_status(f"Filter error: {err_msg}", level="error")

    def apply_filter_to_input(self):
        """Set the recovered filtered signal as the new input."""
        if len(self.app.filtered_signal) == 0:
            self.apply_filter()
            return
        self.app.noisy_signal = self.app.filtered_signal.copy()
        self._noise_counts = {"awgn": 0, "hum": 0, "clicks": 0}
        self._update_noise_badge()
        self.status_lbl.configure(text="Clean filtered signal set as input! Noise vanished. You can add noise back anytime.")
        self.app.show_status("Noise vanished from input! You can add noise back anytime.", level="success")
        self.redraw()

    # External Audio Loading & Routing

    def load_clean_signal(self):
        """Load clean audio signal from Signal Lab (Tab 2)."""
        sig = self.app.current_signal
        sr = self.app.current_sample_rate
        carrier = self.app.signal_tab.freq_var.get()
        self._set_carrier_freq(carrier)
        self.sync_bandwidth_to_carrier(carrier)

        self._clean_ref = sig.copy()
        self.app.noisy_signal = sig.copy()
        self.app.filtered_signal = sig.copy()
        self._noise_counts = {"awgn": 0, "hum": 0, "clicks": 0}
        self._update_noise_badge()

        self.status_lbl.configure(
            text=f"Loaded from Signal Lab: {len(sig):,} samples @ {sr} Hz (carrier {carrier:.0f} Hz). Ready to inject noise."
        )
        self.redraw()

    def load_external_noisy_audio(self, audio: np.ndarray, sr: int, source_name: str = "External Audio"):
        """Load noisy audio from an external source or recording."""
        cleaned = sp.preprocess_mic_audio(audio, sr)
        self._clean_ref = cleaned.copy()
        self.app.noisy_signal = cleaned
        self.app.current_sample_rate = sr
        self._noise_counts = {"awgn": 1, "hum": 0, "clicks": 0}

        detected_freq = sp.detect_carrier_frequency(cleaned, sr)
        self._set_carrier_freq(detected_freq)
        self.sync_bandwidth_to_carrier(detected_freq)
        self._update_noise_badge()

        self.status_lbl.configure(
            text=f"Loaded {source_name}: {len(cleaned):,} samples @ {sr} Hz • Carrier detected: {detected_freq:.0f} Hz"
        )
        self.app.show_status(f"Noisy audio loaded into Noise Lab (carrier: {detected_freq:.0f} Hz)", level="info")
        self.apply_filter()

    def import_from_mic(self):
        """Load recorded audio from the Mic Decoder (Tab 4)."""
        audio = self.app.recorded_signal
        sr = self.app.recorded_sample_rate or 44100
        if len(audio) == 0:
            self.app.show_status("No mic recording available — record or load audio in Mic tab first", level="warning")
            return
        self.load_external_noisy_audio(audio, sr, source_name="Mic Recording")

    # Visualization & Plots

    def redraw(self):
        """Plot input vs filtered signals in either Time Domain or Frequency Domain."""
        colors = get_plot_theme()
        self.figure.patch.set_facecolor(colors['fig_bg'])
        self.ax_before.clear()
        self.ax_after.clear()

        sr = self.app.current_sample_rate or 44100
        noisy = self.app.noisy_signal
        filtered = self.app.filtered_signal
        plot_type = self.plot_type_var.get()
        freq = self._get_carrier_freq()
        bw = self.bw_var.get()
        order = int(self.order_var.get().split()[0].replace('nd', '').replace('th', ''))

        noise_parts = []
        if self._noise_counts.get("awgn", 0) > 0:
            noise_parts.append(f"{self._noise_counts['awgn']}× AWGN")
        if self._noise_counts.get("hum", 0) > 0:
            noise_parts.append(f"{self._noise_counts['hum']}× Hum")
        if self._noise_counts.get("clicks", 0) > 0:
            noise_parts.append(f"{self._noise_counts['clicks']}× Clicks")
        noise_title = ", ".join(noise_parts) if noise_parts else "Clean Channel"

        if "Time Domain" in plot_type:
            # Time-domain waveforms before and after filtering
            if len(noisy) > 0:
                step = max(1, len(noisy) // 20000)
                t = np.arange(0, len(noisy), step) / sr
                self.ax_before.plot(t, noisy[::step], linewidth=0.7, color=colors['noisy'])
                style_axis(self.ax_before, colors, title=f"1. Input Signal ({noise_title})", ylabel="Amp")
            else:
                style_axis(self.ax_before, colors, title="Input Signal (Click '+ AWGN / Hum / Clicks')", ylabel="Amp")

            if len(filtered) > 0:
                step = max(1, len(filtered) // 20000)
                t = np.arange(0, len(filtered), step) / sr
                self.ax_after.plot(t, filtered[::step], linewidth=0.7, color=colors['filtered'])
                style_axis(self.ax_after, colors, title=f"2. Filtered Signal (Butterworth Band-pass: Noise Vanished — {freq:.0f} ± {bw/2:.0f} Hz)", xlabel="Time (s)", ylabel="Amp")
            else:
                style_axis(self.ax_after, colors, title="Filtered Signal (Click 'Apply Filter (Vanish Noise)')", xlabel="Time (s)", ylabel="Amp")

        else:
            # Frequency spectrum (FFT) and filter response curve |H(f)|
            if len(noisy) > 0:
                f_noisy, mag_noisy = sp.compute_fft(noisy, sr)
                self.ax_before.plot(f_noisy, mag_noisy, linewidth=0.9, color=colors['noisy'])
                self.ax_before.set_xlim(0, min(3000, sr / 2.0))
                style_axis(self.ax_before, colors, title=f"Input Signal Spectrum (FFT) — {noise_title}", ylabel="Magnitude")

            if len(filtered) > 0:
                f_filt, mag_filt = sp.compute_fft(filtered, sr)
                self.ax_after.plot(f_filt, mag_filt, linewidth=0.9, color=colors['filtered'], label='Filtered Spectrum (Noise Vanished)')

                w, h_mag = sp.filter_frequency_response(freq, sr, bandwidth=bw, order=order, num_points=1024)
                if len(w) > 0 and len(mag_filt) > 0:
                    peak_mag = np.max(mag_filt) if np.max(mag_filt) > 0 else 1.0
                    h_scaled = h_mag * peak_mag
                    self.ax_after.plot(w, h_scaled, linewidth=1.5, linestyle='--', color=colors['response'], label='|H(f)| Filter Passband')

                self.ax_after.set_xlim(0, min(3000, sr / 2.0))
                self.ax_after.legend(loc='upper right', fontsize=8)
                style_axis(self.ax_after, colors, title=f"Filtered Spectrum vs Butterworth Response |H(f)| ({order}th order)", xlabel="Frequency (Hz)", ylabel="Magnitude")

        self.figure.tight_layout()
        self.canvas.draw()

    # Decoding & Audition Controls

    def decode_filtered(self):
        """Demodulate and decode the filtered audio into text."""
        filtered = self.app.filtered_signal
        if len(filtered) == 0:
            if len(self.app.noisy_signal) > 0:
                self.apply_filter()
                filtered = self.app.filtered_signal
            else:
                self.app.show_status("No signal available — generate or add noise first", level="warning")
                return

        sr = self.app.current_sample_rate or 44100
        carrier = self._get_carrier_freq()
        self.app.set_busy(True, "Decoding filtered Morse signal...")

        def _worker():
            try:
                morse, details = sp.decode_signal_to_morse(filtered, sr, carrier_freq=carrier, return_details=True)
                self.app.run_on_ui_thread(lambda: self._on_decode_filtered_complete(morse, details))
            except Exception as e:
                self.app.run_on_ui_thread(lambda: self.app.show_status(f"Decode error: {e}", level="error"))
            finally:
                self.app.run_on_ui_thread(lambda: self.app.set_busy(False))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_decode_filtered_complete(self, morse: str, details: dict):
        """Show decode results and comparison with expected message."""
        text = mc.morse_to_text(morse) if morse else ""
        orig_morse = getattr(self.app, 'current_morse', '')
        orig_text = mc.morse_to_text(orig_morse)

        accuracy_str = "100% Exact match!" if text == orig_text else f"Expected: '{orig_text}'"
        speed = details.get('wpm', 0)
        self.status_lbl.configure(text=f"Decoded: '{text}' ({speed:.1f} WPM) • {accuracy_str}")
        self.app.show_status(f"Decoded: '{text}' ({speed:.1f} WPM) • {accuracy_str}", level="success", duration_ms=5000)

    def send_to_mic_decode(self):
        """Send filtered signal to Mic Decoder (Tab 4)."""
        filtered = self.app.filtered_signal
        if len(filtered) == 0:
            self.apply_filter()
            filtered = self.app.filtered_signal
        self.app.mic_tab.load_audio_data(
            filtered,
            self.app.current_sample_rate or 44100,
            source_name="Noise Lab (Filtered Audio)"
        )
        self.app.tabview.set("4. Mic & Audio Decode")
        self.app.show_status("Filtered audio routed to Audio Decoder", level="info")

    def send_decoded_to_converter(self):
        """Send decoded message to Converter (Tab 1)."""
        status_text = self.status_lbl.cget('text')
        import re
        match = re.search(r"Decoded:\s*'([^']*)'", status_text)
        if match:
            decoded = match.group(1)
            self.app.converter_tab.load_text_or_morse(decoded, is_morse=False)
            self.app.tabview.set("1. Converter")
            self.app.show_status(f"Decoded text '{decoded}' sent to Converter", level="success")
        else:
            self.app.show_status("No decoded text available — decode filtered signal first", level="warning")

    def _play_selected(self):
        """Play selected audio version (Clean, Noisy, or Filtered)."""
        src = self.audition_source_var.get()
        if src == "Clean":
            data = self._get_clean_signal()
        elif src == "Noisy":
            data = self.app.noisy_signal
        else:
            data = self.app.filtered_signal
        self._play(data, src)

    def _play(self, data: np.ndarray, label: str = "Audio"):
        """Play audio waveform through speakers."""
        if len(data) == 0:
            self.app.show_status(f"Nothing to play — generate {label.lower()} signal first", level="warning")
            return
        try:
            sp.play_signal(data, self.app.current_sample_rate or 44100)
            self._is_playing = True
            dur = len(data) / float(self.app.current_sample_rate or 44100)
            self.app.show_status(f"Playing {label} signal ({dur:.2f}s)...", level="info")
            self.after(int(dur * 1000) + 100, lambda: setattr(self, '_is_playing', False))
        except RuntimeError as e:
            self.app.show_status(f"Playback error: {e}", level="error")

    # WAV Export Operations

    def export_wav(self, target: Optional[str] = None, filepath: Optional[str] = None) -> Optional[str]:
        """
        Save the noisy, filtered, or clean audio waveform to a standard 16-bit PCM WAV file.

        Parameters
        ----------
        target : str, optional
            Which signal to export: 'noisy', 'filtered', or 'clean'.
            If None, exports whichever signal is currently selected in Audition.
        filepath : str, optional
            Target file path. If None, opens a standard file dialog.

        Returns
        -------
        str or None
            Saved file path if successful, None otherwise.
        """
        if target is None:
            source = self.audition_source_var.get().strip().lower()
        else:
            source = target.strip().lower()

        sr = self.app.current_sample_rate or 44100

        if "noisy" in source:
            data = self.app.noisy_signal
            if len(data) == 0:
                clean = self._get_clean_signal()
                if len(self.app.noisy_signal) == 0:
                    self.app.noisy_signal = clean.copy()
                data = self.app.noisy_signal
            label = "Noisy"
            default_fname = "noisy_morse_signal.wav"
        elif "clean" in source:
            data = self._get_clean_signal()
            label = "Clean"
            default_fname = "clean_morse_signal.wav"
        else:
            # Default to filtered signal
            data = self.app.filtered_signal
            if len(data) == 0:
                # If filter has not yet been computed, compute it on demand from noisy signal
                noisy = self.app.noisy_signal
                if len(noisy) == 0:
                    clean = self._get_clean_signal()
                    if len(self.app.noisy_signal) == 0:
                        self.app.noisy_signal = clean.copy()
                    noisy = self.app.noisy_signal
                freq = self._get_carrier_freq()
                bw = self.bw_var.get()
                order = int(self.order_var.get().split()[0].replace('nd', '').replace('th', ''))
                try:
                    data = sp.bandpass_filter(noisy, freq, sr, bandwidth=bw, order=order, squelch=True)
                    self.app.filtered_signal = data
                    self.redraw()
                except Exception as e:
                    self.app.show_status(f"Filter failed prior to export: {e}", level="error")
                    return None
            label = "Filtered"
            default_fname = "filtered_morse_signal.wav"

        if len(data) == 0:
            self.app.show_status(f"Nothing to export — no {label.lower()} audio available", level="warning")
            return None

        if filepath is None:
            path = filedialog.asksaveasfilename(
                title=f"Export {label} Audio to WAV",
                defaultextension=".wav",
                filetypes=[("WAV Audio (*.wav)", "*.wav"), ("All Files (*.*)", "*.*")],
                initialfile=default_fname
            )
            if not path:
                return None
        else:
            path = filepath

        try:
            sp.save_wav(path, data, sr)
            fname = Path(path).name
            dur = len(data) / float(sr)
            self.app.show_status(
                f"{label} WAV exported successfully: {fname} ({dur:.2f}s @ {sr} Hz)",
                level="success",
                duration_ms=4500
            )
            return path
        except Exception as e:
            self.app.show_status(f"Failed to export {label} WAV: {e}", level="error")
            return None

    def export_noisy_wav(self, filepath: Optional[str] = None) -> Optional[str]:
        """Export the noisy audio waveform to a WAV file."""
        return self.export_wav(target="noisy", filepath=filepath)

    def export_filtered_wav(self, filepath: Optional[str] = None) -> Optional[str]:
        """Export the filtered audio waveform to a WAV file."""
        return self.export_wav(target="filtered", filepath=filepath)
