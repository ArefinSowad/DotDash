"""
DotDash — Main Application Window.
Coordinates tabs for Morse text conversion, signal synthesis, noise filtering, and mic decoding.
"""

import json
from pathlib import Path
import queue
import sys
import tkinter as tk
from tkinter import messagebox
from typing import Optional
import numpy as np
import customtkinter as ctk

import morse_code as mc
import signal_proc as sp

# UI tabs and theme
from ui.theme import get_plot_theme, style_axis
from ui.converter_tab import ConverterTab
from ui.signal_tab import SignalLabTab
from ui.noise_tab import NoiseFilterTab
from ui.mic_tab import MicDecodeTab

# App appearance defaults
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class MorseSignalLab(ctk.CTk):
    """
    Main application window managing shared signals, settings persistence,
    status bar messages, and tabs.
    """

    def __init__(self):
        super().__init__()
        self.title("DotDash — DSP & Systems (CSE-220)")
        self.geometry("1080x820")
        self.minsize(940, 700)

        self._status_timer: Optional[str] = None
        self._task_queue: queue.Queue = queue.Queue()
        self._poll_ui_queue()

        # Shared signals accessible across all tabs
        self.current_signal: np.ndarray = np.array([])
        self.current_sample_rate: int = 44100
        self.current_morse: str = mc.text_to_morse("SOS")
        self.noisy_signal: np.ndarray = np.array([])
        self.filtered_signal: np.ndarray = np.array([])
        self.recorded_signal: np.ndarray = np.array([])
        self.recorded_sample_rate: int = 44100
        self.recorded_details: dict = {}

        # Build top header and bottom status bar
        self._build_header()
        self._build_status_bar()

        # Build tab container
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.pack(fill='both', expand=True, padx=10, pady=(0, 4))

        tab_conv = self.tabview.add("1. Converter")
        tab_sig = self.tabview.add("2. Signal Lab")
        tab_noise = self.tabview.add("3. Noise & Filter")
        tab_mic = self.tabview.add("4. Mic & Audio Decode")

        # Initialize tab contents
        self.converter_tab = ConverterTab(tab_conv, self)
        self.signal_tab = SignalLabTab(tab_sig, self)
        self.noise_tab = NoiseFilterTab(tab_noise, self)
        self.mic_tab = MicDecodeTab(tab_mic, self)

        # Restore saved window size and theme
        self._restore_window_state()

        # Clean exit handler
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Alert if audio hardware is missing
        if not sp.SOUND_AVAILABLE:
            self.after(250, lambda: self.show_status(
                "Audio hardware unavailable: Sound recording/playback muted. DSP synthesis/decode fully functional.",
                level="warning",
                duration_ms=6000
            ))

    # Configuration persistence

    @staticmethod
    def _get_config_path() -> Path:
        """Find the config file path in the user's home folder."""
        try:
            p = Path.home() / ".dotdash_config.json"
            legacy = Path.home() / ".morse_signal_lab_config.json"
            if not p.exists() and legacy.exists():
                return legacy
            return p
        except Exception:
            return Path(__file__).parent / ".dotdash_config.json"

    def _restore_window_state(self):
        """Restore window size, position, and theme from config file."""
        cfg_path = self._get_config_path()
        if not cfg_path.exists():
            return
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            geom = data.get("geometry")
            if geom and isinstance(geom, str) and "x" in geom:
                self.geometry(geom)
            mode = data.get("appearance_mode")
            if mode in ("Dark", "Light", "System"):
                ctk.set_appearance_mode(mode)
                if hasattr(self, 'theme_switch'):
                    self.theme_switch.set(mode)
        except Exception:
            pass

    def _save_window_state(self):
        """Save window size and current theme to config file."""
        try:
            cfg_path = self._get_config_path()
            data = {
                "geometry": self.geometry(),
                "appearance_mode": ctk.get_appearance_mode()
            }
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _on_close(self):
        """Save settings and stop audio before closing the app."""
        self._save_window_state()
        sp.stop_playback()
        self.destroy()

    def run_on_ui_thread(self, callback):
        """Schedule a background thread callback to safely run on the main UI thread."""
        self._task_queue.put(callback)

    def _poll_ui_queue(self):
        """Check for and run queued background callbacks."""
        try:
            while not self._task_queue.empty():
                try:
                    cb = self._task_queue.get_nowait()
                    cb()
                except Exception as e:
                    print(f"Error executing UI callback: {e}", file=sys.stderr)
        finally:
            self.after(25, self._poll_ui_queue)

    # Header and status bar

    def _build_header(self):
        """Create header with title, audio status indicator, and theme switcher."""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill='x', padx=12, pady=(8, 2))

        title_lbl = ctk.CTkLabel(
            header_frame,
            text="DOTDASH",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_lbl.pack(side='left')

        subtitle_lbl = ctk.CTkLabel(
            header_frame,
            text="CSE-220 · Signals & Systems",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        subtitle_lbl.pack(side='left', padx=10)

        # Show if microphone / speakers are available
        sound_status = "● Audio Ready" if sp.SOUND_AVAILABLE else "○ Muted"
        sound_color = "#10b981" if sp.SOUND_AVAILABLE else "#94a3b8"
        sound_lbl = ctk.CTkLabel(
            header_frame,
            text=sound_status,
            font=ctk.CTkFont(size=10),
            text_color=sound_color
        )
        sound_lbl.pack(side='left', padx=6)

        # Dark / Light mode selector
        self.theme_switch = ctk.CTkOptionMenu(
            header_frame,
            values=["Dark", "Light", "System"],
            width=80,
            height=24,
            command=self._change_appearance_mode
        )
        self.theme_switch.set("Dark")
        self.theme_switch.pack(side='right')

    def _build_status_bar(self):
        """Create bottom status bar for alerts and progress messages."""
        self.status_bar = ctk.CTkFrame(self, height=24, corner_radius=0, fg_color=("gray85", "gray17"))
        self.status_bar.pack(side='bottom', fill='x')

        self.status_icon = ctk.CTkLabel(
            self.status_bar,
            text="●",
            font=ctk.CTkFont(size=11),
            text_color="#10b981",
            width=14
        )
        self.status_icon.pack(side='left', padx=(8, 3), pady=2)

        self.status_msg = ctk.CTkLabel(
            self.status_bar,
            text="Ready",
            font=ctk.CTkFont(size=10),
            text_color=("gray20", "gray80")
        )
        self.status_msg.pack(side='left', padx=2, pady=2)

        # Progress bar shown during slow calculations
        self.busy_bar = ctk.CTkProgressBar(self.status_bar, width=110, height=6, mode="indeterminate")

    # Status notifications

    def show_status(self, text: str, level: str = "info", duration_ms: int = 3500):
        """
        Display a status message in the bottom bar.
        Levels: 'info', 'success', 'warning', 'error', 'busy', 'ready'.
        """
        colors = {
            'info': ("#0284c7", "ℹ"),
            'success': ("#10b981", "✓"),
            'warning': ("#f59e0b", "⚠"),
            'error': ("#ef4444", "✖"),
            'ready': ("#10b981", "●"),
            'busy': ("#8b5cf6", "◌"),
        }
        color, icon = colors.get(level, ("#0284c7", "ℹ"))
        self.status_icon.configure(text=icon, text_color=color)
        self.status_msg.configure(text=text)

        if self._status_timer is not None:
            self.after_cancel(self._status_timer)
            self._status_timer = None

        if duration_ms > 0 and level not in ('busy', 'ready'):
            self._status_timer = self.after(duration_ms, self._reset_status)

    def set_busy(self, is_busy: bool, message: str = "Computing..."):
        """Show or hide the busy progress bar during background tasks."""
        if is_busy:
            self.show_status(message, level="busy", duration_ms=0)
            self.busy_bar.pack(side='right', padx=12, pady=8)
            self.busy_bar.start()
        else:
            self.busy_bar.stop()
            self.busy_bar.pack_forget()
            self.show_status("Ready", level="ready", duration_ms=0)

    def _reset_status(self):
        """Reset status bar back to Ready."""
        self.status_icon.configure(text="●", text_color="#10b981")
        self.status_msg.configure(text="Ready")
        self._status_timer = None

    # Theme and clipboard helpers

    def _change_appearance_mode(self, new_mode: str):
        """Switch appearance mode (Dark, Light, System) and refresh all plots."""
        ctk.set_appearance_mode(new_mode)
        self.converter_tab.redraw()
        self.signal_tab.redraw()
        self.noise_tab.redraw()
        self.mic_tab.redraw()
        self.show_status(f"Appearance theme switched to {new_mode}", level="info", duration_ms=2500)

    def copy_to_clipboard(self, text: str, desc: str = "Copied to clipboard!"):
        """Copy text to system clipboard and display status confirmation."""
        if not text:
            self.show_status("Nothing to copy", level="warning")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()
        self.show_status(desc, level="success")


__all__ = [
    "MorseSignalLab",
    "ConverterTab",
    "SignalLabTab",
    "NoiseFilterTab",
    "MicDecodeTab",
    "get_plot_theme",
    "style_axis"
]


if __name__ == '__main__':
    app = MorseSignalLab()
    app.mainloop()