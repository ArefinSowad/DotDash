"""
Tab 1: Plain text and Morse code converter.
Provides two-way text conversion, audio preview, timing stats, and a reference table.
"""

import tkinter as tk
from tkinter import messagebox
from typing import TYPE_CHECKING
import customtkinter as ctk

import morse_code as mc
import signal_proc as sp
from ui.theme import (
    BTN_PRIMARY, BTN_PRIMARY_HOVER,
    BTN_PLAY, BTN_PLAY_HOVER,
    BTN_STOP, BTN_STOP_HOVER,
    BTN_SECONDARY, BTN_SECONDARY_HOVER, BTN_SECONDARY_TEXT,
    BTN_TRANSFER, BTN_TRANSFER_HOVER, BTN_TRANSFER_TEXT,
    BTN_HEIGHT_MD, BTN_HEIGHT_SM, BTN_CHIP_RADIUS
)

if TYPE_CHECKING:
    from app import MorseSignalLab


class ConverterTab(ctk.CTkFrame):
    """Tab for converting between plain text and Morse code."""

    def __init__(self, parent: ctk.CTkBaseClass, app: "MorseSignalLab"):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.pack(fill='both', expand=True, padx=2, pady=2)

        # Build UI sections
        self._build_top_bar()
        self._build_input_section()
        self._build_action_toolbar()
        self._build_output_section()
        self._build_reference_section()

        # Run initial conversion with default text
        self.convert()

    # UI Layout Construction

    def _build_top_bar(self):
        """Header bar with title and direction toggle (Text->Morse vs Morse->Text)."""
        top_row = ctk.CTkFrame(self, fg_color="transparent")
        top_row.pack(fill='x', pady=(2, 6))

        ctk.CTkLabel(
            top_row,
            text="Text ↔ International Morse Converter",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side='left')

        self.mode_var = ctk.StringVar(value="Text → Morse")
        self.mode_seg = ctk.CTkSegmentedButton(
            top_row,
            values=["Text → Morse", "Morse → Text"],
            variable=self.mode_var,
            command=lambda _: self.on_mode_change(),
            width=200,
            height=26
        )
        self.mode_seg.pack(side='right')

    def _build_input_section(self):
        """Input textbox with quick example buttons."""
        input_header = ctk.CTkFrame(self, fg_color="transparent")
        input_header.pack(fill='x', pady=(0, 2))
        self.input_lbl = ctk.CTkLabel(
            input_header,
            text="Input:",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.input_lbl.pack(side='left')

        preset_frame = ctk.CTkFrame(input_header, fg_color="transparent")
        preset_frame.pack(side='right')
        ctk.CTkLabel(preset_frame, text="Presets:", font=ctk.CTkFont(size=10), text_color="gray").pack(side='left', padx=4)

        for preset in ["HELLO WORLD", "SOS", "CQ CQ CQ"]:
            ctk.CTkButton(
                preset_frame,
                text=preset,
                font=ctk.CTkFont(size=10),
                width=60,
                height=20,
                corner_radius=BTN_CHIP_RADIUS,
                fg_color=BTN_SECONDARY,
                hover_color=BTN_SECONDARY_HOVER,
                text_color=BTN_SECONDARY_TEXT,
                command=lambda p=preset: self.load_preset(p)
            ).pack(side='left', padx=2)

        self.input_box = ctk.CTkTextbox(self, height=55, font=('Consolas', 12))
        self.input_box.pack(fill='x', pady=(0, 4))
        self.input_box.insert('1.0', 'HELLO WORLD')
        self.input_box.bind('<KeyRelease>', lambda _: self.on_text_change())

    def _build_action_toolbar(self):
        """Action buttons for converting, previewing sound, and routing text."""
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill='x', pady=(0, 4))

        # Conversion and playback buttons
        left_box = ctk.CTkFrame(btn_frame, fg_color="transparent")
        left_box.pack(side='left')

        self.convert_btn = ctk.CTkButton(
            left_box,
            text="Convert",
            width=85,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=BTN_PRIMARY,
            hover_color=BTN_PRIMARY_HOVER,
            command=self.convert
        )
        self.convert_btn.pack(side='left', padx=(0, 4))

        self.listen_btn = ctk.CTkButton(
            left_box,
            text="▶ Preview",
            width=85,
            height=BTN_HEIGHT_MD,
            fg_color=BTN_PLAY,
            hover_color=BTN_PLAY_HOVER,
            command=self.listen_preview
        )
        self.listen_btn.pack(side='left', padx=(0, 4))

        self.stop_btn = ctk.CTkButton(
            left_box,
            text="■ Stop",
            width=58,
            height=BTN_HEIGHT_MD,
            fg_color=BTN_STOP,
            hover_color=BTN_STOP_HOVER,
            command=sp.stop_playback
        )
        self.stop_btn.pack(side='left', padx=(0, 4))

        self.clear_btn = ctk.CTkButton(
            left_box,
            text="Clear",
            width=58,
            height=BTN_HEIGHT_MD,
            fg_color=BTN_SECONDARY,
            hover_color=BTN_SECONDARY_HOVER,
            text_color=BTN_SECONDARY_TEXT,
            command=self.clear
        )
        self.clear_btn.pack(side='left')

        # Send/receive buttons connecting other tabs
        right_box = ctk.CTkFrame(btn_frame, fg_color="transparent")
        right_box.pack(side='right')

        self.pull_btn = ctk.CTkButton(
            right_box,
            text="📥 Pull Decoder",
            width=110,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11),
            fg_color=BTN_TRANSFER,
            hover_color=BTN_TRANSFER_HOVER,
            text_color=BTN_TRANSFER_TEXT,
            command=self.pull_from_decoder
        )
        self.pull_btn.pack(side='left', padx=(0, 4))

        self.send_btn = ctk.CTkButton(
            right_box,
            text="To Signal Lab →",
            width=120,
            height=BTN_HEIGHT_MD,
            font=ctk.CTkFont(size=11),
            fg_color=BTN_TRANSFER,
            hover_color=BTN_TRANSFER_HOVER,
            text_color=BTN_TRANSFER_TEXT,
            command=self.send_to_signal_lab
        )
        self.send_btn.pack(side='left')

    def _build_output_section(self):
        """Output box displaying conversion results and timing metrics."""
        output_header = ctk.CTkFrame(self, fg_color="transparent")
        output_header.pack(fill='x', pady=(0, 2))
        ctk.CTkLabel(output_header, text="Output:", font=ctk.CTkFont(size=11, weight="bold")).pack(side='left')

        copy_btn = ctk.CTkButton(
            output_header,
            text="📋 Copy",
            width=60,
            height=20,
            font=ctk.CTkFont(size=10),
            fg_color=BTN_SECONDARY,
            hover_color=BTN_SECONDARY_HOVER,
            text_color=BTN_SECONDARY_TEXT,
            corner_radius=6,
            command=self.copy_output
        )
        copy_btn.pack(side='right')

        self.stats_var = ctk.StringVar(value="")
        self.stats_lbl = ctk.CTkLabel(output_header, textvariable=self.stats_var, font=ctk.CTkFont(size=10), text_color="gray")
        self.stats_lbl.pack(side='right', padx=8)

        self.output_box = ctk.CTkTextbox(self, height=55, font=('Consolas', 12))
        self.output_box.pack(fill='x', pady=(0, 4))

    def _build_reference_section(self):
        """Searchable visual reference chart of Morse code characters."""
        ref_header = ctk.CTkFrame(self, fg_color="transparent")
        ref_header.pack(fill='x', pady=(2, 3))
        ctk.CTkLabel(
            ref_header,
            text="Morse Code Reference:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(side='left')

        self.filter_var = ctk.StringVar()
        self.filter_entry = ctk.CTkEntry(
            ref_header,
            textvariable=self.filter_var,
            placeholder_text="🔍 Filter...",
            width=120,
            height=24
        )
        self.filter_entry.pack(side='right')
        self.filter_entry.bind('<KeyRelease>', lambda _: self.render_reference_table())

        self.scroll_frame = ctk.CTkScrollableFrame(self, height=175, corner_radius=6)
        self.scroll_frame.pack(fill='both', expand=True)
        self.render_reference_table()

    # Event Handlers & Operations

    def on_mode_change(self):
        """Update label text when toggling between Text->Morse and Morse->Text."""
        mode = self.mode_var.get()
        if mode == "Text → Morse":
            self.input_lbl.configure(text="Input Text (Plaintext):")
        else:
            self.input_lbl.configure(text="Input Morse Code (dots/dashes):")
        self.convert()

    def on_text_change(self):
        """Automatically convert as the user types."""
        self.convert()

    def load_preset(self, text: str):
        """Load an example word or phrase into the input box."""
        self.mode_var.set("Text → Morse")
        self.on_mode_change()
        self.input_box.delete('1.0', 'end')
        self.input_box.insert('1.0', text)
        self.convert()

    def convert(self):
        """Convert input text and display length and playback timing."""
        raw = self.input_box.get('1.0', 'end').strip()
        if not raw:
            self.output_box.delete('1.0', 'end')
            self.stats_var.set("Enter text to convert")
            return

        if self.mode_var.get() == "Text → Morse":
            result = mc.text_to_morse(raw)
            units = mc.estimate_transmission_units(result)
            dur = mc.estimate_duration_seconds(result, wpm=20.0)
            self.stats_var.set(f"{len(raw)} characters • {units} units • ~{dur:.1f}s @ 20 WPM")
        else:
            if not mc.is_valid_morse(raw):
                self.output_box.delete('1.0', 'end')
                self.output_box.insert('1.0', "(Invalid Morse: only '.', '-', spaces, and '/' allowed)")
                self.stats_var.set("Invalid characters detected")
                return
            result = mc.morse_to_text(raw)
            units = mc.estimate_transmission_units(raw)
            dur = mc.estimate_duration_seconds(raw, wpm=20.0)
            self.stats_var.set(f"{len(result)} characters • {units} units • ~{dur:.1f}s @ 20 WPM")

        self.output_box.delete('1.0', 'end')
        self.output_box.insert('1.0', result)

    def toggle_preview(self):
        """Toggle audio preview on or off."""
        if getattr(self, '_is_playing', False):
            self.stop_preview()
        else:
            self.listen_preview()

    def stop_preview(self):
        """Stop any playing audio preview."""
        sp.stop_playback()
        self._is_playing = False
        self.app.show_status("Audio playback stopped", level="info")

    def listen_preview(self):
        """Synthesize and play audio for the current Morse code."""
        out = self.output_box.get('1.0', 'end').strip()
        morse = out if self.mode_var.get() == "Text → Morse" else self.input_box.get('1.0', 'end').strip()
        if not morse or not mc.is_valid_morse(morse):
            self.app.show_status("Please convert valid text to Morse first", level="warning")
            return
        try:
            sig = sp.morse_to_signal(morse, freq=700, wpm=20, sample_rate=44100, amplitude=0.6)
            sp.play_signal(sig, 44100)
            self._is_playing = True
            dur = len(sig) / 44100.0
            self.app.show_status(f"Playing audio preview ({dur:.1f}s)...", level="info")
            self.after(int(dur * 1000) + 100, lambda: setattr(self, '_is_playing', False))
        except RuntimeError as e:
            self.app.show_status(f"Audio playback error: {e}", level="error")

    def copy_output(self):
        """Copy converted output to the clipboard."""
        text = self.output_box.get('1.0', 'end').strip()
        if text:
            self.app.copy_to_clipboard(text, desc="Converted output copied to clipboard!")
        else:
            self.app.show_status("No converted text to copy", level="warning")

    def clear(self):
        """Clear both input and output text boxes."""
        self.input_box.delete('1.0', 'end')
        self.output_box.delete('1.0', 'end')
        self.stats_var.set("Ready")
        self.app.show_status("Cleared converter textboxes", level="info")

    def send_to_signal_lab(self):
        """Forward current Morse code string to Signal Lab (Tab 2)."""
        out = self.output_box.get('1.0', 'end').strip()
        if not out and self.mode_var.get() == "Text → Morse":
            self.convert()
            out = self.output_box.get('1.0', 'end').strip()
        if self.mode_var.get() == "Morse → Text":
            out = self.input_box.get('1.0', 'end').strip()
        if not out or not mc.is_valid_morse(out):
            self.app.show_status("Convert valid text to Morse first", level="warning")
            return
        self.app.signal_tab.set_morse_input(out)
        self.app.tabview.set("2. Signal Lab")
        self.app.show_status("Morse code forwarded to Signal Lab", level="success")

    def load_text_or_morse(self, text: str, is_morse: bool = False):
        """Load text from another tab and trigger automatic conversion."""
        if is_morse:
            self.mode_var.set("Morse → Text")
            self.input_lbl.configure(text="Input Morse Code (dots/dashes):")
        else:
            self.mode_var.set("Text → Morse")
            self.input_lbl.configure(text="Input Text (Plaintext):")
        self.input_box.delete('1.0', 'end')
        self.input_box.insert('1.0', text)
        self.convert()

    def pull_from_decoder(self):
        """Import decoded text or Morse from the Mic Decoder (Tab 4)."""
        decoded_text = self.app.mic_tab.text_out.get('1.0', 'end').strip()
        decoded_morse = self.app.mic_tab.morse_out.get('1.0', 'end').strip()
        if decoded_text and decoded_text not in ('(none)', '(no recognizable Morse tone detected)'):
            self.load_text_or_morse(decoded_text, is_morse=False)
            self.app.show_status(f"Decoded text '{decoded_text}' pulled from Decoder", level="success")
        elif decoded_morse and decoded_morse != '(no recognizable Morse tone detected)':
            self.load_text_or_morse(decoded_morse, is_morse=True)
            self.app.show_status("Decoded Morse pulled from Decoder", level="success")
        else:
            self.app.show_status("No decoded output available — demodulate audio in Mic tab first", level="warning")

    def redraw(self):
        """Redraw theme-dependent elements when appearance mode changes."""
        self.render_reference_table()

    def render_reference_table(self):
        """Populate the reference grid with Morse character cards."""
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        rows = mc.morse_table_rows()
        filter_text = self.filter_var.get().strip().upper()
        if filter_text:
            rows = [(ch, code) for ch, code in rows if filter_text in ch or filter_text in code]

        num_cols = 4
        chunk_size = max(1, (len(rows) + num_cols - 1) // num_cols)

        for c in range(num_cols):
            self.scroll_frame.grid_columnconfigure(c, weight=1, uniform="group1")

        for col_idx in range(num_cols):
            start = col_idx * chunk_size
            chunk = rows[start:start + chunk_size]
            for row_idx, (char, code) in enumerate(chunk):
                item_frame = ctk.CTkFrame(self.scroll_frame, fg_color=("gray90", "gray18"), corner_radius=6)
                item_frame.grid(row=row_idx, column=col_idx, padx=4, pady=3, sticky="ew")

                lbl_char = ctk.CTkLabel(
                    item_frame,
                    text=f" {char} ",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    fg_color=("#3b82f6", "#1d4ed8"),
                    text_color="white",
                    corner_radius=4,
                    width=28
                )
                lbl_char.pack(side='left', padx=(4, 8), pady=2)

                lbl_code = ctk.CTkLabel(
                    item_frame,
                    text=code,
                    font=('Consolas', 12, 'bold'),
                    text_color=("gray20", "gray80")
                )
                lbl_code.pack(side='left', padx=2, pady=2)
