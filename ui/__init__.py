"""
UI package for DotDash.
Contains user interface tabs and styling themes.
"""

from ui.theme import get_plot_theme, style_axis
from ui.converter_tab import ConverterTab
from ui.signal_tab import SignalLabTab
from ui.noise_tab import NoiseFilterTab
from ui.mic_tab import MicDecodeTab

__all__ = [
    "get_plot_theme",
    "style_axis",
    "ConverterTab",
    "SignalLabTab",
    "NoiseFilterTab",
    "MicDecodeTab",
]
