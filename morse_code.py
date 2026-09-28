"""
Morse code dictionary and text conversion helpers.
Converts between plain text and Morse code strings (. and -).
"""

# Standard International Morse Code mapping
MORSE_CODE_DICT = {
    'A': '.-',    'B': '-...',  'C': '-.-.',  'D': '-..',   'E': '.',
    'F': '..-.',  'G': '--.',   'H': '....',  'I': '..',    'J': '.---',
    'K': '-.-',   'L': '.-..',  'M': '--',    'N': '-.',    'O': '---',
    'P': '.--.',  'Q': '--.-',  'R': '.-.',   'S': '...',   'T': '-',
    'U': '..-',   'V': '...-',  'W': '.--',   'X': '-..-',  'Y': '-.--',
    'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.',
    '!': '-.-.--', '/': '-..-.', '(': '-.--.',  ')': '-.--.-',
    '&': '.-...',  ':': '---...', ';': '-.-.-.', '=': '-...-',
    '+': '.-.-.',  '-': '-....-', '_': '..--.-', '"': '.-..-.',
    '$': '...-..-', '@': '.--.-.',
}

# Reverse lookup: Morse code -> character
REVERSE_MORSE_CODE_DICT = {v: k for k, v in MORSE_CODE_DICT.items()}

# Separators: single space between letters, slash between words
LETTER_SEP = ' '
WORD_SEP = ' / '


def text_to_morse(text: str) -> str:
    """Convert plain text to Morse code. Unknown characters become '?'."""
    text = text.strip().upper()
    if not text:
        return ''

    words = text.split()
    morse_words = []
    for word in words:
        if not word:
            continue
        letters = [MORSE_CODE_DICT.get(ch, '?') for ch in word]
        morse_words.append(LETTER_SEP.join(letters))
    return WORD_SEP.join(morse_words)


def morse_to_text(morse: str) -> str:
    """Convert a Morse code string back into readable plain text."""
    morse = morse.strip()
    if not morse:
        return ''

    # Normalize different slash styles into standard ' / '
    normalized = morse.replace('|', '/').replace('/', ' / ')
    words = normalized.split('/')

    out_words = []
    for word in words:
        letters = word.strip().split()
        if not letters:
            continue
        chars = [REVERSE_MORSE_CODE_DICT.get(code, '?') for code in letters]
        out_words.append(''.join(chars))
    return ' '.join(out_words)


def is_valid_morse(morse: str) -> bool:
    """Check if text contains only valid Morse characters."""
    return all(ch in '.-/| \t\n' for ch in morse)


def morse_table_rows():
    """Return character and Morse pairs for the UI reference table."""
    order = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?\'!/()&:;=+-_"$@'
    return [(ch, MORSE_CODE_DICT[ch]) for ch in order if ch in MORSE_CODE_DICT]


def estimate_transmission_units(morse_code: str) -> int:
    """
    Calculate total timing units using the standard PARIS formula.
    Dot = 1 unit, Dash = 3 units, space between symbols = 1 unit,
    space between letters = 3 units, space between words = 7 units.
    """
    words = [w for w in morse_code.strip().split(' / ') if w]
    total_units = 0
    for wi, word in enumerate(words):
        letters = [l for l in word.strip().split(' ') if l]
        for li, letter in enumerate(letters):
            for si, symbol in enumerate(letter):
                if symbol == '.':
                    total_units += 1
                elif symbol == '-':
                    total_units += 3
                if si < len(letter) - 1:
                    total_units += 1  # gap between dots/dashes in same letter
            if li < len(letters) - 1:
                total_units += 3      # gap between letters
        if wi < len(words) - 1:
            total_units += 7          # gap between words
    return total_units


def estimate_duration_seconds(morse_code: str, wpm: float = 20.0) -> float:
    """Estimate transmission duration in seconds at a given Words Per Minute (WPM)."""
    units = estimate_transmission_units(morse_code)
    unit_dur = 1.2 / max(wpm, 1e-6)
    return units * unit_dur