#!/usr/bin/env python3
"""
Command infrastructure for interactive control during recording.

Provides thread-safe command queue and stdin reader for hotkey support.
"""

import queue
import sys
import threading
from enum import Enum
from typing import Optional


class Command(Enum):
    """Available commands during recording."""
    QUIT = 'q'
    RESUME = ''  # Empty string = just hit ENTER
    BREAK_WITH_GAP = 'b'
    BREAK_IMMEDIATE = 'B'
    SHOW_HISTOGRAM = 'h'
    TOGGLE_SPECTROGRAM = 's'
    TOGGLE_MONITOR = 'm'  # Monitor mode: spectrogram on, recording paused
    SHOW_HELP = '?'
    ENTER_COMMAND_MODE = '\x03'  # Ctrl+C


class CommandQueue:
    """Thread-safe command queue for inter-thread communication."""

    def __init__(self, maxsize: int = 100):
        self._queue: queue.Queue[Command] = queue.Queue(maxsize=maxsize)

    def put(self, cmd: Command, block: bool = True, timeout: Optional[float] = None):
        """Put a command in the queue."""
        try:
            self._queue.put(cmd, block=block, timeout=timeout)
        except queue.Full:
            # Silently drop commands if queue is full (prevents blocking stdin reader)
            pass

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Command]:
        """Get a command from the queue. Returns None on timeout."""
        try:
            return self._queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    def clear(self):
        """Clear all pending commands."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


class CommandMode:
    """
    Command mode state - entered via Ctrl+C, exited via ENTER or quit.

    When in command mode:
    - Recording is paused
    - User can enter single-key commands
    - ENTER resumes recording
    - 'q' or second Ctrl+C quits
    """

    def __init__(self):
        self._in_command_mode = False
        self._lock = threading.Lock()

    def enter(self):
        """Enter command mode."""
        with self._lock:
            self._in_command_mode = True

    def exit(self):
        """Exit command mode."""
        with self._lock:
            self._in_command_mode = False

    def is_active(self) -> bool:
        """Check if in command mode."""
        with self._lock:
            return self._in_command_mode


# No background stdin reader needed - Ctrl+C is handled by signal handler
# and command reading happens synchronously in main loop when in command mode


def print_command_menu_help():
    """Print command menu help (shown when entering command mode)."""
    help_text = """
╔═══════════════════════════════════════════════════════════╗
║                    COMMAND MENU                           ║
╠═══════════════════════════════════════════════════════════╣
║  ?      │ Show this help                                  ║
║  ENTER  │ Resume recording                                ║
║  q      │ Quit - Stop recording and exit gracefully       ║
║  b      │ Break with gap - Cut segment at next good gap   ║
║  B      │ Break immediate - Force segment cut now         ║
║  h      │ Histogram - Show current gap histogram          ║
║  s      │ Toggle spectrogram - Show/hide spectrogram      ║
║  m      │ Monitor mode - Toggle spectrogram while paused  ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(help_text, flush=True)


def print_startup_help():
    """Print startup help message."""
    help_text = """
╔═══════════════════════════════════════════════════════════╗
║  Press Ctrl+C to pause and enter command menu             ║
║  In command menu: ? for help, ENTER to resume, q to quit  ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(help_text, flush=True)


def parse_command(input_str: str) -> Optional[Command]:
    """Parse input string to Command enum."""
    if not input_str or input_str == '\n':
        return Command.RESUME

    # Take first character
    ch = input_str[0] if input_str else ''

    # Map to command
    command_map = {
        'q': Command.QUIT,
        'b': Command.BREAK_WITH_GAP,
        'B': Command.BREAK_IMMEDIATE,
        'h': Command.SHOW_HISTOGRAM,
        's': Command.TOGGLE_SPECTROGRAM,
        'm': Command.TOGGLE_MONITOR,
        '?': Command.SHOW_HELP,
        '': Command.RESUME,
    }

    # Handle uppercase B specially
    if ch == 'B':
        return Command.BREAK_IMMEDIATE

    return command_map.get(ch.lower())


def read_command() -> Optional[Command]:
    """
    Read a single command from stdin (blocking).

    Returns None if interrupted or EOF.
    """
    try:
        line = input()  # Blocking read with prompt support
        return parse_command(line)
    except (EOFError, KeyboardInterrupt):
        # Second Ctrl+C while in command mode = quit
        return Command.QUIT
    except Exception:
        return None
