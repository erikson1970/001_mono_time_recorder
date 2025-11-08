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
    PAUSE_RESUME = 'p'
    BREAK_WITH_GAP = 'b'
    BREAK_IMMEDIATE = 'B'
    SHOW_HISTOGRAM = 'h'
    TOGGLE_SPECTROGRAM = 's'


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


class StdinReader(threading.Thread):
    """
    Background thread that reads from stdin and enqueues commands.

    Attempts to use readchar for single-key input (no Enter required).
    Falls back to line-based input if readchar is unavailable.
    """

    def __init__(self, cmd_queue: CommandQueue):
        super().__init__(daemon=True, name="StdinReader")
        self.cmd_queue = cmd_queue
        self._stop_event = threading.Event()
        self._use_readchar = False

        # Try to import readchar
        try:
            import readchar
            self._readchar = readchar
            self._use_readchar = True
        except ImportError:
            self._readchar = None

    def stop(self):
        """Request thread to stop."""
        self._stop_event.set()

    def _read_single_char(self) -> Optional[str]:
        """Read a single character using readchar."""
        try:
            ch = self._readchar.readchar()
            return ch
        except Exception:
            return None

    def _read_line(self) -> Optional[str]:
        """Read a line from stdin."""
        try:
            line = sys.stdin.readline()
            if line:
                return line.strip()
            return None
        except Exception:
            return None

    def _parse_command(self, input_str: str) -> Optional[Command]:
        """Parse input string to Command enum."""
        if not input_str:
            return None

        # Take first character
        ch = input_str[0].lower() if input_str else ''

        # Map to command
        command_map = {
            'q': Command.QUIT,
            'p': Command.PAUSE_RESUME,
            'b': Command.BREAK_WITH_GAP,
            'B': Command.BREAK_IMMEDIATE,
            'h': Command.SHOW_HISTOGRAM,
            's': Command.TOGGLE_SPECTROGRAM,
        }

        # Handle uppercase B specially
        if input_str[0] == 'B':
            return Command.BREAK_IMMEDIATE

        return command_map.get(ch)

    def run(self):
        """Main loop: read from stdin and enqueue commands."""
        if self._use_readchar:
            self._run_readchar()
        else:
            self._run_line_based()

    def _run_readchar(self):
        """Run loop with single-character input (readchar)."""
        while not self._stop_event.is_set():
            ch = self._read_single_char()
            if ch:
                cmd = self._parse_command(ch)
                if cmd:
                    self.cmd_queue.put(cmd, block=False)

    def _run_line_based(self):
        """Run loop with line-based input (fallback)."""
        while not self._stop_event.is_set():
            line = self._read_line()
            if line:
                cmd = self._parse_command(line)
                if cmd:
                    self.cmd_queue.put(cmd, block=False)


def print_hotkey_help():
    """Print available hotkeys to console."""
    help_text = """
╔══════════════════════════════════════════════════════════╗
║                   HOTKEY COMMANDS                        ║
╠══════════════════════════════════════════════════════════╣
║  q  │ Quit - Stop recording and exit gracefully          ║
║  p  │ Pause/Resume - Toggle recording pause              ║
║  b  │ Break with gap - Cut segment at next good gap      ║
║  B  │ Break immediate - Force segment cut now            ║
║  h  │ Histogram - Show current gap histogram (if enabled)║
║  s  │ Toggle spectrogram - Show/hide spectrogram         ║
╚══════════════════════════════════════════════════════════╝
"""
    print(help_text, flush=True)
