#!/usr/bin/env python3
"""
Session timing infrastructure for recording with pause support.

Tracks active recording time (excludes pauses) vs total elapsed time.
"""

import threading
import time
from typing import Optional


class SessionTimer:
    """
    Manages session timing with pause support.

    Separates active recording time from total elapsed time, allowing pauses
    to extend the max runtime without counting against recording limits.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._active_time: float = 0.0  # Accumulated active recording time
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused
        self._last_resume_time: Optional[float] = None
        self._is_paused: bool = False

    def start(self):
        """Start the timer."""
        with self._lock:
            now = time.monotonic()
            self._start_time = now
            self._last_resume_time = now
            self._active_time = 0.0
            self._is_paused = False
            self._pause_event.set()

    def pause(self):
        """Pause the timer. Accumulated active time is preserved."""
        with self._lock:
            if self._is_paused:
                return  # Already paused

            now = time.monotonic()
            if self._last_resume_time is not None:
                # Accumulate the time since last resume
                self._active_time += (now - self._last_resume_time)

            self._is_paused = True
            self._pause_event.clear()

    def resume(self):
        """Resume the timer. Starts accumulating active time again."""
        with self._lock:
            if not self._is_paused:
                return  # Already running

            self._is_paused = False
            self._last_resume_time = time.monotonic()
            self._pause_event.set()

    def is_paused(self) -> bool:
        """Check if currently paused."""
        with self._lock:
            return self._is_paused

    def get_active_time(self) -> float:
        """
        Get accumulated active recording time in seconds.

        This excludes time spent in paused state.
        """
        with self._lock:
            active = self._active_time
            if not self._is_paused and self._last_resume_time is not None:
                # Add time since last resume
                active += (time.monotonic() - self._last_resume_time)
            return active

    def get_total_time(self) -> float:
        """
        Get total elapsed time in seconds since start.

        This includes time spent in paused state.
        """
        with self._lock:
            if self._start_time is None:
                return 0.0
            return time.monotonic() - self._start_time

    def wait_if_paused(self, timeout: Optional[float] = None) -> bool:
        """
        Block if paused, return when resumed.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            True if not paused or was resumed, False if timeout occurred while paused.
        """
        return self._pause_event.wait(timeout=timeout)

    def add_active_time(self, seconds: float):
        """
        Manually add time to active counter.

        Useful for incorporating time tracked externally (e.g., audio callback).
        """
        with self._lock:
            self._active_time += seconds

    def reset(self):
        """Reset all timers."""
        with self._lock:
            self._start_time = None
            self._active_time = 0.0
            self._last_resume_time = None
            self._is_paused = False
            self._pause_event.set()


class SegmentTimer:
    """
    Timer for individual segment tracking.

    Simpler than SessionTimer - just tracks elapsed time for current segment.
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._elapsed: float = 0.0

    def start(self):
        """Start segment timer."""
        self._start_time = time.monotonic()
        self._elapsed = 0.0

    def get_elapsed(self) -> float:
        """Get elapsed time for current segment in seconds."""
        if self._start_time is None:
            return self._elapsed
        return self._elapsed + (time.monotonic() - self._start_time)

    def add_time(self, seconds: float):
        """Add time to elapsed counter."""
        self._elapsed += seconds

    def reset(self):
        """Reset segment timer."""
        self._start_time = time.monotonic()
        self._elapsed = 0.0
