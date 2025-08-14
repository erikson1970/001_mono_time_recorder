#!/usr/bin/env python3
"""
Continuous mono recorder with silence-aware segmentation and async LAME encoding.

Behavior
--------
- Record mono audio into an in-memory buffer.
- After H minutes elapse (default 60), begin watching for a "pause" of N seconds
  (default 2.0) determined by an RMS threshold (default 0.01, ~ -40 dBFS).
- On the first such pause, dump the *previous* buffer to WAV, and continue
  recording seamlessly into a fresh buffer. A background worker writes the WAV
  and launches `lame --preset mw-eu` to create an MP3.

Dependencies
------------
pip install sounddevice soundfile numpy
System: LAME must be installed (command: `lame`)

Example
-------
python segment_recorder.py --minutes 30 --pause-seconds 1.5 --sr 48000 --silence-threshold 0.008
"""


import argparse
import csv
import datetime as dt
import math
import re
import queue
import signal
import subprocess
import sys
import threading
import time
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Prepare the jet colormap
JET_CMAP = cm.get_cmap("jet")


def ts_to_date(ts: float) -> str:  # DDMMYY
    return dt.datetime.fromtimestamp(ts).strftime("%y%m%d")


def ts_to_time(ts: float) -> str:  # HHMMSS
    return dt.datetime.fromtimestamp(ts).strftime("%H%M%S")


def secs_to_hhmmss(s: float) -> str:
    total = int(round(s))
    h = total // 3600
    m = (total % 3600) // 60
    sec = total % 60
    return f"{h:02d}{m:02d}{sec:02d}"


def ensure_file_exists(path: Path, header: List[str] | None = None):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if header:
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)


# Define a function to get the terminal dimensions
def get_terminal_size():
    return shutil.get_terminal_size()


# Get initial terminal size
TERMINAL_WIDTH, TERMINAL_HEIGHT = get_terminal_size()
RIGHT_HALF_START_COL = TERMINAL_WIDTH // 2


# --- Color Gradient Mapping ---
# Create a list of ANSI color codes for a cool-to-warm gradient
# These are ANSI 256-color codes for the background
def generate_color_gradient(num_colors):
    """Generates a gradient of 256-color ANSI background codes."""
    colors = []
    # Dark blue -> blue
    for i in range(num_colors // 4):
        colors.append(f"\033[48;5;{232 + i}m")
    # Blue -> cyan
    for i in range(num_colors // 4):
        colors.append(f"\033[48;5;{17 + i}m")
    # Cyan -> green
    for i in range(num_colors // 4):
        colors.append(f"\033[48;5;{46 + i}m")
    # Green -> yellow -> red
    for i in range(num_colors // 4):
        colors.append(f"\033[48;5;{11 + i}m")
    return colors


COLOR_GRADIENT = generate_color_gradient(256)
GRADIENT_SIZE = len(COLOR_GRADIENT)


def map_amplitude_to_color(amplitude: float, map: str | None = None) -> str:
    """Maps a normalized amplitude (0-1) to an ANSI color code from the gradient."""
    # Clip amplitude to stay within the 0-1 range
    clipped_amplitude = np.clip(amplitude, 0, 1)
    if map is None:
        # Scale the amplitude to the size of the color gradient
        color_index = int(clipped_amplitude * (GRADIENT_SIZE - 1))
        return COLOR_GRADIENT[color_index]
    elif map == "jet":
        r, g, b, _ = JET_CMAP(clipped_amplitude)  # returns floats 0–1
        R, G, B = int(r * 255), int(g * 255), int(b * 255)
        return f"\033[48;2;{R};{G};{B}m"  # ANSI truecolor background
    else:
        raise ValueError(f"Unknown color map: {map}")


# --- Configuration ---
# # Set the desired audio sample rate
# SAMPLE_RATE = 44100
# # Define the size of each audio block to process
# BLOCK_SIZE = 1024
# Set the number of frequency bins to display
NUM_BINS = 80
# Set a scaling factor for the spectrogram intensity
INTENSITY_SCALE = 0.8
# Set the refresh rate of the spectrogram
REFRESH_RATE = 20

# --- ANSI Escape Sequences ---
# Escape code to reset all attributes
RESET = "\033[0m"
# Escape code to move the cursor to a specific position (row;column)
MOVE_CURSOR = "\033[{row};{col}H"
# Escape code to erase the line from the cursor
ERASE_LINE = "\033[K"
# Escape code to hide the cursor
HIDE_CURSOR = "\033[?25l"
# Escape code to show the cursor
SHOW_CURSOR = "\033[?25h"


class SessionIO:
    """Single-threaded writer for WAV/CSV/M3U and async MP3 encodes."""

    def __init__(
        self,
        outdir: Path,
        base_pattern: str,
        overall_start_ts: float,
        delete_wav: bool,
        sr: int,
    ):
        self.outdir = outdir
        self.base_pattern = base_pattern
        self.overall_start_ts = overall_start_ts
        self.delete_wav = delete_wav
        self.sr = sr

        m3u_path = base_pattern
        for i in ["C", "D", "T", "d", "t", "E", "e", "L", "l"]:
            m3u_path = m3u_path.replace(f"~{i}", f"")
        m3u_path = re.sub(re.compile(r"_+[^_]*$"), "_", m3u_path)

        # Sidecar file paths, include overall start tag for uniqueness
        session_tag = f"{ts_to_date(overall_start_ts)}_{ts_to_time(overall_start_ts)}"
        self.csv_path = outdir / f"{m3u_path}_segments_{session_tag}.csv".replace(
            "__", "_"
        )
        self.m3u_mp3_path = outdir / f"{m3u_path}_pl_mp3_{session_tag}.m3u".replace(
            "__", "_"
        )
        self.m3u_wav_path = outdir / f"{m3u_path}_pl_wav_{session_tag}.m3u".replace(
            "__", "_"
        )

        ensure_file_exists(
            self.csv_path,
            header=[
                "index",
                "t0_epoch",
                "t1_epoch",
                "duration_sec",
                "length_hhmmss",
                "wav_file",
                "mp3_file",
                "sr",
            ],
        )
        # Create empty playlists if needed
        self.m3u_mp3_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.m3u_mp3_path.exists():
            self.m3u_mp3_path.write_text("#EXTM3U\n")
        if not delete_wav:
            if not self.m3u_wav_path.exists():
                self.m3u_wav_path.write_text("#EXTM3U\n")

    def _format_base(
        self, seg_idx: int, seg_t0: float, seg_t1: float, seg_len_sec: float
    ) -> str:
        # Tokens:
        # ~D overall date, ~T overall time
        # ~d segment date, ~t segment time
        # ~E segment end date, ~e segment end time
        # ~C counter, ~L length HHMMSS, ~l length SSSS
        overall_D = ts_to_date(self.overall_start_ts)
        overall_T = ts_to_time(self.overall_start_ts)
        seg_d = ts_to_date(seg_t0)
        seg_t = ts_to_time(seg_t0)
        seg_end_d = ts_to_date(seg_t1)
        seg_end_t = ts_to_time(seg_t1)
        L = secs_to_hhmmss(seg_len_sec)
        s = self.base_pattern
        s = s.replace("~D", overall_D).replace("~T", overall_T)
        s = s.replace("~d", seg_d).replace("~t", seg_t)
        s = s.replace("~E", seg_end_d).replace("~e", seg_end_t)
        s = s.replace("~C", f"{seg_idx:03d}")  # Zero-padded counter
        s = s.replace("~L", L)
        s = s.replace("~l", f"{int(seg_len_sec+0.5):04d}")  # Length in seconds

        # Simple filesystem safety
        s = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)
        return s

    def write_segment_and_spawn_encode(
        self, samples: np.ndarray, t0: float, t1: float, seg_idx: int
    ):
        seg_len = max(0.0, (len(samples) / self.sr))  # or (t1 - t0)
        base = self._format_base(seg_idx, t0, t1, seg_len)
        wav_path = self.outdir / f"{base}.wav"
        mp3_path = self.outdir / f"{base}.mp3"

        # Write WAV
        sf.write(str(wav_path), samples, self.sr, subtype="PCM_16")

        # Update CSV
        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    seg_idx,
                    f"{t0:.6f}",
                    f"{t1:.6f}",
                    f"{seg_len:.3f}",
                    secs_to_hhmmss(seg_len),
                    wav_path.name,
                    mp3_path.name,
                    self.sr,
                ]
            )

        # Update playlists (relative basenames keep files portable)
        with self.m3u_mp3_path.open("a") as f:
            f.write(f"{mp3_path.name}\n")
        if not self.delete_wav:
            with self.m3u_wav_path.open("a") as f:
                f.write(f"{wav_path.name}\n")

        # Spawn LAME encode
        cmd = ["lame", "--preset", "mw-eu", "--quiet", str(wav_path), str(mp3_path)]
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print(
                "[WARN] 'lame' not found in PATH; skipping MP3 encode.", file=sys.stderr
            )

        # Optionally delete WAV immediately after spawning encode
        if self.delete_wav:
            time.sleep(0.2)
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass


class SegmentWriter(threading.Thread):
    """Background worker driven by a queue from the audio callback."""

    def __init__(
        self,
        task_q: "queue.Queue[Tuple[np.ndarray, int, float, float, int]]",
        io: SessionIO,
    ):
        super().__init__(daemon=True)
        self.task_q = task_q
        self.io = io
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                samples, sr, t0, t1, seg_idx = self.task_q.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                self.io.write_segment_and_spawn_encode(samples, t0, t1, seg_idx)
            except Exception as e:
                print(f"[SegmentWriter] Error: {e}", file=sys.stderr)
            finally:
                self.task_q.task_done()


class ContinuousRecorder:
    def __init__(
        self,
        minutes: float = 60.0,
        pause_seconds: float = 2.0,
        sr: int = 48000,
        blocksize: int = 2048,
        device: str | int | None = None,
        silence_rms_threshold: float = 0.01,
        output_dir: Path = Path("./recordings"),
        delete_wav_after_encode: bool = False,
        max_pause_minutes: float | None = None,
        base_filename: str = "segment_~C_~d~t_~l_secs",
        max_total_minutes: float | None = None,
        spectrogram_frequency: float = 0.6,
        spectrogram_colors: str | None = None,
        start_seg_number: int = 0,
    ):
        self._epoch0 = None  # UNIX epoch baseline
        self._mono0 = None  # perf_counter baseline
        self._pa0 = None  # PortAudio time baseline (if available)
        self.start_seg_number = start_seg_number
        self.minutes = minutes
        self.pause_seconds = pause_seconds
        self.sr = sr
        self.blocksize = blocksize
        self.spec_freq = spectrogram_frequency
        self.device = device
        self.threshold = silence_rms_threshold
        self.output_dir = output_dir
        self.delete_wav_after_encode = delete_wav_after_encode
        self.max_pause_minutes = (
            math.ceil(0.1 * self.minutes)
            if max_pause_minutes is None
            else max_pause_minutes
        )
        self.base_filename = base_filename
        self.max_total_minutes = (
            math.ceil(10 * self.minutes)
            if max_total_minutes is None
            else max_total_minutes
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._active_frames: List[np.ndarray] = []
        self._active_start_ts: float | None = None

        self._elapsed_since_start = 0.0
        self._silence_run = 0.0
        self._total_elapsed = 0.0
        self.spec_colors = spectrogram_colors

        self._eligible_for_split = False
        self._pause_watch_elapsed = 0.0
        self._pause_notice_printed = False
        self._pause_expire_notice_printed = False

        self._overall_start_ts = time.time()
        self._segment_index = 0

        # IO & writer
        self._io = SessionIO(
            outdir=self.output_dir,
            base_pattern=self.base_filename,
            overall_start_ts=self._overall_start_ts,
            delete_wav=self.delete_wav_after_encode,
            sr=self.sr,
        )
        self._task_q: "queue.Queue[Tuple[np.ndarray, int, float, float, int]]" = (
            queue.Queue()
        )
        self._writer = SegmentWriter(self._task_q, self._io)

        self._stop_event = threading.Event()
        self._request_stop = False
        self._stream: sd.InputStream | None = None

    # --- add helper ---
    def _epoch_now_from_callback(self, time_info):
        import math, time

        # Grab current monotonic
        now_mono = time.perf_counter()

        # First-call init if needed
        if self._epoch0 is None:
            self._epoch0 = time.time()
            self._mono0 = now_mono
            t_pa = getattr(time_info, "inputBufferAdcTime", None)
            self._pa0 = (
                float(t_pa)
                if (t_pa is not None and math.isfinite(t_pa) and t_pa > 0)
                else None
            )
            # Return epoch0 for this very first block
            return self._epoch0

        # Prefer PortAudio time if we have a stable baseline
        t_pa = getattr(time_info, "inputBufferAdcTime", None)
        if (
            self._pa0 is not None
            and t_pa is not None
            and math.isfinite(t_pa)
            and t_pa > 0
        ):
            return self._epoch0 + (float(t_pa) - self._pa0)

        # Fallback: monotonic delta from our baseline
        return self._epoch0 + (now_mono - self._mono0)

    def _reset_active_buffer(self, t_start: float | None = None):
        with self._lock:
            self._active_frames = []
            self._active_start_ts = t_start

    def _append_block(self, block: np.ndarray, block_start_ts: float):
        with self._lock:
            if self._active_start_ts is None:
                self._active_start_ts = block_start_ts
            self._active_frames.append(block)

    def _collect_active(self) -> Tuple[np.ndarray, float, float] | None:
        with self._lock:
            if not self._active_frames or self._active_start_ts is None:
                return None
            samples = np.concatenate(self._active_frames, axis=0)
            t0 = self._active_start_ts
            t1 = t0 + len(samples) / self.sr
            return samples, t0, t1

    def _rms(self, x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))

    def _cut_and_enqueue(self, next_buffer_start_ts: float, reason: str):
        collected = self._collect_active()

        # Swap immediately to avoid gaps
        self._reset_active_buffer(t_start=next_buffer_start_ts)
        self._elapsed_since_start = 0.0
        self._silence_run = 0.0
        self._eligible_for_split = False
        self._pause_watch_elapsed = 0.0
        self._pause_notice_printed = False
        self._pause_expire_notice_printed = False

        if collected:
            samples, t0, t1 = collected
            seg_idx = self._segment_index
            self._segment_index += 1
            self._task_q.put(
                (samples.copy(), self.sr, t0, t1, seg_idx + self.start_seg_number)
            )
            print(
                ("\n" if self.spec_freq > 0.0 else "")
                + f"[INFO] Segment cut ({reason}). idx={seg_idx+self.start_seg_number:04d}  samples:{len(samples)}",
                flush=True,
            )

    def _force_finalize_current(self):
        """Flush current active buffer (if any) as a final segment and request stop."""
        collected = self._collect_active()
        if collected:
            samples, t0, t1 = collected
            seg_idx = self._segment_index
            self._segment_index += 1
            self._task_q.put((samples.copy(), self.sr, t0, t1, seg_idx))
            print(
                f"[INFO] Finalizing current segment and stopping. idx={seg_idx}",
                flush=True,
            )
        self._request_stop = True

    def show_spectrogram(self, indata, N: int = 1):
        if N <= 0:
            return
        # Calculate the FFT of the audio data
        fft_data = np.fft.rfft(indata, n=N)
        # Compute the magnitude of the FFT result
        fft_magnitude = np.abs(fft_data)

        # Scale and normalize the magnitude for visualization
        fft_magnitude = np.log10(fft_magnitude * INTENSITY_SCALE + 1e-12)
        fft_magnitude = np.clip(fft_magnitude, 0, 1)

        # Downsample the frequency data to fit the number of display bins
        display_bins = np.logspace(
            np.log10(1),
            np.log10(len(fft_magnitude)),
            num=NUM_BINS,
            endpoint=False,
            dtype=int,
        )
        downsampled_data = fft_magnitude[display_bins]

        # Get the current terminal size in case it changed
        current_width, current_height = get_terminal_size()
        right_half_start_col = current_width // 2

        # Construct the spectrogram line for the right half of the screen
        spectrogram_line = ""
        # Ensure the line fits the right half of the terminal
        display_width = current_width - right_half_start_col
        for bin_amplitude in downsampled_data[:display_width]:
            color_code = map_amplitude_to_color(bin_amplitude, map=self.spec_colors)
            spectrogram_line += (
                f"{color_code} "  # Use a space to create the colored block
            )
        spectrogram_line += RESET

        # --- Rendering to the Console ---
        # Move the cursor to the bottom line, starting in the right half
        sys.stdout.write(
            MOVE_CURSOR.format(row=current_height, col=right_half_start_col)
        )
        # Erase the old content on the line
        # sys.stdout.write(ERASE_LINE)
        # Write the new spectrogram line
        sys.stdout.write(spectrogram_line)
        # Flush the output to ensure it's displayed immediately
        sys.stdout.flush()

    def _on_audio(self, indata, frames, time_info, status):
        try:
            if status:
                print(f"[AudioStatus] {status}", file=sys.stderr)

            # mono = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
            mono = np.array(indata[:, 0], dtype=np.float32, copy=True)

            t_candidate = getattr(time_info, "inputBufferAdcTime", None)
            # block_start_ts = (
            #     float(t_candidate)
            #     if (t_candidate and math.isfinite(t_candidate) and t_candidate > 0)
            #     else time.monotonic()
            # )
            block_start_ts = self._epoch_now_from_callback(
                time_info
            )  # <-- epoch-seconds

            self._append_block(mono, block_start_ts)

            block_duration = frames / self.sr
            self._elapsed_since_start += block_duration
            self._total_elapsed += block_duration

            # Max total recording time logic
            if (
                self._total_elapsed >= self.max_total_minutes * 60.0
                and not self._request_stop
            ):
                print(
                    ("\n" if self.spec_freq > 0.0 else "")
                    + f"[INFO] Max total time reached ({self.max_total_minutes} min). Forcing split and exit.",
                    flush=True,
                )
                next_start = block_start_ts + block_duration
                # Cut if we have enough to form a segment; otherwise just finalize whatever is there
                self._cut_and_enqueue(next_start, reason="max total time")
                self._request_stop = True
                return

            # After H minutes, start pause watching
            if (
                not self._eligible_for_split
                and self._elapsed_since_start >= self.minutes * 60.0
            ):
                self._eligible_for_split = True
                self._pause_watch_elapsed = 0.0
                if not self._pause_notice_printed:
                    print(
                        ("\n" if self.spec_freq > 0.0 else "")
                        + f"[INFO] Segment length reached {self.minutes:.2f} min; looking for a pause "
                        f"(≥ {self.pause_seconds:.2f}s, thr={self.threshold}).",
                        flush=True,
                    )
                    print(
                        ("\n" if self.spec_freq > 0.0 else "")
                        + f"[INFO] Pause search window: up to {self.max_pause_minutes} minute(s) before forced cut.",
                        flush=True,
                    )
                    self._pause_notice_printed = True

            if self.spec_freq > 0.0 and np.random.rand() < self.spec_freq:
                NN = min(len(indata[:, 0]), 2048)
                if NN > 0:
                    # Show spectrogram in the console
                    self.show_spectrogram(indata[:NN, 0].copy(), N=NN)

            # If eligible, check pause or window expiry
            if self._eligible_for_split:
                block_rms = self._rms(mono)
                if block_rms < self.threshold:
                    self._silence_run += block_duration
                else:
                    self._silence_run = 0.0

                self._pause_watch_elapsed += block_duration

                if self._silence_run >= self.pause_seconds:
                    next_start = block_start_ts + block_duration
                    self._cut_and_enqueue(next_start, reason="pause detected")
                    return

                if self._pause_watch_elapsed >= self.max_pause_minutes * 60.0:
                    if not self._pause_expire_notice_printed:
                        print(
                            ("\n" if self.spec_freq > 0.0 else "")
                            + "[INFO] Pause search window expired—forcing segment cut.",
                            flush=True,
                        )
                        self._pause_expire_notice_printed = True
                    next_start = block_start_ts + block_duration
                    self._cut_and_enqueue(next_start, reason="pause window expired")
                    return

            # If stop requested (e.g., after max total time), finalize ASAP
            if self._request_stop:
                self._force_finalize_current()

        except Exception as e:
            print(f"[CallbackError] {e}", file=sys.stderr)

    def start(self):
        print("[INFO] Starting writer thread…")
        self._writer.start()
        print("[INFO] Opening input stream…")
        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._on_audio,
            device=self.device,
        )
        self._reset_active_buffer()
        self._stream.start()
        print("[INFO] Recording… Press Ctrl+C to stop.")
        print(
            f"[INFO] Config: segment={self.minutes} min, pause>={self.pause_seconds:.2f}s, "
            f"max-pause-window={self.max_pause_minutes} min, sr={self.sr}, blocksize={self.blocksize}, thr={self.threshold}"
        )
        print(f"[INFO] Output dir: {self.output_dir}")
        print(f"[INFO] CSV: {self._io.csv_path.name}")
        print(f"[INFO] MP3 M3U: {self._io.m3u_mp3_path.name}")
        if not self.delete_wav_after_encode:
            print(f"[INFO] WAV M3U: {self._io.m3u_wav_path.name}")

    def stop(self):
        self._stop_event.set()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

        # Flush remaining buffer (if any)
        final = self._collect_active()
        if final:
            samples, t0, t1 = final
            seg_idx = self._segment_index
            self._segment_index += 1
            self._task_q.put((samples, self.sr, t0, t1, seg_idx))
            print(f"[INFO] Final segment flushed. idx={seg_idx}", flush=True)

        try:
            self._task_q.join()
        except Exception:
            pass
        self._writer.stop()

    def wait_forever(self):
        try:
            while not self._stop_event.is_set():
                if self._request_stop:
                    # Give the queue a moment to drain, then stop
                    time.sleep(0.5)
                    self.stop()
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping…")
            self.stop()


def list_devices_and_exit():
    print(sd.query_devices())
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(
        description="Silence-aware continuous recorder with async LAME encoding + CSV + M3U."
    )
    ap.add_argument(
        "--minutes",
        type=float,
        default=60.0,
        help="Target segment length in minutes before waiting for a pause (default: 60)",
    )
    ap.add_argument(
        "--pause-seconds",
        type=float,
        default=2.0,
        help="Pause length (seconds) that triggers a segment cut once eligible (default: 2.0). Set 0 to split immediately at H.",
    )
    ap.add_argument(
        "--sr", type=int, default=48000, help="Sample rate (default: 48000)"
    )
    ap.add_argument(
        "--blocksize",
        type=int,
        default=2048,
        help="Frames per block callback (default: 2048)",
    )
    ap.add_argument(
        "--start-seg-number",
        type=int,
        default=0,
        help="Starting segment number for file naming (default: 0)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Input device name or index (default: system default)",
    )
    ap.add_argument(
        "--silence-threshold",
        type=float,
        default=0.01,
        help="RMS threshold for silence in [-1,1] float (default: 0.01 ≈ -40 dBFS)",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("./recordings"),
        help="Output folder (default: ./recordings)",
    )
    ap.add_argument(
        "--delete-wav-after-encode",
        action="store_true",
        help="Delete WAV after spawning MP3 encode",
    )

    ap.add_argument(
        "--specrogram-frequency",
        type=float,
        default=0.6,
        help="percent of time to show spectrogram in the console (default: 0.6)",
    )

    ap.add_argument(
        "--specrogram-colors",
        type=str,
        default=None,
        help="Color map for spectrogram: 'jet' for truecolor, or None for ANSI gradient (default: None)",
    )

    ap.add_argument(
        "--list-devices", action="store_true", help="List audio devices and exit"
    )
    ap.add_argument(
        "--max-pause-minutes",
        type=float,
        default=None,
        help="Max time to search for a pause after the segment length is reached before forcing a split. Default is ceil(10%% of --minutes).",
    )
    ap.add_argument(
        "--base-filename",
        type=str,
        default="segment_~C_~d~t_~l_secs",
        help="Base filename pattern with tokens: ~D ~T (overall YYMMDD/HHMMSS), "
        "~d ~t (segment start YYMMDD/HHMMSS),~E ~e (segment end YYMMDD/HHMMSS),"
        " ~C (counter), ~L (segment length HHMMSS).",
    )
    ap.add_argument(
        "--max-total-minutes",
        type=float,
        default=None,
        help="Max total recording time before exit (minutes). Default ceil(10×--minutes).",
    )
    args = ap.parse_args()

    if args.list_devices:
        list_devices_and_exit()

    rec = ContinuousRecorder(
        minutes=args.minutes,
        pause_seconds=args.pause_seconds,
        sr=args.sr,
        blocksize=args.blocksize,
        device=args.device,
        silence_rms_threshold=args.silence_threshold,
        output_dir=args.outdir,
        delete_wav_after_encode=args.delete_wav_after_encode,
        max_pause_minutes=args.max_pause_minutes,
        base_filename=args.base_filename,
        max_total_minutes=args.max_total_minutes,
        spectrogram_frequency=args.specrogram_frequency,
        spectrogram_colors=args.specrogram_colors,
        start_seg_number=args.start_seg_number,
    )

    def _handle_sig(signum, frame):
        print("\n[INFO] Signal received, shutting down…")
        rec.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    rec.start()
    rec.wait_forever()


if __name__ == "__main__":
    main()
