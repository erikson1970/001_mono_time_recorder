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
import math

import argparse
import datetime as dt
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf


def timestamp_for_filename(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")


class SegmentWriter(threading.Thread):
    """
    Background worker that receives (samples, sr, start_ts, end_ts, outdir)
    and writes a WAV, then spawns LAME to encode to MP3 with --preset mw-eu.
    """

    def __init__(
        self, task_q: "queue.Queue[Tuple[np.ndarray, int, float, float, Path, bool]]"
    ):
        super().__init__(daemon=True)
        self.task_q = task_q
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                samples, sr, t0, t1, outdir, delete_wav = self.task_q.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                start_tag = timestamp_for_filename(t0)
                end_tag = timestamp_for_filename(t1)
                base = f"segment_{start_tag}_to_{end_tag}"
                wav_path = outdir / f"{base}.wav"
                mp3_path = outdir / f"{base}.mp3"

                # Write WAV (PCM_16)
                sf.write(str(wav_path), samples, sr, subtype="PCM_16")

                # Kick off LAME encode
                # Note: if your LAME build doesn't know "mw-eu", you can change it to e.g. "--preset medium"
                cmd = ["lame", "--preset", "mw-eu", str(wav_path), str(mp3_path)]
                try:
                    # Use Popen to avoid blocking; let LAME run independently
                    subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                except FileNotFoundError:
                    print(
                        "[WARN] 'lame' not found in PATH; skipping MP3 encode.",
                        file=sys.stderr,
                    )

                # Optionally delete WAV after encode starts (encode is reading from its own handle)
                if delete_wav:
                    # A small delay reduces risk of racing with encoder open on some OSes
                    time.sleep(0.2)
                    try:
                        wav_path.unlink(missing_ok=True)
                    except Exception:
                        pass

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
    ):
        self.minutes = minutes
        self.pause_seconds = pause_seconds
        self.sr = sr
        self.blocksize = blocksize
        self.device = device
        self.threshold = silence_rms_threshold
        self.output_dir = output_dir
        self.delete_wav_after_encode = delete_wav_after_encode

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Double-buffer-ish: active buffer accumulates blocks until we swap on a pause
        self._lock = threading.Lock()
        self._active_frames: List[np.ndarray] = []
        self._active_start_ts: float | None = None

        # For pause detection
        self._elapsed_since_start = 0.0
        self._silence_run = 0.0
        self._eligible_for_split = False  # set true once minutes threshold exceeded

        # Worker for writing/encoding
        self._task_q: (
            "queue.Queue[Tuple[np.ndarray, int, float, float, Path, bool]]"
        ) = queue.Queue()
        self._writer = SegmentWriter(self._task_q)

        # Graceful shutdown
        self._stop_event = threading.Event()

        self._stream: sd.InputStream | None = None

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
        # x is float32 mono in [-1, 1]
        return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))

    def _on_audio(self, indata, frames, time_info, status):
        try:
            if status:
                # Dropouts or overflows will be reported here
                print(f"[AudioStatus] {status}", file=sys.stderr)

            # indata shape: (frames, channels); we open mono so channels=1
            mono = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
            # block_start_ts = time_info["input_buffer_adc_time"] or time.time()
            t_candidate = getattr(time_info, "inputBufferAdcTime", None)
            if (
                t_candidate is None
                or not math.isfinite(t_candidate)
                or t_candidate <= 0
            ):
                # Fall back if PortAudio time isn’t available yet on this platform/driver
                block_start_ts = time.monotonic()
            else:
                block_start_ts = float(t_candidate)

            # Append to active buffer
            self._append_block(mono, block_start_ts)

            # Update timers
            block_duration = frames / self.sr
            self._elapsed_since_start += block_duration

            # After H minutes, watch for pause runs
            if self._elapsed_since_start >= self.minutes * 60.0:
                self._eligible_for_split = True

            # Silence detection
            block_rms = self._rms(mono)
            if block_rms < self.threshold:
                self._silence_run += block_duration
            else:
                self._silence_run = 0.0

            # If eligible and pause run exceeded N seconds, split
            if self._eligible_for_split and self._silence_run >= self.pause_seconds:
                collected = self._collect_active()
                # Start a fresh buffer immediately to avoid gaps
                next_buffer_start_ts = block_start_ts + block_duration
                self._reset_active_buffer(t_start=next_buffer_start_ts)
                self._elapsed_since_start = 0.0
                self._silence_run = 0.0
                self._eligible_for_split = False

                if collected:
                    samples, t0, t1 = collected
                    # Queue for writer: keep mono shape (N,) -> (N,1) not needed for WAV; soundfile accepts (N,)
                    self._task_q.put(
                        (
                            samples.copy(),
                            self.sr,
                            t0,
                            t1,
                            self.output_dir,
                            self.delete_wav_after_encode,
                        )
                    )
        except Exception as e:
            print(f"[CallbackError] {e}", file=sys.stderr)

    def start(self):
        print("[INFO] Starting writer thread…")
        self._writer.start()
        print("[INFO] Opening input stream…")
        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,  # mono
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._on_audio,
            device=self.device,
        )
        self._reset_active_buffer()
        self._stream.start()
        print("[INFO] Recording… Press Ctrl+C to stop.")

    def stop(self):
        self._stop_event.set()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        # Flush the final active buffer to disk if it has content
        final = self._collect_active()
        if final:
            samples, t0, t1 = final
            self._task_q.put(
                (
                    samples,
                    self.sr,
                    t0,
                    t1,
                    self.output_dir,
                    self.delete_wav_after_encode,
                )
            )
        # Allow the queue to drain a bit
        try:
            self._task_q.join()
        except Exception:
            pass
        self._writer.stop()

    def wait_forever(self):
        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping…")
            self.stop()


def list_devices_and_exit():
    print(sd.query_devices())
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(
        description="Silence-aware continuous recorder with async LAME encoding."
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
        help="Pause length (seconds) that triggers a segment cut once eligible (default: 2.0)",
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
        default=False,
        help="Delete WAV after spawning MP3 encode",
    )
    ap.add_argument(
        "--list-devices", action="store_true", help="List audio devices and exit"
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
    )

    # Handle SIGINT/SIGTERM gracefully
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
