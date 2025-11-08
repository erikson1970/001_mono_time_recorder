
## Goals
- Make long-form, unattended recordings reliable and controllable from the terminal.
- Improve gap-aware segmentation (minimize mid-word cuts).
- Produce tidy artifacts (MP3/WAV, CSV, playlists, tags) with consistent naming and metadata.
- Keep the CLI-first UX; **TUI is optional** and must have **zero impact** on audio-thread timing.
- Prefer **LAME** for MP3; **fallback to FFmpeg** if LAME is unavailable.

## Non-Goals (for now)
- Heavy DSP beyond robust gap finding (no ASR/VAD pipeline).
- Network streaming; multi-channel support.

---

## Deliverables
- Reliable CLI recorder with hotkeys (quit/pause/break now/break with gap).
- Clean shutdown (no zero-length segments, flushed encodes).
- Adaptive gap detection with optional **3D histogram finder**.
- Forced-split overlap (`--overlap-seconds`).
- Status line with timestamps; improved logging format.
- Extended M3U and PLS playlists; ID3 tags via `mutagen` (default genre `audiobook`).
- Pytest suite for segmentation, filenames, playlists/tags, and shutdown behavior.
- Refactored package layout with `uv` build/install.

---

## Work Environment
- macOS primary; Linux secondary; windows tertiary.
- Keep VS Code dev flow intact.
- Git flow: feature branches → PRs → tests → merge → periodic pushes to GitHub remote.

## Build & Deploy
- Python 3.11+, `uv` for env/build.
- External: `lame` must be installed (Homebrew/Apt). Assume it's there - throw an error if not 
- CLI entry point via `pyproject.toml` so `uvx` can run from the repo.

---


## Phase Plan

### Phase 1 — Stabilize controls & shutdown
**Tasks**
1. **Command plane**
   - `CommandQueue` + stdin reader thread (use `readchar` if present, otherwise line-based fallback).
   - Keys: `q` quit • `p` pause/resume • `b` break with gap finder • `B` immediate break • `h` show current gap histogram (snapshot only) • `s` toggle spectrogram.
2. **Session timing**
   - `SessionTimer` accumulates **active** time only (pauses extend max-total runtime).
3. **Shutdown semantics**
   - `shutdown_event`; handle `SIGINT/SIGTERM` to request graceful stop.
   - Writer thread consumes until sentinel; join LAME/FFmpeg encoders (timeout + warning).
   - Guard against `min_duration_seconds`; skip empty/near-empty files.

**Acceptance**
- Hotkeys react immediately without audio underruns.
- On quit or when max time (active) elapses: process exits cleanly, **no zero-length segments**.

---

### Phase 2 — Better gap detection
**Tasks**
1. **Adaptive energy gate**
   - Rolling RMS → percentile threshold with hysteresis/hangover.
   - Optionally blend with **spectral flux** (low-cost FFT already computed for spectrogram).
2. **3D gap histogram (duration × energy × phase)** _(Optional, behind a flag)_
   - Maintain decayed histogram `H[dur_bin, energy_bin, phase_bin]` over trailing window.
   - After target segment length, search the future **gap window** for the most probable long, low-energy gap; progressively relax; finally force split with overlap.
3. **Forced-split overlap**
   - `--overlap-seconds` (default 4 s); duplicate tail of previous file to head of next.

**Acceptance**
- On speech/audio books, ≥ 80–90% of boundaries align with low-energy gaps.
- Timeouts result in overlap at next file start.

---

### Phase 3 — UX polish (CLI-first)
**Tasks**
- 1 Hz status line: `Seg 12 | 00:41:22 / 01:00:00 | paused=no | rms=-38 dBFS`.
- Logging timestamps: `%(asctime)s [%(levelname)s] %(message)s`.
- Device list: index + name + default SR; allow `--device` by index or substring.

---

### Phase 3.5 — Optional TUI (Textual)
**Tasks**
- `--tui` launches a **separate process** TUI (extras: `tui = ["textual>=0.52", "rich>=13"]`).
- Recorder publishes status & spectrogram frames via `StatusBus` (read-only) while consuming key commands via `CommandQueue`.
- Layout: Spectrogram (ASCII) + Status + Key help. **Cap refresh 5–10 fps**.
- If Textual not available / fails, silently fall back to CLI.

**Acceptance**
- With `--tui`, audio underruns do **not** increase vs CLI-only runs.

---

### Phase 4 — Metadata & playlists
**Tasks**
- Extended M3U (`#EXTINF`) using known durations; default `genre=audiobook`.
- PLS playlist output option.
- ID3 tagging with **`mutagen`** (`artist`, `album`, `title`, `track`, and post-session `tracktotal`).

---

### Phase 4.5 — Offline file input
**Tasks**
- `FileInputSource` supporting **WAV/FLAC/OGG** via `soundfile`, **MP3** via `audioread` or FFmpeg pipe.
- Resample to app SR; mono downmix to float32.
- CLI:
  - `--input-file path` (repeatable)
  - `--input-list file.txt` (one path per line)
  - `--offline` (no pacing, allow lookahead)
  - `--real-time-from-file` (pace to wall clock)
  - `--offline-no-lookahead` (mimic live behavior)

**Acceptance**
- Mixed formats in one list process end-to-end; artifacts match live runs.

---

### Phase 5 — Tests, structure, packaging
**Tasks**
- Module split: `recorder.py`, `silence.py`, `writer.py`, `io.py`, `spectro.py`, `cli.py`, `tui.py` (optional).
- **Injectable input** sources for unit tests (no PortAudio needed).
- Pytest on segmentation, overlaps, metadata, shutdown; basic CI (macOS + Linux).
- Build with **`uv`**; LAME preferred; FFmpeg fallback.

---

## CLI Flags (additions)
- `--tui` : launch optional Textual UI.
- `--overlap-seconds N` : duplicate N seconds of audio at forced splits (default 4).
- `--input-file PATH` (repeatable), `--input-list FILE`.
- `--offline` / `--real-time-from-file` / `--offline-no-lookahead`.
- `--spectrogram-fps N` : cap spectrogram/TUI refresh.
- `--min-segment-seconds N` : avoid zero-length or trivial segments.

---
