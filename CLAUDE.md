# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a continuous mono audio recorder with silence-aware segmentation. It records audio into memory, automatically splits recordings based on time and silence detection, and asynchronously encodes segments to MP3 using LAME. Designed for long-term unattended recording (lectures, podcasts, ambient soundscapes).

## Development Commands

### Running the Recorder

Basic usage with interactive hotkeys:
```bash
python main.py --minutes 30 --pause-seconds 1.5
```

Or with `uv`:
```bash
uv run main.py -- --minutes 30 --pause-seconds 1.5
```

Disable hotkeys (Ctrl+C only):
```bash
python main.py --no-hotkeys --minutes 30 --pause-seconds 1.5
```

### Interactive Hotkeys (Phase 1 - Completed)

When running with hotkeys enabled (default), press these keys for runtime control:
- `q`: Quit - Stop recording and exit gracefully
- `p`: Pause/Resume - Toggle recording pause (paused time doesn't count against max runtime)
- `b`: Break with gap - Force segment cut at next good silence gap
- `B`: Break immediate - Force segment cut immediately
- `s`: Toggle spectrogram - Show/hide real-time spectrogram
- `h`: Show histogram - Display gap histogram (Phase 2 feature, placeholder for now)

Note: Requires `readchar` package for single-key input. Falls back to line-based input if unavailable.

### Listing Audio Devices

```bash
python main.py --list-devices
```

### Installing Dependencies

Using pip:
```bash
pip install sounddevice soundfile numpy matplotlib
```

Using uv:
```bash
uv pip install sounddevice soundfile numpy matplotlib
```

### System Requirements

The `lame` MP3 encoder must be installed:
- macOS: `brew install lame`
- Ubuntu/Debian: `sudo apt-get install lame`

## Architecture

### Core Modules (Phase 1 - Completed)

**command.py** - Interactive command infrastructure:
- `CommandQueue`: Thread-safe queue for inter-thread communication
- `StdinReader`: Background thread for reading hotkeys (supports `readchar` or line-based fallback)
- `Command` enum: Available commands (QUIT, PAUSE_RESUME, BREAK_WITH_GAP, BREAK_IMMEDIATE, TOGGLE_SPECTROGRAM, SHOW_HISTOGRAM)

**timing.py** - Session timing with pause support:
- `SessionTimer`: Tracks active recording time (excludes pauses) vs total elapsed time
- `SegmentTimer`: Simple timer for individual segment tracking
- Uses threading.Event for pause signaling to ensure atomic state transitions

**main.py** - Primary recorder implementation with updated classes:

1. **ContinuousRecorder** (~372-836)
   - Audio callback handler that manages in-memory buffer
   - Implements silence detection via RMS threshold analysis
   - **NEW**: Integrates command processing and pause/resume functionality
   - **NEW**: Pause-aware timing (paused time doesn't count against max runtime)
   - **NEW**: Dynamic spectrogram toggle
   - Manages segment timing logic: records for H minutes, then watches for N-second pause
   - Uses PortAudio timestamp when available, falls back to monotonic time
   - Enqueues completed segments to background writer thread

2. **SegmentWriter** (~341-370)
   - Background daemon thread consuming from task queue
   - Prevents recording interruptions by handling I/O asynchronously
   - Writes WAV files and spawns LAME encoding processes

3. **SessionIO** (~156-338)
   - Single-threaded I/O manager for WAV/CSV/M3U playlists
   - **NEW**: Tracks encoder processes for graceful shutdown
   - **NEW**: Minimum duration guard to skip trivial segments
   - **NEW**: Proper encoder completion waiting (fixes WAV deletion race condition)
   - Token-based filename generation supporting multiple date/time/duration patterns
   - Maintains CSV metadata (timestamps, durations, filenames, sample rate)
   - Creates both MP3 and WAV M3U playlists

### Key Design Patterns

**Seamless Buffer Swapping**: When a segment is cut, the recorder immediately swaps to a fresh buffer before enqueueing the old one, preventing audio gaps during I/O operations.

**Async MP3 Encoding**: WAV files are written synchronously, but MP3 encoding is spawned as separate processes via `subprocess.Popen` to avoid blocking. **NEW**: Encoder processes are tracked and joined during shutdown with timeout handling.

**Time Tracking**: Uses three time baselines for accuracy: UNIX epoch, perf_counter monotonic, and PortAudio ADC time when available. **NEW**: SessionTimer separates active recording time from paused time.

**Pause-Aware State Management** (Phase 1):
- Audio callback checks pause state and skips processing when paused (keeps stream alive to avoid underruns)
- SessionTimer tracks active vs total time using threading.Event
- Paused time extends max runtime but doesn't count against recording limits
- Lock-protected pause state transitions ensure thread safety

**Graceful Shutdown** (Phase 1):
- Tracks all spawned LAME encoder processes
- Waits for encoder completion with configurable timeout (default 30s)
- Only deletes WAV files after successful MP3 encode
- Minimum segment duration guard prevents zero-length or trivial segments
- Proper queue draining and thread joining

**Command Processing** (Phase 1):
- Separate stdin reader thread feeds CommandQueue
- Main wait loop polls command queue and dispatches to handler methods
- Commands processed outside audio callback to avoid blocking
- Supports both readchar (single-key) and line-based input modes

**Silence Detection State Machine**:
- Records for target duration (`--minutes`)
- Enters "eligible for split" state
- Watches for pause (`--pause-seconds` of silence below RMS threshold)
- Forces split if no pause found within `--max-pause-minutes`
- Implements max **active** recording time with automatic shutdown

### Spectrogram Visualization

Both main.py and gemini_spectrogram.py implement real-time FFT-based spectrograms:
- **main.py**: Integrated into recorder, displays on right half of terminal, supports both ANSI gradient and truecolor "jet" colormap
- **gemini_spectrogram.py**: Standalone spectrogram tool with simpler implementation
- **spectrogram.py**: Basic reference implementation

### Output Files

The recorder generates per-session:
- Segmented WAV files (optionally deleted after MP3 encoding)
- MP3 files (encoded with `lame --preset mw-eu`)
- CSV manifest with segment metadata
- M3U playlists (separate for MP3 and WAV)

### Filename Token System

The `--base-filename` parameter supports tokens for flexible naming:
- `~D` / `~T`: Overall session start date/time (YYMMDD/HHMMSS)
- `~d` / `~t`: Segment start date/time
- `~E` / `~e`: Segment end date/time
- `~C`: Zero-padded segment counter
- `~L`: Segment length as HHMMSS
- `~l`: Segment length in seconds (zero-padded)

Example: `segment_~C_~d~t_~l_secs` produces `segment_001_2511070830_1234_secs.wav`

## Configuration

Key command-line arguments:

**Recording Control:**
- `--minutes`: Target segment length before waiting for pause (default: 60)
- `--pause-seconds`: Silence duration to trigger split (default: 2.0)
- `--max-pause-minutes`: Max pause search window (default: 10% of --minutes)
- `--max-total-minutes`: Max **active** recording time (default: 10 × --minutes)
- `--min-segment-seconds`: Minimum segment duration to save (default: 1.0) **[Phase 1]**

**Audio Settings:**
- `--sr`: Sample rate (default: 48000)
- `--blocksize`: Frames per audio callback (default: 2048)
- `--device`: Input device name or index (default: system default)
- `--silence-threshold`: RMS threshold in [-1,1] float range (default: 0.01 ≈ -40 dBFS)

**Output Settings:**
- `--outdir`: Output directory (default: ./recordings)
- `--base-filename`: Filename pattern with tokens (default: segment_~C_~d~t_~l_secs)
- `--start-seg-number`: Starting segment number for naming (default: 0)
- `--delete-wav-after-encode`: Remove WAV files after MP3 encode completes **[Phase 1: fixed race condition]**

**UI Settings:**
- `--no-hotkeys`: Disable interactive hotkeys, Ctrl+C only **[Phase 1]**
- `--spectrogram-frequency`: Probability of showing spectrogram per audio block (default: 0.6)
- `--spectrogram-colors`: Use 'jet' for truecolor, None for ANSI gradient
- `--list-devices`: Show available audio devices and exit

## Dependencies

### Required Dependencies

Managed via pyproject.toml with uv:
- sounddevice: Audio I/O via PortAudio
- soundfile: WAV file writing
- numpy: Audio buffer management and FFT
- matplotlib: Colormap support for spectrograms

### Optional Dependencies

- `readchar`: Single-key hotkey input (falls back to line-based input if unavailable) **[Phase 1]**

### System Dependencies

- `lame`: MP3 encoder (required)
  - macOS: `brew install lame`
  - Ubuntu/Debian: `sudo apt-get install lame`

Python version: >=3.12

---

## Development Roadmap

### Phase 1: Controls & Shutdown ✅ COMPLETED
- Interactive hotkeys (q/p/b/B/s/h)
- Pause/resume with active time tracking
- Graceful shutdown with encoder joining
- Minimum segment duration guard
- Fixed WAV deletion race condition

### Phase 2: Adaptive Gap Detection (Planned)
- Silence detector abstraction (RMS, adaptive, spectral flux)
- Optional 3D histogram gap finder
- Forced-split overlap support

### Phase 3: UX Polish (Planned)
- 1 Hz status line with segment info
- Structured logging with timestamps
- Enhanced device listing

### Phase 3.5: Optional TUI (Planned)
- Textual-based UI with separate process
- Zero impact on audio timing
- Spectrogram + status + hotkey help

### Phase 4: Metadata & Playlists (Planned)
- Extended M3U with #EXTINF
- PLS playlist support
- ID3 tagging via mutagen

### Phase 4.5: Offline File Input (Planned)
- Support WAV/FLAC/OGG/MP3 input
- Real-time pacing or fast processing
- Optional lookahead for gap detection

### Phase 5: Structure & Tests (Planned)
- Modular package layout
- Pytest suite with mocked audio sources
- LAME/FFmpeg encoder fallback
- CI setup for macOS + Linux

---

## Important Notes for Development

### Threading Model
- **Audio callback**: Timing-critical, minimal work, lock-free except buffer append
- **Stdin reader**: Background daemon, feeds CommandQueue
- **Segment writer**: Background daemon, consumes task queue, spawns encoders
- **Main loop**: Polls command queue, processes commands, handles shutdown

### Lock Usage
- `_lock`: Protects audio buffer operations (_active_frames, _active_start_ts)
- `_pause_lock`: Protects pause state (_is_paused)
- `_encoder_lock`: Protects encoder process list in SessionIO

### Audio Underrun Prevention
- Keep callback lean - no I/O, minimal computation
- Spectrogram rendering is probabilistic (spec_freq) to reduce overhead
- Command processing happens in main loop, not callback
- Pause skips processing but keeps stream alive
