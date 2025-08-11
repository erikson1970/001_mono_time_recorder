import numpy as np
import sounddevice as sd
import shutil
import os
import sys

# --- Configuration ---
# Set the desired audio sample rate
SAMPLE_RATE = 44100
# Define the size of each audio block to process
BLOCK_SIZE = 1024
# Set the number of frequency bins to display
NUM_BINS = 80
# Set a scaling factor for the spectrogram intensity
INTENSITY_SCALE = 0.5
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


def map_amplitude_to_color(amplitude):
    """Maps a normalized amplitude (0-1) to an ANSI color code from the gradient."""
    # Clip amplitude to stay within the 0-1 range
    clipped_amplitude = np.clip(amplitude, 0, 1)
    # Scale the amplitude to the size of the color gradient
    color_index = int(clipped_amplitude * (GRADIENT_SIZE - 1))
    return COLOR_GRADIENT[color_index]


# --- Audio Processing Callback ---
# This function is called by sounddevice for each block of audio
def callback(indata, frames, time, status):
    """Processes a block of audio and renders the spectrogram to the console."""
    if status:
        print(status, file=sys.stderr)
    if np.random.rand() < 0.4:
        # Calculate the FFT of the audio data
        fft_data = np.fft.rfft(indata[:, 0])
        # Compute the magnitude of the FFT result
        fft_magnitude = np.abs(fft_data)

        # Scale and normalize the magnitude for visualization
        fft_magnitude = np.log10(fft_magnitude * INTENSITY_SCALE)
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
            color_code = map_amplitude_to_color(bin_amplitude)
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
        if np.random.rand() < 0.01:
            print(
                f"\nDebug: Max amplitude in block: {np.max(fft_magnitude):.2f}",
                file=sys.stderr,
            )


# --- Main loop ---
if __name__ == "__main__":
    print(HIDE_CURSOR, end="")  # Hide the cursor for a cleaner display
    try:
        # Start the audio stream with the callback function
        with sd.InputStream(
            callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE
        ):
            print("Spectrogram running. Press Ctrl+C to exit.", end="")
            # Keep the main thread alive while the stream is running
            while True:
                pass
    except KeyboardInterrupt:
        # User pressed Ctrl+C to exit
        print("\nExiting...")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
    finally:
        # Ensure the cursor is shown and the terminal is reset on exit
        print(SHOW_CURSOR, end="")
        print(RESET)
        print("Spectrogram stopped.")
