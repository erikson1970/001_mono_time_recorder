# **Continuous Mono Recorder**

This project provides a continuous mono audio recorder that segments recordings based on silence detection and asynchronously encodes the segments into MP3 files. It's designed for long-term, unattended recording, such as capturing lectures, podcasts, or ambient soundscapes.

## **🚀 Features**

* **Continuous Recording:** Records mono audio into an in-memory buffer for seamless operation.  
* **Silence-Aware Segmentation:** Automatically splits the recording into separate segments after a specified time (--minutes) and when a period of silence (--pause-seconds) is detected.  
* **Asynchronous MP3 Encoding:** A background worker uses lame to encode the segments to MP3, preventing recording interruptions.  
* **Flexible Output:** Saves segments as WAV files (optionally deleted after encoding) and generates CSV and M3U playlists for easy management.  
* **Real-time Spectrogram:** Displays a live, color-coded spectrogram in the console for visual feedback on the audio input.  
* **Extensive Configuration:** Customizable through command-line arguments for segment length, pause detection, sample rate, output directory, and more.

## **🛠️ Dependencies**

### **Python Libraries**

You can install the required Python libraries using pip or uv:

```pip install sounddevice soundfile numpy matplotlib```
<br>
\# or  <br>
```uv pip install sounddevice soundfile numpy matplotlib```

### **System Dependencies**

You must have the lame encoder installed and accessible in your system's PATH.

**On Ubuntu/Debian:**

```sudo apt-get install lame```

**On macOS (with Homebrew):**

```brew install lame```

## **💻 Usage**

To run the recorder, execute the ```main.py``` script. The program offers a variety of command-line arguments to customize its behavior.

### **Basic Example**

This command records audio, creating segments of approximately 30 minutes, and uses a pause of 1.5 seconds to trigger a split. You can run the script directly with Python or use uv.

```python main.py --minutes 30 --pause-seconds 1.5  ``` <br>
\# or  
```uv run main.py -- --minutes 30 --pause-seconds 1.5```

### **Command-Line Arguments**

| Argument | Description | Default | 
|--:|:--|:--:|
| --minutes | Target segment length in minutes before waiting for a pause. | 60.0 |  
| --pause-seconds | Pause length (seconds) that triggers a segment cut once eligible. | 2.0 |  
| --sr | Sample rate. | 48000 |  
| --blocksize | Frames per block callback. | 2048 |  
| --device | Input device name or index. | System default |  
| --silence-threshold | RMS threshold for silence in \[-1,1\] float. | 0.01 |  
| --outdir | Output folder. | ./recordings |  
| --delete-wav-after-encode | Delete WAV after spawning MP3 encode. | False |  
| --spectrogram-frequency | Frequency (as a percentage) to show the spectrogram in the console. | 0.6 |  
| --spectrogram-colors | Color map for spectrogram: 'jet' for truecolor, or None for ANSI gradient. | None |  
| --list-devices | List audio devices and exit. | False |  
| --max-pause-minutes | Max time to search for a pause after the segment length is reached before forcing a split. | 10% of \--minutes |  
| --base-filename | Base filename pattern with a variety of tokens (\~D, \~T, \~d, \~t, etc.). | segment\_\~C\_\~d\~t\_\~l\_secs |  
| --max-total-minutes | Max total recording time before exiting. | 10 \* \--minutes |

## **🤝 Contributing**

We welcome contributions\! If you'd like to contribute, please follow these steps:

1. Fork the repository.  
2. Create a new branch: git checkout \-b feature/your-feature-name  
3. Make your changes and commit them: git commit \-m 'feat: Add new feature'  
4. Push to the branch: git push origin feature/your-feature-name  
5. Open a pull request.

## **📄 License**

This project is licensed under the

LICENSENAME

* see the [LICENSE.md](http://docs.google.com/LICENSE.md) file for details.

