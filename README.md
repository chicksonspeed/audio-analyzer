# 🎵 Advanced Audio Analyzer for Electronic Dance Music

A comprehensive Python toolkit for analyzing electronic dance music (EDM) tracks, extracting energy characteristics, tempo, and mood classifications optimized for house, techno, and EDM genres.

## ✨ Features

- **🎼 Multi-dimensional Analysis**: Energy (bass, mids, highs), tempo, and mood classification
- **🎯 EDM-Optimized**: 15 specialized mood categories for electronic music
- **⚡ High Performance**: Parallel processing with progress tracking
- **📊 Advanced Visualizations**: Energy trends, mood analysis, correlation matrices
- **🔄 Batch Processing**: Analyze multiple files with comprehensive reporting
- **⚙️ Configurable**: JSON-based configuration system
- **📈 Professional Output**: CSV, JSON exports with statistical analysis

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/audio-analyzer.git
cd audio-analyzer
```

2. **Set up virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install essentia soundfile joblib matplotlib numpy seaborn pandas tqdm
```

4. **Install FFmpeg** (required for audio processing):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Basic Usage

**Single File Analysis**:
```bash
python audio_analyzer.py your_track.mp3
```

**Batch Processing**:
```bash
python batch_analyzer.py /path/to/audio/files
```

**With Custom Configuration**:
```bash
python audio_analyzer.py track.mp3 --config config_example.json --n-jobs 8
```

## 📋 Mood Categories

The analyzer classifies tracks into 15 specialized EDM mood categories:

| Category | Description | Characteristics |
|----------|-------------|-----------------|
| **Big Room** | Festival anthems | High party mood, strong bass, 126+ BPM |
| **Circuit** | Underground club | High energy, 125+ BPM, spectral flux |
| **Vocal Chop** | Vocal samples | Happy mood, strong mids/highs |
| **Funky** | Groove-based | Happy mood, balanced frequencies |
| **Tribal** | Percussion-heavy | Party mood, strong highs |
| **Raw** | Hard techno | Dark mood, aggressive bass, 135+ BPM |
| **Electro** | Electronic synthesis | Happy mood, balanced energy |
| **Future** | Modern EDM | Happy mood, full spectrum |
| **Euphoric** | Uplifting | Happy mood, strong mids |
| **Soulful** | Emotional depth | Mixed happy/sad, strong mids |
| **Progressive** | Building energy | Moderate tempo, controlled flux |
| **Dark** | Industrial/techno | Sad mood, bass-heavy, low flux |
| **Deep** | Underground | Sad mood, bass-focused |
| **Ambient** | Atmospheric | Sad mood, minimal energy |
| **Neutral** | Unclassified | Default category |

## 📊 Output Files

### Single File Analysis
- `{filename}_summary.json` - Detailed analysis results
- `{filename}_energy_trends.png` - Energy visualization over time
- `{filename}_mood_trends.png` - Mood probability trends

### Batch Processing
- `batch_summary.csv` - Statistical summary of all files
- `batch_summary.json` - Comprehensive analysis data
- `tempo_distribution.png` - Tempo histogram
- `mood_distribution.png` - Mood category pie chart
- `correlation_heatmap.png` - Energy correlation matrix
- `duration_tempo_scatter.png` - Duration vs tempo analysis

## ⚙️ Configuration

Create a `config.json` file to customize analysis parameters:

```json
{
  "target_sr": 44100,
  "seg_len_s": 30.0,
  "nfft": 2048,
  "hop": 512,
  "n_jobs": 4,
  "tempo_range": [110.0, 160.0],
  "mood_thresholds": {
    "big_room": {
      "mood_party": 0.7,
      "bass": 0.15,
      "tempo": 126,
      "spectral_flux": 0.7
    }
  }
}
```

## 🔧 Advanced Usage

### Command Line Options

**Single File Analyzer**:
```bash
python audio_analyzer.py [OPTIONS] AUDIO_FILE

Options:
  --output-dir DIR     Output directory (default: analysis_results)
  --verbose, -v       Enable verbose logging
```

**Batch Analyzer**:
```bash
python batch_analyzer.py [OPTIONS] INPUT

Options:
  --output-dir DIR     Output directory (default: batch_analysis_results)
  --patterns PATTERN   File patterns to match (default: *.mp3,*.wav,*.flac)
  --max-workers N     Maximum parallel workers (default: 4)
  --config FILE        Configuration file path
  --verbose, -v        Enable verbose logging
```

### Programmatic Usage

```python
import audio_analyzer

# Analyze single file
results = audio_analyzer.analyze_audio('track.mp3', 'output_dir')
```

## 📈 Performance

- **Memory Efficient**: Streaming approach handles large files
- **Parallel Processing**: Configurable multi-core utilization
- **Progress Tracking**: Real-time progress bars for batch operations
- **Error Recovery**: Continues processing even if individual segments fail

## 🛠️ Technical Details

### Audio Processing Pipeline

1. **File Loading**: Efficient duration calculation and segment loading
2. **Energy Analysis**: FFT-based frequency band analysis (bass <250Hz, mids 250-4000Hz, highs >4000Hz)
3. **Tempo Estimation**: Percival BPM estimator optimized for 110-160 BPM range
4. **Mood Extraction**: Essentia MusicExtractor for high-level features
5. **Classification**: Rule-based system with 15 EDM-specific categories
6. **Visualization**: Matplotlib/Seaborn for professional charts

### Dependencies

- **essentia**: Audio analysis and feature extraction
- **soundfile**: Audio I/O operations
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Data visualization
- **joblib**: Parallel processing
- **pandas**: Data analysis and CSV export
- **tqdm**: Progress bars

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Bonus points for adding weird mood categories like "horny minimal."** 🎵

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Essentia**: Audio analysis library by Music Technology Group
- **SoundFile**: Audio I/O library by Bastian Bechtold
- **Matplotlib/Seaborn**: Visualization libraries

## 📞 Support

For questions, issues, or feature requests, please open an issue on GitHub.

---

**Made with ❤️ for the electronic music community**