#!/usr/bin/env python3
"""
Advanced Audio Analyzer for Electronic Dance Music
Analyzes audio files for energy, tempo, and mood characteristics.
"""

import os
import sys
import json
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import numpy as np
import essentia.standard as es
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('audio_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Configuration for audio analysis parameters."""
    target_sr: int = 44100
    seg_len_s: float = 30.0
    nfft: int = 2048
    hop: int = 512
    n_jobs: int = 4
    tempo_range: Tuple[float, float] = (110.0, 160.0)
    
    # Frequency band definitions
    bass_freq: float = 250.0
    mids_freq_low: float = 250.0
    mids_freq_high: float = 4000.0
    
    # Mood classification thresholds
    mood_thresholds: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.mood_thresholds is None:
            self.mood_thresholds = {
                'big_room': {'mood_party': 0.7, 'bass': 0.15, 'tempo': 126, 'spectral_flux': 0.7},
                'circuit': {'mood_party': 0.65, 'tempo': 125, 'spectral_flux': 0.6, 'bass': 0.1},
                'vocal_chop': {'mood_happy': 0.6, 'mood_party': 0.5, 'mids': 0.12, 'highs': 0.05, 'tempo': 122, 'spectral_flux': 0.6},
                'funky': {'mood_happy': 0.6, 'mids': 0.1, 'highs': 0.05, 'tempo': 120, 'spectral_flux': 0.5},
                'tribal': {'mood_party': 0.5, 'highs': 0.06, 'tempo': 122, 'spectral_flux': 0.65},
                'raw': {'mood_sad': 0.6, 'bass': 0.15, 'tempo': 135, 'spectral_flux': 0.7},
                'electro': {'mood_happy': 0.5, 'bass': 0.12, 'mids': 0.08, 'tempo': 128, 'spectral_flux': 0.65},
                'future': {'mood_happy': 0.6, 'bass': 0.1, 'mids': 0.1, 'tempo': 122, 'spectral_flux': 0.5},
                'euphoric': {'mood_happy': 0.55, 'mids': 0.08, 'tempo': 120, 'spectral_flux': 0.4},
                'soulful': {'mood_happy': 0.55, 'mood_sad': 0.3, 'mids': 0.12, 'tempo': 120},
                'progressive': {'mood_happy': 0.5, 'bass': 0.08, 'mids': 0.1, 'spectral_flux_min': 0.4, 'spectral_flux_max': 0.6, 'tempo_min': 120, 'tempo_max': 128},
                'dark': {'mood_sad': 0.5, 'bass': 0.12, 'mids': 0.1, 'spectral_flux': 0.5},
                'deep': {'mood_sad': 0.4, 'bass': 0.1, 'mids': 0.12, 'highs': 0.05, 'tempo': 118},
                'ambient': {'mood_sad': 0.55, 'bass': 0.1, 'mids': 0.08, 'highs': 0.04, 'spectral_flux': 0.3}
            }

@dataclass
class SegmentResult:
    """Result data for a single audio segment."""
    segment: int
    start_min: float
    bass: float
    mids: float
    highs: float
    tempo: float
    mood: Dict[str, Union[float, str]]
    spectral_flux: float

class AudioAnalyzer:
    """Main audio analyzer class with improved error handling and performance."""
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self._setup_essentia_objects()
        
    def _setup_essentia_objects(self):
        """Initialize Essentia objects once for reuse."""
        self.window = es.Windowing(type='hann')
        self.spectrum = es.Spectrum()
        self.bpm_estimator = es.PercivalBpmEstimator()
        self.music_extractor = es.MusicExtractor()
        
    def get_file_duration(self, audio_file: str) -> float:
        """Get audio file duration without loading entire file into memory."""
        try:
            # Use soundfile for faster duration calculation
            info = sf.info(audio_file)
            return info.duration
        except Exception as e:
            logger.warning(f"Could not get duration from soundfile, falling back to Essentia: {e}")
            try:
                loader = es.MonoLoader(filename=audio_file, sampleRate=self.config.target_sr)
                y = loader()
                duration = len(y) / self.config.target_sr
                del y
                return duration
            except Exception as e2:
                logger.error(f"Failed to get file duration: {e2}")
                raise
    
    def calculate_energy_bands(self, y: np.ndarray) -> Tuple[float, float, float]:
        """Calculate energy in bass, mids, and highs frequency bands."""
        try:
            # Compute spectrogram
            frames = []
            for frame in range(0, len(y) - self.config.nfft, self.config.hop):
                frame_data = y[frame:frame + self.config.nfft]
                if len(frame_data) == self.config.nfft:
                    spectrum = self.spectrum(self.window(frame_data))
                    frames.append(spectrum)
            
            if not frames:
                return 0.0, 0.0, 0.0
                
            S = np.array(frames)
            freqs = np.linspace(0, self.config.target_sr / 2, S.shape[1])
            
            # Define frequency masks
            bass_mask = freqs < self.config.bass_freq
            mids_mask = (freqs >= self.config.mids_freq_low) & (freqs < self.config.mids_freq_high)
            highs_mask = freqs >= self.config.mids_freq_high
            
            def band_rms(mask):
                if not np.any(mask):
                    return 0.0
                band = S[:, mask]
                return float(np.mean(np.sqrt(np.mean(band**2, axis=0))))
            
            return band_rms(bass_mask), band_rms(mids_mask), band_rms(highs_mask)
            
        except Exception as e:
            logger.error(f"Energy calculation failed: {e}")
            return 0.0, 0.0, 0.0
    
    def estimate_tempo(self, y: np.ndarray) -> float:
        """Estimate tempo with improved error handling."""
        try:
            tempo = float(self.bpm_estimator(y))
            if self.config.tempo_range[0] <= tempo <= self.config.tempo_range[1]:
                return tempo
            else:
                return float('nan')
        except Exception as e:
            logger.warning(f"Tempo estimation failed: {e}")
            return float('nan')
    
    def extract_mood_features(self, y: np.ndarray, segment_idx: int, output_dir: str) -> Tuple[float, float, float, float]:
        """Extract mood features using MusicExtractor."""
        temp_file = None
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                prefix=f"seg_{segment_idx:04d}_", 
                suffix=".wav", 
                dir=output_dir, 
                delete=False
            )
            sf.write(temp_file.name, y, self.config.target_sr, subtype='PCM_16')
            
            # Extract features
            mood_res = self.music_extractor(temp_file.name)
            
            mood_happy = float(mood_res.get('highlevel.mood_happy.value', 0.0))
            mood_sad = float(mood_res.get('highlevel.mood_sad.value', 0.0))
            mood_party = float(mood_res.get('highlevel.mood_party.value', 0.0))
            spectral_flux = float(mood_res.get('lowlevel.spectral_flux.mean', 0.0))
            
            return mood_happy, mood_sad, mood_party, spectral_flux
            
        except Exception as e:
            logger.warning(f"Mood extraction failed for segment {segment_idx}: {e}")
            return 0.0, 0.0, 0.0, 0.0
        finally:
            if temp_file:
                try:
                    os.remove(temp_file.name)
                except OSError:
                    pass
    
    def classify_mood(self, mood_happy: float, mood_sad: float, mood_party: float,
                     bass: float, mids: float, highs: float, tempo: float, 
                     spectral_flux: float) -> Tuple[str, Dict[str, float]]:
        """Classify mood using rule-based system with improved logic."""
        
        # Check each mood category
        mood_checks = {
            'big_room': (mood_party > 0.7 and bass > 0.15 and tempo >= 126 and spectral_flux > 0.7),
            'circuit': (mood_party > 0.65 and tempo >= 125 and spectral_flux > 0.6 and bass > 0.1),
            'vocal_chop': (mood_happy > 0.6 and mood_party > 0.5 and mids > 0.12 and highs > 0.05 and tempo >= 122 and spectral_flux > 0.6),
            'funky': (mood_happy > 0.6 and mids > 0.1 and highs > 0.05 and tempo >= 120 and spectral_flux > 0.5),
            'tribal': (mood_party > 0.5 and highs > 0.06 and tempo >= 122 and spectral_flux > 0.65),
            'raw': (mood_sad > 0.6 and bass > 0.15 and tempo >= 135 and spectral_flux > 0.7),
            'electro': (mood_happy > 0.5 and bass > 0.12 and mids > 0.08 and tempo >= 128 and spectral_flux > 0.65),
            'future': (mood_happy > 0.6 and bass > 0.1 and mids > 0.1 and tempo >= 122 and spectral_flux > 0.5),
            'euphoric': (mood_happy > 0.55 and mids > 0.08 and tempo >= 120 and spectral_flux > 0.4),
            'soulful': (mood_happy > 0.55 and mood_sad > 0.3 and mids > 0.12 and tempo >= 120),
            'progressive': (mood_happy > 0.5 and bass > 0.08 and mids > 0.1 and 0.4 <= spectral_flux <= 0.6 and 120 <= tempo <= 128),
            'dark': (mood_sad > 0.5 and bass > 0.12 and mids < 0.1 and spectral_flux < 0.5),
            'deep': (mood_sad > 0.4 and bass > 0.1 and mids < 0.12 and highs < 0.05 and tempo >= 118),
            'ambient': (mood_sad > 0.55 and bass < 0.1 and mids < 0.08 and highs < 0.04 and spectral_flux < 0.3)
        }
        
        # Find matching mood (priority order)
        for mood, is_match in mood_checks.items():
            if is_match:
                mood_label = mood
                break
        else:
            mood_label = 'neutral'
        
        # Create mood probabilities
        mood_probs = {
            'dark': mood_sad if mood_checks['dark'] else 0.0,
            'euphoric': mood_happy if mood_checks['euphoric'] else 0.0,
            'circuit': mood_party if mood_checks['circuit'] else 0.0,
            'funky': mood_happy if mood_checks['funky'] else 0.0,
            'soulful': (mood_happy + mood_sad) / 2 if mood_checks['soulful'] else 0.0,
            'deep': mood_sad if mood_checks['deep'] else 0.0,
            'tribal': mood_party if mood_checks['tribal'] else 0.0,
            'progressive': mood_happy if mood_checks['progressive'] else 0.0,
            'raw': mood_sad if mood_checks['raw'] else 0.0,
            'ambient': mood_sad if mood_checks['ambient'] else 0.0,
            'big_room': mood_party if mood_checks['big_room'] else 0.0,
            'electro': mood_happy if mood_checks['electro'] else 0.0,
            'future': mood_happy if mood_checks['future'] else 0.0,
            'vocal_chop': mood_happy if mood_checks['vocal_chop'] else 0.0,
            'label': mood_label
        }
        
        return mood_label, mood_probs
    
    def process_segment(self, args: Tuple[int, str, str]) -> Optional[SegmentResult]:
        """Process a single audio segment with improved error handling."""
        segment_idx, audio_file, output_dir = args
        
        try:
            start_time = segment_idx * self.config.seg_len_s
            seg_dur = max(0.0, min(self.config.seg_len_s, self.file_duration - start_time))
            
            if seg_dur <= 0:
                return None
            
            # Load segment
            loader = es.MonoLoader(
                filename=audio_file, 
                sampleRate=self.config.target_sr, 
                offset=start_time, 
                duration=seg_dur
            )
            y = loader()
            
            if len(y) == 0:
                return None
            
            # Energy analysis
            bass, mids, highs = self.calculate_energy_bands(y)
            
            # Tempo estimation
            tempo = self.estimate_tempo(y)
            
            # Mood analysis
            mood_happy, mood_sad, mood_party, spectral_flux = self.extract_mood_features(
                y, segment_idx, output_dir
            )
            
            # Mood classification
            mood_label, mood_probs = self.classify_mood(
                mood_happy, mood_sad, mood_party, bass, mids, highs, tempo, spectral_flux
            )
            
            return SegmentResult(
                segment=segment_idx + 1,
                start_min=start_time / 60.0,
                bass=bass,
                mids=mids,
                highs=highs,
                tempo=tempo,
                mood=mood_probs,
                spectral_flux=spectral_flux
            )
            
        except Exception as e:
            logger.error(f"Segment {segment_idx + 1} processing failed: {e}")
            return None
    
    def analyze_audio(self, audio_file: str, output_dir: str) -> Dict:
        """Main analysis function with improved structure."""
        logger.info(f"Starting analysis of {audio_file}")
        
        # Get file duration
        self.file_duration = self.get_file_duration(audio_file)
        logger.info(f"File duration: {self.file_duration:.2f} seconds")
        
        # Calculate number of segments
        num_segments = int(np.ceil(self.file_duration / self.config.seg_len_s))
        logger.info(f"Processing {num_segments} segments")
        
        # Process segments sequentially (Essentia objects can't be pickled)
        results = []
        for i in range(num_segments):
            result = self.process_segment((i, audio_file, output_dir))
            if result:
                results.append(result)
            logger.info(f"Processed segment {i+1}/{num_segments}")
        
        if not results:
            raise ValueError("No segments were successfully processed")
        
        logger.info(f"Successfully processed {len(results)} segments")
        
        # Aggregate results
        return self._aggregate_results(results, audio_file)
    
    def _aggregate_results(self, results: List[SegmentResult], audio_file: str) -> Dict:
        """Aggregate segment results into summary statistics."""
        # Extract data arrays
        segment_times = [r.start_min for r in results]
        energy_bass = [r.bass for r in results]
        energy_mids = [r.mids for r in results]
        energy_highs = [r.highs for r in results]
        tempos = [r.tempo for r in results]
        mood_probs = [r.mood for r in results]
        spectral_fluxes = [r.spectral_flux for r in results]
        
        # Compute averages
        avg_bass = float(np.mean([x for x in energy_bass if x > 0])) if any(x > 0 for x in energy_bass) else 0.0
        avg_mids = float(np.mean([x for x in energy_mids if x > 0])) if any(x > 0 for x in energy_mids) else 0.0
        avg_highs = float(np.mean([x for x in energy_highs if x > 0])) if any(x > 0 for x in energy_highs) else 0.0
        median_tempo = float(np.nanmedian(tempos)) if not np.all(np.isnan(tempos)) else 0.0
        
        # Calculate mood averages
        mood_keys = ['dark', 'euphoric', 'circuit', 'funky', 'soulful', 'deep', 'tribal', 
                    'progressive', 'raw', 'ambient', 'big_room', 'electro', 'future', 'vocal_chop']
        avg_mood = {key: float(np.mean([m[key] for m in mood_probs])) for key in mood_keys}
        dominant_mood = max(avg_mood, key=avg_mood.get)
        
        # Print summary
        logger.info(f"Average Bass Energy: {avg_bass:.4f}")
        logger.info(f"Average Mids Energy: {avg_mids:.4f}")
        logger.info(f"Average Highs Energy: {avg_highs:.4f}")
        logger.info(f"Median Tempo: {median_tempo:.2f} BPM")
        logger.info(f"Dominant Mood: {dominant_mood}")
        
        return {
            'file': audio_file,
            'duration_min': self.file_duration / 60.0,
            'avg_bass': avg_bass,
            'avg_mids': avg_mids,
            'avg_highs': avg_highs,
            'median_tempo': median_tempo,
            'avg_mood': avg_mood,
            'dominant_mood': dominant_mood,
            'segment_results': [asdict(r) for r in results],
            'segment_times': segment_times,
            'energy_bass': energy_bass,
            'energy_mids': energy_mids,
            'energy_highs': energy_highs,
            'tempos': tempos,
            'mood_probs': mood_probs,
            'spectral_fluxes': spectral_fluxes
        }
    
    def create_visualizations(self, results: Dict, output_dir: str):
        """Create improved visualizations with better styling."""
        audio_file = results['file']
        base_name = Path(audio_file).stem
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Energy trends plot
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(results['segment_times'], results['energy_bass'], 
                label='Bass', color='#FF6B6B', linewidth=2, alpha=0.8)
        ax.plot(results['segment_times'], results['energy_mids'], 
                label='Mids', color='#4ECDC4', linewidth=2, alpha=0.8)
        ax.plot(results['segment_times'], results['energy_highs'], 
                label='Highs', color='#45B7D1', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('RMS Energy', fontsize=12)
        ax.set_title('Energy Trends Over Time', fontsize=16, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_energy_trends.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Mood trends plot
        fig, ax = plt.subplots(figsize=(16, 10))
        
        mood_colors = {
            'dark': '#2D2D2D', 'euphoric': '#FFD60A', 'circuit': '#8338EC',
            'funky': '#FF006E', 'soulful': '#00D4FF', 'deep': '#3A86FF',
            'tribal': '#FF9F1C', 'progressive': '#06D6A0', 'raw': '#7209B7',
            'ambient': '#118AB2', 'big_room': '#F72585', 'electro': '#7209B7',
            'future': '#4CC9F0', 'vocal_chop': '#F48C06'
        }
        
        for mood, color in mood_colors.items():
            values = [m[mood] for m in results['mood_probs']]
            ax.plot(results['segment_times'], values, label=mood.replace('_', ' ').title(), 
                   color=color, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Mood Probability', fontsize=12)
        ax.set_title('Mood Trends Over Time (House, Techno, EDM)', fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_mood_trends.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations created successfully")

def main():
    """Main function with improved argument parsing and error handling."""
    parser = argparse.ArgumentParser(
        description="Advanced Audio Analyzer for Electronic Dance Music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audio_analyzer_improved.py song.mp3
  python audio_analyzer_improved.py song.mp3 --output-dir results --n-jobs 8
  python audio_analyzer_improved.py song.mp3 --config config.json
        """
    )
    
    parser.add_argument("audio_file", type=str, help="Path to the input audio file")
    parser.add_argument("--output-dir", type=str, default="analysis_results", 
                   help="Output directory for results")
    parser.add_argument("--n-jobs", type=int, default=4, 
                       help="Number of parallel jobs")
    parser.add_argument("--config", type=str, 
                       help="Path to configuration JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.isfile(args.audio_file):
        logger.error(f"Audio file '{args.audio_file}' does not exist.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load configuration
        config = AudioConfig()
        if args.config and os.path.isfile(args.config):
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                config = AudioConfig(**config_data)
        
        config.n_jobs = args.n_jobs
        
        # Initialize analyzer
        analyzer = AudioAnalyzer(config)
        
        # Perform analysis
        results = analyzer.analyze_audio(args.audio_file, args.output_dir)
        
        # Create visualizations
        analyzer.create_visualizations(results, args.output_dir)
        
        # Save results
        output_json = os.path.join(args.output_dir, f"{Path(args.audio_file).stem}_summary.json")
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Analysis complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
