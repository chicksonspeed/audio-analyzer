# -*- coding: utf-8 -*-
import os, json, tempfile, argparse, logging
from typing import Dict, Any, Tuple
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
import essentia.standard as es

# -----------------------
# Helpers
# -----------------------
def compute_freq_masks(sr: int, nfft: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    freqs = librosa.fft_frequencies(sr=sr, n_fft=nfft)
    bass_mask = freqs < 250
    mids_mask = (freqs >= 250) & (freqs < 4000)
    highs_mask = freqs >= 4000
    return freqs, bass_mask, mids_mask, highs_mask

def compute_energy_bands(y_seg: np.ndarray, sr: int, masks: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[float, float, float]:
    _freqs, bass_mask, mids_mask, highs_mask = masks
    S = np.abs(librosa.stft(y_seg, n_fft=NFFT, hop_length=HOP))
    def band_rms(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        band = S[mask, :]
        return float(np.mean(np.sqrt(np.mean(band**2, axis=0))))
    return band_rms(bass_mask), band_rms(mids_mask), band_rms(highs_mask)

def estimate_tempo_safe(y_seg: np.ndarray, sr: int) -> float:
    try:
        max_len = int(sr * 30.0)
        y_tempo = y_seg[:max_len] if y_seg.size > max_len else y_seg
        onset_strength = librosa.onset.onset_strength(y=y_tempo, sr=sr, hop_length=HOP)
        tempo_candidates = librosa.beat.tempo(
            onset_envelope=onset_strength,
            sr=sr,
            hop_length=HOP,
            aggregate=np.median,
            ac_size=8.0,
            max_tempo=180.0
        )
        tempo = float(np.atleast_1d(tempo_candidates)[0])
        return tempo if (110.0 <= tempo <= 160.0) else float('nan')
    except Exception:
        return float('nan')

# -----------------------
# Command-line arguments
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Audio Analysis Tool - Extract energy, tempo, and mood from audio files')
    parser.add_argument('audio_file', help='Path to audio file to analyze')
    parser.add_argument('--output-dir', default='analysis_results', help='Output directory for results (default: analysis_results)')
    parser.add_argument('--segment-length', type=int, default=30, help='Segment length in seconds (default: 30)')
    parser.add_argument('--n-jobs', type=int, default=8, help='Number of parallel jobs (default: 8)')
    parser.add_argument('--target-sr', type=int, default=44100, help='Target sample rate (default: 44100)')
    parser.add_argument('--backend', default='threading', choices=['threading', 'loky'], help='Joblib backend (default: threading)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

# -----------------------
# Setup logging
# -----------------------
def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('audio_analysis.log')
        ]
    )
    return logging.getLogger(__name__)

def main():
    args = parse_args()
    logger = setup_logging(args.verbose)

    audio_file = args.audio_file
    base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create subfolder named after filename + "analysis"
    filename = os.path.splitext(os.path.basename(audio_file))[0]
    output_dir = os.path.join(base_output_dir, f"{filename}_analysis")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    global HOP, NFFT
    HOP = 512
    NFFT = 4096
    target_sr = args.target_sr
    seg_len_s = args.segment_length
    n_jobs = args.n_jobs

    logger.info(f"Analyzing: {audio_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Segment length: {seg_len_s}s, Jobs: {n_jobs}, Sample rate: {target_sr}Hz")

    # Load audio once
    y, sr = librosa.load(audio_file, sr=target_sr, mono=True)
    file_duration = y.size / sr
    n_segments = int(np.ceil(file_duration / seg_len_s))

    if file_duration > 3600:
        logger.warning(f"Large file detected ({file_duration/60:.1f} minutes). Consider reducing segment length for faster processing.")
    if n_segments > 200:
        logger.warning(f"Many segments ({n_segments}). This may take a while. Consider increasing segment length.")

    masks = compute_freq_masks(sr, NFFT)

# -----------------------
# Segment worker
# -----------------------
    def process_segment(i: int):
        start_time = i * seg_len_s
        start = int(start_time * sr)
        end = int(min((start_time + seg_len_s) * sr, y.size))
        if end <= start:
            return None
        y_seg = y[start:end]

    # --- Energy bands (RMS on magnitude spectrogram buckets)
        bass_seg, mids_seg, highs_seg = compute_energy_bands(y_seg, sr, masks)

        # --- Tempo (bpm) - safer method avoiding numba beat_tracker segfaults
        try:
            onset_strength = librosa.onset.onset_strength(y=y_seg[:int(sr*30.0)] if y_seg.size > int(sr*30.0) else y_seg, sr=sr, hop_length=HOP)
            tempo_candidates = librosa.beat.tempo(
                onset_envelope=onset_strength,
                sr=sr,
                hop_length=HOP,
                aggregate=np.median,
                ac_size=8.0,
                max_tempo=180.0
            )
            tempo = float(np.atleast_1d(tempo_candidates)[0])
            if not (110.0 <= tempo <= 160.0):
                tempo = float('nan')
        except Exception as e:
            logger.warning(f"[tempo] onset-based method failed for segment {i+1}: {e}")
            tempo = float('nan')

        # --- Spectral flux for mood classification (no Essentia mood detection needed)
        spectral_flux = 0.0
        try:
            # Compute spectral flux using librosa
            stft = librosa.stft(y_seg, hop_length=HOP, n_fft=NFFT)
            magnitude = np.abs(stft)
            spectral_flux = float(np.mean(librosa.onset.onset_strength(S=magnitude)))
        except Exception as e:
            logger.warning(f"[spectral_flux] segment {i+1} failed: {e}")
        
        # Use rule-based mood classification (no Essentia required)
        mood_happy = mood_sad = mood_party = 0.0

        # Rule-based mood classification using audio features (no Essentia required)
        # Calculate derived mood indicators from audio features
        energy_ratio = bass_seg / (mids_seg + highs_seg + 1e-6)
        brightness = highs_seg / (bass_seg + mids_seg + 1e-6)
        activity = spectral_flux * tempo / 100.0  # Normalized activity
        
        # Mood classification based on audio features (adjusted thresholds for actual energy scale)
        is_big_room = (bass_seg > 30) and (tempo >= 126) and (spectral_flux > 0.5) and (energy_ratio > 1.5)
        is_circuit = (tempo >= 125) and (spectral_flux > 0.4) and (bass_seg > 25) and (activity > 1.5) and not is_big_room
        is_vocal_chop = (mids_seg > 8) and (highs_seg > 0.5) and (tempo >= 122) and (spectral_flux > 0.4) and (brightness > 0.05)
        is_funky = (mids_seg > 8) and (highs_seg > 0.5) and (tempo >= 120) and (spectral_flux > 0.3) and (activity > 1.0) and not is_vocal_chop
        is_tribal = (highs_seg > 1.0) and (tempo >= 122) and (spectral_flux > 0.5) and (bass_seg > 25) and (activity > 1.8)
        is_raw = (bass_seg > 35) and (tempo >= 130) and (spectral_flux > 0.5) and (energy_ratio > 2.0) and (activity > 2.0)
        is_hard_techno = (bass_seg > 30) and (tempo >= 140) and (spectral_flux > 0.5) and (energy_ratio > 1.8)
        is_electro = (bass_seg > 25) and (mids_seg > 8) and (tempo >= 128) and (spectral_flux > 0.5) and (brightness > 0.03)
        is_future = (bass_seg > 25) and (mids_seg > 8) and (tempo >= 122) and (spectral_flux > 0.3) and (brightness > 0.04) and not is_vocal_chop
        is_euphoric = (mids_seg > 8) and (tempo >= 120) and (spectral_flux > 0.3) and (brightness > 0.03) and (activity > 1.2)
        is_soulful = (mids_seg > 8) and (tempo >= 120) and (spectral_flux > 0.3) and (energy_ratio < 2.5) and (brightness > 0.02)
        is_progressive = (bass_seg > 20) and (mids_seg > 8) and (0.3 <= spectral_flux <= 0.5) and (120 <= tempo <= 128) and (brightness > 0.015)
        is_dark = (bass_seg > 25) and (mids_seg < 12) and (spectral_flux < 0.4) and (brightness < 0.01) and (energy_ratio > 2.5)
        is_deep = (bass_seg > 20) and (mids_seg < 12) and (highs_seg < 2.0) and (tempo >= 118) and (spectral_flux < 0.3) and (energy_ratio > 2.0)
        is_ambient = (bass_seg < 15) and (mids_seg < 5) and (highs_seg < 1.0) and (spectral_flux < 0.2) and (tempo < 100)

        if is_big_room:
            mood_label = 'big_room'
        elif is_hard_techno:
            mood_label = 'hard_techno'
        elif is_circuit:
            mood_label = 'circuit'
        elif is_vocal_chop:
            mood_label = 'vocal_chop'
        elif is_funky:
            mood_label = 'funky'
        elif is_tribal:
            mood_label = 'tribal'
        elif is_raw:
            mood_label = 'raw'
        elif is_electro:
            mood_label = 'electro'
        elif is_future:
            mood_label = 'future'
        elif is_euphoric:
            mood_label = 'euphoric'
        elif is_soulful:
            mood_label = 'soulful'
        elif is_progressive:
            mood_label = 'progressive'
        elif is_dark:
            mood_label = 'dark'
        elif is_deep:
            mood_label = 'deep'
        elif is_ambient:
            mood_label = 'ambient'
        else:
            mood_label = 'neutral'

        mood_dict = {
            'dark': mood_sad if is_dark else 0.0,
            'euphoric': mood_happy if is_euphoric else 0.0,
            'circuit': mood_party if is_circuit else 0.0,
            'funky': mood_happy if is_funky else 0.0,
            'soulful': (mood_happy + mood_sad) / 2 if is_soulful else 0.0,
            'deep': mood_sad if is_deep else 0.0,
            'tribal': mood_party if is_tribal else 0.0,
            'progressive': mood_happy if is_progressive else 0.0,
            'raw': mood_sad if is_raw else 0.0,
            'ambient': mood_sad if is_ambient else 0.0,
            'big_room': mood_party if is_big_room else 0.0,
            'hard_techno': mood_sad if is_hard_techno else 0.0,
            'electro': mood_happy if is_electro else 0.0,
            'future': mood_happy if is_future else 0.0,
            'vocal_chop': mood_happy if is_vocal_chop else 0.0,
            'label': mood_label
        }

        return {
            'segment': i + 1,
            'start_min': start_time / 60.0,
            'bass': bass_seg,
            'mids': mids_seg,
            'highs': highs_seg,
            'tempo': tempo,
            'mood': mood_dict,
            'spectral_flux': spectral_flux
        }

    logger.info(f"Processing {n_segments} segments over ~{file_duration/60:.1f} minutes with {n_jobs} jobs...")
    with tqdm_joblib(tqdm(total=n_segments, desc="Processing segments", unit="seg")):
        results = Parallel(n_jobs=n_jobs, backend=args.backend, verbose=0)(
            delayed(process_segment)(i) for i in range(n_segments)
        )
        results = [r for r in results if r is not None]

        # Guard for empty
        if not results:
            raise RuntimeError("No segments processed. Check file path, codecs, or permissions.")

        # -----------------------
        # Aggregate
        # -----------------------
        energy_bass = np.array([r['bass'] for r in results], float)
        energy_mids = np.array([r['mids'] for r in results], float)
        energy_highs = np.array([r['highs'] for r in results], float)
        tempos = np.array([r['tempo'] for r in results], float)
        segment_times = np.array([r['start_min'] for r in results], float)
        mood_probs = [r['mood'] for r in results]

        avg_bass = float(np.mean(energy_bass))
        avg_mids = float(np.mean(energy_mids))
        avg_highs = float(np.mean(energy_highs))
        median_tempo = float(np.nanmedian(tempos))

        avg_mood = {
            'dark': float(np.mean([m.get('dark', 0.0) for m in mood_probs])),
            'euphoric': float(np.mean([m.get('euphoric', 0.0) for m in mood_probs])),
            'circuit': float(np.mean([m.get('circuit', 0.0) for m in mood_probs])),
            'funky': float(np.mean([m.get('funky', 0.0) for m in mood_probs])),
            'soulful': float(np.mean([m.get('soulful', 0.0) for m in mood_probs])),
            'deep': float(np.mean([m.get('deep', 0.0) for m in mood_probs])),
            'tribal': float(np.mean([m.get('tribal', 0.0) for m in mood_probs])),
            'progressive': float(np.mean([m.get('progressive', 0.0) for m in mood_probs])),
            'raw': float(np.mean([m.get('raw', 0.0) for m in mood_probs])),
            'ambient': float(np.mean([m.get('ambient', 0.0) for m in mood_probs])),
            'big_room': float(np.mean([m.get('big_room', 0.0) for m in mood_probs])),
            'hard_techno': float(np.mean([m.get('hard_techno', 0.0) for m in mood_probs])),
            'electro': float(np.mean([m.get('electro', 0.0) for m in mood_probs])),
            'future': float(np.mean([m.get('future', 0.0) for m in mood_probs])),
            'vocal_chop': float(np.mean([m.get('vocal_chop', 0.0) for m in mood_probs]))
        }
        dominant_mood = max(avg_mood, key=avg_mood.get)

        logger.info("=== OVERALL ANALYSIS ===")
        logger.info(f"Average Energy  bass={avg_bass:.3f}, mids={avg_mids:.3f}, highs={avg_highs:.3f}")
        logger.info(f"Median Tempo    {median_tempo:.0f} BPM")
        logger.info(f"Mood Averages   dark={avg_mood['dark']:.2f}, euphoric={avg_mood['euphoric']:.2f}, circuit={avg_mood['circuit']:.2f}")
        logger.info(f"Dominant Mood   {dominant_mood.capitalize()}")

        # -----------------------
        # Save JSON
        # -----------------------
        summary = {
        'file': audio_file,
        'duration_min': file_duration / 60.0,
        'avg_bass': avg_bass,
        'avg_mids': avg_mids,
        'avg_highs': avg_highs,
        'median_tempo': median_tempo,
        'avg_mood': avg_mood,
        'dominant_mood': dominant_mood,
        'mood_distribution': {mood: int(np.sum([1 for r in results if r['mood'].get('label') == mood])) for mood in avg_mood.keys()},
        'segment_results': results
    }
        with open(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        # -----------------------
        # Plots (quick QC)
        # -----------------------
        plt.figure(figsize=(12, 6))
    plt.plot(segment_times, energy_bass, label='Bass')
    plt.plot(segment_times, energy_mids, label='Mids')
    plt.plot(segment_times, energy_highs, label='Highs')
    plt.xlabel('Time (minutes)')
    plt.ylabel('RMS Energy (banded)')
    plt.title('Energy Trends Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_energy_trends.png"))
    plt.close()

    plt.figure(figsize=(12, 10))
    plt.plot(segment_times, [m.get('dark', 0.0) for m in mood_probs], label='Dark', color='#2D2D2D')
    plt.plot(segment_times, [m.get('euphoric', 0.0) for m in mood_probs], label='Euphoric', color='#FFD60A')
    plt.plot(segment_times, [m.get('circuit', 0.0) for m in mood_probs], label='Circuit', color='#8338EC')
    plt.plot(segment_times, [m.get('funky', 0.0) for m in mood_probs], label='Funky', color='#FF006E')
    plt.plot(segment_times, [m.get('soulful', 0.0) for m in mood_probs], label='Soulful', color='#00D4FF')
    plt.plot(segment_times, [m.get('deep', 0.0) for m in mood_probs], label='Deep', color='#3A86FF')
    plt.plot(segment_times, [m.get('tribal', 0.0) for m in mood_probs], label='Tribal', color='#FF9F1C')
    plt.plot(segment_times, [m.get('progressive', 0.0) for m in mood_probs], label='Progressive', color='#06D6A0')
    plt.plot(segment_times, [m.get('raw', 0.0) for m in mood_probs], label='Raw', color='#7209B7')
    plt.plot(segment_times, [m.get('ambient', 0.0) for m in mood_probs], label='Ambient', color='#118AB2')
    plt.plot(segment_times, [m.get('big_room', 0.0) for m in mood_probs], label='Big Room', color='#F72585')
    plt.plot(segment_times, [m.get('hard_techno', 0.0) for m in mood_probs], label='Hard Techno', color='#D00000')
    plt.plot(segment_times, [m.get('electro', 0.0) for m in mood_probs], label='Electro', color='#9B59B6')
    plt.plot(segment_times, [m.get('future', 0.0) for m in mood_probs], label='Future', color='#4CC9F0')
    plt.plot(segment_times, [m.get('vocal_chop', 0.0) for m in mood_probs], label='Vocal Chop', color='#F48C06')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Mood Probability')
    plt.title('Mood Trends Over Time (House, Techno, EDM)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_mood_trends.png"))
    plt.close()

    # Tempo trends plot
    plt.figure(figsize=(12, 6))
    valid_tempos = ~np.isnan(tempos)
    if np.any(valid_tempos):
        plt.plot(segment_times[valid_tempos], tempos[valid_tempos], 'o-', label='Tempo', color='red')
        plt.xlabel('Time (minutes)')
        plt.ylabel('BPM')
        plt.title('Tempo Over Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_tempo_trends.png"))
        logger.info(f"Saved tempo trends plot: {os.path.join(output_dir, f'{os.path.splitext(os.path.basename(audio_file))[0]}_tempo_trends.png')}")
    else:
        logger.warning("No valid tempo data found, skipping tempo plot")
    plt.close()

    logger.info(f"Analysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()