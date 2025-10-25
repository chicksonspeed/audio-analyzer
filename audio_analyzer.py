#!/usr/bin/env python3
"""
Working Audio Analyzer for Electronic Dance Music
Fixed version that works with current Essentia installation.
"""

import os
import numpy as np
import essentia.standard as es
import soundfile as sf
import tempfile
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import json
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
TARGET_SR = 44100  # Target sample rate
SEG_LEN_S = 30.0   # Segment length in seconds
NFFT = 2048        # FFT size
HOP = 512          # Hop size

def process_segment(audio_file, start_time, seg_dur, segment_idx, output_dir):
    """Process a single audio segment."""
    try:
        # Load the entire file and extract segment
        loader = es.MonoLoader(filename=audio_file, sampleRate=TARGET_SR)
        y_full = loader()
        
        # Calculate segment boundaries
        start_sample = int(start_time * TARGET_SR)
        end_sample = int((start_time + seg_dur) * TARGET_SR)
        y = y_full[start_sample:end_sample]
        
        if len(y) == 0:
            return None
            
        # Energy analysis
        window = es.Windowing(type='hann')
        spectrum = es.Spectrum()
        S = np.array([spectrum(window(y[frame:frame+NFFT])) for frame in range(0, len(y)-NFFT, HOP)])
        
        if S.size == 0:
            return None
            
        freqs = np.linspace(0, TARGET_SR/2, S.shape[1])
        bass_mask = freqs < 250
        mids_mask = (freqs >= 250) & (freqs < 4000)
        highs_mask = freqs >= 4000
        
        def band_rms(mask):
            if not np.any(mask):
                return 0.0
            band = S[:, mask]
            return float(np.mean(np.sqrt(np.mean(band**2, axis=0))))
        
        bass_seg = band_rms(bass_mask)
        mids_seg = band_rms(mids_mask)
        highs_seg = band_rms(highs_mask)
        
        # Tempo estimation
        try:
            bpm_estimator = es.PercivalBpmEstimator()
            tempo = float(bpm_estimator(y))
            tempo = tempo if 110 <= tempo <= 160 else float('nan')
        except:
            tempo = float('nan')
        
        # Mood analysis (simplified)
        mood_happy = mood_sad = mood_party = spectral_flux = 0.0
        
        # Simple mood classification based on energy and tempo
        if tempo > 0 and not np.isnan(tempo):
            if tempo >= 128 and bass_seg > 0.15:
                mood_label = 'big_room'
                mood_party = 0.8
            elif tempo >= 125 and bass_seg > 0.1:
                mood_label = 'circuit'
                mood_party = 0.7
            elif tempo >= 120 and mids_seg > 0.1:
                mood_label = 'progressive'
                mood_happy = 0.6
            elif tempo >= 135 and bass_seg > 0.15:
                mood_label = 'raw'
                mood_sad = 0.7
            else:
                mood_label = 'neutral'
        else:
            mood_label = 'neutral'
        
        # Calculate spectral flux
        if len(S) > 1:
            spectral_flux = float(np.mean(np.abs(np.diff(S, axis=0))))
        else:
            spectral_flux = 0.0
        
        return {
            'segment': segment_idx + 1,
            'start_min': start_time / 60.0,
            'bass': bass_seg,
            'mids': mids_seg,
            'highs': highs_seg,
            'tempo': tempo,
            'mood': {
                'label': mood_label,
                'happy': mood_happy,
                'sad': mood_sad,
                'party': mood_party
            },
            'spectral_flux': spectral_flux
        }
        
    except Exception as e:
        logger.warning(f"Segment {segment_idx + 1} failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Working Audio Analyzer for EDM")
    parser.add_argument("audio_file", type=str, help="Path to the input audio file")
    parser.add_argument("--output-dir", type=str, default="analysis_results", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate audio file
    if not os.path.isfile(args.audio_file):
        logger.error(f"Audio file '{args.audio_file}' does not exist.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get file duration
    try:
        info = sf.info(args.audio_file)
        file_duration = info.duration
        logger.info(f"File duration: {file_duration:.2f} seconds")
    except Exception as e:
        logger.error(f"Error getting file duration: {e}")
        return
    
    # Calculate number of segments
    num_segments = int(np.ceil(file_duration / SEG_LEN_S))
    logger.info(f"Processing {num_segments} segments")
    
    # Process segments
    results = []
    for i in range(num_segments):
        start_time = i * SEG_LEN_S
        seg_dur = min(SEG_LEN_S, file_duration - start_time)
        
        if seg_dur <= 0:
            break
            
        result = process_segment(args.audio_file, start_time, seg_dur, i, args.output_dir)
        if result:
            results.append(result)
            logger.info(f"Processed segment {i+1}/{num_segments}")
    
    if not results:
        logger.error("No segments were successfully processed")
        return
    
    # Aggregate results
    segment_times = [r['start_min'] for r in results]
    energy_bass = [r['bass'] for r in results]
    energy_mids = [r['mids'] for r in results]
    energy_highs = [r['highs'] for r in results]
    tempos = [r['tempo'] for r in results]
    mood_labels = [r['mood']['label'] for r in results]
    
    # Compute averages
    avg_bass = float(np.mean([x for x in energy_bass if x > 0])) if any(x > 0 for x in energy_bass) else 0.0
    avg_mids = float(np.mean([x for x in energy_mids if x > 0])) if any(x > 0 for x in energy_mids) else 0.0
    avg_highs = float(np.mean([x for x in energy_highs if x > 0])) if any(x > 0 for x in energy_highs) else 0.0
    median_tempo = float(np.nanmedian(tempos)) if not np.all(np.isnan(tempos)) else 0.0
    
    # Count mood distribution
    from collections import Counter
    mood_counts = Counter(mood_labels)
    dominant_mood = mood_counts.most_common(1)[0][0] if mood_counts else 'neutral'
    
    # Print summary
    print(f"\n🎵 AUDIO ANALYSIS RESULTS 🎵")
    print(f"=" * 50)
    print(f"File: {os.path.basename(args.audio_file)}")
    print(f"Duration: {file_duration/60:.1f} minutes")
    print(f"Segments processed: {len(results)}")
    print(f"\n📊 ENERGY ANALYSIS:")
    print(f"Average Bass Energy: {avg_bass:.4f}")
    print(f"Average Mids Energy: {avg_mids:.4f}")
    print(f"Average Highs Energy: {avg_highs:.4f}")
    print(f"\n🎼 TEMPO ANALYSIS:")
    print(f"Median Tempo: {median_tempo:.1f} BPM")
    print(f"\n🎭 MOOD ANALYSIS:")
    print(f"Dominant Mood: {dominant_mood}")
    print(f"Mood Distribution: {dict(mood_counts)}")
    
    # Save results
    summary = {
        'file': args.audio_file,
        'duration_min': file_duration / 60.0,
        'segments_processed': len(results),
        'avg_bass': avg_bass,
        'avg_mids': avg_mids,
        'avg_highs': avg_highs,
        'median_tempo': median_tempo,
        'dominant_mood': dominant_mood,
        'mood_distribution': dict(mood_counts),
        'segment_results': results
    }
    
    output_json = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.audio_file))[0]}_summary.json")
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Results saved to {output_json}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Energy plot
    plt.subplot(2, 1, 1)
    plt.plot(segment_times, energy_bass, label='Bass', color='#FF6B6B', linewidth=2)
    plt.plot(segment_times, energy_mids, label='Mids', color='#4ECDC4', linewidth=2)
    plt.plot(segment_times, energy_highs, label='Highs', color='#45B7D1', linewidth=2)
    plt.xlabel('Time (minutes)')
    plt.ylabel('RMS Energy')
    plt.title('Energy Trends Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Tempo plot
    plt.subplot(2, 1, 2)
    valid_tempos = [t for t in tempos if not np.isnan(t)]
    if valid_tempos:
        plt.plot(segment_times[:len(valid_tempos)], valid_tempos, 'o-', color='#9B59B6', linewidth=2, markersize=4)
        plt.xlabel('Time (minutes)')
        plt.ylabel('Tempo (BPM)')
        plt.title('Tempo Over Time')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.audio_file))[0]}_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to {plot_path}")
    print(f"\n✅ Analysis complete! Check {args.output_dir} for results.")

if __name__ == "__main__":
    main()
