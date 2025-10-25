import os
import numpy as np
import essentia.standard as es
import soundfile as sf
import tempfile
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import json
import argparse

# Configuration
TARGET_SR = 44100  # Target sample rate
SEG_LEN_S = 30.0   # Segment length in seconds
NFFT = 2048        # FFT size
HOP = 512          # Hop size

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Analyze audio file for energy, tempo, and mood.")
parser.add_argument("audio_file", type=str, help="Path to the input audio file (e.g., MP3, WAV)")
parser.add_argument("--output-dir", type=str, default="analysis_results", help="Output directory for results")
args = parser.parse_args()

# Validate audio file
audio_file = args.audio_file
if not os.path.isfile(audio_file):
    print(f"Error: Audio file '{audio_file}' does not exist.")
    exit(1)

# Set output directory
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Get file duration
try:
    loader = es.MonoLoader(filename=audio_file, sampleRate=TARGET_SR)
    y = loader()
    file_duration = len(y) / TARGET_SR
    del y
except Exception as e:
    print(f"Error loading file '{audio_file}': {e}")
    exit(1)

def process_segment(i: int):
    start_time = i * SEG_LEN_S
    seg_dur = max(0.0, min(SEG_LEN_S, file_duration - start_time))
    if seg_dur <= 0:
        return None
    # Load with Essentia
    try:
        loader = es.MonoLoader(filename=audio_file, sampleRate=TARGET_SR, offset=start_time, duration=seg_dur)
        y = loader()
        sr_local = TARGET_SR
    except Exception as e:
        print(f"[load] segment {i+1} failed: {e}")
        return None
    if len(y) == 0:
        return None
    # Energy analysis
    window = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    S = np.array([spectrum(window(y[frame:frame+NFFT])) for frame in range(0, len(y)-NFFT, HOP)])
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
    # Tempo estimation (optimized for house/techno/EDM, extended to 160 BPM)
    try:
        bpm_estimator = es.PercivalBpmEstimator()
        tempo = float(bpm_estimator(y))
        tempo = tempo if 110 <= tempo <= 160 else float('nan')  # Extended for hard techno/hardstyle
    except Exception as e:
        print(f"[tempo] segment {i+1} failed: {e}")
        tempo = float('nan')
    # Mood analysis
    mood_happy = mood_sad = mood_party = spectral_flux = 0.0
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(prefix=f"seg_{i:04d}_", suffix=".wav", dir=output_dir, delete=False)
        sf.write(tmp.name, y, sr_local, subtype='PCM_16')
        mex = es.MusicExtractor(highlevel=True)
        mood_res = mex(tmp.name)
        mood_happy = float(mood_res['highlevel.mood_happy.value'])
        mood_sad = float(mood_res['highlevel.mood_sad.value'])
        mood_party = float(mood_res['highlevel.mood_party.value'])
        spectral_flux = float(mood_res['lowlevel.spectral_flux.mean'])
    except Exception as e:
        print(f"[mood] segment {i+1} failed: {e}")
    finally:
        if tmp is not None:
            try:
                os.remove(tmp.name)
            except OSError:
                pass
    # Mood logic with vocal chop, adjusted for extended tempo range
    is_big_room = (mood_party > 0.7) and (bass_seg > 0.15) and (tempo >= 126) and (spectral_flux > 0.7)
    is_circuit = (mood_party > 0.65) and (tempo >= 125) and (spectral_flux > 0.6) and (bass_seg > 0.1) and not is_big_room
    is_vocal_chop = (mood_happy > 0.6) and (mood_party > 0.5) and (mids_seg > 0.12) and (highs_seg > 0.05) and (tempo >= 122) and (spectral_flux > 0.6)
    is_funky = (mood_happy > 0.6) and (mids_seg > 0.1) and (highs_seg > 0.05) and (tempo >= 120) and (spectral_flux > 0.5) and not is_vocal_chop
    is_tribal = (mood_party > 0.5) and (highs_seg > 0.06) and (tempo >= 122) and (spectral_flux > 0.65)
    is_raw = (mood_sad > 0.6) and (bass_seg > 0.15) and (tempo >= 135) and (spectral_flux > 0.7)  # Adjusted for faster techno
    is_electro = (mood_happy > 0.5) and (bass_seg > 0.12) and (mids_seg > 0.08) and (tempo >= 128) and (spectral_flux > 0.65)
    is_future = (mood_happy > 0.6) and (bass_seg > 0.1) and (mids_seg > 0.1) and (tempo >= 122) and (spectral_flux > 0.5) and not is_vocal_chop
    is_euphoric = (mood_happy > 0.55) and (mids_seg > 0.08) and (tempo >= 120) and (spectral_flux > 0.4)
    is_soulful = (mood_happy > 0.55) and (mood_sad > 0.3) and (mids_seg > 0.12) and (tempo >= 120)
    is_progressive = (mood_happy > 0.5) and (bass_seg > 0.08) and (mids_seg > 0.1) and (0.4 <= spectral_flux <= 0.6) and (120 <= tempo <= 128)
    is_dark = (mood_sad > 0.5) and (bass_seg > 0.12) and (mids_seg < 0.1) and (spectral_flux < 0.5)
    is_deep = (mood_sad > 0.4) and (bass_seg > 0.1) and (mids_seg < 0.12) and (highs_seg < 0.05) and (tempo >= 118)
    is_ambient = (mood_sad > 0.55) and (bass_seg < 0.1) and (mids_seg < 0.08) and (highs_seg < 0.04) and (spectral_flux < 0.3)
    if is_big_room:
        mood_label = 'big_room'
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
    return {
        'segment': i + 1,
        'start_min': start_time / 60.0,
        'bass': bass_seg,
        'mids': mids_seg,
        'highs': highs_seg,
        'tempo': tempo,
        'mood': {
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
            'electro': mood_happy if is_electro else 0.0,
            'future': mood_happy if is_future else 0.0,
            'vocal_chop': mood_happy if is_vocal_chop else 0.0,
            'label': mood_label
        },
        'spectral_flux': spectral_flux
    }

# Process segments in parallel
num_segments = int(np.ceil(file_duration / SEG_LEN_S))
results = Parallel(n_jobs=8)(delayed(process_segment)(i) for i in range(num_segments))
results = [r for r in results if r is not None]

# Aggregate results
segment_times = [r['start_min'] for r in results]
energy_bass = [r['bass'] for r in results]
energy_mids = [r['mids'] for r in results]
energy_highs = [r['highs'] for r in results]
tempos = [r['tempo'] for r in results]
mood_probs = [r['mood'] for r in results]
spectral_fluxes = [r['spectral_flux'] for r in results]

# Compute averages
avg_bass = float(np.mean([x for x in energy_bass if x > 0]))
avg_mids = float(np.mean([x for x in energy_mids if x > 0]))
avg_highs = float(np.mean([x for x in energy_highs if x > 0]))
median_tempo = float(np.nanmedian(tempos))
avg_mood = {
    'dark': float(np.mean([m['dark'] for m in mood_probs])),
    'euphoric': float(np.mean([m['euphoric'] for m in mood_probs])),
    'circuit': float(np.mean([m['circuit'] for m in mood_probs])),
    'funky': float(np.mean([m['funky'] for m in mood_probs])),
    'soulful': float(np.mean([m['soulful'] for m in mood_probs])),
    'deep': float(np.mean([m['deep'] for m in mood_probs])),
    'tribal': float(np.mean([m['tribal'] for m in mood_probs])),
    'progressive': float(np.mean([m['progressive'] for m in mood_probs])),
    'raw': float(np.mean([m['raw'] for m in mood_probs])),
    'ambient': float(np.mean([m['ambient'] for m in mood_probs])),
    'big_room': float(np.mean([m['big_room'] for m in mood_probs])),
    'electro': float(np.mean([m['electro'] for m in mood_probs])),
    'future': float(np.mean([m['future'] for m in mood_probs])),
    'vocal_chop': float(np.mean([m['vocal_chop'] for m in mood_probs]))
}
dominant_mood = max(avg_mood, key=avg_mood.get)

# Print summary
print(f"Average Bass Energy: {avg_bass:.4f}")
print(f"Average Mids Energy: {avg_mids:.4f}")
print(f"Average Highs Energy: {avg_highs:.4f}")
print(f"Median Tempo: {median_tempo:.2f} BPM")
print(f"Mood Averages: dark={avg_mood['dark']:.2f}, euphoric={avg_mood['euphoric']:.2f}, circuit={avg_mood['circuit']:.2f}, "
      f"funky={avg_mood['funky']:.2f}, soulful={avg_mood['soulful']:.2f}, deep={avg_mood['deep']:.2f}, "
      f"tribal={avg_mood['tribal']:.2f}, progressive={avg_mood['progressive']:.2f}, raw={avg_mood['raw']:.2f}, "
      f"ambient={avg_mood['ambient']:.2f}, big_room={avg_mood['big_room']:.2f}, electro={avg_mood['electro']:.2f}, "
      f"future={avg_mood['future']:.2f}, vocal_chop={avg_mood['vocal_chop']:.2f}")
print(f"Dominant Mood: {dominant_mood}")

# Save results
summary = {
    'file': audio_file,
    'duration_min': file_duration / 60.0,
    'avg_bass': avg_bass,
    'avg_mids': avg_mids,
    'avg_highs': avg_highs,
    'median_tempo': median_tempo,
    'avg_mood': avg_mood,
    'dominant_mood': dominant_mood,
    'segment_results': results
}
output_json = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_summary.json")
with open(output_json, 'w') as f:
    json.dump(summary, f, indent=4)

# Plot energy trends
plt.figure(figsize=(12, 6))
plt.plot(segment_times, energy_bass, label='Bass', color='#FF6B6B')
plt.plot(segment_times, energy_mids, label='Mids', color='#4ECDC4')
plt.plot(segment_times, energy_highs, label='Highs', color='#45B7D1')
plt.xlabel('Time (minutes)')
plt.ylabel('RMS Energy')
plt.title('Energy Trends Over Time')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_energy_trends.png"))
plt.close()

# Plot mood trends
plt.figure(figsize=(12, 10))
plt.plot(segment_times, [m['dark'] for m in mood_probs], label='Dark', color='#2D2D2D')
plt.plot(segment_times, [m['euphoric'] for m in mood_probs], label='Euphoric', color='#FFD60A')
plt.plot(segment_times, [m['circuit'] for m in mood_probs], label='Circuit', color='#8338EC')
plt.plot(segment_times, [m['funky'] for m in mood_probs], label='Funky', color='#FF006E')
plt.plot(segment_times, [m['soulful'] for m in mood_probs], label='Soulful', color='#00D4FF')
plt.plot(segment_times, [m['deep'] for m in mood_probs], label='Deep', color='#3A86FF')
plt.plot(segment_times, [m['tribal'] for m in mood_probs], label='Tribal', color='#FF9F1C')
plt.plot(segment_times, [m['progressive'] for m in mood_probs], label='Progressive', color='#06D6A0')
plt.plot(segment_times, [m['raw'] for m in mood_probs], label='Raw', color='#7209B7')
plt.plot(segment_times, [m['ambient'] for m in mood_probs], label='Ambient', color='#118AB2')
plt.plot(segment_times, [m['big_room'] for m in mood_probs], label='Big Room', color='#F72585')
plt.plot(segment_times, [m['electro'] for m in mood_probs], label='Electro', color='#7209B7')
plt.plot(segment_times, [m['future'] for m in mood_probs], label='Future', color='#4CC9F0')
plt.plot(segment_times, [m['vocal_chop'] for m in mood_probs], label='Vocal Chop', color='#F48C06')
plt.xlabel('Time (minutes)')
plt.ylabel('Mood Probability')
plt.title('Mood Trends Over Time (House, Techno, EDM)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_mood_trends.png"))
plt.close()
