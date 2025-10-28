#!/usr/bin/env python3
"""
Batch Audio Analyzer
Process multiple audio files with progress tracking and comprehensive reporting.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from audio_analyzer_improved import AudioAnalyzer, AudioConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

class BatchAnalyzer:
    """Batch processing analyzer with progress tracking and comprehensive reporting."""
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.analyzer = AudioAnalyzer(self.config)
        self.results = []
        
    def find_audio_files(self, input_dir: str, patterns: List[str] = None) -> List[str]:
        """Find audio files in directory with optional pattern matching."""
        if patterns is None:
            patterns = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.aac']
        
        audio_files = []
        input_path = Path(input_dir)
        
        if input_path.is_file():
            return [str(input_path)]
        
        for pattern in patterns:
            audio_files.extend(input_path.glob(pattern))
            audio_files.extend(input_path.glob(pattern.upper()))
        
        return [str(f) for f in audio_files if f.is_file()]
    
    def analyze_single_file(self, args: tuple) -> Optional[Dict]:
        """Analyze a single audio file (for parallel processing)."""
        audio_file, output_dir = args
        
        try:
            logger.info(f"Processing: {Path(audio_file).name}")
            results = self.analyzer.analyze_audio(audio_file, output_dir)
            self.analyzer.create_visualizations(results, output_dir)
            
            # Save individual results
            base_name = Path(audio_file).stem
            output_json = os.path.join(output_dir, f"{base_name}_summary.json")
            with open(output_json, 'w') as f:
                json.dump(results, f, indent=4)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")
            return None
    
    def analyze_batch(self, input_dir: str, output_dir: str, 
                     file_patterns: List[str] = None, 
                     max_workers: int = None) -> List[Dict]:
        """Analyze multiple audio files with progress tracking."""
        
        # Find audio files
        audio_files = self.find_audio_files(input_dir, file_patterns)
        
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process files
        results = []
        max_workers = max_workers or self.config.n_jobs
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.analyze_single_file, (f, output_dir)): f 
                for f in audio_files
            }
            
            # Process with progress bar
            with tqdm(total=len(audio_files), desc="Processing files") as pbar:
                for future in as_completed(future_to_file):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)
        
        self.results = results
        logger.info(f"Successfully processed {len(results)} files")
        
        return results
    
    def create_batch_report(self, results: List[Dict], output_dir: str):
        """Create comprehensive batch analysis report."""
        
        if not results:
            logger.warning("No results to report")
            return
        
        # Create summary DataFrame
        summary_data = []
        for result in results:
            summary_data.append({
                'file': Path(result['file']).name,
                'duration_min': result['duration_min'],
                'avg_bass': result['avg_bass'],
                'avg_mids': result['avg_mids'],
                'avg_highs': result['avg_highs'],
                'median_tempo': result['median_tempo'],
                'dominant_mood': result['dominant_mood']
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save CSV report
        csv_path = os.path.join(output_dir, 'batch_summary.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV report saved to {csv_path}")
        
        # Create visualizations
        self._create_batch_visualizations(df, results, output_dir)
        
        # Create JSON summary
        batch_summary = {
            'total_files': len(results),
            'total_duration_min': sum(r['duration_min'] for r in results),
            'avg_duration_min': df['duration_min'].mean(),
            'tempo_stats': {
                'mean': df['median_tempo'].mean(),
                'std': df['median_tempo'].std(),
                'min': df['median_tempo'].min(),
                'max': df['median_tempo'].max()
            },
            'mood_distribution': df['dominant_mood'].value_counts().to_dict(),
            'energy_stats': {
                'bass': {'mean': df['avg_bass'].mean(), 'std': df['avg_bass'].std()},
                'mids': {'mean': df['avg_mids'].mean(), 'std': df['avg_mids'].std()},
                'highs': {'mean': df['avg_highs'].mean(), 'std': df['avg_highs'].std()}
            },
            'files': summary_data
        }
        
        json_path = os.path.join(output_dir, 'batch_summary.json')
        with open(json_path, 'w') as f:
            json.dump(batch_summary, f, indent=4)
        
        logger.info(f"JSON summary saved to {json_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total files processed: {len(results)}")
        print(f"Total duration: {batch_summary['total_duration_min']:.1f} minutes")
        print(f"Average tempo: {batch_summary['tempo_stats']['mean']:.1f} BPM")
        print(f"Most common mood: {max(batch_summary['mood_distribution'], key=batch_summary['mood_distribution'].get)}")
        print("="*60)
    
    def _create_batch_visualizations(self, df: pd.DataFrame, results: List[Dict], output_dir: str):
        """Create comprehensive batch visualizations."""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Tempo distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(df['median_tempo'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Tempo (BPM)')
        ax.set_ylabel('Frequency')
        ax.set_title('Tempo Distribution Across Files')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tempo_distribution.png'), dpi=300)
        plt.close()
        
        # 2. Mood distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        mood_counts = df['dominant_mood'].value_counts()
        ax.pie(mood_counts.values, labels=mood_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('Mood Distribution Across Files')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mood_distribution.png'), dpi=300)
        plt.close()
        
        # 3. Energy correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        energy_cols = ['avg_bass', 'avg_mids', 'avg_highs', 'median_tempo']
        correlation_matrix = df[energy_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Energy and Tempo Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
        plt.close()
        
        # 4. Duration vs Tempo scatter
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(df['duration_min'], df['median_tempo'], 
                           c=df['avg_bass'], cmap='viridis', alpha=0.7, s=100)
        ax.set_xlabel('Duration (minutes)')
        ax.set_ylabel('Tempo (BPM)')
        ax.set_title('Duration vs Tempo (colored by bass energy)')
        plt.colorbar(scatter, label='Bass Energy')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'duration_tempo_scatter.png'), dpi=300)
        plt.close()
        
        logger.info("Batch visualizations created successfully")

def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch Audio Analyzer for Electronic Dance Music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_analyzer.py /path/to/audio/files
  python batch_analyzer.py /path/to/audio/files --output-dir results --max-workers 8
  python batch_analyzer.py song.mp3 --patterns "*.mp3" "*.wav"
        """
    )
    
    parser.add_argument("input", type=str, help="Input directory or file")
    parser.add_argument("--output-dir", type=str, default="batch_analysis_results",
                       help="Output directory for results")
    parser.add_argument("--patterns", nargs='+', default=['*.mp3', '*.wav', '*.flac'],
                       help="File patterns to match")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum number of parallel workers")
    parser.add_argument("--config", type=str,
                       help="Path to configuration JSON file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = AudioConfig()
        if args.config and os.path.isfile(args.config):
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                config = AudioConfig(**config_data)
        
        # Initialize batch analyzer
        batch_analyzer = BatchAnalyzer(config)
        
        # Process files
        results = batch_analyzer.analyze_batch(
            args.input, 
            args.output_dir, 
            args.patterns, 
            args.max_workers
        )
        
        # Create batch report
        batch_analyzer.create_batch_report(results, args.output_dir)
        
        logger.info(f"Batch analysis complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
