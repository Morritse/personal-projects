import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

class PatternStats:
    """Maintain running statistics of pattern features across symbols."""
    
    def __init__(self, stats_file='pattern_statistics.json'):
        self.stats_file = stats_file
        self.feature_stats = self._load_stats()
        
        # Features that are always trivial/noisy
        self.blacklist = {
            'autocorrelation__lag_0',
            'partial_autocorrelation__lag_0',
            'symmetry_looking__r_0.05',
            'symmetry_looking__r_0.1',
            'symmetry_looking__r_0.15',
            'symmetry_looking__r_0.2',
        }
        
        # Load existing or initialize new statistics
        if self.feature_stats is None:
            print("Initializing new statistics file...")
            self.feature_stats = {
                'metadata': {
                    'symbols_analyzed': 0,
                    'last_updated': None,
                    'first_run': datetime.now().isoformat()
                },
                'features': {},
                'metrics': {}
            }
        else:
            print(f"Loaded existing statistics from {self.stats_file}")
            print(f"Baseline data from {self.feature_stats['metadata'].get('first_run', 'unknown')}")
            print(f"Last updated: {self.feature_stats['metadata'].get('last_updated', 'never')}")
            print(f"Symbols analyzed: {self.feature_stats['metadata'].get('symbols_analyzed', 0)}")
    
    def _load_stats(self):
        """Load existing statistics from file."""
        try:
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def _save_stats(self):
        """Save current statistics to file with backup."""
        # Create backup of existing file
        if Path(self.stats_file).exists():
            backup_file = self.stats_file + '.backup'
            try:
                import shutil
                shutil.copy2(self.stats_file, backup_file)
            except Exception as e:
                print(f"Warning: Could not create backup: {str(e)}")
        
        # Update metadata
        self.feature_stats['metadata']['last_updated'] = datetime.now().isoformat()
        
        # Save new stats
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.feature_stats, f, indent=2)
        except Exception as e:
            print(f"Error saving stats: {str(e)}")
            # Try to restore from backup
            if Path(backup_file).exists():
                shutil.copy2(backup_file, self.stats_file)
                print("Restored from backup")
    
    def update_stats(self, features: dict, pattern_metrics: dict):
        """Update running statistics with new data."""
        stats = self.feature_stats
        current_count = stats['metadata']['symbols_analyzed']
        stats['metadata']['symbols_analyzed'] = current_count + 1
        
        print(f"\nUpdating statistics (symbol {current_count + 1})...")
        
        # Update feature statistics
        for feat_name, feat_val in features.items():
            if feat_name in self.blacklist or pd.isna(feat_val):
                continue
                
            if feat_name not in stats['features']:
                stats['features'][feat_name] = {
                    'count': 0,
                    'mean': 0,
                    'M2': 0,  # For online variance calculation
                    'min': float('inf'),
                    'max': float('-inf'),
                    'values': []  # Keep last 1000 values for percentile calculation
                }
            
            feat_stats = stats['features'][feat_name]
            feat_stats['count'] += 1
            
            # Update running mean and variance (Welford's online algorithm)
            delta = feat_val - feat_stats['mean']
            feat_stats['mean'] += delta / feat_stats['count']
            delta2 = feat_val - feat_stats['mean']
            feat_stats['M2'] += delta * delta2
            
            # Update min/max
            feat_stats['min'] = min(feat_stats['min'], feat_val)
            feat_stats['max'] = max(feat_stats['max'], feat_val)
            
            # Update recent values for percentiles
            feat_stats['values'].append(feat_val)
            if len(feat_stats['values']) > 1000:
                feat_stats['values'].pop(0)
        
        # Update metric statistics similarly
        for metric_name, metric_val in pattern_metrics.items():
            if pd.isna(metric_val):
                continue
                
            if metric_name not in stats['metrics']:
                stats['metrics'][metric_name] = {
                    'count': 0,
                    'mean': 0,
                    'M2': 0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'values': []
                }
            
            metric_stats = stats['metrics'][metric_name]
            metric_stats['count'] += 1
            
            delta = metric_val - metric_stats['mean']
            metric_stats['mean'] += delta / metric_stats['count']
            delta2 = metric_val - metric_stats['mean']
            metric_stats['M2'] += delta * delta2
            
            metric_stats['min'] = min(metric_stats['min'], metric_val)
            metric_stats['max'] = max(metric_stats['max'], metric_val)
            
            metric_stats['values'].append(metric_val)
            if len(metric_stats['values']) > 1000:
                metric_stats['values'].pop(0)
        
        self._save_stats()
    
    def get_thresholds(self, percentile=95, min_samples=20):
        """
        Calculate thresholds based on collected statistics.
        Requires minimum number of samples for reliable thresholds.
        """
        if self.feature_stats['metadata']['symbols_analyzed'] < min_samples:
            print(f"\nWarning: Only {self.feature_stats['metadata']['symbols_analyzed']} samples")
            print(f"Thresholds may not be reliable until {min_samples} samples are collected")
        thresholds = {}
        
        for feat_name, feat_stats in self.feature_stats['features'].items():
            if feat_stats['count'] > 10:  # Need some minimum data
                values = np.array(feat_stats['values'])
                thresholds[feat_name] = np.percentile(values, percentile)
        
        for metric_name, metric_stats in self.feature_stats['metrics'].items():
            if metric_stats['count'] > 10:
                values = np.array(metric_stats['values'])
                thresholds[metric_name] = np.percentile(values, percentile)
        
        return thresholds
    
    def get_zscore(self, name: str, value: float) -> float:
        """Calculate z-score for a feature/metric value."""
        if name in self.feature_stats['features']:
            stats = self.feature_stats['features'][name]
        elif name in self.feature_stats['metrics']:
            stats = self.feature_stats['metrics'][name]
        else:
            return 0.0
        
        if stats['count'] < 2:  # Need at least 2 samples for variance
            return 0.0
            
        variance = stats['M2'] / (stats['count'] - 1)
        if variance == 0:
            return 0.0
            
        return (value - stats['mean']) / np.sqrt(variance)
    
    def analyze_log(self, log_file='standout_patterns.log'):
        """Analyze pattern log to find most common/interesting patterns."""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Split into entries
            entries = content.split('=' * 80)
            entries = [e.strip() for e in entries if e.strip()]
            
            # Parse entries
            patterns = []
            for entry in entries:
                lines = entry.split('\n')
                pattern = {}
                
                for line in lines:
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        pattern[key] = value
                
                if pattern:
                    patterns.append(pattern)
            
            # Analyze patterns
            analysis = {
                'total_entries': len(patterns),
                'symbols_with_high_scores': [],
                'common_patterns': {},
                'extreme_scores': []
            }
            
            for pattern in patterns:
                if 'Symbol' in pattern and 'Pattern Score' in pattern:
                    score = float(pattern['Pattern Score'])
                    if score >= 8:
                        analysis['symbols_with_high_scores'].append({
                            'symbol': pattern['Symbol'],
                            'score': score,
                            'time': pattern.get('Time', 'Unknown')
                        })
                
                if 'Major Patterns' in pattern:
                    major_patterns = pattern['Major Patterns'].split('\n')
                    for p in major_patterns:
                        if p.startswith('- '):
                            p = p[2:]  # Remove '- ' prefix
                            analysis['common_patterns'][p] = analysis['common_patterns'].get(p, 0) + 1
            
            # Sort and format results
            analysis['symbols_with_high_scores'].sort(key=lambda x: x['score'], reverse=True)
            analysis['common_patterns'] = dict(sorted(
                analysis['common_patterns'].items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            return analysis
            
        except FileNotFoundError:
            return {"error": "Log file not found"}
