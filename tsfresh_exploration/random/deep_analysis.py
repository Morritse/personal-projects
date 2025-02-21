import numpy as np
import pandas as pd
from tsfresh import extract_features

def extract_time_series_features(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    """
    Extract deep, fundamental patterns from raw price and volume data.
    Uses tsfresh to compute hundreds of time series characteristics.
    """
    print(f"\nExtracting deep patterns for {symbol}...")
    print("This may take a while...")
    
    # Calculate returns for pattern analysis
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    
    # Prepare data structures for pattern extraction
    features_dict = {}
    pattern_metrics = {}
    
    # Extract features from raw data
    for column in ['close', 'volume', 'returns', 'high', 'low']:
        print(f"\nAnalyzing {column} patterns...")
        data_for_tsfresh = pd.DataFrame({
            'id': symbol,
            'time': range(len(df)),
            'value': df[column].fillna(0)
        })
        
        # Use comprehensive feature extraction (all possible features)
        extracted = extract_features(
            data_for_tsfresh,
            column_id='id',
            column_sort='time',
            column_value='value',
            disable_progressbar=True
        )
        features_dict[column] = extracted
        
        # Calculate fundamental metrics
        series = df[column].dropna()
        if len(series) > 0:
            # Structural breaks (changes in underlying distribution)
            changes = np.abs(np.diff(series))
            pattern_metrics[f'{column}_structural_breaks'] = np.sum(changes > np.std(changes) * 2) / len(changes)
            
            # Distribution characteristics
            pattern_metrics[f'{column}_skew'] = series.skew()
            pattern_metrics[f'{column}_kurtosis'] = series.kurtosis()
            
            # Entropy (measure of randomness/predictability)
            hist, _ = np.histogram(series, bins='auto', density=True)
            pattern_metrics[f'{column}_entropy'] = -np.sum(hist * np.log(hist + 1e-10))
    
    # Analyze price-volume relationships
    price_volume_corr = df['close'].corr(df['volume'])
    pattern_metrics['price_volume_correlation'] = price_volume_corr
    
    # Analyze high-low range behavior
    df['range'] = df['high'] - df['low']
    range_series = df['range'].dropna()
    if len(range_series) > 0:
        pattern_metrics['range_variability'] = range_series.std() / range_series.mean()
    
    # Combine all features with unique column names
    all_features = pd.DataFrame()
    for column, feature_df in features_dict.items():
        feature_df.columns = [f"{column}_{col}" for col in feature_df.columns]
        if all_features.empty:
            all_features = feature_df
        else:
            all_features = pd.concat([all_features, feature_df], axis=1)
    
    print("\nFeature extraction complete!")
    return all_features, pattern_metrics

from pattern_stats import PatternStats

# Global stats tracker
stats = PatternStats()

def detect_patterns(feature_row: pd.DataFrame, pattern_metrics: dict) -> dict:
    """
    Detect fundamental patterns using unsupervised analysis of time series characteristics.
    Uses dynamic thresholds based on collected statistics.
    """
    print("\nAnalyzing patterns deeply...")
    
    patterns = {
        'score': 0.0,
        'notes': [],
        'major_patterns': [],
        'correlations': [],
        'regime_changes': []
    }
    
    # Update running statistics
    row_dict = feature_row.to_dict(orient='index')
    if len(row_dict) == 0:
        return patterns
    
    features = list(row_dict.values())[0]
    stats.update_stats(features, pattern_metrics)
    
    # Get current thresholds based on collected statistics
    thresholds = stats.get_thresholds(percentile=98)  # More stringent percentile
    min_zscore = 2.0  # Only flag patterns with z-score > 2 (roughly top 2.5%)
    
    # Track z-scores for different aspects
    aspect_scores = {
        'energy': [],
        'entropy': [],
        'autocorrelation': [],
        'asymmetry': [],
        'time_reversal': []
    }
    
    # Analyze features using z-scores and dynamic thresholds
    for feat_name, feat_val in features.items():
        if pd.isna(feat_val) or feat_name in stats.blacklist:
            continue
            
        # Get z-score for this feature
        zscore = stats.get_zscore(feat_name, feat_val)
        threshold = thresholds.get(feat_name)
        
        if 'abs_energy' in feat_name:
            aspect_scores['energy'].append(abs(zscore))
            if threshold and feat_val > threshold and abs(zscore) > min_zscore:
                patterns['notes'].append(
                    f"Extreme energy in {feat_name}: {feat_val:.2f} (z-score: {zscore:.2f})"
                )
        
        elif 'entropy' in feat_name:
            aspect_scores['entropy'].append(abs(zscore))
            if threshold and feat_val > threshold and abs(zscore) > min_zscore:
                patterns['notes'].append(
                    f"Extreme entropy in {feat_name}: {feat_val:.2f} (z-score: {zscore:.2f})"
                )
        
        elif 'autocorrelation' in feat_name and 'lag' in feat_name:
            # Extract lag number
            try:
                lag = int(feat_name.split('lag_')[1].split('__')[0])
                if lag > 1:  # Ignore short lags
                    aspect_scores['autocorrelation'].append(abs(zscore))
                    if threshold and abs(feat_val) > threshold and abs(zscore) > min_zscore:
                        patterns['notes'].append(
                            f"Strong autocorrelation at lag {lag}: {feat_val:.2f} (z-score: {zscore:.2f})"
                        )
            except ValueError:
                continue
        
        elif 'time_reversal_asymmetry' in feat_name:
            aspect_scores['time_reversal'].append(abs(zscore))
            if threshold and abs(feat_val) > threshold and abs(zscore) > min_zscore:
                patterns['notes'].append(
                    f"Extreme time asymmetry in {feat_name}: {feat_val:.2f} (z-score: {zscore:.2f})"
                )
        
        elif 'symmetry' in feat_name:
            aspect_scores['asymmetry'].append(abs(zscore))
            if threshold and abs(feat_val) > threshold and abs(zscore) > min_zscore:
                patterns['notes'].append(
                    f"Strong asymmetry in {feat_name}: {feat_val:.2f} (z-score: {zscore:.2f})"
                )
    
    # Analyze metrics using z-scores and dynamic thresholds
    print("Analyzing structural patterns...")
    for key, value in pattern_metrics.items():
        zscore = stats.get_zscore(key, value)
        threshold = thresholds.get(key)
        
        if 'structural_breaks' in key:
            if threshold and value > threshold and abs(zscore) > min_zscore:
                patterns['regime_changes'].append(
                    f"Major structural breaks in {key.split('_')[0]} "
                    f"({value:.1%} of moves, z-score: {zscore:.2f})"
                )
        
        elif 'skew' in key:
            if threshold and abs(value) > threshold and abs(zscore) > min_zscore:
                patterns['major_patterns'].append(
                    f"Extreme skew in {key.split('_')[0]} distribution "
                    f"({value:.2f}, z-score: {zscore:.2f})"
                )
        
        elif 'kurtosis' in key:
            if threshold and abs(value) > threshold and abs(zscore) > min_zscore:
                patterns['major_patterns'].append(
                    f"Extreme tail behavior in {key.split('_')[0]} distribution "
                    f"({value:.2f}, z-score: {zscore:.2f})"
                )
    
    # Analyze correlations and variability
    corr_val = abs(pattern_metrics.get('price_volume_correlation', 0))
    corr_zscore = stats.get_zscore('price_volume_correlation', corr_val)
    if (thresholds.get('price_volume_correlation') and 
        corr_val > thresholds['price_volume_correlation'] and 
        abs(corr_zscore) > min_zscore):
        patterns['correlations'].append(
            f"Extreme price-volume correlation: {pattern_metrics['price_volume_correlation']:.2f} "
            f"(z-score: {corr_zscore:.2f})"
        )
    
    var_val = pattern_metrics.get('range_variability', 0)
    var_zscore = stats.get_zscore('range_variability', var_val)
    if (thresholds.get('range_variability') and 
        var_val > thresholds['range_variability'] and 
        abs(var_zscore) > min_zscore):
        patterns['major_patterns'].append(
            f"Extreme price range variability: {var_val:.2f}x normal "
            f"(z-score: {var_zscore:.2f})"
        )
    
    # Calculate overall pattern score from z-scores
    aspect_weights = {
        'energy': 0.2,
        'entropy': 0.2,
        'autocorrelation': 0.2,
        'asymmetry': 0.2,
        'time_reversal': 0.2
    }
    
    for aspect, scores in aspect_scores.items():
        if scores:  # If we have any scores for this aspect
            # Use the maximum absolute z-score for each aspect
            max_zscore = max(scores)
            # Use z-score directly for scoring, scaled to 0-1
            score = min(1.0, abs(max_zscore) / 4)  # z-score of 4 maps to 1.0
            patterns['score'] += score * aspect_weights[aspect]
    
    # Cap the score
    patterns['score'] = min(10, patterns['score'])
    
    print("Pattern detection complete!")
    return patterns
