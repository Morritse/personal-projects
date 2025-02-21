import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
import pandas_ta as ta
import pytz
import matplotlib.pyplot as plt

from momentum_ai_trading.utils.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH
)
from momentum_ai_trading.utils.feature_engineering import prepare_features

def safe_numeric_convert(data):
    """
    Safely convert data to numeric with robust handling.
    
    Args:
        data (pd.Series or pd.DataFrame): Input data to convert
    
    Returns:
        pd.Series: Numeric series with non-convertible entries replaced
    """
    try:
        # Handle DataFrame input by converting all columns
        if isinstance(data, pd.DataFrame):
            converted_data = data.apply(pd.to_numeric, errors='coerce')
        else:
            # Handle Series input
            converted_data = pd.to_numeric(data, errors='coerce')
        
        # Fill NaN values with 0
        return converted_data.fillna(0)
    
    except Exception as e:
        print(f"Numeric conversion error: {e}")
        
        # Fallback conversion strategy
        try:
            # Convert to string, remove non-numeric characters
            if isinstance(data, pd.DataFrame):
                converted_data = data.astype(str).apply(lambda x: x.str.replace(r'[^\d.-]', '', regex=True))
            else:
                converted_data = data.astype(str).str.replace(r'[^\d.-]', '', regex=True)
            
            # Convert cleaned strings to numeric
            return pd.to_numeric(converted_data, errors='coerce').fillna(0)
        
        except Exception as e:
            print(f"Advanced conversion failed: {e}")
            # Return zeros if all conversion attempts fail
            return pd.Series([0] * len(data))

def handle_missing_values(data):
    """
    Robust handling of missing values with advanced type conversion.

    Args:
        data (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    # Create a copy to avoid modifying the original
    df = data.copy()

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Convert all columns to numeric
    df = df.apply(safe_numeric_convert)

    # Forward fill for trend indicators
    trend_cols = [col for col in df.columns if any(x in col.lower() for x in ['sma', 'ema', 'macd', 'adx'])]
    if trend_cols:
        df[trend_cols] = df[trend_cols].fillna(method='ffill')

    # Backward fill for remaining technical indicators
    tech_cols = [col for col in df.columns if any(x in col.lower() for x in ['rsi', 'volume', 'returns', 'volatility', 'lag'])]
    if tech_cols:
        # Preserve original order while removing duplicates
        unique_tech_cols = []
        for col in tech_cols:
            if col not in unique_tech_cols:
                unique_tech_cols.append(col)
        
        # Debug print statements
        print("Columns before backward fill:", df[unique_tech_cols].columns)
        
        # Fill missing values in place
        df[unique_tech_cols] = df[unique_tech_cols].fillna(method='bfill')
        
        # Debug print statements
        print("Columns after backward fill:", df[unique_tech_cols].columns)
    else:
        print("Warning: No technical columns found for backward filling.")

    # Final fill for any remaining NaNs
    df = df.fillna(0)

    # Replace infinite values
    df = df.replace([np.inf, -np.inf], [1e10, -1e10])

    return df


def add_lag_features(data, features, lags=[1, 2, 3, 5]):
    """
    Add lagged versions of important features with robust error handling.

    Args:
        data (pd.DataFrame): Input dataframe
        features (list): List of features to create lags for
        lags (list): List of lag values to create

    Returns:
        pd.DataFrame: DataFrame with added lag features
    """
    # Create a copy to avoid modifying the original DataFrame
    df = data.copy()

    # Ensure numeric data types
    for feature in features:
        if feature in df.columns:
            df[feature] = safe_numeric_convert(df[feature])

    for feature in features:
        # Check if feature exists in the DataFrame
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found. Skipping lag generation.")
            continue

        for lag in lags:
            lag_column = f'{feature}_lag_{lag}'

            # Check if lag column already exists
            if lag_column in df.columns:
                print(f"Warning: Lag column '{lag_column}' already exists. Skipping.")
                continue

            # Safely create lag feature
            try:
                # Debug statement to check the feature being processed
                print(f"Creating lag feature for {feature} with lag {lag}")
                shifted_series = df[feature].shift(lag).fillna(0)
                # Debug statement to check the shape and type
                print(f"Shifted series shape: {shifted_series.shape}, type: {type(shifted_series)}")
                
                # Ensure shifted_series is a Series
                if isinstance(shifted_series, pd.DataFrame):
                    shifted_series = shifted_series.iloc[:, 0]
                    print(f"Converted shifted_series to Series: shape {shifted_series.shape}, type: {type(shifted_series)}")
                
                df[lag_column] = shifted_series
            except Exception as e:
                print(f"Error creating lag feature for {feature}: {e}")

    return df

def add_interaction_features(data):
    """
    Add interaction features between important indicators with robust error handling.
    
    Args:
        data (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: DataFrame with added interaction features
    """
    # Create a copy to avoid modifying the original DataFrame
    df = data.copy()
    
    # Ensure numeric data types for all potential interaction columns
    numeric_columns = [
        'rsi', 'macd', 'adx', 'returns_volatility', 'volume', 
        'sma_50', 'sma_200', 'sma_20', 'trend_strength', 
        'di_plus', 'di_minus', 'close'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = safe_numeric_convert(df[col])
    
    # Safely add interaction features with fallback
    try:
        if 'rsi' in df.columns and 'macd' in df.columns:
            df['rsi_macd'] = df['rsi'] * df['macd']
    except Exception as e:
        print(f"Error creating rsi_macd interaction: {e}")
    
    # Trend strength interactions
    try:
        if all(col in df.columns for col in ['adx', 'di_plus', 'di_minus']):
            df['adx_di_plus'] = df['adx'] * df['di_plus']
            df['trend_strength'] = df['adx'] * (df['di_plus'] - df['di_minus'])
    except Exception as e:
        print(f"Error creating trend strength interactions: {e}")
    
    # Volume and price momentum
    try:
        if all(col in df.columns for col in ['volume', 'returns']):
            df['volume_price_momentum'] = df['volume'] * df['returns']
    except Exception as e:
        print(f"Error creating volume price momentum: {e}")
    
    # Moving average convergence
    try:
        if all(col in df.columns for col in ['sma_50', 'sma_200']):
            df['ma_convergence'] = (df['sma_50'] - df['sma_200']) / df['sma_200']
    except Exception as e:
        print(f"Error creating moving average convergence: {e}")
    
    # Additional trend features
    try:
        if all(col in df.columns for col in ['close', 'sma_50']):
            df['price_to_sma50'] = df['close'] / df['sma_50'] - 1
    except Exception as e:
        print(f"Error creating price to SMA50 feature: {e}")
    
    try:
        if all(col in df.columns for col in ['sma_20', 'sma_50']):
            df['sma20_to_sma50'] = df['sma_20'] / df['sma_50'] - 1
    except Exception as e:
        print(f"Error creating SMA20 to SMA50 feature: {e}")
    
    # Volatility and momentum interactions
    try:
        if all(col in df.columns for col in ['returns_volatility', 'rsi']):
            df['volatility_rsi'] = df['returns_volatility'] * df['rsi']
    except Exception as e:
        print(f"Error creating volatility RSI interaction: {e}")
    
    try:
        if all(col in df.columns for col in ['returns_volatility', 'trend_strength']):
            df['volatility_trend'] = df['returns_volatility'] * df['trend_strength']
    except Exception as e:
        print(f"Error creating volatility trend interaction: {e}")
    
    return df

def get_market_context(data):
    """
    Get market context for daily trading.

    Parameters:
    - data: DataFrame with recent daily price data and indicators
    """
    latest = data.iloc[-1]
    lookback = data.iloc[-20:]  # Look back 20 days

    context = {}

    # Trend analysis using moving averages and ADX
    context['sma_50_200_ratio'] = latest.get('sma_50', 1) / latest.get('sma_200', 1)
    context['adx'] = latest.get('adx', np.nan)
    context['di_plus'] = latest.get('di_plus', np.nan)
    context['di_minus'] = latest.get('di_minus', np.nan)
    context['ma_convergence'] = latest.get('ma_convergence', np.nan)
    context['rsi'] = latest.get('rsi', np.nan)
    context['macd'] = latest.get('macd', np.nan)

    # Determine trend status with more nuanced criteria
    if context['sma_50_200_ratio'] > 1.05 and context['adx'] > 25 and context['di_plus'] > context['di_minus']:
        trend_status = 'Strong Uptrend'
    elif context['sma_50_200_ratio'] > 1.02 and context['di_plus'] > context['di_minus']:
        trend_status = 'Moderate Uptrend'
    elif context['sma_50_200_ratio'] > 1.0:
        trend_status = 'Weak Uptrend'
    elif context['sma_50_200_ratio'] < 0.95 and context['adx'] > 25 and context['di_plus'] < context['di_minus']:
        trend_status = 'Strong Downtrend'
    elif context['sma_50_200_ratio'] < 0.98:
        trend_status = 'Moderate Downtrend'
    else:
        trend_status = 'Sideways'

    context['trend'] = trend_status

    # Volatility analysis with percentile-based thresholds
    current_vol = latest.get('returns_volatility', 0)
    vol_percentiles = np.percentile(lookback['returns_volatility'].dropna(), [20, 40, 60, 80])
    
    if current_vol > vol_percentiles[3]:
        vol_status = 'Very High'
    elif current_vol > vol_percentiles[2]:
        vol_status = 'High'
    elif current_vol > vol_percentiles[1]:
        vol_status = 'Moderate'
    elif current_vol > vol_percentiles[0]:
        vol_status = 'Low'
    else:
        vol_status = 'Very Low'
    
    context['volatility'] = vol_status

    # Enhanced momentum analysis
    rsi = latest.get('rsi', 50)
    macd = latest.get('macd', 0)
    macd_signal = latest.get('macd_signal', 0)
    rsi_trend = latest.get('rsi', 50) - lookback['rsi'].mean()
    
    if rsi > 70 and macd > macd_signal and rsi_trend > 0:
        momentum = 'Strong Bullish'
    elif rsi > 60 and macd > macd_signal:
        momentum = 'Moderate Bullish'
    elif rsi < 30 and macd < macd_signal and rsi_trend < 0:
        momentum = 'Strong Bearish'
    elif rsi < 40 and macd < macd_signal:
        momentum = 'Moderate Bearish'
    else:
        momentum = 'Neutral'
    
    context['momentum'] = momentum

    # Enhanced volume analysis
    volume_ma = data['volume'].rolling(window=20).mean().iloc[-1]
    volume_std = data['volume'].rolling(window=20).std().iloc[-1]
    volume_zscore = (latest['volume'] - volume_ma) / volume_std
    
    if volume_zscore > 2:
        volume_status = 'Very High'
    elif volume_zscore > 1:
        volume_status = 'High'
    elif volume_zscore < -2:
        volume_status = 'Very Low'
    elif volume_zscore < -1:
        volume_status = 'Low'
    else:
        volume_status = 'Normal'
    
    context['volume_status'] = volume_status

    return context

def get_signal_strength(prob, threshold, context):
    """Get signal strength and confidence level."""
    signal_strength = (prob - threshold) / threshold
    
    # Base signal on probability vs threshold
    if prob > threshold * 1.5:
        base_signal = 'STRONG BUY'
        confidence = 'Very High'
    elif prob > threshold * 1.2:
        base_signal = 'BUY'
        confidence = 'High'
    elif prob > threshold:
        base_signal = 'WEAK BUY'
        confidence = 'Moderate'
    else:
        base_signal = 'HOLD'
        confidence = 'Low'
    
    # Adjust confidence based on market context
    if base_signal != 'HOLD':
        if context['trend'] in ['Strong Uptrend', 'Moderate Uptrend']:
            confidence = 'Very High' if confidence == 'High' else 'High'
        elif context['momentum'] in ['Strong Bullish', 'Moderate Bullish']:
            confidence = 'High' if confidence == 'Moderate' else confidence
        elif context['volatility'] in ['High', 'Very High']:
            confidence = 'Moderate' if confidence == 'High' else confidence
    
    return base_signal, confidence, signal_strength

def get_summary_recommendation(context, recent_signals, importance):
    """Generate a summary recommendation based on all factors."""
    # Count signal types
    signal_counts = pd.Series([s[0] for s in recent_signals]).value_counts()
    
    # Calculate average probability and strength
    avg_prob = np.mean([s[2] for s in recent_signals])
    avg_strength = np.mean([s[3] for s in recent_signals])
    
    # Analyze trend
    prob_trend = [s[2] for s in recent_signals]
    strength_increasing = prob_trend[-1] > prob_trend[0]
    
    # Generate summary
    summary = []
    
    # Overall recommendation
    if 'STRONG BUY' in signal_counts and signal_counts['STRONG BUY'] >= 3:
        summary.append("STRONG BUY RECOMMENDATION")
    elif 'BUY' in signal_counts and avg_prob > 0.2:
        summary.append("BUY RECOMMENDATION")
    else:
        summary.append("HOLD RECOMMENDATION")
    
    # Market context support
    context_support = []
    if context['trend'] in ['Strong Uptrend', 'Moderate Uptrend']:
        context_support.append(f"- {context['trend']}")
    if context['momentum'] in ['Strong Bullish', 'Moderate Bullish']:
        context_support.append(f"- {context['momentum']} momentum")
    if context['rsi'] > 50 and context['rsi'] < 70:
        context_support.append(f"- Healthy RSI at {context['rsi']:.1f}")
    if context['macd'] > 0:
        context_support.append("- Positive MACD")
    
    if context_support:
        summary.append("\nSupporting Factors:")
        summary.extend(context_support)
    
    # Signal analysis
    summary.append("\nSignal Analysis:")
    summary.append(f"- Average probability: {avg_prob:.3f}")
    summary.append(f"- Average strength: {avg_strength:.2f}")
    if strength_increasing:
        summary.append("- Signal strength is increasing")
    
    # Risk factors
    risk_factors = []
    if context['volatility'] in ['High', 'Very High']:
        risk_factors.append(f"- {context['volatility']} volatility")
    if context['volume_status'] == 'Low':
        risk_factors.append("- Low volume")
    if context['rsi'] > 65:
        risk_factors.append("- RSI approaching overbought")
    
    if risk_factors:
        summary.append("\nRisk Factors to Consider:")
        summary.extend(risk_factors)
    
    return "\n".join(summary)

def test_daily_model(symbol):
    """Test model for daily trading."""
    print(f"\nTesting daily model on {symbol}...")

    # Load data
    data_path = os.path.join(PROCESSED_DATA_PATH, f"{symbol}_processed_daily.csv")
    data = pd.read_csv(data_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.sort_values('datetime', inplace=True)

    # Regenerate features using the original feature engineering function
    X, y = prepare_features(data)

    # Load saved features and importance
    saved_features = joblib.load(f"{MODEL_PATH}_features.pkl")
    importance_df = pd.read_csv(f"{MODEL_PATH}_importance.csv")

    # Handle feature name changes and compatibility
    if 'volatility' in saved_features and 'returns_volatility' not in X.columns:
        print("Warning: 'volatility' feature not found. Regenerating features.")
        X, y = prepare_features(data)
    
    # Replace 'volatility' with 'returns_volatility' if needed
    saved_features = ['returns_volatility' if f == 'volatility' else f for f in saved_features]

    # Ensure all saved features are present
    missing_features = set(saved_features) - set(X.columns)
    if missing_features:
        print(f"Warning: Missing features {missing_features}. Regenerating features.")
        # Regenerate features with additional indicators
        X, y = prepare_features(data)

    # Select only the saved features, in the original order
    X = X[saved_features]

    # Add lag and interaction features with robust handling
    important_features = ['rsi', 'macd', 'adx', 'returns_volatility', 'volume']
    X = add_lag_features(X, important_features)  # Call add_lag_features function

    # Handle missing values
    X = handle_missing_values(X)
    X = add_interaction_features(X)

    # Load model and scaler
    model = lgb.Booster(model_file=f"{MODEL_PATH}.txt")
    scaler = joblib.load(f"{MODEL_PATH}_scaler.pkl")

    # Ensure feature count matches saved model
    X = X[saved_features]

    # Scale features
    X_scaled = scaler.transform(X)

    # Get predictions
    probabilities = model.predict(X_scaled)

    # Get recent data for context
    recent_data = data.iloc[-50:].copy()  # Use last 50 days for context
    context = get_market_context(recent_data)
    
    # Print market context in a more organized format
    print("\n" + "="*50)
    print("CURRENT MARKET CONTEXT")
    print("="*50)
    context_format = {
        'trend': 'Market Trend',
        'momentum': 'Momentum',
        'volatility': 'Volatility',
        'volume_status': 'Volume',
        'rsi': 'RSI',
        'macd': 'MACD',
        'sma_50_200_ratio': 'SMA 50/200 Ratio',
        'adx': 'ADX',
        'di_plus': 'DI+',
        'di_minus': 'DI-'
    }
    
    for key, label in context_format.items():
        if key in context:
            value = context[key]
            if isinstance(value, float):
                print(f"{label:<20}: {value:>10.2f}")
            else:
                print(f"{label:<20}: {str(value):>10}")
    print("="*50 + "\n")

    # Calculate dynamic threshold based on recent probabilities
    recent_probs = probabilities[-20:]  # Last 20 days
    base_threshold = 0.3  # Lower base threshold
    dynamic_threshold = np.mean(recent_probs) + 0.5 * np.std(recent_probs)
    threshold = min(base_threshold, dynamic_threshold)  # Use the lower of the two

    # Adjust threshold based on market context
    if context['trend'] == 'Strong Uptrend':
        threshold *= 0.8  # More aggressive in strong uptrend
    elif context['trend'] == 'Moderate Uptrend':
        threshold *= 0.9
    elif context['momentum'] == 'Strong Bullish':
        threshold *= 0.85
    elif context['momentum'] == 'Moderate Bullish':
        threshold *= 0.95
    elif context['volatility'] in ['High', 'Very High']:
        threshold *= 1.1  # More conservative in high volatility

    # Plot probability distribution
    plt.figure(figsize=(12, 6))
    plt.hist(probabilities[-100:], bins=50, alpha=0.7, label='All Probabilities')  # Last 100 days
    plt.hist(recent_probs, bins=20, alpha=0.7, label='Recent Probabilities')  # Last 20 days
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
    plt.title('Probability Distribution (Last 100 Days)')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/probability_distribution.png')
    plt.close()

    # Store signals for summary
    recent_signals = []
    
    # Generate signals for the last 5 days with improved formatting
    print(f"\nPREDICTIONS (Last 5 Days) - Threshold: {threshold:.3f}")
    print("="*85)
    print(f"{'Date':<12} {'Signal':<12} {'Confidence':<10} {'Probability':>10} {'Strength':>10} {'Key Indicators':<30}")
    print("-" * 85)
    
    for i in range(-5, 0):
        prob = probabilities[i]
        signal, confidence, strength = get_signal_strength(prob, threshold, context)
        
        # Get key indicators
        key_indicators = []
        if context['rsi'] > 50:
            key_indicators.append("RSI>50")
        if context['macd'] > 0:
            key_indicators.append("MACD+")
        if context['sma_50_200_ratio'] > 1:
            key_indicators.append("MA+")
        
        date_str = data.iloc[i]['datetime'].strftime('%Y-%m-%d')
        print(f"{date_str:<12} {signal:<12} {confidence:<10} {prob:>10.3f} {strength:>10.2f}  {', '.join(key_indicators):<30}")
        
        recent_signals.append((signal, confidence, prob, strength))
    print("="*85 + "\n")

    # Print feature importance with better formatting
    print("\nTOP FEATURE IMPORTANCE")
    print("="*40)
    print(f"{'Feature':<15} {'Importance':>10}")
    print("-"*40)
    for _, row in importance_df.head().iterrows():
        print(f"{row['feature']:<15} {row['importance']:>10.0f}")
    print("="*40 + "\n")
    
    # Generate and print summary recommendation with clear formatting
    print("\n" + "="*50)
    print("SUMMARY RECOMMENDATION")
    print("="*50)
    print(get_summary_recommendation(context, recent_signals, importance_df))
    print("="*50)

if __name__ == "__main__":
    test_daily_model("AAPL")
