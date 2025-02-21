import numpy as np
import pandas as pd
import ta

def prepare_features(df, target_column='close', forward_period=1):
    """
    Prepare features for machine learning model.
    
    Args:
        df (pd.DataFrame): Input dataframe with financial data
        target_column (str): Column to use for creating target
        forward_period (int): Number of periods forward for returns calculation
    
    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable (forward returns)
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Ensure dataframe is sorted by datetime
    df = df.sort_values('datetime')
    
    # Calculate forward returns as target
    df['forward_returns'] = df[target_column].pct_change(periods=forward_period).shift(-forward_period)
    
    # Technical Indicators
    # Moving Averages
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
    
    df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
    
    # Price to Moving Average Ratios
    df['price_to_sma10'] = df['close'] / df['sma_10'] - 1
    df['price_to_sma20'] = df['close'] / df['sma_20'] - 1
    df['price_to_sma50'] = df['close'] / df['sma_50'] - 1
    df['price_to_sma200'] = df['close'] / df['sma_200'] - 1
    
    # Moving Average Crossovers
    df['sma_10_20_cross'] = df['sma_10'] / df['sma_20'] - 1
    df['sma_20_50_cross'] = df['sma_20'] / df['sma_50'] - 1
    df['sma_50_200_cross'] = df['sma_50'] / df['sma_200'] - 1
    
    # Momentum Indicators
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Trend Indicators
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    df['adx'] = adx.adx()
    df['di_plus'] = adx.adx_pos()
    df['di_minus'] = adx.adx_neg()
    
    # Volatility Indicators
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    
    # Volume Indicators
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Returns and Volatility
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log1p(df['returns'])
    df['returns_volatility'] = df['returns'].rolling(window=20).std()
    
    # Trend Strength Features
    df['trend_strength'] = (df['sma_50'] - df['sma_200']) / df['sma_200']
    df['momentum_strength'] = df['rsi'] * df['macd_diff']
    
    # Volatility Regime
    df['volatility_regime'] = (df['returns_volatility'] / 
                              df['returns_volatility'].rolling(window=100).mean())
    
    # Price Patterns
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Advanced Features
    df['rsi_ma_cross'] = df['rsi'] - df['rsi'].rolling(window=10).mean()
    df['volume_price_trend'] = (df['volume'] * (df['close'] - df['close'].shift(1))).cumsum()
    df['price_momentum'] = df['close'] / df['close'].shift(10) - 1
    
    # Interaction Features
    df['rsi_macd'] = df['rsi'] * df['macd']
    df['trend_vol'] = df['trend_strength'] * df['returns_volatility']
    df['volume_volatility'] = df['volume_ratio'] * df['returns_volatility']
    
    # Drop NaN values
    df_features = df.dropna()
    
    # Separate features and target
    feature_columns = [col for col in df_features.columns 
                      if col not in ['datetime', 'forward_returns', target_column]]
    
    X = df_features[feature_columns]
    y = df_features['forward_returns']
    
    return X, y

def select_important_features(X, importance_threshold=0.01):
    """
    Select features based on importance threshold.
    
    Args:
        X (pd.DataFrame): Feature matrix
        importance_threshold (float): Minimum feature importance to keep
    
    Returns:
        pd.DataFrame: Filtered feature matrix
    """
    # Load feature importance from saved CSV
    importance_df = pd.read_csv('models/saved_models/momentum_model_importance.csv')
    
    # Filter features above threshold
    important_features = importance_df[importance_df['importance'] >= importance_threshold]['feature']
    
    return X[important_features]
