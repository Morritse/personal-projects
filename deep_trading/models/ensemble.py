
import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

def calculate_features(df, all_data=None):
    """Calculate technical indicators and cross-instrument features."""
    # Basic features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log1p(df['returns'])
    df['volatility'] = df['returns'].rolling(20).std()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']

    # Moving averages and crossovers
    for window in [5, 10, 20, 50, 200]:
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        df[f'distance_to_sma_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']

    # Forward fill NaN values in moving averages
    sma_cols = [col for col in df.columns if 'sma_' in col or 'ema_' in col or 'distance_to_sma_' in col]
    df[sma_cols] = df[sma_cols].ffill().bfill()

    # Volume features
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ma'] = df['volume_ma'].ffill().bfill()
    df['volume_ratio'] = (df['volume'] / df['volume_ma']).fillna(1)

    # Technical indicators
    df['rsi'] = ta.momentum.rsi(df['close']).fillna(50)
    df['macd'] = ta.trend.macd_diff(df['close']).fillna(0)
    df['roc'] = df['close'].pct_change(10).fillna(0)
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close']).fillna(50)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close']).fillna(50)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close']).fillna(25)
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close']).fillna(0)
    df['dpo'] = ta.trend.dpo(df['close']).fillna(0)

    # Volatility indicators
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_width'] = ((bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()).fillna(0)
    df['bb_position'] = ((df['close'] - bb.bollinger_lband()) /
                        (bb.bollinger_hband() - bb.bollinger_lband() + 1e-7)).fillna(0.5)

    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close']).fillna(0)
    df['willr'] = ta.momentum.williams_r(df['high'], df['low'], df['close']).fillna(-50)

    # Cross-instrument features
    if all_data is not None:
        symbol = df['symbol'].iloc[0]

        # Market relative strength (vs SPY)
        spy_data = all_data[all_data['symbol'] == 'SPY'].copy()
        if len(spy_data) > 0:
            spy_returns = spy_data['close'].pct_change().fillna(0)
            df['market_relative_strength'] = (df['returns'] - spy_returns).fillna(0)
        else:
            df['market_relative_strength'] = 0

        # Sector relative strength (vs QQQ)
        qqq_data = all_data[all_data['symbol'] == 'QQQ'].copy()
        if len(qqq_data) > 0:
            qqq_returns = qqq_data['close'].pct_change().fillna(0)
            df['sector_relative_strength'] = (df['returns'] - qqq_returns).fillna(0)
        else:
            df['sector_relative_strength'] = 0

        # Correlation with other tech stocks
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD']
        for other in tech_stocks:
            if other != symbol:
                other_data = all_data[all_data['symbol'] == other].copy()
                if len(other_data) > 0:
                    other_returns = other_data['close'].pct_change().fillna(0)
                    corr = df['returns'].rolling(20).corr(other_returns).fillna(0)
                    df[f'corr_{other}'] = corr
                else:
                    df[f'corr_{other}'] = 0

    # Fill remaining NaN values
    df = df.fillna(0)

    # Target variable (after filling NaNs)
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

    return df

def prepare_data(api_key, api_secret, symbols, start_date, end_date):
    """Prepare data with cross-instrument features."""
    stock_client = StockHistoricalDataClient(api_key, api_secret)

    # First pass: get all raw data
    all_data = pd.DataFrame()
    for symbol in symbols:
        print(f"Processing {symbol}...")
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )

        bars = stock_client.get_stock_bars(request)
        df = bars.df.reset_index()
        df = df.rename(columns={'timestamp': 'datetime'})
        df['symbol'] = symbol
        all_data = pd.concat([all_data, df])

    # Second pass: calculate features with cross-instrument relationships
    processed_data = pd.DataFrame()
    for symbol in symbols:
        print(f"Processing {symbol} with cross-instrument features...")
        symbol_data = all_data[all_data['symbol'] == symbol].copy()
        symbol_data = calculate_features(symbol_data, all_data)
        processed_data = pd.concat([processed_data, symbol_data])

    return processed_data

def train_ensemble(data, test_size=0.2):
    """Train ensemble of models with improved balance."""
    # Prepare features and target
    feature_cols = [col for col in data.columns if col not in ['datetime', 'symbol', 'target']]
    X = data[feature_cols].values
    y = data['target'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time series split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Calculate class weights
    n_samples = len(y_train)
    n_positive = sum(y_train)
    n_negative = n_samples - n_positive
    class_weight = {
        0: n_samples / (2 * n_negative),
        1: n_samples / (2 * n_positive)
    }

    # Train models with balanced settings
    models = {
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=50,
            class_weight=class_weight,
            random_state=42
        ),
        'hist_gb': HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=200,
            learning_rate=0.1,
            min_samples_leaf=50,
            random_state=42
        ),
        'xgb': xgb.XGBClassifier(
            max_depth=3,
            n_estimators=200,
            scale_pos_weight=n_negative/n_positive,
            min_child_weight=50,
            random_state=42
        )
    }

    # Train and evaluate each model
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
        probabilities[name] = model.predict_proba(X_test)[:, 1]

        print(f"\n{name.upper()} Results:")
        print(f"Accuracy: {accuracy_score(y_test, predictions[name]):.4f}")
        print(f"Precision: {precision_score(y_test, predictions[name]):.4f}")
        print(f"Recall: {recall_score(y_test, predictions[name]):.4f}")
        print(f"F1: {f1_score(y_test, predictions[name]):.4f}")

        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nTop 10 Important Features:")
            print(importances.head(10))

    # Ensemble prediction
    ensemble_probs = np.mean([probs for probs in probabilities.values()], axis=0)
    ensemble_preds = (ensemble_probs > 0.6).astype(int)

    print("\nENSEMBLE Results:")
    print(f"Accuracy: {accuracy_score(y_test, ensemble_preds):.4f}")
    print(f"Precision: {precision_score(y_test, ensemble_preds):.4f}")
    print(f"Recall: {recall_score(y_test, ensemble_preds):.4f}")
    print(f"F1: {f1_score(y_test, ensemble_preds):.4f}")

    return models, scaler, feature_cols, ensemble_probs

def predict_next_move(models, scaler, feature_cols, new_data, all_data):
    """Predict next market move with cross-instrument analysis."""
    # Basic prediction
    X = new_data[feature_cols].values
    X_scaled = scaler.transform(X)

    # Get predictions from each model
    probabilities = []
    predictions = []
    for name, model in models.items():
        prob = model.predict_proba(X_scaled)[:, 1]
        pred = model.predict(X_scaled)
        probabilities.append(prob)
        predictions.append(pred)

    # Ensemble probability
    ensemble_prob = np.mean(probabilities, axis=0)

    # Get symbol-specific data
    symbol = new_data['symbol'].iloc[0]

    # Safely get market data
    spy_data = all_data[all_data['symbol'] == 'SPY']
    qqq_data = all_data[all_data['symbol'] == 'QQQ']

    if len(spy_data) > 0 and len(qqq_data) > 0:
        spy_latest = spy_data.iloc[-1]
        qqq_latest = qqq_data.iloc[-1]

        # Calculate market context
        market_condition = {
            'spy_trend': spy_latest['distance_to_sma_200'] if 'distance_to_sma_200' in spy_latest else 0,
            'qqq_trend': qqq_latest['distance_to_sma_200'] if 'distance_to_sma_200' in qqq_latest else 0,
            'sector_volume': qqq_latest['volume_ratio'] if 'volume_ratio' in qqq_latest else 1.0,
            'market_regime': 'Bearish' if spy_latest.get('distance_to_sma_200', 0) < 0 else 'Bullish'
        }

        # Calculate relative strength if we have the data
        if len(new_data) >= 20 and len(qqq_data) >= 20:
            symbol_return = (new_data['close'].iloc[-1] / new_data['close'].iloc[-20])
            qqq_return = (qqq_latest['close'] / qqq_data['close'].iloc[-20])
            market_condition['relative_strength'] = symbol_return / qqq_return
        else:
            market_condition['relative_strength'] = 1.0
    else:
        # Default market condition if we don't have market data
        market_condition = {
            'spy_trend': 0,
            'qqq_trend': 0,
            'relative_strength': 1.0,
            'sector_volume': 1.0,
            'market_regime': 'Neutral'
        }

    # Adjust threshold based on conditions
    base_threshold = 0.6
    if market_condition['market_regime'] == 'Bearish':
        base_threshold += 0.05
    if new_data['adx'].iloc[0] < 20:
        base_threshold += 0.05
    if new_data['volume_ratio'].iloc[0] < 0.7:
        base_threshold += 0.05

    prediction = (ensemble_prob > base_threshold).astype(int)

    return prediction, {
        'probability': ensemble_prob[0],
        'model_agreement': np.mean([1 if p == predictions[0] else 0 for p in predictions]),
        'trend_strength': new_data['adx'].iloc[0],
        'volume_quality': new_data['volume_ratio'].iloc[0],
        'threshold_used': base_threshold,
        'market_context': market_condition
    }

def predict_next_move(models, scaler, feature_cols, new_data, all_data):
    """Predict next market move with enhanced divergence analysis."""
    # Basic prediction
    X = new_data[feature_cols].values
    X_scaled = scaler.transform(X)

    # Get predictions and probabilities
    probabilities = []
    predictions = []
    for name, model in models.items():
        prob = model.predict_proba(X_scaled)[:, 1]
        pred = model.predict(X_scaled)
        probabilities.append(prob)
        predictions.append(pred)

    ensemble_prob = np.mean(probabilities, axis=0)

    # Market context
    spy_data = all_data[all_data['symbol'] == 'SPY'].iloc[-1]
    qqq_data = all_data[all_data['symbol'] == 'QQQ'].iloc[-1]

    market_condition = {
        'spy_trend': spy_data['distance_to_sma_200'],
        'qqq_trend': qqq_data['distance_to_sma_200'],
        'sector_volume': qqq_data['volume_ratio'],
        'market_regime': 'Bullish' if spy_data['distance_to_sma_200'] > 0 else 'Bearish'
    }

    # Calculate relative strength
    symbol = new_data['symbol'].iloc[0]
    symbol_data = all_data[all_data['symbol'] == symbol]
    if len(symbol_data) >= 20:
        symbol_return = (new_data['close'].iloc[-1] / symbol_data['close'].iloc[-20])
        qqq_return = (qqq_data['close'] / all_data[all_data['symbol'] == 'QQQ'].iloc[-20]['close'])
        market_condition['relative_strength'] = symbol_return / qqq_return
    else:
        market_condition['relative_strength'] = 1.0

    # Dynamic threshold with divergence handling
    base_threshold = 0.6

    # Market regime adjustments
    if market_condition['market_regime'] == 'Bullish':
        if market_condition['relative_strength'] > 1.0:
            base_threshold += 0.2  # Strong stocks in bull market need very strong signal
        else:
            base_threshold += 0.15  # Weak stocks in bull market need strong signal

    # Technical condition adjustments
    if new_data['adx'].iloc[0] < 20:
        base_threshold += 0.05  # Weak trend
    if new_data['volume_ratio'].iloc[0] < 0.7:
        base_threshold += 0.05  # Low volume

    # Price level adjustments
    if new_data['distance_to_sma_200'].iloc[0] > 0:
        base_threshold += 0.05  # Above 200 SMA needs stronger signal

    # Momentum adjustments
    if new_data['dpo'].iloc[0] > 0:
        base_threshold += 0.05  # Positive DPO needs stronger signal

    prediction = (ensemble_prob > base_threshold).astype(int)

    return prediction, {
        'probability': ensemble_prob[0],
        'model_agreement': np.mean([1 if p == predictions[0] else 0 for p in predictions]),
        'trend_strength': new_data['adx'].iloc[0],
        'volume_quality': new_data['volume_ratio'].iloc[0],
        'threshold_used': base_threshold,
        'market_context': market_condition
    }

# Update prediction display
print("\nEnhanced Predictions with Market Context:")
print("-" * 50)
for symbol in symbols:
    latest_data = data[data['symbol'] == symbol].tail(1)
    if len(latest_data) > 0:  # Make sure we have data for this symbol
        prediction, metrics = predict_next_move(models, scaler, feature_cols, latest_data, data)

        print(f"\n{symbol}:")
        print(f"Direction: {'UP' if prediction[0] == 1 else 'DOWN'}")
        print(f"Base Probability: {metrics['probability']:.2%}")
        print(f"Model Agreement: {metrics['model_agreement']:.2%}")

        print(f"\nSignal Quality Metrics:")
        print(f"- Trend Strength (ADX): {metrics['trend_strength']:.1f}")
        print(f"- Volume Quality: {metrics['volume_quality']:.2f}")
        print(f"- Threshold Used: {metrics['threshold_used']:.2f}")

        print(f"\nMarket Context:")
        print(f"- Market Regime: {metrics['market_context']['market_regime']}")
        print(f"- SPY Trend: {metrics['market_context']['spy_trend']:.2%}")
        print(f"- QQQ Trend: {metrics['market_context']['qqq_trend']:.2%}")
        print(f"- Relative Strength: {metrics['market_context']['relative_strength']:.2f}")
        print(f"- Sector Volume: {metrics['market_context']['sector_volume']:.2f}")

        print("\nKey Technical Indicators:")
        feature_values = latest_data[['volume_ratio', 'dpo', 'adx', 'distance_to_sma_5', 'distance_to_sma_200']].iloc[0]
        for feature, value in feature_values.items():
            print(f"{feature}: {value:.4f}")
    else:
        print(f"\n{symbol}: No data available")

# Define parameters
symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'SPY', 'QQQ']
start_date = datetime(2019, 1, 1)
end_date = datetime.now()

# Get data
data = prepare_data("PKMNC0G9A58H9C2IVKP2", "Z0pq7QurDcxCMcgCO8LAtFdWqHy6RHro6lr6fmxi", symbols, start_date, end_date)

# Verify no NaN values
print("\nChecking for NaN values before training:")
print(data.isna().sum().sum())

# Train models
models, scaler, feature_cols, ensemble_probs = train_ensemble(data)

# Make predictions
print("\nEnhanced Predictions with Market Context:")
print("-" * 50)
for symbol in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD']:  # Only predict for tech stocks
    latest_data = data[data['symbol'] == symbol].tail(1)
    if len(latest_data) > 0:
        prediction, metrics = predict_next_move(models, scaler, feature_cols, latest_data, data)

        print(f"\n{symbol}:")
        print(f"Direction: {'UP' if prediction[0] == 1 else 'DOWN'}")
        print(f"Base Probability: {metrics['probability']:.2%}")
        print(f"Model Agreement: {metrics['model_agreement']:.2%}")

        print(f"\nSignal Quality Metrics:")
        print(f"- Trend Strength (ADX): {metrics['trend_strength']:.1f}")
        print(f"- Volume Quality: {metrics['volume_quality']:.2f}")
        print(f"- Threshold Used: {metrics['threshold_used']:.2f}")

        print(f"\nMarket Context:")
        print(f"- Market Regime: {metrics['market_context']['market_regime']}")
        print(f"- SPY Trend: {metrics['market_context']['spy_trend']:.2%}")
        print(f"- QQQ Trend: {metrics['market_context']['qqq_trend']:.2%}")
        print(f"- Relative Strength: {metrics['market_context']['relative_strength']:.2f}")
        print(f"- Sector Volume: {metrics['market_context']['sector_volume']:.2f}")

        print("\nKey Technical Indicators:")
        feature_values = latest_data[['volume_ratio', 'dpo', 'adx', 'distance_to_sma_5', 'distance_to_sma_200']].iloc[0]
        for feature, value in feature_values.items():
            print(f"{feature}: {value:.4f}")
    else:
        print(f"\n{symbol}: No data available")