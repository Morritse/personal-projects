import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class QuantModels:
    def __init__(self, data):
        self.data = data.copy()
        self.models = {}
        self.scaler = StandardScaler()
        
    def add_features(self):
        """Add advanced technical features for ML models"""
        df = self.data.copy()
        
        # Momentum features
        df['returns'] = df['close'].pct_change()
        df['returns_volatility'] = df['returns'].rolling(window=20).std()
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['price_to_sma20'] = df['close'] / df['sma_20']
        df['price_to_sma50'] = df['close'] / df['sma_50']
        
        # Volatility
        df['atr'] = self._calculate_atr(df)
        df['bollinger_upper'] = df['sma_20'] + (df['close'].rolling(window=20).std() * 2)
        df['bollinger_lower'] = df['sma_20'] - (df['close'].rolling(window=20).std() * 2)
        df['bollinger_position'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Momentum indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        
        # Clean up NaN values
        df = df.dropna()
        
        self.data = df
        return df
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD and Signal line"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        return macd, signal_line

    def prepare_ml_data(self, prediction_window=5):
        """Prepare data for ML models"""
        df = self.data.copy()
        
        # Features for prediction
        feature_columns = [
            'returns', 'returns_volatility', 'price_to_sma20', 'price_to_sma50',
            'atr', 'bollinger_position', 'volume_ratio', 'rsi', 'macd'
        ]
        
        # Create target: 1 if price increases over prediction window, 0 otherwise
        df['target'] = (df['close'].shift(-prediction_window) > df['close']).astype(int)
        
        # Prepare features and target
        X = df[feature_columns].values[:-prediction_window]
        y = df['target'].values[:-prediction_window]
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train/test split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train Random Forest and XGBoost models"""
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        self.models['rf'] = rf_model
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            eval_metric='logloss',
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgb'] = xgb_model
        
        return self.models

class Backtester:
    def __init__(self, data, models, scaler, initial_capital=100000):
        self.data = data
        self.models = models
        self.scaler = scaler  # Store the scaler
        self.initial_capital = initial_capital
        self.positions = []
        self.portfolio_value = []

        # Add these methods to the Backtester class in quant_models.py:

    def calculate_metrics(self):
        """Calculate backtest performance metrics"""
        portfolio_values = pd.Series(self.portfolio_value)
        returns = portfolio_values.pct_change()
        
        metrics = {
            'total_return': (portfolio_values.iloc[-1] - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 and returns.std() != 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(),
            'num_trades': len(self.positions),
            'final_value': portfolio_values.iloc[-1] if len(portfolio_values) > 0 else self.initial_capital
        }
        
        return metrics

    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        if not self.portfolio_value:
            return 0
            
        portfolio_values = pd.Series(self.portfolio_value)
        rolling_max = portfolio_values.expanding().max()
        drawdowns = portfolio_values / rolling_max - 1
        return drawdowns.min()

    def run_backtest(self):
        """Run backtest using trained models"""
        df = self.data.copy()
        capital = self.initial_capital
        position = 0  # 0: no position, 1: long
        
        feature_columns = [
            'returns', 'returns_volatility', 'price_to_sma20', 'price_to_sma50',
            'atr', 'bollinger_position', 'volume_ratio', 'rsi', 'macd'
        ]
        
        for i in range(len(df)-1):
            # Skip if not enough data for features
            if i < 50:  # Need enough data for moving averages
                continue
                
            # Prepare features
            features = df[feature_columns].iloc[i:i+1].values
            features = self.scaler.transform(features)  # Use the passed scaler
            
            # Get predictions from both models
            rf_pred = self.models['rf'].predict(features)[0]
            xgb_pred = self.models['xgb'].predict(features)[0]
            
            # Rest of the method remains the same...
            
            # Ensemble prediction (both models must agree)
            buy_signal = rf_pred == 1 and xgb_pred == 1
            sell_signal = rf_pred == 0 and xgb_pred == 0
            
            # Execute trades
            if buy_signal and position == 0:
                position = 1
                entry_price = df['close'].iloc[i]
                self.positions.append(('BUY', entry_price, df.index[i]))
                
            elif sell_signal and position == 1:
                position = 0
                exit_price = df['close'].iloc[i]
                self.positions.append(('SELL', exit_price, df.index[i]))
            
            # Update portfolio value
            if position == 1:
                capital *= (1 + df['returns'].iloc[i])
            self.portfolio_value.append(capital)
        
        return self.calculate_metrics()