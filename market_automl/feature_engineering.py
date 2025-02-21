import pandas as pd
import pandas_ta as ta
import numpy as np
from config import (
    TECH_STOCKS,
    INDEX_FUNDS,
    TIME_LAGS,
    CORRELATION_WINDOW,
    FAST_PERIOD,
    SLOW_PERIOD,
    SIGNAL_PERIOD,
    RSI_PERIOD,
    BB_PERIOD,
    BB_STD,
    PRICE_FEATURES,
    MOMENTUM_INDICATORS,
    VOLATILITY_INDICATORS,
    TREND_INDICATORS
)

class FeatureEngineer:
    def __init__(self, data_dict):
        """
        Initialize feature engineer with dictionary of dataframes
        data_dict: {symbol: pd.DataFrame}
        """
        self.data_dict = data_dict
        self.features = {}
        
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators using pandas-ta"""
        # Create custom strategy with our indicators
        custom_strategy = ta.Strategy(
            name="custom_strategy",
            description="RSI, MACD, BB, SMA, EMA, ATR",
            ta=[
                {"kind": "rsi", "length": RSI_PERIOD},
                {"kind": "macd", "fast": FAST_PERIOD, "slow": SLOW_PERIOD, "signal": SIGNAL_PERIOD},
                {"kind": "bbands", "length": BB_PERIOD, "std": BB_STD},
                {"kind": "sma", "length": 20},
                {"kind": "sma", "length": 50},
                {"kind": "sma", "length": 200},
                {"kind": "ema", "length": 20},
                {"kind": "atr", "length": 14}
            ]
        )
        
        # Calculate all indicators
        df.ta.strategy(custom_strategy)
        return df
        
    def create_lagged_features(self, df, symbol):
        """Create lagged features for a given symbol"""
        feature_df = pd.DataFrame(index=df.index)
        
        # Create lagged features for price and volume
        for col in PRICE_FEATURES:
            for lag in TIME_LAGS:
                feature_df[f'{symbol}_{col}_lag_{lag}'] = df[col].shift(lag)
                
        # Create lagged returns
        for lag in TIME_LAGS:
            feature_df[f'{symbol}_return_lag_{lag}'] = df['close'].pct_change(lag)
            
        return feature_df
        
    def calculate_cross_correlations(self, tech_stock, index_fund):
        """Calculate rolling correlations between a tech stock and index fund"""
        tech_returns = self.data_dict[tech_stock]['close'].pct_change()
        index_returns = self.data_dict[index_fund]['close'].pct_change()
        
        correlation = tech_returns.rolling(CORRELATION_WINDOW).corr(index_returns)
        return correlation
        
    def create_market_features(self):
        """Create aggregate market features from index funds"""
        market_features = pd.DataFrame()
        
        # Calculate average market return and volatility
        index_returns = pd.DataFrame()
        for index in INDEX_FUNDS:
            index_returns[index] = self.data_dict[index]['close'].pct_change()
            
        market_features['market_return'] = index_returns.mean(axis=1)
        market_features['market_volatility'] = index_returns.std(axis=1)
        
        return market_features
        
    def engineer_features(self):
        """Main method to engineer all features"""
        # Calculate technical indicators for all symbols
        for symbol in TECH_STOCKS + INDEX_FUNDS:
            self.data_dict[symbol] = self.calculate_technical_indicators(self.data_dict[symbol])
            
        # Create feature matrix for each tech stock
        for tech_stock in TECH_STOCKS:
            # Initialize features for this stock
            stock_features = pd.DataFrame(index=self.data_dict[tech_stock].index)
            
            # Add technical indicators
            tech_df = self.data_dict[tech_stock]
            for indicator in MOMENTUM_INDICATORS + VOLATILITY_INDICATORS + TREND_INDICATORS:
                if indicator in tech_df.columns:
                    stock_features[f'{tech_stock}_{indicator}'] = tech_df[indicator]
            
            # Add lagged features
            lagged_features = self.create_lagged_features(tech_df, tech_stock)
            stock_features = pd.concat([stock_features, lagged_features], axis=1)
            
            # Add cross-correlations with index funds
            for index_fund in INDEX_FUNDS:
                correlation = self.calculate_cross_correlations(tech_stock, index_fund)
                stock_features[f'{tech_stock}_{index_fund}_correlation'] = correlation
            
            # Add market features
            market_features = self.create_market_features()
            stock_features = pd.concat([stock_features, market_features], axis=1)
            
            # Store features
            self.features[tech_stock] = stock_features.fillna(method='ffill').fillna(0)
            
        return self.features

    def prepare_for_automl(self, target_horizon=5):
        """Prepare features for AutoML training"""
        all_features = {}
        
        for tech_stock in TECH_STOCKS:
            # Get features for this stock
            features_df = self.features[tech_stock]
            
            # Create target variable (future returns)
            target = self.data_dict[tech_stock]['close'].pct_change(target_horizon).shift(-target_horizon)
            
            # Combine features and target
            final_df = pd.concat([features_df, target.rename('target')], axis=1)
            
            # Remove rows with NaN values
            final_df = final_df.dropna()
            
            all_features[tech_stock] = final_df
            
        return all_features

def main():
    """Test feature engineering pipeline"""
    from data_collection import DataCollector
    
    # Collect data
    collector = DataCollector()
    data = collector.fetch_all_data()
    
    # Engineer features
    engineer = FeatureEngineer(data)
    features = engineer.engineer_features()
    
    # Prepare for AutoML
    automl_features = engineer.prepare_for_automl()
    
    # Save features
    for symbol, df in automl_features.items():
        df.to_csv(f'data/processed/{symbol}_features.csv')

if __name__ == '__main__':
    main()
