import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from quant_models import QuantModels, Backtester
from model_optimizer import ModelOptimizer

def fetch_data(symbol, start_date, end_date):
    """
    Fetch historical data directly using requests
    """
    print(f"Fetching data for {symbol}...")
    
    # Convert dates to UNIX timestamps
    start_timestamp = int(time.mktime(start_date.timetuple()))
    end_timestamp = int(time.mktime(end_date.timetuple()))
    
    # Yahoo Finance API URL
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    
    params = {
        "period1": start_timestamp,
        "period2": end_timestamp,
        "interval": "1d",
        "events": "history"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        # Extract price data
        chart_data = data['chart']['result'][0]
        timestamps = pd.to_datetime(chart_data['timestamp'], unit='s')
        quotes = chart_data['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Datetime': timestamps,
            'open': quotes['open'],
            'high': quotes['high'],
            'low': quotes['low'],
            'close': quotes['close'],
            'volume': quotes['volume']
        })
        
        # Clean any NaN values
        df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        print("Full error details:", e.__class__.__name__)
        return None

def optimize_models(df_with_features):
    """Run model optimization"""
    print("\nOptimizing models...")
    
    feature_sets = {
        'basic': [
            'returns', 'returns_volatility', 'price_to_sma20', 'price_to_sma50'
        ],
        'volume': [
            'returns', 'returns_volatility', 'volume_ratio', 'price_to_sma20'
        ],
        'technical': [
            'returns', 'returns_volatility', 'price_to_sma20', 'price_to_sma50',
            'atr', 'bollinger_position', 'volume_ratio', 'rsi', 'macd'
        ]
    }
    
    best_results = {}
    
    for set_name, features in feature_sets.items():
        print(f"\n{'='*50}")
        print(f"Testing feature set: {set_name}")
        print(f"Features being tested: {', '.join(features)}")
        
        optimizer = ModelOptimizer(df_with_features, features)
        
        print("\nOptimizing Random Forest...")
        best_rf = optimizer.optimize_random_forest()
        print(f"Best Random Forest parameters found: {optimizer.results[-1]['best_params']}")
        print(f"Random Forest accuracy score: {optimizer.results[-1]['best_score']:.4f}")
        
        print("\nOptimizing XGBoost...")
        best_xgb = optimizer.optimize_xgboost()
        print(f"Best XGBoost parameters found: {optimizer.results[-1]['best_params']}")
        print(f"XGBoost accuracy score: {optimizer.results[-1]['best_score']:.4f}")
        
        best_results[set_name] = {
            'features': features,
            'rf_model': best_rf,
            'xgb_model': best_xgb,
            'accuracy_rf': optimizer.results[-2]['best_score'],  # -2 for RF, -1 for XGB
            'accuracy_xgb': optimizer.results[-1]['best_score']
        }
        
        print(f"\nAverage accuracy for {set_name}: {(best_results[set_name]['accuracy_rf'] + best_results[set_name]['accuracy_xgb'])/2:.4f}")
        
        optimizer.plot_optimization_results()
    
    # Find best overall feature set
    best_set = max(best_results.items(), 
                  key=lambda x: (x[1]['accuracy_rf'] + x[1]['accuracy_xgb'])/2)
    
    print(f"\n{'='*50}")
    print(f"Best overall feature set: {best_set[0]}")
    print(f"Average accuracy: {(best_set[1]['accuracy_rf'] + best_set[1]['accuracy_xgb'])/2:.4f}")
    print(f"Features: {', '.join(best_set[1]['features'])}")
    
    return best_set[1]['rf_model'], best_set[1]['xgb_model'], best_set[1]['features']

def run_trading_workflow(symbol, initial_capital=100000):
    """
    Execute the complete trading workflow
    """
    # 1. Fetch Historical Data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    
    df = fetch_data(symbol, start_date, end_date)
    if df is None:
        print("Failed to fetch data. Exiting.")
        return None, None, None
        
    print(f"Fetched {len(df)} days of data")
    
    # 2. Initialize and Prepare Models
    print("\nPreparing models...")
    quant_models = QuantModels(df)
    
    # 3. Add Technical Features
    df_with_features = quant_models.add_features()
    print("Added technical features")
    
    # 4. Optimize models and select best feature set
    best_rf, best_xgb, best_features = optimize_models(df_with_features)
    
    # Update models with optimized versions
    quant_models.models['rf'] = best_rf
    quant_models.models['xgb'] = best_xgb
    
    # 5. Prepare Data for ML Models with best features
    print("\nPreparing data for ML models...")
    X_train, X_test, y_train, y_test = quant_models.prepare_ml_data()
    
    # 6. Run Backtest
    print("\nRunning backtest...")
    backtester = Backtester(df_with_features, quant_models.models, scaler=quant_models.scaler, initial_capital=initial_capital)
    metrics = backtester.run_backtest()
    
    # 7. Display Results
    print("\nBacktest Results:")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
    
    # 8. Generate plots
    print("\nGenerating plots...")
    plt.figure(figsize=(15, 10))
    
    # Portfolio value plot
    plt.subplot(2, 1, 1)
    plt.plot(backtester.portfolio_value)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    
    # Price and signals plot
    plt.subplot(2, 1, 2)
    plt.plot(df['close'])
    for pos in backtester.positions:
        if pos[0] == 'BUY':
            plt.scatter(pos[2], pos[1], color='green', marker='^')
        else:
            plt.scatter(pos[2], pos[1], color='red', marker='v')
    plt.title('Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    
    plt.tight_layout()
    plt.show()
    
    return quant_models, backtester, metrics

if __name__ == "__main__":
    # Run with different parameters for testing
    test_params = [
        #{'symbol': 'SPY', 'initial_capital': 100000},
        #{'symbol': 'QQQ', 'initial_capital': 100000},
        {'symbol': 'AAPL', 'initial_capital': 100000}
    ]
    
    results = {}
    for params in test_params:
        print(f"\nTesting {params['symbol']}...")
        models, backtester, metrics = run_trading_workflow(**params)
        results[params['symbol']] = metrics
    
    # Compare results across symbols
    print("\nComparison across symbols:")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df)