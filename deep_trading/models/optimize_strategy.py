import os
import numpy as np
import pandas as pd
from itertools import product
import joblib
import lightgbm as lgb
from datetime import datetime
import backtrader as bt
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from momentum_ai_trading.utils.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    RESULTS_PATH
)

def safe_get(obj, *keys, default=0.0):
    """Safely get nested dictionary values."""
    try:
        for key in keys:
            if hasattr(obj, '__getitem__'):
                obj = obj[key]
            elif hasattr(obj, key):
                obj = getattr(obj, key)
            else:
                return default
        return obj if obj is not None else default
    except (KeyError, AttributeError, TypeError):
        return default

class OptimizedMLStrategy(bt.Strategy):
    params = (
        ('model_path', f"{MODEL_PATH}.txt"),
        ('scaler_path', f"{MODEL_PATH}_scaler.pkl"),
        ('features_path', f"{MODEL_PATH}_features.pkl"),
        ('pred_threshold', 0.001),
        ('stop_loss', 0.02),
        ('trail_percent', 0.015),
        ('position_size', 0.2),
        ('profit_take', 0.025),
    )

    def __init__(self):
        self.model = lgb.Booster(model_file=self.p.model_path)
        self.scaler = joblib.load(self.p.scaler_path)
        self.features = joblib.load(self.p.features_path)

        # Price data
        self.data_close = self.datas[0].close
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low
        self.data_volume = self.datas[0].volume

        # Technical Indicators
        self.sma_20 = bt.indicators.SMA(self.data_close, period=20)
        self.sma_50 = bt.indicators.SMA(self.data_close, period=50)
        self.sma_200 = bt.indicators.SMA(self.data_close, period=200)
        self.rsi = bt.indicators.RSI(self.data_close, period=14)
        self.macd = bt.indicators.MACD(self.data_close)
        self.atr = bt.indicators.ATR(self.datas[0], period=14)
        self.bbands = bt.indicators.BollingerBands(self.data_close, period=20)

        # Position tracking
        self.orders = {}
        self.stops = {}
        self.profit_targets = {}
        self.trade_count = 0

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_count += 1

    def get_features(self):
        features = {
            'close': self.data_close[0],
            'sma_20': self.sma_20[0],
            'sma_50': self.sma_50[0],
            'sma_200': self.sma_200[0],
            'rsi': self.rsi[0],
            'macd': self.macd.macd[0],
            'macd_signal': self.macd.signal[0],
            'atr': self.atr[0],
            'volume': self.data_volume[0],
            'high_low_range': (self.data_high[0] - self.data_low[0]) / self.data_close[0],
            'price_to_sma50': self.data_close[0] / self.sma_50[0] - 1,
            'price_to_sma200': self.data_close[0] / self.sma_200[0] - 1,
            'sma_50_200_cross': self.sma_50[0] / self.sma_200[0] - 1,
        }
        return features

    def next(self):
        if len(self) < 200:
            return

        # Get prediction
        features = self.get_features()
        df_features = pd.DataFrame([features])
        df_features = df_features.reindex(columns=self.features, fill_value=0)
        X_scaled = self.scaler.transform(df_features)
        pred_return = self.model.predict(X_scaled)[0]

        # Update stops and targets
        for data, pos in self.positions.items():
            if pos:
                if self.data_close[0] >= self.profit_targets.get(data, float('inf')):
                    self.close(data)
                    continue

                current_stop = self.stops.get(data, None)
                if current_stop:
                    new_stop = max(
                        current_stop,
                        self.data_close[0] * (1 - self.p.trail_percent)
                    )
                    self.stops[data] = new_stop
                    if self.data_close[0] <= new_stop:
                        self.close(data)
                        continue

        # Entry logic
        if not self.position:
            if pred_return > self.p.pred_threshold:
                trend_ok = (
                    self.data_close[0] > self.sma_20[0] and
                    self.rsi[0] > 30 and self.rsi[0] < 70 and
                    self.data_close[0] > self.bbands.mid[0]  # Price above BB middle band
                )
                
                volume_ok = self.data_volume[0] > np.mean([self.data_volume[i] for i in range(-5, 0)])
                
                if trend_ok and volume_ok and not self.orders.get(self.datas[0], None):
                    # Position sizing based on volatility
                    volatility = self.atr[0] / self.data_close[0]
                    position_size = self.p.position_size * (1 - volatility)
                    
                    cash = self.broker.get_cash()
                    value = self.broker.getvalue()
                    target_value = value * position_size
                    size = int(target_value / self.data_close[0])
                    
                    if size > 0:
                        self.orders[self.datas[0]] = self.buy(size=size)
                        self.stops[self.datas[0]] = self.data_close[0] * (1 - self.p.stop_loss)
                        self.profit_targets[self.datas[0]] = self.data_close[0] * (1 + self.p.profit_take)

        # Exit logic
        elif pred_return < -self.p.pred_threshold:
            exit_ok = (
                self.rsi[0] > 70 or
                self.data_close[0] < self.sma_20[0] or
                self.data_close[0] < self.bbands.bot[0] or  # Price below BB lower band
                (self.macd.macd[0] < self.macd.signal[0] and 
                 self.macd.macd[-1] > self.macd.signal[-1])
            )
            
            if exit_ok and not self.orders.get(self.datas[0], None):
                self.close()
                self.stops.pop(self.datas[0], None)
                self.profit_targets.pop(self.datas[0], None)

def analyze_backtest(strat, initial_capital):
    """Analyze backtest results with proper error handling."""
    final_value = strat.broker.getvalue()
    pnl = final_value - initial_capital
    ret = (pnl / initial_capital) * 100

    sharpe_ratio = safe_get(strat.analyzers.sharpe.get_analysis(), 'sharperatio', default=0.0)
    max_drawdown = safe_get(strat.analyzers.drawdown.get_analysis(), 'max', 'drawdown', default=0.0)
    trades = strat.analyzers.trades.get_analysis()
    
    total_trades = strat.trade_count
    won_trades = safe_get(trades, 'won', 'total', default=0)
    lost_trades = safe_get(trades, 'lost', 'total', default=0)
    
    win_rate = won_trades / total_trades if total_trades > 0 else 0.0
    avg_win = safe_get(trades, 'won', 'pnl', 'average', default=0.0)
    avg_loss = safe_get(trades, 'lost', 'pnl', 'average', default=0.0)

    return {
        'return': ret,
        'sharpe': sharpe_ratio,
        'drawdown': max_drawdown,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'won_trades': won_trades,
        'lost_trades': lost_trades
    }

def optimize_strategy(symbols=['AAPL', 'MSFT', 'GOOGL'], initial_capital=100000):
    """
    Optimize strategy parameters across multiple symbols.
    """
    print(f"Optimizing strategy parameters for symbols: {', '.join(symbols)}")
    
    # Focused parameter grid
    param_grid = {
        'pred_threshold': [0.001, 0.002],
        'stop_loss': [0.015, 0.02],
        'trail_percent': [0.01, 0.015],
        'position_size': [0.2, 0.3],
        'profit_take': [0.02, 0.025]
    }
    
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in product(*param_grid.values())]
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    best_params = None
    best_avg_sharpe = float('-inf')
    all_results = []
    
    # Progress bar for total iterations
    total_iterations = len(param_combinations) * len(symbols)
    with tqdm(total=total_iterations) as pbar:
        for params in param_combinations:
            symbol_results = []
            
            for symbol in symbols:
                try:
                    # Load data
                    data_path = os.path.join(PROCESSED_DATA_PATH, f"{symbol}_processed_daily.csv")
                    data = pd.read_csv(data_path, parse_dates=['datetime'])
                    data = data.sort_values('datetime').tail(500)  # Last 500 days
                    data.set_index('datetime', inplace=True)
                    
                    cerebro = bt.Cerebro(stdstats=False)
                    cerebro.broker.setcash(initial_capital)
                    cerebro.broker.setcommission(commission=0.001)
                    
                    data_feed = bt.feeds.PandasData(dataname=data)
                    cerebro.adddata(data_feed)
                    
                    cerebro.addstrategy(OptimizedMLStrategy, **params)
                    
                    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
                    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
                    
                    results = cerebro.run()
                    strat = results[0]
                    
                    metrics = analyze_backtest(strat, initial_capital)
                    metrics['symbol'] = symbol
                    metrics['params'] = params
                    
                    symbol_results.append(metrics)
                    all_results.append(metrics)
                
                except Exception as e:
                    print(f"\nError with {symbol}, parameters {params}: {str(e)}")
                    continue
                
                finally:
                    pbar.update(1)
            
            # Calculate average performance across symbols
            if symbol_results:
                avg_sharpe = np.mean([r['sharpe'] for r in symbol_results])
                avg_return = np.mean([r['return'] for r in symbol_results])
                min_trades = min([r['total_trades'] for r in symbol_results])
                
                # Update best parameters if better performance found
                if (avg_sharpe > best_avg_sharpe and 
                    min_trades >= 5 and    # Minimum trades per symbol
                    avg_return > 0):       # Must be profitable on average
                    best_avg_sharpe = avg_sharpe
                    best_params = params
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(RESULTS_PATH, f'optimization_results_multi_{datetime.now().strftime("%Y%m%d")}.csv')
    results_df.to_csv(results_path, index=False)
    
    if best_params is None:
        print("\nNo suitable parameter combination found.")
        return None
    
    print("\nOptimization Results:")
    print("====================")
    print("\nBest Parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    print("\nPerformance by Symbol:")
    best_results = results_df[results_df['params'].apply(lambda x: x == best_params)]
    for _, row in best_results.iterrows():
        print(f"\n{row['symbol']}:")
        print(f"Return: {row['return']:.2f}%")
        print(f"Sharpe: {row['sharpe']:.2f}")
        print(f"Trades: {row['total_trades']}")
        print(f"Win Rate: {row['win_rate']*100:.1f}%")
    
    return best_params

if __name__ == "__main__":
    best_params = optimize_strategy()
