import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime
import gc

import backtrader as bt

from momentum_ai_trading.utils.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    PLOTS_PATH,
    LOGS_PATH
)

os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

class PctChange(bt.Indicator):
    lines = ('pctchange',)
    params = dict(period=1)
    plotinfo = dict(plot=False)
    
    def __init__(self):
        super(PctChange, self).__init__()
        self.addminperiod(self.params.period + 1)
    
    def next(self):
        if len(self) > self.p.period:
            self.lines.pctchange[0] = (self.data[0] - self.data[-self.p.period]) / self.data[-self.p.period]
        else:
            self.lines.pctchange[0] = 0.0

class OBV(bt.Indicator):
    lines = ('obv',)
    plotinfo = dict(plot=False)

    def __init__(self):
        super(OBV, self).__init__()
        self.addminperiod(2)
        self.cumv = 0.0
        
    def next(self):
        c = self.data.close[0]
        v = self.data.volume[0]
        
        if len(self) > 1:
            c1 = self.data.close[-1]
            if c > c1:
                self.cumv += v
            elif c < c1:
                self.cumv -= v
        else:
            self.cumv = v
            
        self.lines.obv[0] = self.cumv

class MLStrategy(bt.Strategy):
    params = (
        ('model_path', f"{MODEL_PATH}.txt"),
        ('scaler_path', f"{MODEL_PATH}_scaler.pkl"),
        ('features_path', f"{MODEL_PATH}_features.pkl"),
        ('pred_threshold', 0.001),  # Lowered threshold for more trades
        ('stop_loss', 0.02),       # 2% stop loss
        ('trail_percent', 0.015),  # 1.5% trailing stop
        ('position_size', 0.2),    # 20% of portfolio per trade
        ('max_positions', 3),      # Maximum concurrent positions
        ('profit_take', 0.025),    # 2.5% profit target
    )

    def __init__(self):
        self.model = lgb.Booster(model_file=self.p.model_path)
        self.scaler = joblib.load(self.p.scaler_path)
        self.features = joblib.load(self.p.features_path)

        # Price data
        self.data_close = self.datas[0].close
        self.data_open = self.datas[0].open
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low
        self.data_volume = self.datas[0].volume

        # Technical Indicators
        self.sma_10 = bt.indicators.SMA(self.data_close, period=10)
        self.sma_20 = bt.indicators.SMA(self.data_close, period=20)
        self.sma_50 = bt.indicators.SMA(self.data_close, period=50)
        self.sma_200 = bt.indicators.SMA(self.data_close, period=200)
        
        self.ema_10 = bt.indicators.EMA(self.data_close, period=10)
        self.ema_20 = bt.indicators.EMA(self.data_close, period=20)
        
        self.rsi = bt.indicators.RSI(self.data_close, period=14)
        self.macd = bt.indicators.MACD(self.data_close)
        self.atr = bt.indicators.ATR(self.datas[0], period=14)
        
        # Position tracking
        self.orders = {}  # Track orders by data
        self.stops = {}   # Track stops by data
        self.profit_targets = {}  # Track profit targets
        self.trade_log = []

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}')
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}')

            self.trade_log.append({
                'datetime': self.data.datetime.date(0),
                'type': 'BUY' if order.isbuy() else 'SELL',
                'price': order.executed.price,
                'size': order.executed.size,
                'value': order.executed.value,
                'commission': order.executed.comm
            })

        self.orders[order.data] = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        log_path = os.path.join(LOGS_PATH, f'backtest_{datetime.now().strftime("%Y%m%d")}.log')
        with open(log_path, 'a') as f:
            f.write(f'{dt.isoformat()} {txt}\n')

    def get_features(self):
        features = {
            'close': self.data_close[0],
            'sma_10': self.sma_10[0],
            'sma_20': self.sma_20[0],
            'sma_50': self.sma_50[0],
            'sma_200': self.sma_200[0],
            'ema_10': self.ema_10[0],
            'ema_20': self.ema_20[0],
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
        # Skip if not enough data
        if len(self) < 200:
            return

        # Get prediction
        features = self.get_features()
        df_features = pd.DataFrame([features])
        df_features = df_features.reindex(columns=self.features, fill_value=0)
        X_scaled = self.scaler.transform(df_features)
        pred_return = self.model.predict(X_scaled)[0]

        # Clean up memory
        del df_features, X_scaled
        gc.collect()

        # Update stops and targets for existing positions
        for data, pos in self.positions.items():
            if pos:
                # Check if we hit profit target
                if self.data_close[0] >= self.profit_targets.get(data, float('inf')):
                    self.close(data)
                    continue

                # Update trailing stop
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
        if len(self.positions) < self.p.max_positions:  # Check if we can take new positions
            if pred_return > self.p.pred_threshold:
                # Relaxed entry filters
                trend_ok = (
                    self.data_close[0] > self.sma_20[0] and  # Price above short-term MA
                    self.rsi[0] > 30 and self.rsi[0] < 70    # RSI not extreme
                )
                
                # Volume filter - more lenient
                volume_ok = self.data_volume[0] > 0.8 * self.data_volume[-1]
                
                if trend_ok and volume_ok and not self.orders.get(self.datas[0], None):
                    # Calculate position size
                    cash = self.broker.get_cash()
                    value = self.broker.getvalue()
                    target_value = value * self.p.position_size
                    size = int(target_value / self.data_close[0])
                    
                    if size > 0:
                        # Place buy order
                        self.orders[self.datas[0]] = self.buy(size=size)
                        
                        # Set initial stop loss
                        self.stops[self.datas[0]] = self.data_close[0] * (1 - self.p.stop_loss)
                        
                        # Set profit target
                        self.profit_targets[self.datas[0]] = self.data_close[0] * (1 + self.p.profit_take)
                        
                        self.log(f'BUY CREATE, Signal: {pred_return:.3f}, Size: {size}')

        # Exit logic for existing positions
        for data, pos in self.positions.items():
            if pos and pred_return < -self.p.pred_threshold:
                # Relaxed exit filters
                exit_ok = (
                    self.rsi[0] > 75 or  # More lenient overbought
                    self.data_close[0] < self.sma_10[0] or  # Price below shorter MA
                    (self.macd.macd[0] < self.macd.signal[0] and 
                     self.macd.macd[-1] > self.macd.signal[-1])  # Fresh MACD crossover
                )
                
                if exit_ok and not self.orders.get(data, None):
                    self.close(data)
                    self.stops.pop(data, None)
                    self.profit_targets.pop(data, None)
                    self.log(f'SELL CREATE, Signal: {pred_return:.3f}')

    def stop(self):
        trade_log_df = pd.DataFrame(self.trade_log)
        if not trade_log_df.empty:
            trade_log_path = os.path.join(LOGS_PATH, f'trades_{datetime.now().strftime("%Y%m%d")}.csv')
            trade_log_df.to_csv(trade_log_path, index=False)

def backtest_daily_strategy(symbol, initial_capital=100000):
    print(f"\nStarting backtest for {symbol}...")

    data_path = os.path.join(PROCESSED_DATA_PATH, f"{symbol}_processed_daily.csv")
    data = pd.read_csv(data_path, parse_dates=['datetime'])
    data = data.sort_values('datetime').tail(500)
    data.set_index('datetime', inplace=True)

    data_feed = bt.feeds.PandasData(dataname=data)

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)
    
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.Value)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    cerebro.addstrategy(MLStrategy)

    print("\nRunning backtest...")
    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    pnl = final_value - initial_capital
    
    print("\nBacktest Results:")
    print("================")
    print(f"Initial Portfolio Value: ${initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Net PnL: ${pnl:,.2f}")
    print(f"Return: {(pnl/initial_capital)*100:.2f}%")
    
    print(f"\nPerformance Metrics:")
    try:
        sharpe = strat.analyzers.sharpe.get_analysis()
        print(f"Sharpe Ratio: {sharpe.get('sharperatio', 0.0):.2f}")
    except:
        print("Sharpe Ratio: N/A")
        
    try:
        drawdown = strat.analyzers.drawdown.get_analysis()
        print(f"Max Drawdown: {drawdown.get('max', {'drawdown': 0.0})['drawdown']:.2f}%")
    except:
        print("Max Drawdown: N/A")
        
    try:
        returns = strat.analyzers.returns.get_analysis()
        print(f"Annual Return: {returns.get('rnorm100', 0.0):.2f}%")
    except:
        print("Annual Return: N/A")
    
    trades = strat.analyzers.trades.get_analysis()
    print(f"\nTrade Analysis:")
    print(f"Total Trades: {getattr(trades, 'total', {'total': 0})['total']}")
    
    try:
        if hasattr(trades, 'won') and trades.won.total > 0:
            print(f"Winning Trades: {trades.won.total}")
            print(f"Average Winning Trade: ${trades.won.pnl.average:.2f}")
    except:
        print("No winning trades in this period")
        
    try:
        if hasattr(trades, 'lost') and trades.lost.total > 0:
            print(f"Losing Trades: {trades.lost.total}")
            print(f"Average Losing Trade: ${trades.lost.pnl.average:.2f}")
    except:
        print("No losing trades in this period")

    results_path = os.path.join(LOGS_PATH, f'backtest_results_{symbol}_{datetime.now().strftime("%Y%m%d")}.txt')
    with open(results_path, 'w') as f:
        f.write("Backtest Results\n")
        f.write("================\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Period: {data.index[0].date()} to {data.index[-1].date()}\n\n")
        f.write(f"Initial Portfolio Value: ${initial_capital:,.2f}\n")
        f.write(f"Final Portfolio Value: ${final_value:,.2f}\n")
        f.write(f"Net PnL: ${pnl:,.2f}\n")
        f.write(f"Return: {(pnl/initial_capital)*100:.2f}%\n\n")
        
        f.write(f"Performance Metrics:\n")
        try:
            f.write(f"Sharpe Ratio: {sharpe.get('sharperatio', 0.0):.2f}\n")
        except:
            f.write("Sharpe Ratio: N/A\n")
            
        try:
            f.write(f"Max Drawdown: {drawdown.get('max', {'drawdown': 0.0})['drawdown']:.2f}%\n")
        except:
            f.write("Max Drawdown: N/A\n")
            
        try:
            f.write(f"Annual Return: {returns.get('rnorm100', 0.0):.2f}%\n")
        except:
            f.write("Annual Return: N/A\n")
            
        f.write(f"\nTrade Analysis:\n")
        f.write(f"Total Trades: {getattr(trades, 'total', {'total': 0})['total']}\n")
        
        try:
            if hasattr(trades, 'won') and trades.won.total > 0:
                f.write(f"Winning Trades: {trades.won.total}\n")
                f.write(f"Average Winning Trade: ${trades.won.pnl.average:.2f}\n")
        except:
            f.write("No winning trades in this period\n")
            
        try:
            if hasattr(trades, 'lost') and trades.lost.total > 0:
                f.write(f"Losing Trades: {trades.lost.total}\n")
                f.write(f"Average Losing Trade: ${trades.lost.pnl.average:.2f}\n")
        except:
            f.write("No losing trades in this period\n")

    plot_path = os.path.join(PLOTS_PATH, f'backtest_{symbol}_{datetime.now().strftime("%Y%m%d")}.png')
    try:
        cerebro.plot(style='candlestick',
                    barup='green',
                    bardown='red',
                    volume=False,
                    numfigs=1,
                    tight=True,
                    savefig=dict(fname=plot_path, dpi=150))
        print(f"\nBacktest plot saved to: {plot_path}")
    except Exception as e:
        print(f"\nWarning: Could not generate plot due to: {str(e)}")
        print("This is likely due to memory constraints but does not affect the backtest results.")

    del data, cerebro
    gc.collect()

if __name__ == "__main__":
    backtest_daily_strategy("AAPL")
