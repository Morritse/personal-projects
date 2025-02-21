###################################################
# backtester.py
###################################################
from strategy import IchimokuSuperTrendMACDStrategy
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class Backtester:
    def __init__(self, 
                 initial_capital=100_000, 
                 symbols=None,
                 timeframe='15Min',
                 lookback=300):
        """
        Basic backtester for Ichimoku+SuperTrend+MACD strategy.
        We'll do a single pass approach:
         - load data
         - calc indicators
         - step through bars for signals
         - track trades/pnl
        """
        self.strategy = IchimokuSuperTrendMACDStrategy()
        self.initial_capital = initial_capital
        self.portfolio_cash = initial_capital
        
        # Get all available symbols from historical_data folder
        if symbols is None:
            data_folder = 'historical_data'
            if os.path.exists(data_folder):
                symbols = [f.split('.')[0] for f in os.listdir(data_folder) 
                          if f.endswith('.csv')]
                print(f"Found {len(symbols)} symbols in {data_folder}")
            else:
                symbols = ['AAPL','MSFT','GOOGL','NVDA']  # fallback
        
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback = lookback
        
        # Store open positions as {symbol: quantity}
        self.positions = {}
        # Store trade logs as list of dicts
        self.trades = []
        # Store portfolio value at each timestamp
        self.equity_curve = []
        # Cache for symbol data
        self.symbol_data = {}
    
    def run_backtest(self, start_date=None, end_date=None):
        """
        1) Load all symbol data into unified timeline
        2) Calculate indicators for all symbols
        3) Process signals on unified timeline
        """
        print("Loading data for all symbols...")
        
        # Load and align all symbol data
        all_data = {}
        unified_index = pd.DatetimeIndex([])  # Will hold all timestamps
        
        for symbol in self.symbols:
            data = self.strategy.get_historical_data(symbol,
                                                   timeframe=self.timeframe,
                                                   lookback=self.lookback,
                                                   use_local=True)
            if data is not None and not data.empty:
                # Ensure UTC timezone
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
                elif data.index.tz != 'UTC':
                    data.index = data.index.tz_convert('UTC')
                    
                # Filter dates if specified
                if start_date:
                    start_date = pd.to_datetime(start_date).tz_localize('UTC')
                    data = data[data.index >= start_date]
                if end_date:
                    end_date = pd.to_datetime(end_date).tz_localize('UTC')
                    data = data[data.index <= end_date]
                
                # Remove any duplicate timestamps
                data = data[~data.index.duplicated(keep='last')]
                
                if len(data) >= 52:  # Minimum bars needed for indicators
                    all_data[symbol] = data
                    unified_index = unified_index.union(data.index)
                    print(f"Added {len(data)} bars for {symbol}")
                else:
                    print(f"Insufficient bars (<52) for {symbol}, skipping.")
            else:
                print(f"No data for {symbol}, skipping.")
        
        if not all_data:
            print("No valid data found for any symbols")
            return self.generate_report()
            
        print(f"Processing {len(all_data)} symbols...")
        
        # Create unified DataFrame with all symbols' close prices
        closes = pd.DataFrame(index=unified_index)
        indicators = {}
        
        for symbol, data in all_data.items():
            # Reindex data to unified timeline
            symbol_data = data.reindex(unified_index)
            closes[symbol] = symbol_data['close']
            
            # Pre-calculate indicators on original data
            ichimoku = self.strategy.calculate_ichimoku(data)
            supertrend_vals, supertrend_dirs = self.strategy.calculate_supertrend(data)
            macd_vals, signal_vals, hist_vals = self.strategy.calculate_macd(data)
            
            # Reindex all indicators to unified timeline
            indicators[symbol] = {
                'ichimoku': {
                    'tenkan_sen': ichimoku['tenkan_sen'].reindex(unified_index),
                    'kijun_sen': ichimoku['kijun_sen'].reindex(unified_index),
                    'cloud': {
                        'senkou_span_a': ichimoku['cloud']['senkou_span_a'].reindex(unified_index),
                        'senkou_span_b': ichimoku['cloud']['senkou_span_b'].reindex(unified_index)
                    }
                },
                'supertrend_vals': pd.Series(supertrend_vals, index=data.index).reindex(unified_index),
                'supertrend_dirs': pd.Series(supertrend_dirs, index=data.index).reindex(unified_index),
                'macd_vals': macd_vals.reindex(unified_index),
                'signal_vals': signal_vals.reindex(unified_index)
            }
        
        # Store for portfolio calculations
        self.symbol_data = all_data
        
        # Process signals on unified timeline
        for timestamp in unified_index[52:]:  # Skip first 52 bars needed for indicators
            self.process_signals_at_time(timestamp, closes, indicators)
        
        return self.generate_report()
    
    def process_signals_at_time(self, timestamp, closes, indicators):
        """Process all symbols at a given timestamp."""
        # Calculate portfolio value using vectorized operations
        portfolio_value = self.portfolio_cash
        for symbol, qty in self.positions.items():
            if not pd.isna(closes.loc[timestamp, symbol]):
                portfolio_value += qty * closes.loc[timestamp, symbol]
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value
        })
        
        # Evaluate signals for each symbol
        for symbol in closes.columns:
            price = closes.loc[timestamp, symbol]
            if pd.isna(price):  # Skip if no data for this symbol at this time
                continue
                
            # Get indicator values for this timestamp using loc with NaN checks
            try:
                tenkan = indicators[symbol]['ichimoku']['tenkan_sen'].loc[timestamp]
                kijun = indicators[symbol]['ichimoku']['kijun_sen'].loc[timestamp]
                span_a = indicators[symbol]['ichimoku']['cloud']['senkou_span_a'].loc[timestamp]
                span_b = indicators[symbol]['ichimoku']['cloud']['senkou_span_b'].loc[timestamp]
                st_val = indicators[symbol]['supertrend_vals'].loc[timestamp]
                st_dir = indicators[symbol]['supertrend_dirs'].loc[timestamp]
                macd_val = indicators[symbol]['macd_vals'].loc[timestamp]
                sig_val = indicators[symbol]['signal_vals'].loc[timestamp]
                
                # Skip if any indicator is NaN
                if pd.isna([tenkan, kijun, span_a, span_b, st_val, st_dir, macd_val, sig_val]).any():
                    continue
                
                current_ichimoku = {
                    'tenkan_sen': tenkan,
                    'kijun_sen': kijun,
                    'cloud': {
                        'senkou_span_a': span_a,
                        'senkou_span_b': span_b
                    }
                }
            except (KeyError, ValueError) as e:
                # Skip if any indicator access fails
                continue
            
            self.evaluate_signals(symbol, timestamp, price, current_ichimoku,
                                st_val, st_dir, macd_val, sig_val)
    
    def evaluate_signals(self, symbol, timestamp, price, ichimoku,
                         st_val, st_dir, macd_val, sig_val):
        """Check bullish/bearish conditions and place trades."""
        pos_qty = self.positions.get(symbol, 0)
        
        # Attempt to find cloud top/bottom
        try:
            cloud_top = max(ichimoku['cloud']['senkou_span_a'],
                            ichimoku['cloud']['senkou_span_b'])
            cloud_bottom = min(ichimoku['cloud']['senkou_span_a'],
                               ichimoku['cloud']['senkou_span_b'])
        except:
            # fallback if cloud is NaN
            cloud_top = max(ichimoku['tenkan_sen'], ichimoku['kijun_sen'])
            cloud_bottom = min(ichimoku['tenkan_sen'], ichimoku['kijun_sen'])
        
        # Stronger bullish confirmation logic
        bullish_ichimoku = (
            price > ichimoku['tenkan_sen'] and 
            price > ichimoku['kijun_sen'] and 
            price > cloud_top and
            ichimoku['tenkan_sen'] > ichimoku['kijun_sen']  # Require bullish TK cross
        )
        bullish_supertrend = (st_dir == 1)
        bullish_macd = (
            macd_val > sig_val and 
            macd_val > 0 and
            macd_val - sig_val > 0.2  # Require stronger MACD momentum
        )
        
        # Basic exit logic
        # If any major indicator flips negative
        exit_ichimoku = (price < ichimoku['kijun_sen'] or price < cloud_bottom)
        exit_supertrend = (st_dir == -1)
        exit_macd = (macd_val < sig_val)
        
        # Print debug
        # (Comment out if too verbose)
        # print(f"[{timestamp}] {symbol} Price={price:.2f} Tenk={ichimoku['tenkan_sen']:.2f} Kij={ichimoku['kijun_sen']:.2f} "
        #       f"ST_dir={st_dir}, MACD={macd_val:.2f}, Sig={sig_val:.2f}, pos={pos_qty}")
        
        # Check if enough time has passed since last trade
        min_bars_between_trades = 20  # Minimum 5 hours (20 x 15min bars)
        recent_trades = [t for t in self.trades if t['symbol'] == symbol]
        if recent_trades:
            bars_since_last_trade = (timestamp - recent_trades[-1]['ts']).total_seconds() / (15 * 60)
            if bars_since_last_trade < min_bars_between_trades:
                return

        # Entry with stronger confirmation
        if (bullish_ichimoku and bullish_supertrend and bullish_macd and
            price > cloud_top * 1.01):  # Price must be 1% above cloud
            # Enter if no existing long
            if pos_qty <= 0:
                # buy 5% of capital
                buy_qty = int((self.portfolio_cash * 0.05) // price)
                if buy_qty > 0:
                    cost = buy_qty * price
                    self.portfolio_cash -= cost
                    self.positions[symbol] = pos_qty + buy_qty
                    trade_id = len(self.trades)
                    self.trades.append({
                        'id': trade_id,
                        'ts': timestamp,
                        'symbol': symbol,
                        'side': 'BUY',
                        'qty': buy_qty,
                        'price': price,
                        'cost': cost,
                        'proceeds': 0,
                        'portfolio': self.portfolio_cash,
                        'pnl': None,  # Will be set when trade is closed
                        'closed': False
                    })
                    print(f"BUY {symbol}: {buy_qty} @ {price:.2f}, Cash => {self.portfolio_cash:.2f}")
        
        # Exit
        elif exit_ichimoku or exit_supertrend or exit_macd:
            # If we have a long pos, sell entire
            if pos_qty > 0:
                proceeds = pos_qty * price
                self.portfolio_cash += proceeds
                # Find the matching open buy trade
                open_buys = [t for t in self.trades 
                            if t['symbol'] == symbol 
                            and t['side'] == 'BUY' 
                            and not t['closed']]
                
                if open_buys:
                    buy_trade = open_buys[-1]  # Get most recent open buy
                    buy_trade['closed'] = True
                    buy_trade['pnl'] = proceeds - buy_trade['cost']
                    
                    self.trades.append({
                        'id': len(self.trades),
                        'ts': timestamp,
                        'symbol': symbol,
                        'side': 'SELL',
                        'qty': pos_qty,
                        'price': price,
                        'cost': buy_trade['cost'],
                        'proceeds': proceeds,
                        'portfolio': self.portfolio_cash,
                        'pnl': proceeds - buy_trade['cost'],
                        'closed': True,
                        'buy_trade_id': buy_trade['id']
                    })
                self.positions[symbol] = 0
                print(f"SELL {symbol}: {pos_qty} @ {price:.2f}, Cash => {self.portfolio_cash:.2f}")
    
    def calculate_performance(self):
        """
        End-of-run calculations: final value, returns, trade stats, etc.
        """
        final_val = self.portfolio_cash
        
        # Mark to market any open positions using cached data
        for sym, qty in self.positions.items():
            if qty > 0 and sym in self.symbol_data:
                last_price = self.symbol_data[sym]['close'].iloc[-1]
                final_val += (qty * last_price)
        
        total_return = (final_val - self.initial_capital) / self.initial_capital
        
        # Group trades by symbol and analyze each symbol's trades
        trades_by_symbol = {}
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)
        
        # Initialize counters for overall stats
        overall_wins = 0
        overall_trades = 0
        overall_pnl = []  # Store all trade PnLs
        
        # Analyze completed trades
        print("\nAnalyzing Completed Trades:")
        for symbol, symbol_trades in trades_by_symbol.items():
            symbol_wins = 0
            symbol_trades_count = 0
            symbol_completed = []
            
            # Find all completed trades using buy_trade_id
            for trade in symbol_trades:
                if trade['side'] == 'SELL' and 'buy_trade_id' in trade:
                    buy_trade = next((t for t in symbol_trades if t['id'] == trade['buy_trade_id']), None)
                    if buy_trade:
                        trade_pnl = trade['proceeds'] - buy_trade['cost']
                        is_win = trade_pnl > 0
                        
                        # Update overall stats
                        overall_trades += 1
                        if is_win:
                            overall_wins += 1
                        overall_pnl.append(trade_pnl)  # Store actual PnL (positive or negative)
                        
                        # Update symbol stats
                        symbol_trades_count += 1
                        if is_win:
                            symbol_wins += 1
                        symbol_completed.append((buy_trade, trade))
            
            print(f"{symbol}: {symbol_trades_count} trades, {symbol_wins} wins, Win Rate: {(symbol_wins/symbol_trades_count if symbol_trades_count > 0 else 0):.1%}")
        
        # Calculate overall stats
        win_rate = overall_wins / overall_trades if overall_trades > 0 else 0
        print(f"\nOverall Stats:")
        print(f"Total Completed Trades: {overall_trades}")
        print(f"Total Wins: {overall_wins}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Total PnL: ${sum(overall_pnl):.2f}")
        
        # Calculate averages and totals
        winning_trades = [pnl for pnl in overall_pnl if pnl > 0]
        losing_trades = [abs(pnl) for pnl in overall_pnl if pnl < 0]
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        gross_win = sum(winning_trades)
        gross_loss = sum(losing_trades)
        profit_factor = gross_win/gross_loss if gross_loss else np.inf
        
        # Calculate performance metrics using equity curve
        if not self.equity_curve:
            sharpe_ratio = sortino_ratio = max_drawdown = 0
            avg_trade_duration = pd.Timedelta(0)
            annualized_return = 0
            trades_per_day = 0
        else:
            # Create portfolio series
            portfolio_series = pd.Series(
                [point['portfolio_value'] for point in self.equity_curve],
                index=[point['timestamp'] for point in self.equity_curve]
            )
            
            # Calculate returns
            returns = portfolio_series.pct_change().dropna()
            trading_bars_per_year = 252 * 26  # ~6552 15-min bars per year
            
            # Annualized metrics
            ann_return = returns.mean() * trading_bars_per_year
            ann_vol = returns.std() * np.sqrt(trading_bars_per_year)
            
            # Risk metrics
            sharpe_ratio = ann_return / ann_vol if ann_vol != 0 else 0
            downside_returns = returns[returns < 0]
            sortino_ratio = (ann_return / (downside_returns.std() * np.sqrt(trading_bars_per_year))) if len(downside_returns) > 0 else np.inf
            
            # Maximum drawdown
            rolling_max = portfolio_series.expanding().max()
            drawdowns = (portfolio_series - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0
            
            # Trading metrics
            trading_days = (portfolio_series.index[-1] - portfolio_series.index[0]).days
            trades_per_day = len(self.trades) / max(trading_days, 1)
            
            # Calculate average trade duration using matched trades
            durations = []
            for symbol, trades in trades_by_symbol.items():
                for trade in trades:
                    if trade['side'] == 'SELL' and 'buy_trade_id' in trade:
                        buy_trade = next((t for t in trades if t['id'] == trade['buy_trade_id']), None)
                        if buy_trade:
                            duration = trade['ts'] - buy_trade['ts']
                            durations.append(duration)
            avg_trade_duration = pd.Timedelta(seconds=np.mean([d.total_seconds() for d in durations])) if durations else pd.Timedelta(0)
            
            # Calculate success rate by symbol using matched trades
            symbol_stats = {}
            print("\nDetailed Trade Analysis:")
            for symbol, trades in trades_by_symbol.items():
                print(f"\n{symbol}:")
                # Get completed trades (both buy and sell)
                completed_trades = []
                for trade in trades:
                    if trade['side'] == 'SELL' and 'buy_trade_id' in trade:
                        # Find matching buy trade
                        buy_trade = next((t for t in trades if t['id'] == trade['buy_trade_id']), None)
                        if buy_trade:
                            completed_trades.append((buy_trade, trade))
                
                wins = 0
                total = len(completed_trades)
                
                # Analyze completed trades
                for buy, sell in completed_trades:
                    pnl = sell['proceeds'] - buy['cost']
                    is_win = pnl > 0
                    if is_win:
                        wins += 1
                    print(f"Trade {buy['id']}: Buy @ {buy['price']:.2f}, Sell @ {sell['price']:.2f}, PnL: ${pnl:.2f} ({'WIN' if is_win else 'LOSS'})")
                
                win_rate = wins / total if total > 0 else 0
                print(f"Total Completed Trades: {total}, Wins: {wins}, Win Rate: {win_rate:.1%}")
                symbol_stats[symbol] = win_rate
        
        return {
            'InitialCap': self.initial_capital,
            'FinalVal': final_val,
            'TotalReturn': total_return,
            'Trades': len(self.trades),
            'WinRate': win_rate,
            'AvgWin': avg_win,
            'AvgLoss': avg_loss,
            'ProfitFactor': profit_factor,
            'SharpeRatio': sharpe_ratio,
            'MaxDrawdown': max_drawdown,
            'AvgDuration': avg_trade_duration,
            'AnnReturn': ann_return,
            'SortinoRatio': sortino_ratio,
            'TradesPerDay': trades_per_day,
            'SymbolStats': symbol_stats
        }
    
    def generate_report(self):
        stats = self.calculate_performance()
        print("\n=== Backtest Summary ===")
        print(f"Initial Capital: ${stats['InitialCap']:,.2f}")
        print(f"Final Value: ${stats['FinalVal']:,.2f}")
        print(f"Total Return: {stats['TotalReturn']:.2%}")
        print(f"Number of Trades: {stats['Trades']}")
        print(f"Win Rate: {stats['WinRate']:.1%}")
        print(f"Avg Win: ${stats['AvgWin']:.2f}")
        print(f"Avg Loss: ${stats['AvgLoss']:.2f}")
        print(f"Profit Factor: {stats['ProfitFactor']:.2f}")
        print(f"Sharpe Ratio: {stats['SharpeRatio']:.2f}")
        print(f"Max Drawdown: {stats['MaxDrawdown']:.1%}")
        print(f"Avg Trade Duration: {stats['AvgDuration']}")
        print(f"Annualized Return: {stats['AnnReturn']:.1%}")
        print(f"Sortino Ratio: {stats['SortinoRatio']:.2f}")
        print(f"Trades per Day: {stats['TradesPerDay']:.1f}")
        print("\nWin Rate by Symbol:")
        for symbol, win_rate in stats['SymbolStats'].items():
            print(f"{symbol}: {win_rate:.1%}")
        return stats


if __name__=="__main__":
    backtester = Backtester(
        initial_capital=100000,
        timeframe='15Min',
        lookback=300
    )  # Let it auto-detect symbols
    # Optionally define date range
    # start = "2024-12-01"
    # end = "2025-01-10"
    # results = backtester.run_backtest(start, end)
    
    results = backtester.run_backtest()  # No date filter => entire data
