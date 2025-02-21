import backtrader as bt
import pandas as pd
from datetime import datetime
import itertools
from bt_strategy import VAMEStrategy
from config import config
import numpy as np

def expand_param_grid(config):
    """Convert config with arrays into list of individual configs"""
    def _expand_dict(d):
        param_grid = {}
        for key, value in d.items():
            if isinstance(value, dict):
                expanded = _expand_dict(value)
                param_combinations = []
                keys = expanded.keys()
                values = expanded.values()
                for combo in itertools.product(*values):
                    param_combinations.append(dict(zip(keys, combo)))
                param_grid[key] = param_combinations
            elif isinstance(value, (list, np.ndarray)):
                param_grid[key] = value
            else:
                param_grid[key] = [value]
        return param_grid
    
    expanded = _expand_dict(config)
    keys = expanded.keys()
    values = expanded.values()
    
    configs = []
    for combo in itertools.product(*values):
        config_instance = {}
        for k, v in zip(keys, combo):
            if isinstance(v, dict):
                # For nested dictionaries, ensure all values are lists
                nested_dict = {}
                for sub_k, sub_v in v.items():
                    nested_dict[sub_k] = [sub_v]
                config_instance[k] = nested_dict
            else:
                # For non-dict values, always wrap in list
                config_instance[k] = [v]
        configs.append(config_instance)
    
    return configs

def run_backtest(data, test_config, plot=False, verbose=True):
    # Configure Cerebro
    cerebro = bt.Cerebro(stdstats=True)
    cerebro.broker.set_coc(True)  # Cheat-On-Close for more accurate execution
    cerebro.broker.set_cash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    # Add data
    data = bt.feeds.PandasData(
        dataname=data,
        datetime=None,  # Use index as datetime
        timeframe=bt.TimeFrame.Minutes,
        compression=1
    )
    cerebro.adddata(data)
    
    # Add strategy with test parameters
    strat_params = {
        # Core parameters
        'mfi_period': test_config['MFI Period'][0],
        'vwap_window': test_config['VWAP Window'][0],
        'atr_period': test_config['ATR Period'][0],
        'min_stop_pct': test_config['Min Stop Pct'][0],
        'max_stop_pct': test_config['Max Stop Pct'][0],
        'risk_per_trade': test_config['Risk Per Trade'][0],
        'vwap_exit_buffer_bull': test_config['VWAP Exit Buffer Bull'][0],
        'vwap_exit_buffer_bear': test_config['VWAP Exit Buffer Bear'][0],
        'min_hold_bars': test_config['Min Hold Bars'][0],
        
        # Trend parameters
        'ema_short_span': test_config['trend_params']['ema_short_span'][0],
        'ema_long_span': test_config['trend_params']['ema_long_span'][0],
        'min_trend_strength': test_config['trend_params']['min_trend_strength'][0],
        'current_window': test_config['Current Window'][0],
        'historical_window': test_config['Historical Window'][0],
        'volatility_multiplier': test_config['Volatility Multiplier'][0],
        
        # MFI thresholds
        'mfi_bull_entry': test_config['mfi_entry']['bull'][0],
        'mfi_bear_entry': test_config['mfi_entry']['bear'][0],
        'mfi_bull_exit': test_config['mfi_exit']['bull'][0],
        'mfi_bear_exit': test_config['mfi_exit']['bear'][0],
        
        # Position management
        'allow_shorts': test_config.get('allow_shorts', [False])[0],
        'profit_target_pct': test_config['position_params']['profit_target_pct'][0],
        'profit_lock_pct': test_config['position_params']['profit_lock_pct'][0],
        
        # Indicator parameters
        'obv_period': test_config['indicator_params']['obv_period'][0],
        'obv_ema_period': test_config['indicator_params']['obv_ema_period'][0],
        'vol_ema_period': test_config['indicator_params']['vol_ema_period'][0],
        'price_vol_period': test_config['indicator_params']['price_vol_period'][0],
        'min_vol_increases': test_config['indicator_params']['min_vol_increases'][0],
        'max_spread_pct': test_config['indicator_params']['max_spread_pct'][0],
        'max_price_volatility': test_config['indicator_params']['max_price_volatility'][0],
        'volatility_cap': test_config['indicator_params']['volatility_cap'][0],
        'vol_trend_threshold': test_config['indicator_params']['vol_trend_threshold'][0],
        'vol_threshold_multiplier': test_config['indicator_params']['vol_threshold_multiplier'][0],
        'verbose': verbose  # Pass verbose flag to strategy
    }
    cerebro.addstrategy(VAMEStrategy, **strat_params)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, 
        _name='sharpe',
        riskfreerate=0.0,  # Assuming cash baseline for crypto
        annualize=True,    # Annualize the ratio
        timeframe=bt.TimeFrame.Minutes,  # Using minute data
        compression=1,     # No compression
        factor=365*24*60   # Annualization factor for minute data (minutes in a year)
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')  # System Quality Number
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')  # Variability-Weighted Return
    
    # Run backtest
    results = cerebro.run()
    
    # Plot if requested
    if plot:
        cerebro.plot(style='candlestick', volume=True)
    strat = results[0]
    
    # Get analyzer results with robust error handling
    analysis = {'trades': strat.analyzers.trades.get_analysis()}
    
    # Safely get Sharpe ratio
    try:
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        analysis['sharpe'] = sharpe_analysis.get('sharperatio', 0) if sharpe_analysis else 0
    except:
        analysis['sharpe'] = 0
        
    # Safely get SQN
    try:
        sqn_analysis = strat.analyzers.sqn.get_analysis()
        analysis['sqn'] = sqn_analysis.get('sqn', 0) if sqn_analysis else 0
    except:
        analysis['sqn'] = 0
        
    # Safely get VWR
    try:
        vwr_analysis = strat.analyzers.vwr.get_analysis()
        analysis['vwr'] = vwr_analysis.get('vwr', 0) if vwr_analysis else 0
    except:
        analysis['vwr'] = 0
        
    # Safely get drawdown
    try:
        dd_analysis = strat.analyzers.drawdown.get_analysis()
        analysis['drawdown'] = dd_analysis.get('max', {}).get('drawdown', 0) if dd_analysis else 0
    except:
        analysis['drawdown'] = 0
    
    # Calculate detailed metrics with robust error handling
    trades = analysis['trades']
    
    # Initialize metrics
    total_trades = 0
    total_won = 0
    win_rate = 0
    avg_trade = 0
    avg_win = 0
    avg_loss = 0
    
    try:
        # Get total trades
        if hasattr(trades, 'total'):
            if hasattr(trades.total, 'closed'):
                total_trades = trades.total.closed
        
        # Get winning trades
        if hasattr(trades, 'won') and hasattr(trades.won, 'total'):
            total_won = trades.won.total
        
        # Calculate win rate
        win_rate = (total_won / total_trades) if total_trades > 0 else 0
        
        # Calculate trade metrics
        if hasattr(trades, 'pnl'):
            if hasattr(trades.pnl, 'net') and hasattr(trades.pnl.net, 'average'):
                avg_trade = trades.pnl.net.average
            
            if hasattr(trades.won, 'pnl') and hasattr(trades.won.pnl, 'average'):
                avg_win = trades.won.pnl.average
                
            if hasattr(trades.lost, 'pnl') and hasattr(trades.lost.pnl, 'average'):
                avg_loss = trades.lost.pnl.average
    except Exception as e:
        if verbose:
            print(f"Warning: Error calculating trade metrics: {str(e)}")
    
    # Calculate metrics with safe division
    metrics = {
        'sharpe': analysis['sharpe'],
        'sqn': analysis['sqn'],
        'vwr': analysis['vwr'],
        'max_drawdown': analysis['drawdown'],
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': -avg_win/avg_loss if avg_loss and avg_loss < 0 else 0,
        'final_value': cerebro.broker.getvalue(),
        'return': (cerebro.broker.getvalue() / 100000.0 - 1) * 100
    }
    
    if total_trades == 0 and verbose:
        print("\nWarning: No trades were executed during this backtest")
    
    return metrics

def optimize_strategy(data, plot_best=False):
    """Test strategy with parameter combinations"""
    global config
    original_config = config.copy()  # Save original config
    configs = expand_param_grid(config)
    total_configs = len(configs)
    print(f"\nTesting {total_configs} parameter combinations...")
    
    best_metrics = None
    best_config = None
    best_score = float('-inf')
    
    # Create optimization log file
    log_filename = f"optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_filename, 'w') as log_file:
        log_file.write("=== Optimization Log ===\n\n")
        
        for i, test_config in enumerate(configs, 1):
            print(f"\rProgress: {i}/{total_configs} combinations tested", end='')
            
            metrics = run_backtest(data, test_config, verbose=False)  # Suppress output during optimization
            
            # Calculate comprehensive score using multiple metrics
            score = 0
            
            try:
                # Convert metrics to float with safe defaults
                sharpe = float(metrics.get('sharpe', 0) if metrics.get('sharpe') is not None else 0)
                returns = float(metrics.get('return', 0) if metrics.get('return') is not None else 0)
                sqn = float(metrics.get('sqn', 0) if metrics.get('sqn') is not None else 0)
                drawdown = float(metrics.get('max_drawdown', 0) if metrics.get('max_drawdown') is not None else 0)
                profit_factor = float(metrics.get('profit_factor', 0) if metrics.get('profit_factor') is not None else 0)
            except (TypeError, ValueError):
                # If any conversion fails, use safe defaults
                sharpe, returns, sqn, drawdown, profit_factor = 0, 0, 0, 0, 0
            
            if sharpe is not None and returns is not None and sharpe >= 0 and returns > 0:
                # Base score components
                sharpe_score = sharpe * returns
                sqn_score = max(0, sqn) * 100  # Scale SQN to be comparable
                
                # Penalize based on drawdown (more penalty for larger drawdowns)
                drawdown_factor = max(0, 1 - (drawdown / 100))
                
                # Reward consistent profitability
                profit_factor_score = min(profit_factor * 10, 100) if profit_factor > 1 else 0
                
                # Combine scores with weights
                score = (
                    0.4 * sharpe_score +      # Sharpe ratio weighted return
                    0.3 * sqn_score +         # System quality
                    0.2 * profit_factor_score + # Profit consistency
                    0.1 * drawdown_factor * 100  # Drawdown penalty
                )
            
            # Log this iteration with safe formatting
            log_file.write(f"\n{'='*50}\n")
            log_file.write(f"Configuration {i}/{total_configs}:\n")
            log_file.write("\nParameters:\n")
            for k, v in test_config.items():
                if isinstance(v, dict):
                    log_file.write(f"{k}:\n")
                    for sub_k, sub_v in v.items():
                        log_file.write(f"  {sub_k}: {sub_v}\n")
                else:
                    log_file.write(f"{k}: {v}\n")
            
            # Safe metric logging with defaults
            def safe_format(value, format_str, default="N/A"):
                try:
                    if value is None:
                        return default
                    return format_str.format(float(value))
                except (ValueError, TypeError):
                    return default
            
            log_file.write(f"Score: {safe_format(score, '{:.4f}')}\n")
            log_file.write(f"Sharpe: {safe_format(metrics.get('sharpe'), '{:.2f}')}\n")
            log_file.write(f"SQN: {safe_format(metrics.get('sqn'), '{:.2f}')}\n")
            log_file.write(f"VWR: {safe_format(metrics.get('vwr'), '{:.2f}')}\n")
            log_file.write(f"Return: {safe_format(metrics.get('return'), '{:.2f}')}%\n")
            log_file.write(f"Win Rate: {safe_format(metrics.get('win_rate'), '{:.2%}')}\n")
            log_file.write(f"Profit Factor: {safe_format(metrics.get('profit_factor'), '{:.2f}')}\n")
            log_file.write(f"Avg Trade: ${safe_format(metrics.get('avg_trade'), '{:.2f}')}\n")
            log_file.write("\nSignals:\n")
            # Capture stdout to get signal output
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            
            # Run backtest with verbose output
            metrics = run_backtest(data, test_config, verbose=True)
            
            # Restore stdout and get captured output
            sys.stdout = old_stdout
            signal_output = mystdout.getvalue()
            log_file.write(signal_output)
            
            log_file.write("\n" + "="*50 + "\n")
            
            if score > best_score:
                best_score = score
                best_metrics = metrics
                best_config = test_config
                
                # Log new best found
                log_file.write("\n!!! New Best Configuration Found !!!\n")
                log_file.write(f"Score: {score:.4f}\n")
                log_file.write("-" * 50 + "\n")
        
        if best_config is not None and best_metrics is not None:
            # Log final best configuration
            log_file.write("\n=== Final Best Configuration ===\n")
            log_file.write("Parameters:\n")
            for k, v in best_config.items():
                if isinstance(v, dict):
                    log_file.write(f"{k}:\n")
                    for sub_k, sub_v in v.items():
                        log_file.write(f"  {sub_k}: {sub_v}\n")
                else:
                    log_file.write(f"{k}: {v}\n")
            log_file.write(f"\nFinal Score: {safe_format(best_score, '{:.4f}')}\n")
            log_file.write(f"Sharpe: {safe_format(best_metrics.get('sharpe'), '{:.2f}')}\n")
            log_file.write(f"SQN: {safe_format(best_metrics.get('sqn'), '{:.2f}')}\n")
            log_file.write(f"VWR: {safe_format(best_metrics.get('vwr'), '{:.2f}')}\n")
            log_file.write(f"Return: {safe_format(best_metrics.get('return'), '{:.2f}')}%\n")
            log_file.write(f"Win Rate: {safe_format(best_metrics.get('win_rate'), '{:.2%}')}\n")
            log_file.write(f"Profit Factor: {safe_format(best_metrics.get('profit_factor'), '{:.2f}')}\n")
            log_file.write(f"Avg Trade: ${safe_format(best_metrics.get('avg_trade'), '{:.2f}')}\n")
        else:
            log_file.write("\nNo valid configurations found\n")
    
    print("\nOptimization complete! Results saved to", log_filename)
    
    # Plot the best result if requested and save metrics from final run
    if plot_best:
        metrics = run_backtest(data, best_config, plot=True, verbose=True)  # Enable output for final run
    
    # Restore original config
    config.update(original_config)
    
    return best_config, metrics

def main(plot_results=True):
    from download_data import load_cached_data
    
    # Load ETH/USD data from CoinAPI cache
    print("\nOptimizing ETH/USD...")
    
    # Read last 50000 rows to allow for proper indicator warmup
    symbol_data = pd.read_csv('cache_data/eth_usd_coinapi.csv', 
                             index_col='time_period_start', 
                             parse_dates=True,
                             nrows=10000)  # Use less data for faster testing
    
    if symbol_data is None or symbol_data.empty:
        print("No cached data found")
        return
    
    # Clean and validate data
    # Remove rows with zero or NaN values in critical columns
    symbol_data = symbol_data[
        (symbol_data['close'] > 0) & 
        (symbol_data['high'] > 0) & 
        (symbol_data['low'] > 0) & 
        (symbol_data['volume'] > 0)
    ].copy()
    
    # Sort by index to ensure chronological order
    symbol_data.sort_index(inplace=True)
    
    print(f"Using {len(symbol_data)} validated data points for backtest")
    
    # Run optimization
    best_config, metrics = optimize_strategy(symbol_data, plot_best=plot_results)
    
    # Print results
    print(f"\nResults for ETH/USD:")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"System Quality Number: {metrics['sqn']:.2f}")
    print(f"Variability-Weighted Return: {metrics['vwr']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Average Trade: ${metrics['avg_trade']:.2f}")
    print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
    print(f"Total Return: {metrics['return']:.2f}%")

if __name__ == '__main__':
    main()
