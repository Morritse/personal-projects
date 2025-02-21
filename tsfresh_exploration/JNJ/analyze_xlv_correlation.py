import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
from scipy import stats

def fetch_data(symbols, days=365*3):
    """Fetch historical data for multiple symbols, using cached data if available."""
    cache_file = 'data/market_data.pkl'
    os.makedirs('data', exist_ok=True)
    
    data = {}
    
    # Try to load cached data first
    if os.path.exists(cache_file):
        print("Loading cached market data...")
        try:
            cached_data = pd.read_pickle(cache_file)
            # Only use cached data that we need
            for symbol in symbols:
                if symbol in cached_data:
                    data[symbol] = cached_data[symbol]
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    # Check which symbols we still need
    missing_symbols = [s for s in symbols if s not in data]
    if not missing_symbols:
        return data
    
    print(f"Fetching data for: {', '.join(missing_symbols)}")
    
    # Fetch missing data
    if missing_symbols:
        load_dotenv()
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        base_url = 'https://paper-api.alpaca.markets'
        
        if not api_key or not api_secret:
            raise ValueError("API credentials not found in environment variables")
            
        api = tradeapi.REST(
            api_key,
            api_secret,
            base_url=base_url,
            api_version='v2'
        )
        
        end = datetime.now()
        start = end - timedelta(days=days)
        
        for symbol in missing_symbols:
            print(f"\nFetching {days} days of data for {symbol}...")
            bars = api.get_bars(
                symbol,
                tradeapi.TimeFrame.Hour,
                start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d'),
                adjustment='raw'
            ).df
            # Remove timezone info
            bars.index = bars.index.tz_localize(None)
            data[symbol] = bars
            print(f"Got {len(bars)} hours of data")
        
        # Cache all data
        print("Caching market data...")
        pd.to_pickle(data, cache_file)
    
    return data

def get_xlv_components():
    """Get top components of XLV by weight."""
    return [
        'UNH',  # UnitedHealth Group
        'JNJ',  # Johnson & Johnson
        'LLY',  # Eli Lilly
        'ABBV', # AbbVie
        'MRK',  # Merck
        'PFE',  # Pfizer
        'TMO',  # Thermo Fisher
        'ABT',  # Abbott Labs
        'DHR',  # Danaher
        'BMY'   # Bristol Myers Squibb
    ]

def safe_forward_index(df, timestamp, periods):
    """Safely get a future index value, handling edge cases."""
    try:
        current_idx = df.index.get_loc(timestamp)
        if current_idx + periods >= len(df.index):
            return None
        return df.index[current_idx + periods]
    except:
        return None

def analyze_relationships():
    # Fetch data for JNJ, ETFs, and XLV components (excluding duplicate JNJ)
    components = [s for s in get_xlv_components() if s != 'JNJ']
    symbols = ['JNJ', 'XLV', 'VHT', 'IYH'] + components
    data = fetch_data(symbols)
    
    # Create DataFrame with resampled data to handle duplicate timestamps
    dfs = []
    
    # Process each symbol's data
    for symbol in symbols:
        # Resample to hourly data and forward fill
        symbol_data = data[symbol].resample('1h').last()
        
        # Create dataframe for this symbol
        temp_df = pd.DataFrame({
            f'{symbol.lower()}_close': symbol_data['close'].ffill(),
            f'{symbol.lower()}_volume': symbol_data['volume'].ffill()
        }, index=symbol_data.index)
        
        dfs.append(temp_df)
    
    # Merge all dataframes on index
    df = pd.concat(dfs, axis=1, join='outer')
    df = df.ffill()  # Forward fill any remaining NaN values
    
    # Add hour of day
    df.index = pd.to_datetime(df.index)
    df['hour'] = df.index.hour
    
    # Calculate returns at different frequencies
    timeframes = {
        '1H': 1,    # 1 hour
        '4H': 4,    # 4 hours
        '1D': 24,   # 1 day (24 hours)
        '1W': 24*5  # 1 week (5 trading days)
    }
    
    # Calculate returns for each symbol
    for symbol in symbols:
        symbol_lower = symbol.lower()
        close_col = f'{symbol_lower}_close'
        
        for label, periods in timeframes.items():
            # Calculate returns using pandas shift
            df[f'{symbol_lower}_returns_{label}'] = (
                df[close_col] - df[close_col].shift(periods)
            ) / df[close_col].shift(periods)
    
    # Drop NaN values
    df = df.dropna()
    
    # Calculate correlations with JNJ
    print("\nJNJ Correlations at different timeframes:")
    for timeframe in timeframes.keys():
        print(f"\n{timeframe}:")
        # ETFs
        print("\nETFs:")
        for etf in ['XLV', 'VHT', 'IYH']:
            corr = df[f'jnj_returns_{timeframe}'].corr(df[f'{etf.lower()}_returns_{timeframe}'])
            print(f"{etf}: {corr:.4f}")
        
        # XLV Components
        print("\nXLV Components:")
        correlations = []
        for symbol in components:  # Using the filtered components list from earlier
            corr = df[f'jnj_returns_{timeframe}'].corr(df[f'{symbol.lower()}_returns_{timeframe}'])
            correlations.append((symbol, corr))
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for symbol, corr in correlations:
            print(f"{symbol}: {corr:.4f}")
    
    # Calculate correlations between ETFs
    print("\nETF Cross-correlations (1D timeframe):")
    etfs = ['XLV', 'VHT', 'IYH']
    for i in range(len(etfs)):
        for j in range(i+1, len(etfs)):
            corr = df[f'{etfs[i].lower()}_returns_1D'].corr(df[f'{etfs[j].lower()}_returns_1D'])
            print(f"{etfs[i]}-{etfs[j]}: {corr:.4f}")
    
    # Calculate lead-lag relationships
    max_lag = 12  # hours
    lags = range(-max_lag, max_lag + 1)
    correlations = {}
    
    # Initialize correlation storage for ETFs and components
    for symbol in ['XLV', 'VHT', 'IYH'] + [s for s in get_xlv_components() if s != 'JNJ']:
        correlations[symbol] = {timeframe: [] for timeframe in timeframes.keys()}
    
    print("\nLead-lag relationships:")
    
    # Analyze ETFs
    print("\nETFs:")
    for etf in ['XLV', 'VHT', 'IYH']:
        print(f"\n{etf}:")
        for timeframe in timeframes.keys():
            for lag in lags:
                if lag < 0:
                    # ETF leading JNJ
                    corr = df[f'jnj_returns_{timeframe}'].corr(
                        df[f'{etf.lower()}_returns_{timeframe}'].shift(-lag)
                    )
                else:
                    # JNJ leading ETF
                    corr = df[f'jnj_returns_{timeframe}'].shift(lag).corr(
                        df[f'{etf.lower()}_returns_{timeframe}']
                    )
                correlations[etf][timeframe].append(corr)
        
            # Find strongest correlation and its lag for this timeframe
            max_corr_idx = np.argmax(np.abs(correlations[etf][timeframe]))
            max_corr = correlations[etf][timeframe][max_corr_idx]
            max_corr_lag = lags[max_corr_idx]
            
            print(f"{timeframe}:")
            print(f"Strongest correlation: {max_corr:.4f} at lag {max_corr_lag}")
            if max_corr_lag < 0:
                print(f"{etf} leads JNJ by {abs(max_corr_lag)} periods")
            else:
                print(f"JNJ leads {etf} by {max_corr_lag} periods")
    
    # Calculate beta relative to each ETF
    print("\nBeta calculations:")
    for etf in ['XLV', 'VHT', 'IYH']:
        print(f"\n{etf}:")
        for timeframe in timeframes.keys():
            # Calculate beta using regression
            slope, _, r_value, _, _ = stats.linregress(
                df[f'{etf.lower()}_returns_{timeframe}'],
                df[f'jnj_returns_{timeframe}']
            )
            r_squared = r_value ** 2
            print(f"{timeframe}:")
            print(f"Beta: {slope:.4f}")
            print(f"R-squared: {r_squared:.4f}")
    
    # Analyze correlation in different market regimes
    print("\nMarket Regime Analysis:")
    for timeframe in timeframes.keys():
        # Calculate rolling returns and volatility
        xlv_ret = df[f'xlv_returns_{timeframe}'].rolling(window=20).mean() * 20  # Annualized
        xlv_vol = df[f'xlv_returns_{timeframe}'].rolling(window=20).std() * np.sqrt(20)  # Annualized
        
        # Define market regimes
        df['regime'] = 'normal'
        df.loc[(xlv_ret > 0) & (xlv_vol <= xlv_vol.median()), 'regime'] = 'bull_low_vol'
        df.loc[(xlv_ret > 0) & (xlv_vol > xlv_vol.median()), 'regime'] = 'bull_high_vol'
        df.loc[(xlv_ret <= 0) & (xlv_vol <= xlv_vol.median()), 'regime'] = 'bear_low_vol'
        df.loc[(xlv_ret <= 0) & (xlv_vol > xlv_vol.median()), 'regime'] = 'bear_high_vol'
        
        print(f"\n{timeframe} Timeframe:")
        
        # Calculate correlations by regime
        for regime in ['bull_low_vol', 'bull_high_vol', 'bear_low_vol', 'bear_high_vol']:
            regime_data = df[df['regime'] == regime]
            
            # ETF correlations
            print(f"\n{regime.replace('_', ' ').title()}:")
            for etf in ['XLV', 'VHT', 'IYH']:
                corr = regime_data[f'jnj_returns_{timeframe}'].corr(
                    regime_data[f'{etf.lower()}_returns_{timeframe}']
                )
                print(f"{etf}: {corr:.4f}")
            
            # Mean reversion analysis
            if len(regime_data) > 0:
                diff = regime_data[f'jnj_returns_{timeframe}'] - regime_data[f'xlv_returns_{timeframe}']
                rolling_mean = diff.rolling(window=20).mean()
                rolling_std = diff.rolling(window=20).std()
                zscore = (diff - rolling_mean) / rolling_std
                
                # Count divergences
                divergences = zscore[abs(zscore) > 2.0]
                if len(divergences) > 0:
                    # Calculate convergence rate
                    forward_periods = timeframes[timeframe] * 2
                    convergence_count = 0
                    total_count = 0
                    
                    for timestamp in divergences.index:
                        forward_idx = safe_forward_index(regime_data, timestamp, forward_periods)
                        if forward_idx is None:
                            continue
                            
                        initial_diff = diff[timestamp]
                        forward_diff = diff[forward_idx]
                        
                        if abs(forward_diff) < abs(initial_diff):
                            convergence_count += 1
                        total_count += 1
                    
                    if total_count > 0:
                        conv_rate = convergence_count / total_count
                        print(f"Mean Reversion: {conv_rate:.2%} ({total_count} divergences)")
            
            # Component correlations
            correlations = []
            for symbol in components:
                if symbol != 'JNJ':
                    corr = regime_data[f'jnj_returns_{timeframe}'].corr(
                        regime_data[f'{symbol.lower()}_returns_{timeframe}']
                    )
                    correlations.append((symbol, corr))
            
            # Print top 3 components
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            print("Top Components:")
            for symbol, corr in correlations[:3]:
                print(f"{symbol}: {corr:.4f}")
    
    # Plot results
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Regime distribution
    plt.subplot(2, 2, 1)
    regime_counts = df['regime'].value_counts()
    plt.pie(regime_counts.values, labels=regime_counts.index.str.replace('_', ' ').str.title(),
            autopct='%1.1f%%', startangle=90)
    plt.title('Market Regime Distribution')
    
    # Plot 2: Mean reversion success by regime (1H)
    plt.subplot(2, 2, 2)
    regimes = ['bull_low_vol', 'bull_high_vol', 'bear_low_vol', 'bear_high_vol']
    success_rates = []
    for regime in regimes:
        regime_data = df[df['regime'] == regime]
        diff = regime_data['jnj_returns_1H'] - regime_data['xlv_returns_1H']
        rolling_mean = diff.rolling(window=20).mean()
        rolling_std = diff.rolling(window=20).std()
        zscore = (diff - rolling_mean) / rolling_std
        divergences = zscore[abs(zscore) > 2.0]
        
        if len(divergences) > 0:
            forward_periods = timeframes['1H'] * 2
            convergence_count = 0
            total_count = 0
            
            for timestamp in divergences.index:
                forward_idx = safe_forward_index(regime_data, timestamp, forward_periods)
                if forward_idx is None:
                    continue
                    
                initial_diff = diff[timestamp]
                forward_diff = diff[forward_idx]
                
                if abs(forward_diff) < abs(initial_diff):
                    convergence_count += 1
                total_count += 1
            
            if total_count > 0:
                success_rates.append(convergence_count / total_count * 100)
            else:
                success_rates.append(0)
        else:
            success_rates.append(0)
    
    plt.bar(range(len(regimes)), success_rates)
    plt.xticks(range(len(regimes)), [r.replace('_', ' ').title() for r in regimes], rotation=45)
    plt.title('Mean Reversion Success Rate by Regime (1H)')
    plt.ylabel('Success Rate (%)')
    plt.grid(True)
    
    # Plot 3: Correlation strength by regime (1H)
    plt.subplot(2, 2, 3)
    correlations_by_regime = []
    for regime in regimes:
        regime_data = df[df['regime'] == regime]
        corr = regime_data['jnj_returns_1H'].corr(regime_data['xlv_returns_1H'])
        correlations_by_regime.append(corr)
    
    plt.bar(range(len(regimes)), correlations_by_regime)
    plt.xticks(range(len(regimes)), [r.replace('_', ' ').title() for r in regimes], rotation=45)
    plt.title('JNJ-XLV Correlation by Regime (1H)')
    plt.ylabel('Correlation')
    plt.grid(True)
    
    # Plot 4: Component correlations by regime (1H)
    plt.subplot(2, 2, 4)
    x = range(len(components))  # Include all components
    for regime in regimes:
        regime_data = df[df['regime'] == regime]
        component_corrs = []
        for symbol in components:
            corr = regime_data['jnj_returns_1H'].corr(
                regime_data[f'{symbol.lower()}_returns_1H']
            )
            component_corrs.append(abs(corr))
        plt.plot(x, sorted(component_corrs, reverse=True),
                label=regime.replace('_', ' ').title(), marker='o')
    
    plt.xticks(x, range(1, len(components) + 1))
    plt.title('Component Correlation Strength by Regime (1H)')
    plt.xlabel('Component Rank')
    plt.ylabel('Absolute Correlation')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('xlv_jnj_correlation.png')
    
    return df

if __name__ == "__main__":
    df = analyze_relationships()
