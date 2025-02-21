import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def find_latest_log(directory='.'):
    """Find the latest optimization log file"""
    log_files = []
    for file in os.listdir(directory):
        if file.startswith('optimization_log_') and file.endswith('.txt'):
            log_files.append(file)
    
    if not log_files:
        print("No optimization log files found")
        return None
    
    # Get latest log file
    latest_log = max(log_files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    print(f"Using log file: {latest_log}")
    
    return latest_log

def parse_signals(log_content):
    """Parse signals from log content"""
    signals = []
    
    # Print first few lines of log content for debugging
    print("\nFirst 500 characters of log content:")
    print(log_content[:500])
    
    # Regular expressions for different signal types with more flexible timestamp matching
    long_entry_pattern = r"Long Entry Signal at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):\s*\n\s*Price: \$([0-9.]+)\s*\n\s*Size: ([0-9]+)"
    short_entry_pattern = r"Short Entry Signal at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):\s*\n\s*Price: \$([0-9.]+)\s*\n\s*Size: ([0-9]+)"
    long_exit_pattern = r"Long Exit Signal at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):\s*\n\s*Price: \$([0-9.]+)\s*\n\s*Entry: \$([0-9.]+)\s*\n\s*P&L: \$([0-9.-]+)"
    short_exit_pattern = r"Short Exit Signal at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):\s*\n\s*Price: \$([0-9.]+)\s*\n\s*Entry: \$([0-9.]+)\s*\n\s*P&L: \$([0-9.-]+)"
    
    # Find all signals and print counts for debugging
    long_entries = list(re.finditer(long_entry_pattern, log_content))
    short_entries = list(re.finditer(short_entry_pattern, log_content))
    long_exits = list(re.finditer(long_exit_pattern, log_content))
    short_exits = list(re.finditer(short_exit_pattern, log_content))
    
    print(f"\nFound matches:")
    print(f"Long entries: {len(long_entries)}")
    print(f"Short entries: {len(short_entries)}")
    print(f"Long exits: {len(long_exits)}")
    print(f"Short exits: {len(short_exits)}")
    
    try:
        # Process long entries
        for match in long_entries:
            print(f"\nProcessing long entry match: {match.groups()}")
            try:
                signals.append({
                    'type': 'long_entry',
                    'timestamp': pd.to_datetime(match.group(1)),
                    'price': float(match.group(2)),
                    'size': int(match.group(3))
                })
            except Exception as e:
                print(f"Error processing long entry: {e}")
                print(f"Match groups: {match.groups()}")
        
        # Process short entries
        for match in short_entries:
            print(f"\nProcessing short entry match: {match.groups()}")
            try:
                signals.append({
                    'type': 'short_entry',
                    'timestamp': pd.to_datetime(match.group(1)),
                    'price': float(match.group(2)),
                    'size': int(match.group(3))
                })
            except Exception as e:
                print(f"Error processing short entry: {e}")
                print(f"Match groups: {match.groups()}")
        
        # Process long exits
        for match in long_exits:
            print(f"\nProcessing long exit match: {match.groups()}")
            try:
                signals.append({
                    'type': 'long_exit',
                    'timestamp': pd.to_datetime(match.group(1)),
                    'price': float(match.group(2)),
                    'entry_price': float(match.group(3)),
                    'pnl': float(match.group(4))
                })
            except Exception as e:
                print(f"Error processing long exit: {e}")
                print(f"Match groups: {match.groups()}")
        
        # Process short exits
        for match in short_exits:
            print(f"\nProcessing short exit match: {match.groups()}")
            try:
                signals.append({
                    'type': 'short_exit',
                    'timestamp': pd.to_datetime(match.group(1)),
                    'price': float(match.group(2)),
                    'entry_price': float(match.group(3)),
                    'pnl': float(match.group(4))
                })
            except Exception as e:
                print(f"Error processing short exit: {e}")
                print(f"Match groups: {match.groups()}")
    except Exception as e:
        print(f"Error processing signals: {e}")
    
    return signals

def load_price_data(csv_file='cache_data/eth_usd_coinapi.csv', nrows=10000):
    """Load price data from CSV file"""
    df = pd.read_csv(csv_file, 
                     index_col='time_period_start', 
                     parse_dates=True,
                     nrows=nrows)
    
    # Clean data
    df = df[
        (df['close'] > 0) & 
        (df['high'] > 0) & 
        (df['low'] > 0) & 
        (df['volume'] > 0)
    ].copy()
    
    # Sort by index
    df.sort_index(inplace=True)
    
    return df

def plot_signals(signals, price_data):
    """Plot price data with signals"""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot price data
    price_data['close'].plot(ax=ax, color='black', alpha=0.7, label='Price')
    
    # Convert signals to DataFrame
    df_signals = pd.DataFrame(signals)
    df_signals.set_index('timestamp', inplace=True)
    
    # Plot signals by type
    for signal_type, color, marker, size in [
        ('long_entry', 'lime', '^', 150),
        ('short_entry', 'red', 'v', 150),
        ('long_exit', 'black', 'x', 100),
        ('short_exit', 'black', 'x', 100)
    ]:
        mask = df_signals['type'] == signal_type
        if mask.any():
            signals_to_plot = df_signals[mask]
            ax.scatter(signals_to_plot.index, signals_to_plot['price'],
                      color=color, marker=marker, s=size)
    
    # Add legend
    ax.scatter([], [], color='lime', marker='^', s=150, label='Long Entry')
    ax.scatter([], [], color='red', marker='v', s=150, label='Short Entry')
    ax.scatter([], [], color='black', marker='x', s=100, label='Exit')
    
    # Customize plot
    ax.set_title('Price with Trading Signals')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'signals_plot_{timestamp}.png')
    plt.close()

def main():
    # Find latest log file
    log_file = find_latest_log()
    if not log_file:
        return
    
    # Read log file
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Parse signals
    signals = parse_signals(log_content)
    print(f"Found {len(signals)} signals")
    
    # Load price data
    price_data = load_price_data()
    
    # Plot signals
    plot_signals(signals, price_data)
    print("Signal plot saved")

if __name__ == '__main__':
    main()
