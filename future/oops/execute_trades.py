import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from ib_insync import *
import config  # Import config for contract specs and position limits
import logging

# Configure logging to only show our messages
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress all IB messages
for name in ['ib_insync.wrapper', 'ib_insync.client', 'ib_insync.ib']:
    logging.getLogger(name).setLevel(logging.ERROR)
util.logToConsole(level=logging.ERROR)

def get_current_positions(ib):
    """Get current positions from IB."""
    positions = {}
    for pos in ib.positions():
        symbol = pos.contract.symbol
        positions[symbol] = pos.position
    return positions

def get_account_value(ib):
    """Get account net liquidation value."""
    account = ib.accountSummary()
    for value in account:
        if value.tag == 'NetLiquidation':
            return float(value.value)
    return None

def get_margin_allocation(account_value):
    """Get margin allocation based on account value."""
    # Use 90% of account value for initial margin since this is paper trading
    return account_value * 0.90

def get_margin_requirements(symbol):
    """Get approximate initial margin requirements per contract."""
    margins = {
        'ES': 15000,  # E-mini S&P
        'NQ': 15000,
        'YM': 12000,
        'RTY': 8000,
        'ZB': 4000,
        'ZN': 3000,
        'ZF': 2500,
        'CL': 7000,
        'NG': 3000,
        'GC': 11000,
        'SI': 10000,
        'HG': 8000,
        'ZC': 2500,
        'ZW': 2500,
        'ZS': 3500,
        'LE': 2500,
        'HO': 7000,
        'RB': 7000,
    }
    return margins.get(symbol, 10000)

def get_price_format(symbol):
    """Get price decimal places for each product."""
    formats = {
        'ES': '.25',   # Quarter point
        'NQ': '.25',
        'YM': '.00',
        'RTY': '.10',
        'ZB': '.015625',  # 1/32 of a point
        'ZN': '.015625',
        'ZF': '.015625',
        'CL': '.02', 
        'NG': '.001',
        'GC': '.10',
        'SI': '.001',
        'HG': '.0005',
        'ZC': '.25',
        'ZW': '.25',
        'ZS': '.25',
        'LE': '.025',
        'HO': '.0001',
        'RB': '.0001',
    }
    return formats.get(symbol, '.02')

def get_contract(ib, symbol):
    """Get the appropriate futures contract outside the delivery window."""
    contract_specs = config.FUTURES_SYMBOLS[symbol]
    exchange = contract_specs['exchange']
    contract = Future(symbol, exchange=exchange)
    
    ib.qualifyContracts(contract)
    valid_contracts = ib.reqContractDetails(contract)
    
    if not valid_contracts:
        raise ValueError(f"No valid contracts found for {symbol}")
    
    valid_contracts.sort(key=lambda x: x.contract.lastTradeDateOrContractMonth)
    
    # Skip contracts within delivery window
    today = datetime.now()
    delivery_buffer = {
        'GC': 45,  # Gold needs longer buffer
        'SI': 45,
        'HG': 45,
        'CL': 30,
        'NG': 30,
    }
    buffer_days = delivery_buffer.get(symbol, 20)
    
    for contract_detail in valid_contracts:
        contract = contract_detail.contract
        expiry = datetime.strptime(contract.lastTradeDateOrContractMonth, '%Y%m%d')
        if expiry - today > timedelta(days=buffer_days):
            logger.debug(f"Selected {contract.localSymbol} (Expiry: {expiry.date()})")
            return contract
            
    raise ValueError(f"No valid contracts found for {symbol} outside delivery window")

def get_contract_value(symbol, price):
    """Get approximate notional value per contract."""
    multipliers = {
        'ES': 50,
        'NQ': 20,
        'YM': 5,
        'RTY': 50,
        'ZB': 1000,
        'ZN': 1000,
        'ZF': 1000,
        'CL': 1000,
        'NG': 10000,
        'GC': 100,
        'SI': 5000,
        'HG': 25000,
        'ZC': 5000,
        'ZW': 5000,
        'ZS': 5000,
        'LE': 40000,
        'HO': 42000,
        'RB': 42000,
    }
    multiplier = multipliers.get(symbol, 1000)  # Default to 1000
    return price * multiplier

#
#  -- UPDATED SIZING LOGIC --
#

# You can tweak this value if you have a larger/smaller account.
# This is the maximum notional you'd hold *per symbol* if signal = ±1. 
TARGET_NOTIONAL_PER_SYMBOL = 200_000.0  # e.g. $200k

def get_max_position(symbol, account_value, current_price):
    """
    1) We want to hold up to TARGET_NOTIONAL_PER_SYMBOL at signal=±1.
    2) If signal=±1 => use that full slot of notional,
       else fraction => fraction * that notional.
    3) Also ensure we do not exceed margin limits.
    """
    contract_value = get_contract_value(symbol, current_price)
    
    # Max contracts based on notional
    max_contracts_notional = int(TARGET_NOTIONAL_PER_SYMBOL // contract_value)
    if max_contracts_notional < 1:
        max_contracts_notional = 1  # at least 1 if we truly have a strong signal
    
    # Margin-based limit
    margin_req = get_margin_requirements(symbol)
    margin_available = get_margin_allocation(account_value)
    margin_limit = int(margin_available // margin_req)
    
    return min(max_contracts_notional, margin_limit)

def calculate_target_position(position_fraction, max_contracts):
    """
    `position_fraction` is from eod_signals.csv (vol-scaling or partial).
    If fraction=1 => full position => max_contracts.
    If fraction=0.5 => half => max_contracts//2, etc.
    
    We'll do normal rounding. If the result is >=0.5, we round up to 1.
    """
    raw = abs(position_fraction * max_contracts)
    # Round half up
    target_size = int(raw + 0.5)
    return target_size if position_fraction >= 0 else -target_size

def place_order(ib, contract, quantity, action):
    """Place a market order and wait for fill."""
    order = MarketOrder(action, abs(quantity))
    trade = ib.placeOrder(contract, order)
    
    timeouts = {
        'LE': 45,
        'HE': 45,
        'GF': 45,
        'ZC': 30, 'ZW': 30, 'ZS': 30,
    }
    timeout = timeouts.get(contract.symbol, 20)
    
    start_time = datetime.now()
    filled = 0
    while not trade.isDone():
        if trade.orderStatus.filled > filled:
            filled = trade.orderStatus.filled
            fill_price = trade.orderStatus.avgFillPrice
            logger.debug(f"Partial fill: {filled} @ {fill_price}")
            
        if (datetime.now() - start_time).seconds > timeout:
            logger.warning(f"Order timeout after {filled} fills")
            ib.cancelOrder(order)
            break
        ib.sleep(1)
    
    return trade

def format_price(symbol, price):
    """Format price according to symbol's typical decimal places."""
    fmt = get_price_format(symbol)
    if fmt.startswith('.'):
        decimals = len(fmt) - 1
        return f"{price:.{decimals}f}"
    return f"{price:{fmt}}"

def execute_trades():
    logger.info(f"\n=== Trade Execution {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    try:
        # Connect to IB
        ib = IB()
        ib.connect('127.0.0.1', 7497, clientId=1)
        
        account_value = get_account_value(ib)
        if not account_value:
            raise ValueError("Could not get account value")
        margin_available = get_margin_allocation(account_value)
        logger.info(f"Account Value: ${account_value:,.2f}")
        logger.info(f"Margin Available: ${margin_available:,.2f}")
        logger.info(f"Target Notional/Symbol (signal=±1): ${TARGET_NOTIONAL_PER_SYMBOL:,.0f}\n")
        
        # Read signals and get positions
        signals = pd.read_csv('eod_signals.csv')
        current_positions = get_current_positions(ib)
        
        if current_positions:
            logger.info("Current Positions: " + ", ".join([f"{s}: {p}" for s, p in current_positions.items()]))
        else:
            logger.info("Current Positions: None")
        
        logger.info("\nPosition Analysis:")
        logger.info(f"{'Symbol':<6} {'Signal':<8} {'PosFrac':<10} {'MaxPos':<7} {'TgtPos':<7} {'Current':<8} {'Trade':<6} {'Notional/Contract':<15}")
        logger.info("-" * 85)
        
        total_margin_used = 0
        filled_trades = {}
        
        for _, row in signals.iterrows():
            symbol = row['symbol']
            signal = row['signal']  # -1 to +1
            pos_frac = row['position']  # e.g. -0.3, +0.8, etc.
            price = row['close']
            
            # Skip if basically zero fraction
            if abs(pos_frac) < 0.01:
                continue
            
            contract_value = get_contract_value(symbol, price)
            max_pos = get_max_position(symbol, account_value, price)
            
            # desired final position
            target_position = calculate_target_position(pos_frac, max_pos)
            
            current_position = current_positions.get(symbol, 0)
            trade_size = target_position - current_position
            
            logger.info(
                f"{symbol:<6} {signal:>8.2f} {pos_frac:>10.2f} {max_pos:>7} "
                f"{target_position:>7} {current_position:>8.1f} {trade_size:>6.1f} "
                f"${contract_value:>14,.0f}"
            )
            
            if trade_size == 0:
                continue
                
            # Check margin
            margin_req = get_margin_requirements(symbol)
            margin_impact = abs(trade_size) * margin_req
            new_total_margin = total_margin_used + margin_impact
            
            if new_total_margin > margin_available:
                logger.info(f"SKIP {symbol}: Would exceed margin limit (${new_total_margin:,.0f} > ${margin_available:,.0f})")
                continue
            
            try:
                contract = get_contract(ib, symbol)
                action = 'BUY' if trade_size > 0 else 'SELL'
                trade = place_order(ib, contract, trade_size, action)
                
                if trade.orderStatus.status == 'Filled':
                    fill_price = trade.orderStatus.avgFillPrice
                    filled = trade.orderStatus.filled
                    margin_used = abs(filled) * margin_req
                    total_margin_used += margin_used
                    notional = get_contract_value(symbol, fill_price) * abs(filled)
                    filled_trades[symbol] = (filled, margin_used, margin_req, notional)
                    formatted_price = format_price(symbol, fill_price)
                    logger.info(f"FILL {symbol}: {action} {filled} @ {formatted_price}")
                else:
                    logger.info(f"TIMEOUT {symbol}: {action} {abs(trade_size)} not filled.")
                
            except Exception as e:
                logger.error(f"ERROR {symbol}: {str(e)}")
                continue
        
        # Summaries
        if filled_trades:
            logger.info("\nPosition Breakdown:")
            logger.info(f"{'Symbol':<6} {'Contracts':<10} {'MarginUsed':<12} {'Notional':<12}")
            logger.info("-" * 45)
            total_notional = 0
            for symb, (contracts, margin, _, notional) in filled_trades.items():
                logger.info(f"{symb:<6} {contracts:>10.0f} ${margin:>11,.0f} ${notional:>11,.0f}")
                total_notional += notional
            logger.info("-" * 45)
            logger.info(f"TOTAL      {'' :<10} ${total_margin_used:>11,.0f} ${total_notional:>11,.0f}")
        
        pct_margin = (total_margin_used / margin_available)*100 if margin_available else 0
        logger.info(f"\nFinal Margin Used: ${total_margin_used:,.0f} ({pct_margin:.1f}%)")
        
    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        raise
        
    finally:
        if 'ib' in locals() and ib.isConnected():
            ib.disconnect()

if __name__ == "__main__":
    execute_trades()
