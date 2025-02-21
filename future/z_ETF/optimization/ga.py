import os
import glob
import numpy as np
import pandas as pd
import itertools
import datetime
import pygad  # pip install pygad

from typing import Dict, Any, List

##############################################################
# 0) Provide or import your helper functions here
#    1) load_data(data_folder) -> dict of {symbol: DataFrame}
#    2) run_backtest_on_universe(dict, param_dict) -> DataFrame with 'portfolio'
#    3) compute_metrics(Series_of_returns) -> {annual_return, sharpe_ratio, max_drawdown, ...}
##############################################################

# For demonstration, we'll define placeholders:
def load_data(data_folder: str) -> Dict[str, pd.DataFrame]:
    """Loads CSV from data_folder into a dict: {symbol:df}."""
    data_dict = {}
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    for path in csv_files:
        symbol = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df.sort_values('Date', inplace=True)
            df.set_index('Date', inplace=True)
        data_dict[symbol] = df
    return data_dict

# Import the real backtest logic with trading parameters
from optimize_trading import compute_indicators, backtest_symbol, apply_exits

def run_backtest_on_universe(data_dict: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> pd.DataFrame:
    """
    Run backtest across all symbols using the real strategy logic with trading parameters.
    """
    all_rets = pd.DataFrame()
    
    for symbol, df_raw in data_dict.items():
        try:
            # Use real backtest logic with trading parameters
            df = compute_indicators(df_raw.copy(), params)
            df = apply_exits(df, params)  # Apply trading rules
            daily_strat, _ = backtest_symbol(df, params)
            daily_strat.name = symbol
            all_rets = pd.concat([all_rets, daily_strat], axis=1)
        except Exception as e:
            print(f"Error backtesting {symbol}: {str(e)}")
            continue
    
    all_rets.dropna(how='all', inplace=True)
    all_rets['portfolio'] = all_rets.mean(axis=1)
    return all_rets

def compute_metrics(returns: pd.Series) -> Dict[str,float]:
    """
    Example stub. Compute annual_return, sharpe_ratio, max_drawdown, etc.
    """
    rets = returns.dropna()
    if len(rets) < 2:
        return dict(annual_return=0, sharpe_ratio=0, max_drawdown=0)
    ann_factor = 252
    daily_mean = rets.mean()
    daily_std  = rets.std()
    annual_return = daily_mean*ann_factor
    annual_vol    = daily_std*np.sqrt(ann_factor)
    sharpe        = annual_return/annual_vol if annual_vol>1e-9 else 0
    # max drawdown
    cum = (1+rets).cumprod()
    peak= cum.cummax()
    dd  = (cum/peak -1).min()
    return dict(
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=dd
    )

##############################################################
# 1) Define GA parameter space
##############################################################
PARAM_SPACE = {
    # MA params
    "lookback_fast":      [10, 20, 50, 100],
    "lookback_slow":      [50, 100, 200, 300],
    "vol_lookback":       [20, 60, 120],
    "vol_target":         [0.10, 0.15, 0.20],
    "adx_threshold":      [None, 10, 20],

    # Trading params
    "stop_atr_multiple":  [1.5, 2.0, 2.5, 3.0],
    "partial_exit_1":     [0.75, 1.0, 1.25],
    "partial_exit_2":     [1.5, 2.0],
    "time_stop":          [0, 10, 20, 30],
    "trailing_stop_factor":[0.0, 1.5, 2.0, 3.0],
    "scaling_mode":       ["none", "pyramid"],
    "corr_filter":        [False, 0.60, 0.80],
}

GA_KEYS   = list(PARAM_SPACE.keys())   # e.g. ['lookback_fast','lookback_slow',...]
GA_RANGES = [PARAM_SPACE[k] for k in GA_KEYS]  # list of lists

# We'll store the data globally for faster access
DATA_DICT = {}

##############################################################
# 2) Helper to decode GA chromosome -> param dictionary
##############################################################
def param_index_to_dict(param_genes):
    """Given a list of len(GA_KEYS), each in [0..(range_n-1)], build param dict."""
    param_dict = {}
    for i, key in enumerate(GA_KEYS):
        valid_values = GA_RANGES[i]
        # clamp the gene
        idx = int(round(param_genes[i]))
        idx = max(0, min(idx, len(valid_values)-1))
        param_dict[key] = valid_values[idx]
    return param_dict

##############################################################
# 3) GA Fitness Function
##############################################################
def fitness_func(ga_instance, solution, sol_idx):
    # decode solution -> param dict
    param_dict = param_index_to_dict(solution)

    # run backtest
    results_df = run_backtest_on_universe(DATA_DICT, param_dict)
    if "portfolio" not in results_df.columns:
        return -9999.0  # just in case
    
    # compute metrics
    mets = compute_metrics(results_df['portfolio'])
    ann_ret = mets['annual_return']
    sharpe  = mets['sharpe_ratio']
    dd      = mets['max_drawdown']

    # Weighted fitness: emphasize returns more than Sharpe
    # for example:
    fit = 2.0*ann_ret + sharpe  # might yield bigger numbers

    # optional penalty for too large DD
    if dd < -0.25:  # if drawdown worse than -25%
        fit -= 2.0

    return fit

##############################################################
# 4) GA Callback for each generation
##############################################################
def on_generation(ga_inst):
    best_sol, best_fit, best_sol_idx = ga_inst.best_solution()
    p_dict = param_index_to_dict(best_sol)
    print(f"\n== Generation {ga_inst.generations_completed}/{ga_inst.num_generations} ==")
    print(f"  Best Fitness: {best_fit:.3f}")
    print("  Genes:", best_sol)
    print("  Decoded:", p_dict)

##############################################################
# 5) main() - orchestrate everything
##############################################################
def main():
    # 1) load data
    global DATA_DICT
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_folder")
    DATA_DICT = load_data(data_folder)
    if not DATA_DICT:
        print("No CSV data found or loaded!")
        return

    # 2) Build gene_space for each param
    gene_space = []
    for i, key in enumerate(GA_KEYS):
        length = len(GA_RANGES[i])
        # param i can be in [0..length-1]
        gene_space.append({"low":0, "high": length-1, "step":1})

    # 3) configure GA
    ga_instance = pygad.GA(
        num_generations= 20,          # choose
        num_parents_mating= 10,       # choose
        fitness_func= fitness_func,
        sol_per_pop= 30,             # population size
        num_genes= len(GA_KEYS),
        gene_type= float,
        init_range_low= 0,
        init_range_high= 1,  # each gene starts in [0..1], then mutated
        gene_space= gene_space,
        crossover_type= "single_point",
        mutation_type= "random",
        mutation_probability= 0.2,
        keep_elitism= 2,
        on_generation= on_generation
    )

    # 4) run GA
    ga_instance.run()

    # 5) final results
    best_sol, best_fit, best_sol_idx = ga_instance.best_solution()
    p_dict = param_index_to_dict(best_sol)
    print("\n=== GA Complete ===")
    print(f"Best Fitness: {best_fit:.3f}")
    print("Genes:", best_sol)
    print("Decoded Param Set:")
    for k,v in p_dict.items():
        print(f"  {k}: {v}")

    # re-run backtest with best param
    final_df = run_backtest_on_universe(DATA_DICT, p_dict)
    if "portfolio" in final_df.columns:
        final_mets = compute_metrics(final_df["portfolio"])
        print("\nFinal Metrics:")
        for mk, mv in final_mets.items():
            if mk in ("annual_return","max_drawdown"):
                print(f"  {mk}: {mv:.2%}")
            else:
                print(f"  {mk}: {mv:.3f}")

if __name__ == "__main__":
    main()
