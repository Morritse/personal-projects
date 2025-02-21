"""
ga_optimize.py

A simple GA that searches both MA and trading parameter space, in parallel.
Requires:
  - config_ga.py with GA_PARAM_BOUNDS, GA_PARAM_CHOICES, etc.
  - strategy.py with RefinedFuturesStrategy & analyze_portfolio
  - Data loaded from fetch_futures_data or local CSV

Usage:
  python ga_optimize.py

Adjust population size, generations, mutation rates, etc. as needed.
"""

import random
import math
import copy
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import your parameter config
from config_ga import GA_PARAM_BOUNDS, GA_PARAM_CHOICES, DEFAULT_FIXED_PARAMS

# Suppose you have a FUTURES dict or define your instruments
from fetch_futures_data import FUTURES

# Import your refined strategy
from strategy import RefinedFuturesStrategy, analyze_portfolio


# ------------------------------------------------------------------------------
# 1) Load data
# ------------------------------------------------------------------------------
def load_data() -> Dict[str, pd.DataFrame]:
    """
    Loads each symbol's CSV, forces a tz-naive DatetimeIndex.
    Returns a dict of { symbol: DataFrame }.
    """
    dfs = {}
    for symbol in FUTURES.keys():
        try:
            filename = f"data/{symbol.replace('=', '_')}_daily.csv"
            # 1) Read CSV with parse_dates=True so pandas tries to parse the index as datetime
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            
            # 2) Force the index to be treated as UTC if it’s not recognized
            #    This step ensures a tz-aware index, or re-localizes it to UTC if already aware.
            df.index = pd.to_datetime(df.index, utc=True, errors='coerce')

            # 3) Convert to tz-naive (removing the UTC offset)
            df.index = df.index.tz_convert(None)
            
            # 4) If desired, drop any rows with a NaT index (in case of parsing issues):
            df = df[~df.index.isna()]

            # 5) We now have a tz-naive DatetimeIndex
            dfs[symbol] = df

        except FileNotFoundError:
            print(f"No data file found for {symbol}")
        except Exception as e:
            print(f"Error loading {symbol} from {filename}: {e}")

    return dfs


# ------------------------------------------------------------------------------
# 2) GA Parameter Routines
# ------------------------------------------------------------------------------
def random_param_value(param_name: str) -> Any:
    """
    Generate a random value for one parameter, based on GA_PARAM_BOUNDS or GA_PARAM_CHOICES.
    If it's numeric, we sample within (min, max).
    If it's a list of choices, we pick from that list.
    """
    # If param_name is in GA_PARAM_CHOICES, pick from a discrete list
    if param_name in GA_PARAM_CHOICES:
        return random.choice(GA_PARAM_CHOICES[param_name])
    
    # Else we assume param_name is in GA_PARAM_BOUNDS
    (low, high, ptype) = GA_PARAM_BOUNDS[param_name]
    if ptype == 'int':
        return random.randint(low, high)
    elif ptype == 'float':
        return random.uniform(low, high)
    else:
        raise ValueError(f"Unsupported type {ptype} for param {param_name}")


def clip_param(param_name: str, value: Any) -> Any:
    """
    Ensure the parameter remains in range. If it's a list param, we pick
    the nearest valid choice, or re-sample. For numeric, we clip into (low, high).
    """
    # If param_name in GA_PARAM_CHOICES, no "clip" needed, but we can fix if invalid
    if param_name in GA_PARAM_CHOICES:
        # If already in the list, fine; else pick the closest or re-sample
        if value not in GA_PARAM_CHOICES[param_name]:
            # fallback or pick random
            return random.choice(GA_PARAM_CHOICES[param_name])
        return value
    
    # It's numeric
    (low, high, ptype) = GA_PARAM_BOUNDS[param_name]
    if ptype == 'int':
        clipped = max(low, min(high, int(round(value))))
        return clipped
    elif ptype == 'float':
        clipped = max(low, min(high, float(value)))
        return clipped
    else:
        raise ValueError(f"Unknown ptype {ptype} for param {param_name}")


# ------------------------------------------------------------------------------
# 3) Individual Representation + Initialization
# ------------------------------------------------------------------------------
ALL_PARAMS = (
    list(GA_PARAM_BOUNDS.keys()) +
    list(GA_PARAM_CHOICES.keys())
)

def create_random_individual() -> dict:
    """
    Create a random 'individual' param dict, sampling from GA_PARAM_BOUNDS and GA_PARAM_CHOICES.
    """
    indiv = {}
    for p in ALL_PARAMS:
        indiv[p] = random_param_value(p)
    # Merge in fixed params (like debug=False)
    indiv.update(DEFAULT_FIXED_PARAMS)
    return indiv

def mutate_individual(indiv: dict, mutation_prob: float=0.2, mutation_scale: float=0.3) -> dict:
    """
    Randomly mutate each param with probability=mutation_prob.
    For numeric params, we add a random delta scaled by mutation_scale*(range).
    For list-based, we just pick a new choice with that probability.
    """
    new_indiv = copy.deepcopy(indiv)
    for p in ALL_PARAMS:
        if random.random() < mutation_prob:
            # mutate
            if p in GA_PARAM_CHOICES:
                # pick a new choice
                new_indiv[p] = random.choice(GA_PARAM_CHOICES[p])
            else:
                # numeric
                (low, high, ptype) = GA_PARAM_BOUNDS[p]
                if ptype == 'int':
                    val_range = high - low
                    delta = random.randint(-int(val_range*mutation_scale),
                                            int(val_range*mutation_scale))
                    new_val = new_indiv[p] + delta
                else:
                    # float
                    val_range = high - low
                    delta = random.uniform(-val_range*mutation_scale, val_range*mutation_scale)
                    new_val = new_indiv[p] + delta
                # clip
                new_indiv[p] = clip_param(p, new_val)
    return new_indiv


def crossover(indiv1: dict, indiv2: dict) -> dict:
    """
    Simple 1-point or uniform crossover: for each param, 50% chance take from indiv1 or indiv2.
    """
    child = {}
    for p in ALL_PARAMS:
        if random.random() < 0.5:
            child[p] = indiv1[p]
        else:
            child[p] = indiv2[p]
    # Also add in fixed params again
    child.update(DEFAULT_FIXED_PARAMS)
    return child


# ------------------------------------------------------------------------------
# 4) Evaluate Individuals in Parallel
# ------------------------------------------------------------------------------
def evaluate_individual(dfs: Dict[str, pd.DataFrame], indiv: dict) -> dict:
    """
    Build the strategy from 'indiv' params, run analyze_portfolio,
    and return a result dict with { 'sharpe': X, 'annual_return': Y, etc. }
    """
    strategy = RefinedFuturesStrategy(**indiv)
    metrics = analyze_portfolio(dfs, strategy)
    return {
        'params': indiv,
        'sharpe_ratio': metrics['sharpe_ratio'],
        'annual_return': metrics['annual_return'],
        'annual_vol': metrics['annual_vol'],
        'max_drawdown': metrics['max_drawdown']
    }


# ------------------------------------------------------------------------------
# 5) Main GA Loop
# ------------------------------------------------------------------------------
def main_ga_optimization(
    dfs: Dict[str, pd.DataFrame],
    pop_size: int=20,
    generations: int=10,
    crossover_rate: float=0.7,
    mutation_prob: float=0.2,
    mutation_scale: float=0.3
):
    """
    GA main loop:
      1) Initialize population
      2) Evaluate in parallel
      3) Sort by Sharpe
      4) Breed top, mutate, etc.
      5) Repeat for multiple generations
    """
    # 1) Initialize population
    population = [create_random_individual() for _ in range(pop_size)]

    # Evaluate population
    population_results = evaluate_population_parallel(dfs, population)

    for gen in range(generations):
        print(f"\n=== Generation {gen+1}/{generations} ===")

        # Sort by Sharpe desc
        population_results.sort(key=lambda r: r['sharpe_ratio'], reverse=True)

        # Print top 1 in the generation
        top_result = population_results[0]
        print(f"Best so far: Sharpe={top_result['sharpe_ratio']:.2f}, "
              f"Return={top_result['annual_return']:.2%}, "
              f"Vol={top_result['annual_vol']:.2%}, "
              f"DD={top_result['max_drawdown']:.2%}")
        # Generate next generation
        next_pop = []

        # Elitism: keep top 2
        next_pop.append(copy.deepcopy(population_results[0]['params']))
        next_pop.append(copy.deepcopy(population_results[1]['params']))

        # Fill the rest by crossover + mutation
        while len(next_pop) < pop_size:
            # Selection: pick 2 parents via e.g. tournament or top
            parent1 = tournament_selection(population_results, k=5)
            parent2 = tournament_selection(population_results, k=5)

            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)

            # mutate
            child = mutate_individual(child, mutation_prob, mutation_scale)
            next_pop.append(child)

        # Evaluate new population
        population = next_pop
        population_results = evaluate_population_parallel(dfs, population)

    # Final sort
    population_results.sort(key=lambda r: r['sharpe_ratio'], reverse=True)
    return population_results


def evaluate_population_parallel(dfs, population) -> List[dict]:
    """
    Evaluate a list of individuals in parallel using ProcessPoolExecutor.
    """
    results = []
    with ProcessPoolExecutor() as executor:
        futures_map = {
            executor.submit(evaluate_individual, dfs, indiv): indiv for indiv in population
        }
        for future in as_completed(futures_map):
            indiv = futures_map[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Error evaluating {indiv}: {e}")
    return results


def tournament_selection(results_list: List[dict], k: int=5) -> dict:
    """
    Basic tournament: pick k random individuals, return the best by Sharpe.
    Returns the 'params' dict for that best individual.
    """
    selected = random.sample(results_list, min(k, len(results_list)))
    selected.sort(key=lambda r: r['sharpe_ratio'], reverse=True)
    return copy.deepcopy(selected[0]['params'])


# ------------------------------------------------------------------------------
# 6) Main Script
# ------------------------------------------------------------------------------
def main():
    print("Loading data...")
    dfs = load_data()
    if not dfs:
        print("No data loaded. Exiting.")
        return

    # GA Settings
    POP_SIZE = 20
    GENERATIONS = 10
    CROSSOVER_RATE = 0.7
    MUTATION_PROB = 0.2
    MUTATION_SCALE = 0.3

    print(f"Starting GA with pop_size={POP_SIZE}, generations={GENERATIONS}...")

    final_results = main_ga_optimization(
        dfs,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        crossover_rate=CROSSOVER_RATE,
        mutation_prob=MUTATION_PROB,
        mutation_scale=MUTATION_SCALE
    )

    # final_results is sorted by Sharpe desc
    print("\n=== GA Optimization Complete ===")
    best = final_results[0]
    print("Best Individual:")
    for k,v in best['params'].items():
        print(f"  {k}: {v}")

    print(f"Sharpe: {best['sharpe_ratio']:.2f}")
    print(f"Annual Return: {best['annual_return']:.2%}")
    print(f"Annual Vol: {best['annual_vol']:.2%}")
    print(f"Max Drawdown: {best['max_drawdown']:.2%}")

    # Save entire final population results
    df = pd.DataFrame(final_results)
    df.to_csv("ga_final_results.csv", index=False)
    print("\nSaved final population results to ga_final_results.csv")

if __name__ == "__main__":
    main()
