import random
import pandas as pd
import numpy as np
import multiprocessing

from deap import base, creator, tools

# Import your backtest functions
from backtest import (
    compute_indicators_and_signal,
    run_backtest_on_df,
    compute_equity_curve_and_stats
)

###############################################################################
# 1) Load Data
###############################################################################
csv_file = "historical_BTC.csv"
df_main = pd.read_csv(csv_file)

df_main['open'] = df_main['open'].astype(float)
df_main['high'] = df_main['high'].astype(float)
df_main['low'] = df_main['low'].astype(float)
df_main['close'] = df_main['close'].astype(float)
df_main['volume'] = df_main['volume'].astype(float)

###############################################################################
# 2) Define Parameter Ranges
###############################################################################
# Suppose we want to optimize:
# - rsi_period in [5..30]
# - macd_fast in [5..20]
# - macd_slow in [20..40]
# - macd_signal in [5..15]
# - bb_period in [10..30]
# We'll keep devup/devdn = 2.0 fixed for now, or you can expand it.
# Also we might tune stop_loss_frac in [0.01..0.05], etc.

RSI_MIN, RSI_MAX = 5, 30
MACD_FAST_MIN, MACD_FAST_MAX = 5, 20
MACD_SLOW_MIN, MACD_SLOW_MAX = 20, 40
MACD_SIGNAL_MIN, MACD_SIGNAL_MAX = 5, 15
BB_MIN, BB_MAX = 10, 30
STOPLOSS_MIN, STOPLOSS_MAX = 0.01, 0.05

###############################################################################
# 3) Define Fitness Function
###############################################################################
def backtest_fitness(individual):
    """
    individual = [
        rsi_period,
        macd_fast,
        macd_slow,
        macd_signal,
        bb_period,
        stop_loss_frac
    ]
    We'll run the pipeline and return the 'annual_return' from compute_equity_curve_and_stats.
    DEAP expects a tuple as fitness, so we'll return (score,).
    We'll NEGATE it if we want to minimize, but here we want to MAXIMIZE annual_return => no negation needed.
    """
    (rsi_p, macd_f, macd_s, macd_sig, bb_p, stop_frac) = individual

    # 1) Copy the original DataFrame, so we don't mutate df_main
    df = df_main.copy()

    # 2) Compute Indicators
    df = compute_indicators_and_signal(
        df,
        rsi_period=int(rsi_p),
        macd_fast=int(macd_f),
        macd_slow=int(macd_s),
        macd_signal=int(macd_sig),
        bb_period=int(bb_p),
        bb_devup=2.0,
        bb_devdn=2.0
    )

    # 3) Run Backtest
    trades, total_pnl, pct_return, equity_series = run_backtest_on_df(
        df,
        initial_capital=10000.0,
        fee_rate=0.0005,
        slippage_rate=0.0002,
        max_risk_percent=0.02,
        stop_loss_frac=float(stop_frac),
        cooldown_bars=10,
        max_bars_in_trade=50
    )

    # 4) Compute Stats
    stats = compute_equity_curve_and_stats(equity_series, initial_capital=10000.0, bars_per_year=252)
    ann_return = stats['annual_return']
    if np.isnan(ann_return):
        ann_return = -9999  # penalize invalid

    # In case you want Sharpe, you can do:
    # sharpe = stats['sharpe'] if not np.isnan(stats['sharpe']) else -9999
    # Or combine metrics:
    # fitness_value = sharpe - (stats['max_drawdown'] * 1.0)
    # But let's just do annual_return for simplicity:

    fitness_value = ann_return

    return (fitness_value,)  # DEAP expects a tuple

###############################################################################
# 4) Configure DEAP
###############################################################################
# We'll use a Single Objective -> Maximize 'annual_return'
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 1.0 => maximize
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Define how to create each parameter (gene)
toolbox.register("rsi_gene", random.randint, RSI_MIN, RSI_MAX)
toolbox.register("macd_fast_gene", random.randint, MACD_FAST_MIN, MACD_FAST_MAX)
toolbox.register("macd_slow_gene", random.randint, MACD_SLOW_MIN, MACD_SLOW_MAX)
toolbox.register("macd_signal_gene", random.randint, MACD_SIGNAL_MIN, MACD_SIGNAL_MAX)
toolbox.register("bb_gene", random.randint, BB_MIN, BB_MAX)
toolbox.register("stoploss_gene", random.uniform, STOPLOSS_MIN, STOPLOSS_MAX)

# Structure an Individual as a list of these 6 genes
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (
        toolbox.rsi_gene,
        toolbox.macd_fast_gene,
        toolbox.macd_slow_gene,
        toolbox.macd_signal_gene,
        toolbox.bb_gene,
        toolbox.stoploss_gene
    ),
    n=1
)

# Population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# The evaluation function
toolbox.register("evaluate", backtest_fitness)

# Genetic operators: mate, mutate, select
toolbox.register("mate", tools.cxTwoPoint)
# We'll do small random changes to genes in mutate
def mutation_fn(individual, indpb=0.2):
    # Indpb => probability of mutating each gene
    # For RSI, MACD => random.randint in [Min,Max]
    # For stoploss => random.uniform in [Min,Max]
    # We'll do a simple approach
    for i, val in enumerate(individual):
        if random.random() < indpb:
            if i == 0: # rsi
                individual[i] = random.randint(RSI_MIN, RSI_MAX)
            elif i == 1: # macd_fast
                individual[i] = random.randint(MACD_FAST_MIN, MACD_FAST_MAX)
            elif i == 2: # macd_slow
                individual[i] = random.randint(MACD_SLOW_MIN, MACD_SLOW_MAX)
            elif i == 3: # macd_signal
                individual[i] = random.randint(MACD_SIGNAL_MIN, MACD_SIGNAL_MAX)
            elif i == 4: # bb_period
                individual[i] = random.randint(BB_MIN, BB_MAX)
            elif i == 5: # stop_loss_frac
                individual[i] = random.uniform(STOPLOSS_MIN, STOPLOSS_MAX)
    return (individual,)

toolbox.register("mutate", mutation_fn, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

###############################################################################
# 5) Main GA Loop
###############################################################################
def main():
    # 1) Initialize random seed for reproducibility
    random.seed(42)

    # 2) Create a process pool
    pool = multiprocessing.Pool(processes=4)  # or any desired number of workers

    # 3) Register pool.map with the toolbox to parallelize evaluate calls
    toolbox.register("map", pool.map)

    # GA hyperparameters
    population_size = 20
    n_generations = 10

    # 4) Create initial population
    pop = toolbox.population(n=population_size)

    # 5) Evaluate initial population (in parallel) using the pool's map
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # 6) Main GA loop
    for gen in range(n_generations):
        print(f"--- Generation {gen} ---")

        # (a) Selection
        offspring = toolbox.select(pop, len(pop))

        # (b) Clone
        offspring = list(map(toolbox.clone, offspring))

        # (c) Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:  # 70% crossover
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # (d) Mutation
        for mutant in offspring:
            if random.random() < 0.3:  # 30% chance to mutate each child
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # (e) Evaluate (in parallel) only invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # (f) Replace population
        pop[:] = offspring

        # (g) Print best in generation
        fits = [ind.fitness.values[0] for ind in pop]
        best_ind = pop[np.argmax(fits)]
        print(f"  Best individual: {best_ind}, Fitness: {max(fits):.4f}")

    # 7) After GA completes, close the pool
    pool.close()
    pool.join()

    # 8) Print final best
    final_fits = [ind.fitness.values[0] for ind in pop]
    best_index = np.argmax(final_fits)
    best_ind = pop[best_index]
    print("\n=== Best Overall Individual ===")
    print(best_ind, final_fits[best_index])

    print("RSI:", best_ind[0])
    print("MACD fast:", best_ind[1])
    print("MACD slow:", best_ind[2])
    print("MACD signal:", best_ind[3])
    print("BB period:", best_ind[4])
    print("StopLoss frac:", best_ind[5])

    # Return final population in case you want it
    return pop


if __name__ == "__main__":
    final_pop = main()
