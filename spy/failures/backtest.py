import backtrader as bt
import backtrader.analyzers as btanalyzers

class EmaCrossoverFlipStrategy(bt.Strategy):
    params = (
        ('fast_len', 9),
        ('slow_len', 20),
        ('stop_loss_pct', 0.01),
    )

    def __init__(self):
        self.fast_ema = bt.ind.EMA(period=self.params.fast_len)
        self.slow_ema = bt.ind.EMA(period=self.params.slow_len)
        self.crossover = bt.ind.CrossOver(self.fast_ema, self.slow_ema)

        self.order = None
        self.buy_price = None

    def notify_order(self, order):
        if not order.alive():
            self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Go long if fast crosses above slow
            if self.crossover > 0:
                self.buy_price = self.data.close[0]
                self.order = self.buy()
            # Go short if fast crosses below slow
            elif self.crossover < 0:
                self.buy_price = self.data.close[0]
                self.order = self.sell()
        else:
            # If we're long:
            if self.position.size > 0:
                # Reverse to short on negative crossover
                if self.crossover < 0:
                    self.close()
                    self.order = self.sell()
                    self.buy_price = self.data.close[0]
                # Stop-loss
                elif self.data.close[0] < self.buy_price * (1 - self.params.stop_loss_pct):
                    self.order = self.close()

            # If we're short:
            elif self.position.size < 0:
                # Reverse to long on positive crossover
                if self.crossover > 0:
                    self.close()
                    self.order = self.buy()
                    self.buy_price = self.data.close[0]
                # Stop-loss for short
                elif self.data.close[0] > self.buy_price * (1 + self.params.stop_loss_pct):
                    self.order = self.close()


if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # -------------------------------------------------
    # Instead of .addstrategy(), call .optstrategy()
    # -------------------------------------------------
    cerebro.optstrategy(
        EmaCrossoverFlipStrategy,
        fast_len=range(5, 15, 2),     # e.g. 5, 7, 9, 11, 13
        slow_len=range(15, 31, 5),    # e.g. 15, 20, 25, 30
        stop_loss_pct=[0.005, 0.01, 0.015]
    )

    # cerebro.optstrategy(
    #     EmaCrossoverFlipStrategy,
    #     fast_len=[5, 10],
    #     slow_len=[15],
    #     stop_loss_pct=[0.005]
    # )

    # Add your data feed
    data = bt.feeds.GenericCSVData(
        dataname='spy.csv',
        dtformat='%Y-%m-%d %H:%M:%S%z',
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        timeframe=bt.TimeFrame.Minutes,
        compression=1
    )
    cerebro.adddata(data)

    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0001)

    # Add analyzers to each run
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')

    # Run the entire optimization
    results = cerebro.run()

    # -------------------------------------------------
    # results is a list of lists of Strategy instances
    # since you might have multiple strategy classes
    # each item in `results` is one optimized run
    # -------------------------------------------------

    # Let's parse them:
    best_sharpe = None
    best_params = None
    best_strat_instance = None

    for run in results:
        # 'run' is a list of strategy objects (usually length=1 if you only specify one strategy).
        strat_inst = run[0]
        # Access analyzers
        sharpe_dict = strat_inst.analyzers.sharpe.get_analysis()
        dd_dict = strat_inst.analyzers.drawdown.get_analysis()
        trade_dict = strat_inst.analyzers.trade_analyzer.get_analysis()

        # Example metric: Sharpe Ratio
        sharpe_val = sharpe_dict.get('sharperatio', None)

        # If Sharpe is not None, track the best
        if sharpe_val is not None:
            if best_sharpe is None or sharpe_val > best_sharpe:
                best_sharpe = sharpe_val
                best_params = strat_inst.params.__dict__
                best_strat_instance = strat_inst

    print('---------------------------------')
    print('Best Sharpe Ratio:', best_sharpe)
    print('Best Params:', best_params)
    print('---------------------------------')

    # You could also print other stats for your best strategy instance here
