from copy import deepcopy
import bt
import ffn
import pandas as pd


def run(*backtests):
    # run each backtest
    for bkt in backtests:
        bkt.run()

    return Result(*backtests)


class Backtest(object):

    def __init__(self, strategy, data,
                 name=None,
                 initial_capital=1000000.0,
                 commissions=None):

        # we want to reuse strategy logic - copy it!
        # basically strategy is a template
        self.strategy = deepcopy(strategy)
        self.data = data
        self.dates = data.index
        self.initial_capital = initial_capital
        self.name = name if name is not None else strategy.name
        if commissions:
            self.strategy.set_commissions(commissions)

    def run(self):
        # setup strategy
        self.strategy.setup(self.data)

        # adjust strategy with initial capital
        self.strategy.adjust(self.initial_capital)

        # loop through dates
        for dt in self.dates:
            self.strategy.update(dt)
            self.strategy.run()
