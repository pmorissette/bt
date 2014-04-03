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

        self.stats = {}
        self._original_prices = None
        self._weights = None
        self._sweights = None

    def run(self):
        # setup strategy
        self.strategy.setup(self.data)

        # adjust strategy with initial capital
        self.strategy.adjust(self.initial_capital)

        # loop through dates
        for dt in self.dates:
            self.strategy.update(dt)
            self.strategy.run()
            # need update after to save weights, values and such
            self.strategy.update(dt)

        self.stats = self.strategy.prices.calc_perf_stats()
        self._original_prices = self.strategy.prices

    @property
    def weights(self):
        if self._weights is not None:
            return self._weights
        else:
            vals = pd.DataFrame({x.full_name: x.values for x in
                                 self.strategy.members})
            vals = vals.div(self.strategy.values, axis=0)
            self._weights = vals
            return vals

    @property
    def security_weights(self):
        if self._sweights is not None:
            return self._sweights
        else:
            # get values for all securities in tree and divide by root values
            # for security weights
            vals = pd.DataFrame({x.name: x.values for x in
                                 self.strategy.members if
                                 isinstance(x, bt.core.SecurityBase)})
            vals = vals.div(self.strategy.values, axis=0)

            # combine securities with same ticker
            vals = vals.groupby(vals.columns, axis=1).sum()

            # save for future use
            self._sweights = vals

            return vals


class Result(ffn.GroupStats):

    def __init__(self, *backtests):
        tmp = [pd.DataFrame({x.name: x.strategy.prices}) for x in backtests]
        super(Result, self).__init__(*tmp)
        self.backtests = {x.name: x for x in backtests}

    def set_date_range(self, start=None, end=None):
        self.stats.set_date_range(start, end)
