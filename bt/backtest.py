from copy import deepcopy
import bt
import ffn
import pandas as pd
from matplotlib import pyplot as plt


def run(*backtests):
    # run each backtest
    for bkt in backtests:
        bkt.run()

    return Result(*backtests)


def benchmark_random(backtest, random_strategy, nsim=100):
    # save name for future use
    if backtest.name is None:
        backtest.name = 'original'

    # run if necessary
    if not backtest.has_run:
        backtest.run()

    bts = []
    bts.append(backtest)
    data = backtest.data

    # create and run random backtests
    for i in range(nsim):
        random_strategy.name = 'random_%s' % i
        rbt = bt.Backtest(random_strategy, data)
        rbt.run()

        bts.append(rbt)

    # now create new RandomBenchmarkResult
    res = RandomBenchmarkResult(*bts)

    return res


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
        self.has_run = False

    def run(self):
        # set run flag
        self.has_run = True

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
            vals = {}
            for m in self.strategy.members:
                if isinstance(m, bt.core.SecurityBase):
                    if m.name in vals:
                        vals[m.name] += m.values
                    else:
                        vals[m.name] = m.values
            vals = pd.DataFrame(vals)

            # divide by root strategy values
            vals = vals.div(self.strategy.values, axis=0)

            # save for future use
            self._sweights = vals

            return vals


class Result(ffn.GroupStats):

    def __init__(self, *backtests):
        tmp = [pd.DataFrame({x.name: x.strategy.prices}) for x in backtests]
        super(Result, self).__init__(*tmp)
        self.backtest_list = backtests
        self.backtests = {x.name: x for x in backtests}

    def display_monthly_returns(self, backtest=0):
        key = self._get_backtest(backtest)
        self[key].display_monthly_returns()

    def plot_weights(self, backtest=0, filter=None, **kwds):
        key = self._get_backtest(backtest)

        if filter is not None:
            data = self.backtests[key].weights[filter]
        else:
            data = self.backtests[key].weights

        data.plot(**kwds)

    def plot_security_weights(self, backtest=0, filter=None, **kwds):
        key = self._get_backtest(backtest)

        if filter is not None:
            data = self.backtests[key].security_weights[filter]
        else:
            data = self.backtests[key].security_weights

        data.plot(**kwds)

    def plot_histogram(self, backtest=0, **kwds):
        key = self._get_backtest(backtest)
        self[key].plot_histogram(**kwds)

    def _get_backtest(self, backtest):
        # based on input order
        if type(backtest) == int:
            return self.backtest_list[backtest].name

        # default case assume ok
        return backtest


class RandomBenchmarkResult(Result):

    def __init__(self, *backtests):
        super(RandomBenchmarkResult, self).__init__(*backtests)
        self.base_name = backtests[0].name
        # seperate stats to make
        self.r_stats = self.stats.drop(self.base_name, axis=1)
        self.b_stats = self.stats[self.base_name]

    def plot_histogram(self, statistic='monthly_sharpe',
                       figsize=(15, 5), title=None,
                       bins=20, **kwargs):

        if not statistic in self.r_stats.index:
            raise ValueError("Invalid statistic. Valid statistics"
                             "are the statistics in self.stats")

        if title is None:
            title = '%s histogram' % statistic

        plt.figure(figsize=figsize)

        ser = self.r_stats.ix[statistic]

        ax = ser.hist(bins=bins, figsize=figsize, normed=True, **kwargs)
        ax.set_title(title)
        plt.axvline(self.b_stats[statistic], linewidth=4)
        ser.plot(kind='kde')
