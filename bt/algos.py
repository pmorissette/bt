import bt
from bt.core import Algo, AlgoStack
import pandas as pd
import numpy as np
import random


def run_always(f):
    """
    Run always decorator to be used with Algo
    to ensure stack runs the decorated Algo
    no matter what.
    """
    f.run_always = True
    return f


class PrintDate(Algo):

    def __call__(self, target):
        print target.now
        return True


class PrintAlgoData(Algo):

    def __call__(self, target):
        print target.temp
        return True


class RunOnce(Algo):

    """
    Returns True on first run then returns False

    As the name says, the algo only runs once. Useful in situations
    where we want to run the logic once (buy and hold for example).
    """

    def __init__(self):
        super(RunOnce, self).__init__()
        self.has_run = False

    def __call__(self, target):
        # if it hasn't run then we will
        # run it and set flag
        if not self.has_run:
            self.has_run = True
            return True

        # return false to stop future execution
        return False


class RunWeekly(Algo):

    """
    Returns True on week change.

    Returns True if the target.now's week has changed
    since the last run, if not returns False. Useful for
    weekly rebalancing strategies.
    """

    def __init__(self):
        super(RunWeekly, self).__init__()
        self.last_date = None

    def __call__(self, target):
        # get last date
        now = target.now

        # if none nothing to do - return false
        if now is None:
            return False

        # create pandas.Timestamp for useful .week property
        now = pd.Timestamp(now)

        if self.last_date is None:
            self.last_date = now
            return False

        result = False
        if now.week != self.last_date.week:
            result = True

        self.last_date = now
        return result


class RunMonthly(Algo):

    """
    Returns True on month change.

    Returns True if the target.now's month has changed
    since the last run, if not returns False. Useful for
    monthly rebalancing strategies.
    """

    def __init__(self):
        super(RunMonthly, self).__init__()
        self.last_date = None

    def __call__(self, target):
        # get last date
        now = target.now

        # if none nothing to do - return false
        if now is None:
            return False

        if self.last_date is None:
            self.last_date = now
            return False

        result = False
        if now.month != self.last_date.month:
            result = True

        self.last_date = now
        return result


class RunYearly(Algo):

    """
    Returns True on year change.

    Returns True if the target.now's year has changed
    since the last run, if not returns False. Useful for
    yearly rebalancing strategies.
    """

    def __init__(self):
        super(RunYearly, self).__init__()
        self.last_date = None

    def __call__(self, target):
        # get last date
        now = target.now

        # if none nothing to do - return false
        if now is None:
            return False

        if self.last_date is None:
            self.last_date = now
            return False

        result = False
        if now.year != self.last_date.year:
            result = True

        self.last_date = now
        return result


class RunOnDate(Algo):

    """
    Returns True on a specific set of dates.
    """

    def __init__(self, *dates):
        """
        Args:
            * dates (*args): A list of dates. Dates will be parsed
                by pandas.to_datetime so pass anything that it can
                parse. Typically, you will pass a string 'yyyy-mm-dd'.
        """
        super(RunOnDate, self).__init__()
        # parse dates and save
        self.dates = [pd.to_datetime(d) for d in dates]

    def __call__(self, target):
        return target.now in self.dates


class SelectAll(Algo):

    def __init__(self, include_no_data=False):
        super(SelectAll, self).__init__()
        self.include_no_data = include_no_data

    def __call__(self, target):
        if self.include_no_data:
            target.temp['selected'] = target.universe.columns
        else:
            target.temp['selected'] = list(
                target.universe.ix[target.now].dropna().index)
        return True


class SelectThese(Algo):

    def __init__(self, tickers):
        super(SelectThese, self).__init__()
        self.tickers = tickers

    def __call__(self, target):
        target.temp['selected'] = self.tickers
        return True


class SelectHasData(Algo):

    def __init__(self, lookback=pd.DateOffset(months=3),
                 min_count=None):
        super(SelectHasData, self).__init__()
        self.lookback = lookback
        if min_count is None:
            min_count = bt.ffn.get_num_days_required(lookback)
        self.min_count = min_count

    def __call__(self, target):
        if 'selected' in target.temp:
            selected = target.temp['selected']
        else:
            selected = target.universe.columns

        filt = target.universe[selected].ix[target.now - self.lookback:]
        cnt = filt.count()
        cnt = cnt[cnt >= self.min_count]
        target.temp['selected'] = list(cnt.index)
        return True


class SelectN(Algo):

    def __init__(self, n, sort_descending=True,
                 all_or_none=False):
        super(SelectN, self).__init__()
        if n < 0:
            raise ValueError('n cannot be negative')
        self.n = n
        self.ascending = not sort_descending
        self.all_or_none = all_or_none

    def __call__(self, target):
        stat = target.temp['stat']
        stat.sort(ascending=self.ascending)

        # handle percent n
        keep_n = self.n
        if self.n < 1:
            keep_n = int(self.n * len(stat))

        sel = list(stat[:keep_n].index)

        if self.all_or_none and len(sel) < keep_n:
            sel = []

        target.temp['selected'] = sel

        return True


class SelectMomentum(AlgoStack):

    def __init__(self, n, lookback=pd.DateOffset(months=3)):
        super(SelectMomentum, self).__init__(
            StatTotalReturn(lookback=lookback),
            SelectN(n=n))


class SelectRandomly(AlgoStack):

    def __init__(self, n=None):
        super(SelectRandomly, self).__init__()
        self.n = n

    def __call__(self, target):
        sel = target.temp['selected']

        if self.n is not None:
            sel = random.sample(sel, self.n)

        target.temp['selected'] = sel
        return True


class StatTotalReturn(Algo):

    def __init__(self, lookback=pd.DateOffset(months=3)):
        super(StatTotalReturn, self).__init__()
        self.lookback = lookback

    def __call__(self, target):
        selected = target.temp['selected']
        prc = target.universe[selected].ix[target.now - self.lookback:]
        target.temp['stat'] = prc.calc_total_return()
        return True


class WeighEqually(Algo):

    def __init__(self):
        super(WeighEqually, self).__init__()

    def __call__(self, target):
        selected = target.temp['selected']
        n = len(selected)

        if n == 0:
            target.temp['weights'] = {}
        else:
            w = 1.0 / n
            target.temp['weights'] = {x: w for x in selected}

        return True


class WeighSpecified(Algo):

    def __init__(self, **weights):
        super(WeighSpecified, self).__init__()
        self.weights = weights

    def __call__(self, target):
        # added copy to make sure these are not overwritten
        target.temp['weights'] = self.weights.copy()
        return True


class WeighInvVol(Algo):

    def __init__(self, lookback=pd.DateOffset(months=3)):
        super(WeighInvVol, self).__init__()
        self.lookback = lookback

    def __call__(self, target):
        selected = target.temp['selected']

        if len(selected) == 0:
            target.temp['weights'] = {}
            return True

        if len(selected) == 1:
            target.temp['weights'] = {selected[0]: 1.}
            return True

        prc = target.universe[selected].ix[target.now - self.lookback:]
        target.temp['weights'] = bt.ffn.calc_inv_vol_weights(
            prc.to_returns().dropna())
        return True


class WeighMeanVar(Algo):

    def __init__(self, lookback=pd.DateOffset(months=3),
                 bounds=(0., 1.), covar_method='ledoit-wolf',
                 rf=0.):
        super(WeighMeanVar, self).__init__()
        self.lookback = lookback
        self.bounds = bounds
        self.covar_method = covar_method
        self.rf = rf

    def __call__(self, target):
        selected = target.temp['selected']

        if len(selected) == 0:
            target.temp['weights'] = {}
            return True

        if len(selected) == 1:
            target.temp['weights'] = {selected[0]: 1.}
            return True

        prc = target.universe[selected].ix[target.now - self.lookback:]
        target.temp['weights'] = bt.ffn.calc_mean_var_weights(
            prc.to_returns().dropna(), weight_bounds=self.bounds,
            covar_method=self.covar_method, rf=self.rf)

        return True


class WeighRandomly(Algo):

    def __init__(self, bounds=(0., 1.), weight_sum=1):
        super(WeighRandomly, self).__init__()
        self.bounds = bounds

        low = bounds[0]
        high = bounds[1]

        self.weight_sum = weight_sum

    def __call__(self, target):
        sel = target.temp['selected']
        n = len(sel)

        w = {}
        try:
            rw = bt.ffn.random_weights(
                n, self.bounds, self.weight_sum)
            w = dict(zip(sel, rw))
        except ValueError:
            pass

        target.temp['weights'] = w
        return True


class LimitDeltas(Algo):

    def __init__(self, limit=0.1):
        super(LimitDeltas, self).__init__()
        self.limit = limit
        # determine if global or specific
        self.global_limit = True
        if isinstance(limit, dict):
            self.global_limit = False

    def __call__(self, target):
        tw = target.temp['weights']
        all_keys = set(target.children.keys() + tw.keys())

        for k in all_keys:
            tgt = tw[k] if k in tw else 0.
            cur = target.children[k].weight if k in target.children else 0.
            delta = tgt - cur

            # check if we need to limit
            if self.global_limit:
                if abs(delta) > self.limit:
                    tw[k] = cur + (self.limit * np.sign(delta))
            else:
                # make sure we have a limit defined in case of limit dict
                if k in self.limit:
                    lmt = self.limit[k]
                    if abs(delta) > lmt:
                        tw[k] = cur + (lmt * np.sign(delta))

        return True


class LimitWeights(Algo):

    def __init__(self, limit=0.1):
        super(LimitWeights, self).__init__()
        self.limit = limit

    def __call__(self, target):
        if 'weights' not in target.temp:
            return True

        tw = target.temp['weights']
        tw = bt.ffn.limit_weights(tw, self.limit)
        target.temp['weights'] = tw

        return True


class CapitalFlow(Algo):

    """
    Used to model capital flows. Flows can either be inflows or outflows.

    This Algo can be used to model capital flows. For example, a pension
    fund might have inflows every month or year due to contributions. This
    Algo will affect the capital of the target node without affecting returns
    for the node.
    """

    def __init__(self, amount):
        """
        CapitalFlow constructor.

        Args:
            * amount (float): Amount to adjust by
        """
        super(CapitalFlow, self).__init__()
        self.amount = float(amount)

    def __call__(self, target):
        target.adjust(self.amount)
        return True


class Rebalance(Algo):

    """
    Rebalances capital based on temp weights.

    Rebalances capital based on temp['weights']. Also closes
    positions if open but not in target_weights. This is typically
    the last Algo called once the target weights have been set.
    """

    def __init__(self):
        super(Rebalance, self).__init__()

    def __call__(self, target):
        if 'weights' not in target.temp:
            return True

        targets = target.temp['weights']

        # de-allocate children that are not in targets
        not_in = [x for x in target.children if x not in targets]
        for c in not_in:
            target.close(c)

        # save value because it will change after each call to allocate
        # use it as base in rebalance calls
        base = target.value
        for item in targets.iteritems():
            target.rebalance(item[1], child=item[0], base=base)

        return True


class RebalanceOverTime(Algo):

    def __init__(self, days=10):
        super(RebalanceOverTime, self).__init__()
        self.days = float(days)
        self._rb = Rebalance()
        self._weights = None
        self._days_left = None

    def __call__(self, target):
        # new weights specified - update rebalance data
        if 'weights' in target.temp:
            self._weights = target.temp['weights']
            self._days_left = self.days

        # if _weights are not None, we have some work to do
        if self._weights:
            tgt = {}
            # scale delta relative to # of days left and set that as the new
            # target
            for t in self._weights:
                curr = target.children[t].weight if t in \
                    target.children else 0.
                dlt = (self._weights[t] - curr) / self._days_left
                tgt[t] = curr + dlt

            # mock weights and call real Rebalance
            target.temp['weights'] = tgt
            self._rb(target)

            # dec _days_left. If 0, set to None & set _weights to None
            self._days_left -= 1

            if self._days_left == 0:
                self._days_left = None
                self._weights = None

        return True


class Require(Algo):
    """
    Flow control Algo.

    This algo returns the value of a predicate
    on an temp entry. Useful for controlling
    flow.

    i.e. Stop execution is len(selected) == 0
    """

    def __init__(self, pred, item, if_none=False):
        super(Require, self).__init__()
        self.item = item
        self.pred = pred
        self.if_none = if_none

    def __call__(self, target):
        if self.item not in target.temp:
            return self.if_none

        return self.pred(target.temp[self.item])
