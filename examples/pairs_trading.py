from copy import deepcopy
from future.utils import iteritems
from datetime import date
import pandas as pd
import numpy as np
import bt


class PairsSignal( bt.Algo ):
    """
    Identify pairs whose indicator exceeds some threshold and save them on temp.

    Args:
        * threshold (float): The threshold to use for the indicator between pairs
        * indicator_name (str): The name of the indicator data set
    Sets:
        * pairs
    """

    def __init__( self, threshold, indicator_name):
        super(PairsSignal, self).__init__()
        self.threshold = threshold
        self.indicator_name = indicator_name

    def __call__(self, target):
        t = target.now
        indicators = target.get_data(self.indicator_name)
        columns = indicators.columns
        signal = indicators.loc[t,:].values.reshape(-1,1) - indicators.loc[t,:].values
        pairs = pd.DataFrame( signal, columns = columns, index = columns).stack()
        pairs.name='weight'
        pairs = pairs[ pairs > self.threshold ]
        pairs.index.names = ['sell','buy']
        pairs=pairs.sort_values(ascending=False)
        target.temp['pairs'] = pairs
        return True


class SetupPairsTrades( bt.Algo ):
    """
    Dynamically create a new sub-strategy (with common logic) for every pairs trade.

    Args:
        * trade_algos ([Algo]): List of algos that defines the sub-strategy behavior
    """

    def __init__(self, trade_algos ):
        super(SetupPairsTrades,self).__init__()
        self.trade_algos = trade_algos

    def __call__(self, target):
        pairs = target.temp.get('pairs', None)
        if pairs is None or pairs.empty:
            return True

        target.temp['weights'] = {}
        for (sell,buy), signal in iteritems( target.temp['pairs'] ):
            trade_name = '%s_%s' % (buy,sell)
            if trade_name not in target.children:
                trade = bt.Strategy( trade_name, deepcopy(self.trade_algos), children = [buy, sell], parent = target )
                trade.setup_from_parent( buy=buy, sell=sell )
                target.temp['weights'][ trade_name ] = 0

        return True


class SizePairsTrades( bt.Algo ):
    """
    Size the pairs trades by allocating capital to them.

    Args:
        * pct_of_capital (float): The percentage of current capital to allocate to new trades this timestep
    """

    def __init__(self, pct_of_capital ):
        super(SizePairsTrades,self).__init__()
        self.pct_of_capital = pct_of_capital

    def __call__(self, target):
        weights = target.temp.get('weights')
        if weights:
            trade_capital = target.capital * self.pct_of_capital / float(len(weights))
            for trade_name in weights:
                target.allocate( trade_capital, child=trade_name, update=False )
            target.update( target.now )

        return True


class WeighPair( bt.Algo ):
    """
    Determine the relative weighting and leverage of the pairs trade

    Args:
        * weight( float ): The weight to put on the buy trade
    Sets:
        * weights
    """
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, target):
        target.temp['weights'] = { target.get_data('buy') : self.weight,
                                   target.get_data('sell') : -self.weight }
        return True


class PriceCompare( bt.Algo ):
    """
    Control flow algo that only returns True if the price of the target crosses the threshold

    Args
        * threshold (float): The price threshold
        * is_greater (bool): Whether to do return True when price exceeds the threshold
    """

    def __init__(self, threshold, is_greater):
        self.threshold = threshold
        self.is_greater = is_greater

    def __call__( self, target ):
        if self.is_greater:
            return target.price >= self.threshold
        else:
            return target.price < self.threshold


class ClosePositions( bt.Algo ):
    """
    Closes all positions on a strategy, pulls the capital into the parent
    """
    def __call__( self, target ):
        if target.children and not target.bankrupt:
            target.flatten()
            target.update( target.now ) # Shouldn't be necessary. Need to fix in bt.

            if target.parent != target:
                capital = target.capital
                target.adjust(-capital, update=False, flow=True)
                target.parent.adjust(capital, update=True, flow=False)

        return False


class DebugPortfolioLevel( bt.Algo ):
    """
    Print portfolio level information relevant to this strategy
    """
    def __call__( self, target ):
        flows = target.flows.loc[ target.now ]
        if flows:
            fmt_str = '{now} {name}: Price = {price:>6.2f}, Value = {value:>10,.0f}, Flows = {flows:>8,.0f}'
        else:
            fmt_str = '{now} {name}: Price = {price:>6.2f}, Value = {value:>10,.0f}'
        print( fmt_str.format(
            now = target.now,
            name = target.name,
            price = target.price,
            value = target.value,
            flows = flows
            ) )


class DebugTradeLevel( bt.Algo ):
    """
    Print trade level information
    """
    def __call__( self, target ):
        flows = target.flows.loc[ target.now ]
        # Check that sub-strategy is active (and not paper trading, which is always active)
        if (target.capital > 0 or flows != 0) and target.parent != target:
            if flows:
                fmt_str = '{name:>33}: Price = {price:>6.2f}, Value = {value:>10,.0f}, Flows = {flows:>8,.0f}'
            else:
                fmt_str = '{name:>33}: Price = {price:>6.2f}, Value = {value:>10,.0f}'
            print( fmt_str.format(
               now = target.now,
               name = target.name,
               price = target.price,
               value = target.value,
               flows = flows
               ) )
        return True


def make_data( n_assets=100, n_periods=100, start_date=date(2021,1,1), phi=0.5, corr=1.0, seed=1234 ):
    ''' Randomly generate a data set consisting of non-stationary prices,
        but where the difference between the prices of any two securities is. '''
    np.random.seed(seed)
    dts = pd.date_range( start_date, periods=n_periods)
    T = dts.values.astype('datetime64[D]').astype(float).reshape(-1,1)
    N = n_assets
    columns = ['s%i' %i for i in range(N)]
    cov = corr * np.ones( (N,N) ) + (1-corr) * np.eye(N)
    noise = pd.DataFrame( np.random.multivariate_normal( np.zeros(N), cov, len(dts)), index = dts, columns = columns )
    # Generate an AR(1) process with parameter phi
    eps = pd.DataFrame( np.random.multivariate_normal( np.zeros(N), np.eye(N), len(dts)), index = dts, columns=columns)
    alpha = 1 - phi
    eps.values[1:] = eps.values[1:] / alpha # To cancel out the weighting that ewm puts on the noise term after x0
    ar1 = eps.ewm(alpha=alpha, adjust=False).mean()
    ar1 *= np.sqrt(1.-phi**2) # Re-scale to unit variance, since the standard AR(1) process has variance sigma_eps/(1-phi^2)
    data = 100. + noise.cumsum()*np.sqrt(0.5) + ar1*np.sqrt(0.5)
    # With the current setup, the difference between any two series should follow a mean reverting process with std=1
    return data


def run():
    """ Run the code that illustrates the pairs trading strategy """
    data = make_data()

    # Define the "entry" strategy of the trade. In this case, we give each asset unit weight and trade it
    trade_entry = bt.AlgoStack( bt.algos.RunOnce(), WeighPair(1.), bt.algos.Rebalance() )

    # Define the "exit" strategy of the trade. Here we exit when we cross either an upper/lower
    # threshold on the price of the strategy, or hold it for a fixed length of time.
    trade_exit = bt.AlgoStack(
        bt.algos.Or( [PriceCompare( 96., is_greater=False ),
                    PriceCompare( 104., is_greater=True),
                    bt.algos.RunAfterDays( 5 ) ] ),
        ClosePositions()
        )
    # Combine the entry, exit and debug algos for each trade
    trade_algos = [ bt.algos.Or( [ trade_entry, trade_exit, DebugTradeLevel() ] )]

    # Define the strategy for the master portfolio.
    strategy_algos = [
        PairsSignal( threshold = 4., indicator_name = 'my_indicator' ),
        SetupPairsTrades( trade_algos ),
        SizePairsTrades( pct_of_capital = 0.2 ),
        DebugPortfolioLevel()
    ]

    # Build and run the strategy
    strategy = bt.Strategy( 'PairsStrategy', strategy_algos )
    test = bt.Backtest( strategy, data, additional_data={'my_indicator':data} )
    out = bt.run( test )
    print(out.stats)
    return out

if __name__ == "__main__":
    run()