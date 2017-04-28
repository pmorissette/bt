The Tree Structure
==================

In addition to the concept of :class:`Algos <bt.core.Algo>` and :class:`AlgoStacks <bt.core.AlgoStack>`, a tree structure lies 
at the heart of the framework.  It allows you to mix and match securities and strategies in order to express 
your sophisticated trading ideas.  Here is a very simple diagram to help explain this concept:

.. image:: _static/tree1.png
    :align: center
    :alt: simple tree structure

This diagram represents the strategy we tested in the :doc:`overview example <index>`. A simple :class:`strategy <bt.core.Strategy>` 
with two children that happen to be :class:`securities <bt.core.SecurityBase>`. However, children nodes don't have to be
securities. They can also be strategies. This concept is very powerful as it
allows you to combine strategies together and allocate capital dynamically
between different strategies as time progresses using sophisticated allocation
logic. This is similar to what hedge funds do - they have a portfolio of strategies and dynamically allocate capital
according to a set of rules. 

For example, say we didn't mind having a passive bond allocation (AGG in the
above graph), but we wanted to swap out the equity portion (SPY) for something a
little more sophisticated. In this case, we will swap out the SPY node for another strategy.
This strategy could be a momentum strategy that attempts to pick the best
performing ETF every month (to keep it simple, let's say it picks either the SPY
or the EEM based on total return over the past 3 months).

Here is the updated graph:

.. image:: _static/tree2.png
    :align: center
    :alt: advanced tree structure

This approach allows you to build complex systems even though all of the building
blocks may be relatively simple. Hopefully you can see how powerful this can be
when designing and testing quantitative strategies.

Oh and here's the code for the second example - not much more complex:

.. code:: python

    import bt

    # create the momentum strategy - we will specify the children (3rd argument)
    # to limit the universe the strategy can choose from
    mom_s = bt.Strategy('mom_s', [bt.algos.RunMonthly(),
                                  bt.algos.SelectAll(),
                                  bt.algos.SelectMomentum(1),
                                  bt.algos.WeighEqually(),
                                  bt.algos.Rebalance()],
                        ['spy', 'eem'])

    # create the master strategy - this is the top-most node in the tree
    # Once again, we are also specifying  the children. In this case, one of the
    # children is a Security and the other is a Strategy.
    master = bt.Strategy('master', [bt.algos.RunMonthly(),
                                    bt.algos.SelectAll(),
                                    bt.algos.WeighEqually(),
                                    bt.algos.Rebalance()],
                        [mom_s, 'agg'])

    # create the backtest and run it
    t = bt.Backtest(master, data)
    r = bt.run(t)

So there you have it. Please read the rest of the docs to have a better idea of
all the features packed into bt.

