"""
Contains the core building blocks of the framework.
"""
from __future__ import division

import math
from copy import deepcopy

import cython as cy
import numpy as np
import pandas as pd
from future.utils import iteritems

PAR = 100.0
TOL = 1e-16


@cy.locals(x=cy.double)
def is_zero(x):
    """
    Test for zero that is robust against floating point precision errors
    """
    return abs(x) < TOL


class Node(object):

    """
    The Node is the main building block in bt's tree structure design.
    Both StrategyBase and SecurityBase inherit Node. It contains the
    core functionality of a tree node.

    Args:
        * name (str): The Node name
        * parent (Node): The parent Node
        * children (dict, list): A collection of children. If dict,
            the format is {name: child}, if list then list of children.
            Children can be any type of Node or str.
            String values correspond to children which will be lazily created
            with that name when needed.

    Attributes:
        * name (str): Node name
        * parent (Node): Node parent
        * root (Node): Root node of the tree (topmost node)
        * children (dict): Node's children
        * now (datetime): Used when backtesting to store current date
        * stale (bool): Flag used to determine if Node is stale and need
            updating
        * prices (TimeSeries): Prices of the Node. Prices for a security will
            be the security's price, for a strategy it will be an index that
            reflects the value of the strategy over time.
        * price (float): last price
        * value (float): last value
        * notional_value (float): last notional value. Notional value is used
            when fixed_income=True. It is always positive for strategies, but
            is signed for securities (and typically set to either market value,
            position, or zero).
        * weight (float): weight in parent
        * full_name (str): Name including parents' names
        * members (list): Current Node + node's children
        * fixed_income (bool): Whether the node corresponds to a fixed income
            component, which would use notional-weighting instead of market
            value weighing. See also FixedIncomeStrategy for more details.
    """

    _capital = cy.declare(cy.double)
    _price = cy.declare(cy.double)
    _value = cy.declare(cy.double)
    _notl_value = cy.declare(cy.double)
    _weight = cy.declare(cy.double)
    _issec = cy.declare(cy.bint)
    _has_strat_children = cy.declare(cy.bint)
    _fixed_income = cy.declare(cy.bint)
    _bidoffer_set = cy.declare(cy.bint)
    _bidoffer_paid = cy.declare(cy.double)

    def __init__(self, name, parent=None, children=None):

        self.name = name

        # children helpers
        self.children = {}
        self._lazy_children = {}
        self._universe_tickers = []
        self._childrenv = []  # Shortcut to self.children.values()

        # strategy children helpers
        self._has_strat_children = False
        self._strat_children = []

        if parent is None:
            self.parent = self
            self.root = self
            # by default all positions are integer
            self.integer_positions = True
        else:
            self.parent = parent
            parent._add_children([self], dc=False)

        self._add_children(children, dc=True)

        # set default value for now
        self.now = 0
        # make sure root has stale flag
        # used to avoid unnecessary update
        # sometimes we change values in the tree and we know that we will need
        # to update if another node tries to access a given value (say weight).
        # This avoid calling the update until it is actually needed.
        self.root.stale = False

        # helper vars
        self._price = 0
        self._value = 0
        self._notl_value = 0
        self._weight = 0
        self._capital = 0

        # is security flag - used to avoid updating 0 pos securities
        self._issec = False

        # fixed income flag - used to turn on notional weighing
        self._fixed_income = False
        # flag for whether to do bid/offer accounting
        self._bidoffer_set = False
        self._bidoffer_paid = 0

    def __getitem__(self, key):
        return self.children[key]

    def _add_children(self, children, dc):
        """
        Add the collection of children to the current node, where
        children is either an iterable of children objects/strings, or
        a dictionary

        Args:
            dc (bool): Whether or not to deepcopy nodes before adding them.
        """
        if children is not None:
            if isinstance(children, dict):
                # Preserve the names from the dictionary by renaming the nodes
                tmp = []
                for name, c in iteritems(children):
                    if isinstance(c, str):
                        tmp.append(name)
                    else:
                        if dc:
                            c = deepcopy(c)
                        c.name = name
                        tmp.append(c)
                children = tmp

            for c in children:

                if dc:  # deepcopy object for possible later reuse
                    c = deepcopy(c)

                if type(c) == str:
                    if c in self._universe_tickers:
                        raise ValueError("Child %s already exists" % c)

                    # Create default security with lazy_add
                    c = Security(c, lazy_add=True)

                if getattr(c, "lazy_add", False):
                    self._lazy_children[c.name] = c
                else:
                    if c.name in self.children:
                        raise ValueError("Child %s already exists" % c)

                    c.parent = self
                    c._set_root(self.root)
                    c.use_integer_positions(self.integer_positions)

                    self.children[c.name] = c
                    self._childrenv.append(c)

                # if strategy, turn on flag and add name to list
                # strategy children have special treatment
                if isinstance(c, StrategyBase):
                    self._has_strat_children = True
                    self._strat_children.append(c.name)
                # if not strategy, then we will want to add this to
                # universe_tickers to filter on setup
                elif c.name not in self._universe_tickers:
                    self._universe_tickers.append(c.name)

    def _set_root(self, root):
        self.root = root
        for c in self._childrenv:
            c._set_root(root)

    def use_integer_positions(self, integer_positions):
        """
        Set indicator to use (or not) integer positions for a given strategy or
        security.

        By default all positions in number of stocks should be integer.
        However this may lead to unexpected results when working with adjusted
        prices of stocks. Because of series of reverse splits of stocks, the
        adjusted prices back in time might be high. Thus rounding of desired
        amount of stocks to buy may lead to having 0, and thus ignoring this
        stock from backtesting.
        """
        self.integer_positions = integer_positions
        for c in self._childrenv:
            c.use_integer_positions(integer_positions)

    @property
    def fixed_income(self):
        """
        Whether the node is a fixed income node (using notional weighting).
        """
        return self._fixed_income

    @property
    def prices(self):
        """
        A TimeSeries of the Node's price.
        """
        # can optimize depending on type -
        # securities don't need to check stale to
        # return latest prices, whereas strategies do...
        raise NotImplementedError()

    @property
    def price(self):
        """
        Current price of the Node
        """
        # can optimize depending on type -
        # securities don't need to check stale to
        # return latest prices, whereas strategies do...
        raise NotImplementedError()

    @property
    def value(self):
        """
        Current value of the Node
        """
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._value

    @property
    def notional_value(self):
        """
        Current notional value of the Node
        """
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._notl_value

    @property
    def weight(self):
        """
        Current weight of the Node (with respect to the parent).
        """
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._weight

    def setup(self, universe, **kwargs):
        """
        Setup method used to initialize a Node with a universe, and potentially other information.
        """
        raise NotImplementedError()

    def update(self, date, data=None, inow=None):
        """
        Update Node with latest date, and optionally some data.
        """
        raise NotImplementedError()

    def adjust(self, amount, update=True, flow=True):
        """
        Adjust Node value by amount.
        """
        raise NotImplementedError()

    def allocate(self, amount, update=True):
        """
        Allocate capital to Node.
        """
        raise NotImplementedError()

    @property
    def members(self):
        """
        Node members. Members include current node as well as Node's
        children.
        """
        res = [self]
        for c in list(self.children.values()):
            res.extend(c.members)
        return res

    @property
    def full_name(self):
        if self.parent == self:
            return self.name
        else:
            return "%s>%s" % (self.parent.full_name, self.name)

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, self.full_name)

    def to_dot(self, root=True):
        """
        Represent the node structure in DOT format.
        """
        name = lambda x: x.name or repr(self)  # noqa: E731
        edges = "\n".join(
            '\t"%s" -> "%s"' % (name(self), name(c)) for c in self.children.values()
        )
        below = "\n".join(c.to_dot(False) for c in self.children.values())
        body = "\n".join([edges, below]).rstrip()
        if root:
            return "\n".join(["digraph {", body, "}"])
        return body


class StrategyBase(Node):

    """
    Strategy Node. Used to define strategy logic within a tree.
    A Strategy's role is to allocate capital to it's children
    based on a function.

    Args:
        * name (str): Strategy name
        * children (dict, list): A collection of children. If dict,
            the format is {name: child}, if list then list of children.
            Children can be any type of Node or str.
            String values correspond to children which will be lazily created
            with that name when needed.
        * parent (Node): The parent Node

    Attributes:
        * name (str): Strategy name
        * parent (Strategy): Strategy parent
        * root (Strategy): Root node of the tree (topmost node)
        * children (dict): Strategy's children
        * now (datetime): Used when backtesting to store current date
        * stale (bool): Flag used to determine if Strategy is stale and need
            updating
        * prices (TimeSeries): Prices of the Strategy - basically an index that
            reflects the value of the strategy over time.
        * outlays (DataFrame): Outlays for each SecurityBase child
        * price (float): last price
        * value (float): last value
        * notional_value (float): last notional value
        * weight (float): weight in parent
        * full_name (str): Name including parents' names
        * members (list): Current Strategy + strategy's children
        * securities (list): List of strategy children that are of type
            SecurityBase
        * commission_fn (fn(quantity, price)): A function used to determine the
            commission (transaction fee) amount. Could be used to model
            slippage (implementation shortfall). Note that often fees are
            symmetric for buy and sell and absolute value of quantity should
            be used for calculation.
        * capital (float): Capital amount in Strategy - cash
        * universe (DataFrame): Data universe available at the current time.
            Universe contains the data passed in when creating a Backtest. Use
            this data to determine strategy logic.

    """

    _net_flows = cy.declare(cy.double)
    _last_value = cy.declare(cy.double)
    _last_notl_value = cy.declare(cy.double)
    _last_price = cy.declare(cy.double)
    _last_fee = cy.declare(cy.double)
    _paper_trade = cy.declare(cy.bint)
    bankrupt = cy.declare(cy.bint)

    def __init__(self, name, children=None, parent=None):
        Node.__init__(self, name, children=children, parent=parent)
        self._weight = 1
        self._value = 0
        self._notl_value = 0
        self._price = PAR

        # helper vars
        self._net_flows = 0
        self._last_value = 0
        self._last_notl_value = 0
        self._last_price = PAR
        self._last_fee = 0

        # default commission function
        self.commission_fn = self._dflt_comm_fn

        self._paper_trade = False
        self._positions = None
        self.bankrupt = False

    @property
    def price(self):
        """
        Current price.
        """
        if self.root.stale:
            self.root.update(self.now, None)
        return self._price

    @property
    def prices(self):
        """
        TimeSeries of prices.
        """
        if self.root.stale:
            self.root.update(self.now, None)
        return self._prices.loc[: self.now]

    @property
    def values(self):
        """
        TimeSeries of values.
        """
        if self.root.stale:
            self.root.update(self.now, None)
        return self._values.loc[: self.now]

    @property
    def notional_values(self):
        """
        TimeSeries of notional values.
        """
        if self.root.stale:
            self.root.update(self.now, None)
        return self._notl_values.loc[: self.now]

    @property
    def capital(self):
        """
        Current capital - amount of unallocated capital left in strategy.
        """
        # no stale check needed
        return self._capital

    @property
    def cash(self):
        """
        TimeSeries of unallocated capital.
        """
        # no stale check needed
        return self._cash

    @property
    def fees(self):
        """
        TimeSeries of fees.
        """
        if self.root.stale:
            self.root.update(self.now, None)
        return self._fees.loc[: self.now]

    @property
    def flows(self):
        """
        TimeSeries of flows.
        """
        if self.root.stale:
            self.root.update(self.now, None)
        return self._all_flows.loc[: self.now]

    @property
    def bidoffer_paid(self):
        """
        Bid/offer spread paid on transactions in the current step
        """
        if self._bidoffer_set:
            if self.root.stale:
                self.root.update(self.now, None)
            return self._bidoffer_paid
        else:
            raise Exception(
                "Bid/offer accounting not turned on: "
                '"bidoffer" argument not provided during setup'
            )

    @property
    def bidoffers_paid(self):
        """
        TimeSeries of bid/offer spread paid on transactions in each step
        """
        if self._bidoffer_set:
            if self.root.stale:
                self.root.update(self.now, None)
            return self._bidoffers_paid.loc[: self.now]
        else:
            raise Exception(
                "Bid/offer accounting not turned on: "
                '"bidoffer" argument not provided during setup'
            )

    @property
    def universe(self):
        """
        Data universe available at the current time.
        Universe contains the data passed in when creating a Backtest.
        Use this data to determine strategy logic.
        """
        # avoid windowing every time
        # if calling and on same date return
        # cached value
        if self.now == self._last_chk:
            return self._funiverse
        else:
            self._last_chk = self.now
            self._funiverse = self._universe.loc[: self.now]
            return self._funiverse

    @property
    def securities(self):
        """
        Returns a list of children that are of type SecurityBase
        """
        return [x for x in self.members if isinstance(x, SecurityBase)]

    @property
    def outlays(self):
        """
        Returns a DataFrame of outlays for each child SecurityBase
        """
        if self.root.stale:
            self.root.update(self.root.now, None)
        return pd.DataFrame({x.name: x.outlays for x in self.securities})

    @property
    def positions(self):
        """
        TimeSeries of positions.
        """
        # if accessing and stale - update first
        if self.root.stale:
            self.root.update(self.root.now, None)

        vals = pd.DataFrame(
            {x.name: x.positions for x in self.members if isinstance(x, SecurityBase)}
        )
        self._positions = vals
        return vals

    def setup(self, universe, **kwargs):
        """
        Setup strategy with universe. This will speed up future calculations
        and updates.
        """
        # save full universe in case we need it
        self._original_data = universe
        self._setup_kwargs = kwargs

        # Guard against fixed income children of regular
        # strategies as the "price" is just a reference
        # value and should not be used for capital allocation
        if self.fixed_income and not self.parent.fixed_income:
            raise ValueError(
                "Cannot have fixed income "
                "strategy child (%s) of non-"
                "fixed income strategy (%s)" % (self.name, self.parent.name)
            )

        # determine if needs paper trading
        # and setup if so
        if self is not self.parent:
            self._paper_trade = True
            self._paper_amount = 1000000

            paper = deepcopy(self)
            paper.parent = paper
            paper.root = paper
            paper._paper_trade = False
            paper.setup(self._original_data, **kwargs)
            paper.adjust(self._paper_amount)
            self._paper = paper

        # setup universe
        funiverse = universe

        if self._universe_tickers:
            # if we have universe_tickers defined, limit universe to
            # those tickers
            valid_filter = list(
                set(universe.columns).intersection(self._universe_tickers)
            )

            funiverse = universe[valid_filter].copy()

            # if we have strat children, we will need to create their columns
            # in the new universe
            if self._has_strat_children:
                for c in self._strat_children:
                    funiverse[c] = np.nan

            # must create to avoid pandas warning
            funiverse = pd.DataFrame(funiverse)

        self._universe = funiverse
        # holds filtered universe
        self._funiverse = funiverse
        self._last_chk = None

        # We're not bankrupt yet
        self.bankrupt = False

        # setup internal data
        self.data = pd.DataFrame(
            index=funiverse.index,
            columns=["price", "value", "notional_value", "cash", "fees", "flows"],
            data=0.0,
        )

        self._prices = self.data["price"]
        self._values = self.data["value"]
        self._notl_values = self.data["notional_value"]
        self._cash = self.data["cash"]
        self._fees = self.data["fees"]
        self._all_flows = self.data["flows"]

        if "bidoffer" in kwargs:
            self._bidoffer_set = True
            self.data["bidoffer_paid"] = 0.0
            self._bidoffers_paid = self.data["bidoffer_paid"]

        # setup children as well - use original universe here - don't want to
        # pollute with potential strategy children in funiverse
        if self.children is not None:
            [c.setup(universe, **kwargs) for c in self._childrenv]

    def setup_from_parent(self, **kwargs):
        """
        Setup a strategy from the parent. Used when dynamically creating
        child strategies.

        Args:
            * kwargs: additional arguments that will be passed to setup
                (potentially overriding those from the parent)
        """
        all_kwargs = self.parent._setup_kwargs.copy()
        all_kwargs.update(kwargs)
        self.setup(self.parent._original_data, **all_kwargs)
        if self.name not in self.parent._universe:
            self.parent._universe[self.name] = np.nan

    def get_data(self, key):
        """
        Returns additional data that was passed to the setup function via kwargs,
        for use in the algos. This allows algos to reference data sources "by name",
        where the binding of the data to the name happens at Backtest creation
        time rather than at Strategy definition time, allowing the same strategies
        to be run against different data sets more easily.
        """
        return self._setup_kwargs[key]

    @cy.locals(
        newpt=cy.bint,
        val=cy.double,
        ret=cy.double,
        coupons=cy.double,
        notl_val=cy.double,
        bidoffer_paid=cy.double,
    )
    def update(self, date, data=None, inow=None):
        """
        Update strategy. Updates prices, values, weight, etc.
        """
        # resolve stale state
        self.root.stale = False

        # update helpers on date change
        # also set newpt flag
        newpt = False
        if self.now == 0:
            newpt = True
        elif date != self.now:
            self._net_flows = 0
            self._last_price = self._price
            self._last_value = self._value
            self._last_notl_value = self._notl_value
            self._last_fee = 0.0
            newpt = True

        # update now
        self.now = date
        if inow is None:
            if self.now == 0:
                inow = 0
            else:
                inow = self.data.index.get_loc(date)

        # update children if any and calculate value
        val = self._capital  # default if no children
        notl_val = 0.0  # Capital doesn't count towards notional value

        bidoffer_paid = 0.0
        coupons = 0
        if self.children:
            for c in self._childrenv:
                # Sweep up cash from the security nodes (from coupon payments, etc)
                if c._issec and newpt:
                    coupons += c._capital
                    c._capital = 0

                # avoid useless update call
                if c._issec and not c._needupdate:
                    continue
                c.update(date, data, inow)
                val += c.value
                # Strategies always have positive notional value
                notl_val += abs(c.notional_value)

                if self._bidoffer_set:
                    bidoffer_paid += c.bidoffer_paid

        self._capital += coupons
        val += coupons

        if self.root == self:
            if (
                (val < 0)
                and not self.bankrupt
                and not self.fixed_income
                and not is_zero(val)
            ):
                # Declare a bankruptcy
                self.bankrupt = True
                self.flatten()

        # update data if this value is different or
        # if now has changed - avoid all this if not since it
        # won't change
        if (
            newpt
            or not is_zero(self._value - val)
            or not is_zero(self._notl_value - notl_val)
        ):
            self._value = val
            self._values.values[inow] = val

            self._notl_value = notl_val
            self._notl_values.values[inow] = notl_val

            if self._bidoffer_set:
                self._bidoffer_paid = bidoffer_paid
                self._bidoffers_paid.values[inow] = bidoffer_paid

            if self.fixed_income:
                # For notional weights, we compute additive return
                pnl = self._value - (self._last_value + self._net_flows)
                if not is_zero(self._last_notl_value):
                    ret = pnl / self._last_notl_value * PAR
                elif not is_zero(self._notl_value):
                    # This case happens when paying bid/offer or fees when building an initial position
                    ret = pnl / self._notl_value * PAR
                else:
                    if is_zero(pnl):
                        ret = 0
                    else:
                        raise ZeroDivisionError(
                            "Could not update %s on %s. Last notional value "
                            "was %s and pnl was %s. Therefore, "
                            "we are dividing by zero to obtain the pnl "
                            "per unit notional for the period."
                            % (self.name, self.now, self._last_notl_value, pnl)
                        )

                self._price = self._last_price + ret
                self._prices.values[inow] = self._price

            else:
                bottom = self._last_value + self._net_flows
                if not is_zero(bottom):
                    ret = self._value / (self._last_value + self._net_flows) - 1
                else:
                    if is_zero(self._value):
                        ret = 0
                    else:
                        raise ZeroDivisionError(
                            "Could not update %s on %s. Last value "
                            "was %s and net flows were %s. Current"
                            "value is %s. Therefore, "
                            "we are dividing by zero to obtain the return "
                            "for the period."
                            % (
                                self.name,
                                self.now,
                                self._last_value,
                                self._net_flows,
                                self._value,
                            )
                        )

                self._price = self._last_price * (1 + ret)
                self._prices.values[inow] = self._price

        # update children weights
        if self.children:
            for c in self._childrenv:
                # avoid useless update call
                if c._issec and not c._needupdate:
                    continue

                if self.fixed_income:
                    if not is_zero(notl_val):
                        c._weight = c.notional_value / notl_val
                    else:
                        c._weight = 0.0
                else:
                    if not is_zero(val):
                        c._weight = c.value / val
                    else:
                        c._weight = 0.0

        # if we have strategy children, we will need to update them in universe
        if self._has_strat_children:
            for c in self._strat_children:
                # TODO: optimize ".loc" here as well
                self._universe.loc[date, c] = self.children[c].price

        # Cash should track the unallocated capital at the end of the day, so
        # we should update it every time we call "update".
        # Same for fees and flows
        self._cash.values[inow] = self._capital
        self._fees.values[inow] = self._last_fee
        self._all_flows.values[inow] = self._net_flows

        # update paper trade if necessary
        if self._paper_trade:
            if newpt:
                self._paper.update(date)
                self._paper.run()
                self._paper.update(date)
            # update price
            self._price = self._paper.price
            self._prices.values[inow] = self._price

    @cy.locals(amount=cy.double, update=cy.bint, flow=cy.bint, fees=cy.double)
    def adjust(self, amount, update=True, flow=True, fee=0.0):
        """
        Adjust capital - used to inject capital to a Strategy. This injection
        of capital will have no effect on the children.

        Args:
            * amount (float): Amount to adjust by.
            * update (bool): Force update?
            * flow (bool): Is this adjustment a flow? A flow will not have an
                impact on the performance (price index). Example of flows are
                simply capital injections (say a monthly contribution to a
                portfolio). This should not be reflected in the returns. A
                non-flow (flow=False) does impact performance. A good example
                of this is a commission, or a dividend.

        """
        # adjust capital
        self._capital += amount
        self._last_fee += fee

        # if flow - increment net_flows - this will not affect
        # performance. Commissions and other fees are not flows since
        # they have a performance impact
        if flow:
            self._net_flows += amount

        if update:
            # indicates that data is now stale and must
            # be updated before access
            self.root.stale = True

    @cy.locals(amount=cy.double, update=cy.bint)
    def allocate(self, amount, child=None, update=True):
        """
        Allocate capital to Strategy. By default, capital is allocated
        recursively down the children, proportionally to the children's
        weights.  If a child is specified, capital will be allocated
        to that specific child.

        Allocation also have a side-effect. They will deduct the same amount
        from the parent's "account" to offset the allocation. If there is
        remaining capital after allocation, it will remain in Strategy.

        Args:
            * amount (float): Amount to allocate.
            * child (str): If specified, allocation will be directed to child
                only. Specified by name.
            * update (bool): Force update.

        """
        # allocate to child
        if child is not None:
            self._create_child_if_needed(child)

            # allocate to child
            self.children[child].allocate(amount)
        # allocate to self
        else:
            # adjust parent's capital
            # no need to update now - avoids repetition
            if self.parent == self:
                self.parent.adjust(-amount, update=False, flow=True)
            else:
                # do NOT set as flow - parent will be another strategy
                # and therefore should not incur flow
                self.parent.adjust(-amount, update=False, flow=False)

            # adjust self's capital
            self.adjust(amount, update=False, flow=True)

            # push allocation down to children if any
            # use _weight to avoid triggering an update
            if self.children is not None:
                [c.allocate(amount * c._weight, update=False) for c in self._childrenv]

            # mark as stale if update requested
            if update:
                self.root.stale = True

    @cy.locals(q=cy.double, update=cy.bint)
    def transact(self, q, child=None, update=True):
        """
        Transact a notional amount q in the Strategy. By default, it is allocated
        recursively down the children, proportionally to the children's
        weights. Recursive allocation only works for fixed income strategies.
        If a child is specified, notional will be allocated
        to that specific child.

        Args:
            * q (float): Notional quantity to allocate.
            * child (str): If specified, allocation will be directed to child
                only. Specified by name.
            * update (bool): Force update.

        """
        # allocate to child
        if child is not None:
            self._create_child_if_needed(child)

            # allocate to child
            self.children[child].transact(q)
        # allocate to self
        else:
            # push allocation down to children if any
            # use _weight to avoid triggering an update
            if self.children is not None:
                [c.transact(q * c._weight, update=False) for c in self._childrenv]

            # mark as stale if update requested
            if update:
                self.root.stale = True

    @cy.locals(delta=cy.double, weight=cy.double, base=cy.double, update=cy.bint)
    def rebalance(self, weight, child, base=np.nan, update=True):
        """
        Rebalance a child to a given weight.

        This is a helper method to simplify code logic. This method is used
        when we want to see the weight of a particular child to a set amount.
        It is similar to allocate, but it calculates the appropriate allocation
        based on the current weight. For fixed income strategies, it uses
        transact to rebalance based on notional value instead of capital.

        Args:
            * weight (float): The target weight. Usually between -1.0 and 1.0.
            * child (str): child to allocate to - specified by name.
            * base (float): If specified, this is the base amount all weight
                delta calculations will be based off of. This is useful when we
                determine a set of weights and want to rebalance each child
                given these new weights. However, as we iterate through each
                child and call this method, the base (which is by default the
                current value) will change. Therefore, we can set this base to
                the original value before the iteration to ensure the proper
                allocations are made.
            * update (bool): Force update?

        """
        # if weight is 0 - we want to close child
        if is_zero(weight):
            if child in self.children:
                return self.close(child, update=update)
            else:
                return

        # if no base specified use self's value
        if np.isnan(base):
            if self.fixed_income:
                base = self.notional_value
            else:
                base = self.value

        # else make sure we have child
        self._create_child_if_needed(child)

        # allocate to child
        # figure out weight delta
        c = self.children[child]
        if self.fixed_income:
            # In fixed income strategies, the provided "base" value can be used
            # to upscale/downscale the notional_value of the strategy, whereas
            # in normal strategies the total capital is fixed. Thus, when
            # rebalancing, we must take care to account for differences between
            # previous notional value and passed base value. Note that for
            # updating many weights in sequence, one must pass update=False so
            # that the existing weights and notional_value are not recalculated
            # before finishing.
            if c.fixed_income:
                delta = weight * base - c.weight * self.notional_value
                c.transact(delta, update=update)
            else:
                delta = weight * base - c.weight * self.notional_value
                c.allocate(delta, update=update)
        else:
            delta = weight - c.weight
            c.allocate(delta * base, update=update)

    @cy.locals(update=cy.bint)
    def close(self, child, update=True):
        """
        Close a child position - alias for rebalance(0, child). This will also
        flatten (close out all) the child's children.

        Args:
            * child (str): Child, specified by name.
        """
        c = self.children[child]
        # flatten if children not None
        if c.children is not None and len(c.children) != 0:
            c.flatten()

        if self.fixed_income:
            if c.position != 0.0:
                c.transact(-c.position, update=update)
        else:
            if c.value != 0.0 and not np.isnan(c.value):
                c.allocate(-c.value, update=update)

    def flatten(self):
        """
        Close all child positions.
        """
        # go right to base alloc
        if self.fixed_income:
            [
                c.transact(-c.position, update=False)
                for c in self._childrenv
                if c.position != 0
            ]
        else:
            [
                c.allocate(-c.value, update=False)
                for c in self._childrenv
                if c.value != 0
            ]

        self.root.stale = True

    def run(self):
        """
        This is the main logic method. Override this method to provide some
        algorithm to execute on each date change. This method is called by
        backtester.
        """
        pass

    def set_commissions(self, fn):
        """
        Set commission (transaction fee) function.

        Args:
            fn (fn(quantity, price)): Function used to determine commission
            amount.

        """
        self.commission_fn = fn

        for c in self._childrenv:
            if isinstance(c, StrategyBase):
                c.set_commissions(fn)

    def get_transactions(self):
        """
        Helper function that returns the transactions in the following format:

            Date, Security | quantity, price

        The result is a MultiIndex DataFrame.
        """
        # get prices for each security in the strategy & create unstacked
        # series
        prc = pd.DataFrame({x.name: x.prices for x in self.securities}).unstack()

        # get security positions
        positions = pd.DataFrame({x.name: x.positions for x in self.securities})
        # trades are diff
        trades = positions.diff()
        # must adjust first row
        trades.iloc[0] = positions.iloc[0]
        # now convert to unstacked series, dropping nans along the way
        trades = trades[trades != 0].unstack().dropna()

        # Adjust prices for bid/offer paid if needed
        if self._bidoffer_set:
            bidoffer = pd.DataFrame(
                {x.name: x.bidoffers_paid for x in self.securities}
            ).unstack()
            prc += bidoffer / trades

        res = pd.DataFrame({"price": prc, "quantity": trades}).dropna(
            subset=["quantity"]
        )

        # set names
        res.index.names = ["Security", "Date"]

        # swap levels so that we have (date, security) as index and sort
        res = res.swaplevel().sort_index()

        return res

    @cy.locals(q=cy.double, p=cy.double)
    def _dflt_comm_fn(self, q, p):
        return 0.0

    def _create_child_if_needed(self, child):
        if child not in self.children:
            # Look up name in lazy children, or create a default security
            c = self._lazy_children.pop(child, Security(child))
            c.lazy_add = False

            # add child to tree
            self._add_children([c], dc=False)
            c.setup(self._universe, **self._setup_kwargs)

            # update to bring up to speed
            c.update(self.now)


class SecurityBase(Node):

    """
    Security Node. Used to define a security within a tree.
    A Security's has no children. It simply models an asset that can be bought
    or sold.

    Args:
        * name (str): Security name
        * multiplier (float): security multiplier - typically used for
            derivatives.
        * lazy_add (bool): Flag to control whether instrument should be added
            to strategy children lazily, i.e. only when there is a transaction
            on the instrument. This improves performance of strategies which
            transact on a sparse set of children.

    Attributes:
        * name (str): Security name
        * parent (Security): Security parent
        * root (Security): Root node of the tree (topmost node)
        * now (datetime): Used when backtesting to store current date
        * stale (bool): Flag used to determine if Security is stale and need
            updating
        * prices (TimeSeries): Security prices.
        * price (float): last price
        * outlays (TimeSeries): Series of outlays. Positive outlays mean
            capital was allocated to security and security consumed that
            amount.  Negative outlays are the opposite. This can be useful for
            calculating turnover at the strategy level.
        * value (float): last value - basically position * price * multiplier
        * weight (float): weight in parent
        * full_name (str): Name including parents' names
        * members (list): Current Security + strategy's children
        * position (float): Current position (quantity).
        * bidoffer (float): Current bid/offer spread
        * bidoffers (TimeSeries): Series of bid/offer spreads
        * bidoffer_paid (TimeSeries): Series of bid/offer paid on transactions
    """

    _last_pos = cy.declare(cy.double)
    _position = cy.declare(cy.double)
    multiplier = cy.declare(cy.double)
    _prices_set = cy.declare(cy.bint)
    _needupdate = cy.declare(cy.bint)
    _outlay = cy.declare(cy.double)
    _bidoffer = cy.declare(cy.double)

    @cy.locals(multiplier=cy.double)
    def __init__(self, name, multiplier=1, lazy_add=False):
        Node.__init__(self, name, parent=None, children=None)
        self._value = 0
        self._price = 0
        self._weight = 0
        self._position = 0
        self.multiplier = multiplier
        self.lazy_add = lazy_add

        # opt
        self._last_pos = 0
        self._issec = True
        self._needupdate = True
        self._outlay = 0
        self._bidoffer = 0

    @property
    def price(self):
        """
        Current price.
        """
        # if accessing and stale - update first
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        return self._price

    @property
    def prices(self):
        """
        TimeSeries of prices.
        """
        # if accessing and stale - update first
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        return self._prices.loc[: self.now]

    @property
    def values(self):
        """
        TimeSeries of values.
        """
        # if accessing and stale - update first
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._values.loc[: self.now]

    @property
    def notional_values(self):
        """
        TimeSeries of notional values.
        """
        # if accessing and stale - update first
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._notl_values.loc[: self.now]

    @property
    def position(self):
        """
        Current position
        """
        # no stale check needed
        return self._position

    @property
    def positions(self):
        """
        TimeSeries of positions.
        """
        # if accessing and stale - update first
        if self._needupdate:
            self.update(self.root.now)
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._positions.loc[: self.now]

    @property
    def outlays(self):
        """
        TimeSeries of outlays. Positive outlays (buys) mean this security
        received and consumed capital (capital was allocated to it). Negative
        outlays are the opposite (the security close/sold, and returned capital
        to parent).
        """
        # if accessing and stale - update first
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._outlays.loc[: self.now]

    @property
    def bidoffer(self):
        """
        Current bid/offer spread.
        """
        # if accessing and stale - update first
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        return self._bidoffer

    @property
    def bidoffers(self):
        """
        TimeSeries of bid/offer spread
        """
        if self._bidoffer_set:
            # if accessing and stale - update first
            if self._needupdate or self.now != self.parent.now:
                self.update(self.root.now)
            return self._bidoffers.loc[: self.now]
        else:
            raise Exception(
                "Bid/offer accounting not turned on: "
                '"bidoffer" argument not provided during setup'
            )

    @property
    def bidoffer_paid(self):
        """
        TimeSeries of bid/offer spread paid on transactions in the current step
        """
        # if accessing and stale - update first
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        return self._bidoffer_paid

    @property
    def bidoffers_paid(self):
        """
        TimeSeries of bid/offer spread paid on transactions in the current step
        """
        if self._bidoffer_set:
            # if accessing and stale - update first
            if self._needupdate or self.now != self.parent.now:
                self.update(self.root.now)
            if self.root.stale:
                self.root.update(self.root.now, None)
            return self._bidoffers_paid.loc[: self.now]
        else:
            raise Exception(
                "Bid/offer accounting not turned on: "
                '"bidoffer" argument not provided during setup'
            )

    def setup(self, universe, **kwargs):
        """
        Setup Security with universe. Speeds up future runs.

        Args:
            * universe (DataFrame): DataFrame of prices with security's name as
                one of the columns.
            ** kwargs (DataFrames): DataFrames of additional security level
                information (i.e. bid/ask spread, risk, etc).
        """
        # if we already have all the prices, we will store them to speed up
        # future updates
        try:
            prices = universe[self.name]
        except KeyError:
            prices = None

        # setup internal data
        if prices is not None:
            self._prices = prices
            self.data = pd.DataFrame(
                index=universe.index,
                columns=["value", "position", "notional_value"],
                data=0.0,
            )
            self._prices_set = True
        else:
            self.data = pd.DataFrame(
                index=universe.index,
                columns=["price", "value", "position", "notional_value"],
            )
            self._prices = self.data["price"]
            self._prices_set = False

        self._values = self.data["value"]
        self._notl_values = self.data["notional_value"]
        self._positions = self.data["position"]

        # add _outlay
        self.data["outlay"] = 0.0
        self._outlays = self.data["outlay"]

        # save bidoffer, if provided
        if "bidoffer" in kwargs:
            self._bidoffer_set = True
            self._bidoffers = kwargs["bidoffer"]
            try:
                bidoffers = self._bidoffers[self.name]
            except KeyError:
                bidoffers = None

            if bidoffers is not None:
                if bidoffers.index.equals(universe.index):
                    self._bidoffers = bidoffers
                else:
                    raise ValueError("Index of bidoffer must match universe data")
            else:
                self.data["bidoffer"] = 0.0
                self._bidoffers = self.data["bidoffer"]

            self.data["bidoffer_paid"] = 0.0
            self._bidoffers_paid = self.data["bidoffer_paid"]

    @cy.locals(prc=cy.double)
    def update(self, date, data=None, inow=None):
        """
        Update security with a given date and optionally, some data.
        This will update price, value, weight, etc.
        """
        # filter for internal calls when position has not changed - nothing to
        # do. Internal calls (stale root calls) have None data. Also want to
        # make sure date has not changed, because then we do indeed want to
        # update.
        if date == self.now and self._last_pos == self._position:
            return

        if inow is None:
            if date == 0:
                inow = 0
            else:
                inow = self.data.index.get_loc(date)

        # date change - update price
        if date != self.now:
            # update now
            self.now = date

            if self._prices_set:
                self._price = self._prices.values[inow]
            # traditional data update
            elif data is not None:
                prc = data[self.name]
                self._price = prc
                self._prices.values[inow] = prc

            # update bid/offer
            if self._bidoffer_set:
                self._bidoffer = self._bidoffers.values[inow]
                self._bidoffer_paid = 0.0

        self._positions.values[inow] = self._position
        self._last_pos = self._position

        if np.isnan(self._price):
            if is_zero(self._position):
                self._value = 0
            else:
                raise Exception(
                    "Position is open (non-zero: %s) and latest price is NaN "
                    "for security %s on %s. Cannot update node value."
                    % (self._position, self.name, date)
                )
        else:
            self._value = self._position * self._price * self.multiplier

        self._notl_value = self._value

        self._values.values[inow] = self._value
        self._notl_values.values[inow] = self._notl_value

        if is_zero(self._weight) and is_zero(self._position):
            self._needupdate = False

        # save outlay to outlays
        if self._outlay != 0:
            self._outlays.values[inow] += self._outlay
            # reset outlay back to 0
            self._outlay = 0

        if self._bidoffer_set:
            self._bidoffers_paid.values[inow] = self._bidoffer_paid

    @cy.locals(
        amount=cy.double, update=cy.bint, q=cy.double, outlay=cy.double, i=cy.int
    )
    def allocate(self, amount, update=True):
        """
        This allocates capital to the Security. This is the method used to
        buy/sell the security.

        A given amount of shares will be determined on the current price, a
        commission will be calculated based on the parent's commission fn, and
        any remaining capital will be passed back up  to parent as an
        adjustment.

        Args:
            * amount (float): Amount of adjustment.
            * update (bool): Force update?

        """

        # will need to update if this has been idle for a while...
        # update if needupdate or if now is stale
        # fetch parent's now since our now is stale
        if self._needupdate or self.now != self.parent.now:
            self.update(self.parent.now)

        # ignore 0 alloc
        # Note that if the price of security has dropped to zero, then it
        # should never be selected by SelectAll, SelectN etc. I.e. we should
        # not open the position at zero price. At the same time, we are able
        # to close it at zero price, because at that point amount=0.
        # Note also that we don't erase the position in an asset which price
        # has dropped to zero (though the weight will indeed be = 0)
        if is_zero(amount):
            return

        if self.parent is self or self.parent is None:
            raise Exception("Cannot allocate capital to a parentless security")

        if is_zero(self._price) or np.isnan(self._price):
            raise Exception(
                "Cannot allocate capital to "
                "%s because price is %s as of %s"
                % (self.name, self._price, self.parent.now)
            )

        # buy/sell
        # determine quantity - must also factor in commission
        # closing out?
        if is_zero(amount + self._value):
            q = -self._position
        else:
            q = amount / (self._price * self.multiplier)
            if self.integer_positions:
                if (self._position > 0) or (is_zero(self._position) and (amount > 0)):
                    # if we're going long or changing long position
                    q = math.floor(q)
                else:
                    # if we're going short or changing short position
                    q = math.ceil(q)

        # if q is 0 nothing to do
        if is_zero(q) or np.isnan(q):
            return

        # unless we are closing out a position (q == -position)
        # we want to ensure that
        #
        # - In the event of a positive amount, this indicates the maximum
        # amount a given security can use up for a purchase. Therefore, if
        # commissions push us above this amount, we cannot buy `q`, and must
        # decrease its value
        #
        # - In the event of a negative amount, we want to 'raise' at least the
        # amount indicated, no less. Therefore, if we have commission, we must
        # sell additional units to fund this requirement. As such, q must once
        # again decrease.
        #
        if not q == -self._position:
            full_outlay, _, _, _ = self.outlay(q)

            # if full outlay > amount, we must decrease the magnitude of `q`
            # this can potentially lead to an infinite loop if the commission
            # per share > price per share. However, we cannot really detect
            # that in advance since the function can be non-linear (say a fn
            # like max(1, abs(q) * 0.01). Nevertheless, we want to avoid these
            # situations.
            # cap the maximum number of iterations to 1e4 and raise exception
            # if we get there
            # if integer positions then we know we are stuck if q doesn't change

            # if integer positions is false then we want full_outlay == amount
            # if integer positions is true then we want to be at the q where
            #   if we bought 1 more then we wouldn't have enough cash
            i = 0
            last_q = q
            last_amount_short = full_outlay - amount
            while not np.isclose(full_outlay, amount, rtol=0.0) and q != 0:

                dq_wout_considering_tx_costs = (full_outlay - amount) / (
                    self._price * self.multiplier
                )
                q = q - dq_wout_considering_tx_costs

                if self.integer_positions:
                    q = math.floor(q)

                full_outlay, _, _, _ = self.outlay(q)

                # if our q is too low and we have integer positions
                # then we know that the correct quantity is the one  where
                # the outlay of q + 1 < amount. i.e. if we bought one more
                # position then we wouldn't have enough cash
                if self.integer_positions:

                    full_outlay_of_1_more, _, _, _ = self.outlay(q + 1)

                    if full_outlay < amount and full_outlay_of_1_more > amount:
                        break

                # if not integer positions then we should keep going until
                # full_outlay == amount or is close enough

                i = i + 1
                if i > 1e4:
                    raise Exception(
                        "Potentially infinite loop detected. This occurred "
                        "while trying to reduce the amount of shares purchased"
                        " to respect the outlay <= amount rule. This is most "
                        "likely due to a commission function that outputs a "
                        "commission that is greater than the amount of cash "
                        "a short sale can raise."
                    )

                if self.integer_positions and last_q == q:
                    raise Exception(
                        "Newton Method like root search for quantity is stuck!"
                        " q did not change in iterations so it is probably a bug"
                        " but we are not entirely sure it is wrong! Consider "
                        " changing to warning."
                    )
                last_q = q

                if np.abs(full_outlay - amount) > np.abs(last_amount_short):
                    raise Exception(
                        "The difference between what we have raised with q and"
                        " the amount we are trying to raise has gotten bigger since"
                        " last iteration! full_outlay should always be approaching"
                        " amount! There may be a case where the commission fn is"
                        " not smooth"
                    )
                last_amount_short = full_outlay - amount

        self.transact(q, update, False)

    @cy.locals(
        q=cy.double,
        update=cy.bint,
        update_self=cy.bint,
        outlay=cy.double,
        bidoffer=cy.double,
    )
    def transact(self, q, update=True, update_self=True, price=None):
        """
        This transacts the Security. This is the method used to
        buy/sell the security for a given quantity.

        The amount of shares is explicitly provided, a
        commission will be calculated based on the parent's commission fn, and
        any remaining capital will be passed back up  to parent as an
        adjustment.

        Args:
            * amount (float): Amount of adjustment.
            * update (bool): Force update on parent due to transaction proceeds
            * update_self (bool): Check for update on self
            * price (float): Optional price if the transaction happens at a bespoke level
        """
        # will need to update if this has been idle for a while...
        # update if needupdate or if now is stale
        # fetch parent's now since our now is stale
        if update_self and (self._needupdate or self.now != self.parent.now):
            self.update(self.parent.now)

        # if q is 0 nothing to do
        if is_zero(q) or np.isnan(q):
            return

        if price is not None and not self._bidoffer_set:
            raise ValueError(
                'Cannot transact at custom prices when "bidoffer" has '
                "not been passed during setup to enable bid-offer tracking."
            )

        # this security will need an update, even if pos is 0 (for example if
        # we close the positions, value and pos is 0, but still need to do that
        # last update)
        self._needupdate = True

        # adjust position & value
        self._position += q

        # calculate proper adjustment for parent
        # parent passed down amount so we want to pass
        # -outlay back up to parent to adjust for capital
        # used
        full_outlay, outlay, fee, bidoffer = self.outlay(q, p=price)

        # store outlay for future reference
        self._outlay += outlay
        self._bidoffer_paid += bidoffer

        # call parent
        self.parent.adjust(-full_outlay, update=update, flow=False, fee=fee)

    @cy.locals(q=cy.double, p=cy.double)
    def commission(self, q, p):
        """
        Calculates the commission (transaction fee) based on quantity and
        price.  Uses the parent's commission_fn.

        Args:
            * q (float): quantity
            * p (float): price

        """
        return self.parent.commission_fn(q, p)

    @cy.locals(q=cy.double)
    def outlay(self, q, p=None):
        """
        Determines the complete cash outlay (including commission) necessary
        given a quantity q.
        Second returning parameter is a commission itself.

        Args:
            * q (float): quantity
            * p (float): price override
        """
        if p is None:
            fee = self.commission(q, self._price * self.multiplier)
            bidoffer = abs(q) * 0.5 * self._bidoffer * self.multiplier
        else:
            # price override provided: custom transaction
            fee = self.commission(q, p * self.multiplier)
            bidoffer = q * (p - self._price) * self.multiplier

        outlay = q * self._price * self.multiplier + bidoffer

        return outlay + fee, outlay, fee, bidoffer

    def run(self):
        """
        Does nothing - securities have nothing to do on run.
        """
        pass


class Security(SecurityBase):
    """
    A standard security with no special features, and where notional value
    is measured based on market value (notional times price).
    It exists to be able to identify standard securities from nonstandard
    ones via isinstance, i.e. isinstance( sec, Security ) would only return
    true for a vanilla security
    """

    pass


class FixedIncomeSecurity(SecurityBase):
    """
    A Fixed Income Security is a security where notional value is
    measured only based on the quantity (par value) of the security.
    """

    @cy.locals(coupon=cy.double)
    def update(self, date, data=None, inow=None):
        """
        Update security with a given date and optionally, some data.
        This will update price, value, weight, etc.
        """

        if inow is None:
            if date == 0:
                inow = 0
            else:
                inow = self.data.index.get_loc(date)

        super(FixedIncomeSecurity, self).update(date, data, inow)

        # For fixed income securities (bonds, swaps), notional value is position size, not value!
        self._notl_value = self._position
        self._notl_values.values[inow] = self._notl_value


class CouponPayingSecurity(FixedIncomeSecurity):
    """
    CouponPayingSecurity expands on SecurityBase to handle securities which
    pay (possibly irregular) coupons (or other forms of cash disbursement).
    More generally, this can include instruments with any sort of carry,
    including (potentially asymmetric) holding costs.

    Args:
        * name (str): Security name
        * multiplier (float): security multiplier - typically used for
            derivatives.
        * fixed_income (bool): Flag to control whether notional_value is based
            only on quantity, or on market value (like an equity).
            Defaults to notional weighting for coupon paying instruments.
        * lazy_add (bool): Flag to control whether instrument should be added
            to strategy children lazily, i.e. only when there is a transaction
            on the instrument. This improves performance of strategies which
            transact on a sparse set of children.

    Attributes:
        * SecurityBase attributes
        * coupon (float): Current coupon payment (quantity).
        * holding_cost (float): Current holding cost (quantity).


    Represents a coupon-paying security, where coupon payments adjust
    the capital of the parent. Coupons and costs must be passed in during setup.
    """

    _coupon = cy.declare(cy.double)
    _holding_cost = cy.declare(cy.double)

    @cy.locals(multiplier=cy.double)
    def __init__(self, name, multiplier=1, fixed_income=True, lazy_add=False):
        super(CouponPayingSecurity, self).__init__(name, multiplier)
        self._coupon = 0
        self._holding_cost = 0
        self._fixed_income = fixed_income
        self.lazy_add = lazy_add

    def setup(self, universe, **kwargs):
        """
        Setup Security with universe and coupon data. Speeds up future runs.

        Args:
            * universe (DataFrame): DataFrame of prices with security's name as
                one of the columns.
            ** kwargs (DataFrames): DataFrames of additional security level
                information (i.e. bid/ask spread, risk, etc).
        """
        super(CouponPayingSecurity, self).setup(universe, **kwargs)

        # Handle coupons
        if "coupons" not in kwargs:
            raise Exception(
                '"coupons" must be passed to setup for a CouponPayingSecurity'
            )

        try:
            self._coupons = kwargs["coupons"][self.name]
        except KeyError:
            self._coupons = None

        if self._coupons is None or not self._coupons.index.equals(universe.index):
            raise ValueError("Index of coupons must match universe data")

        # Handle holding costs
        try:
            self._cost_long = kwargs["cost_long"][self.name]
        except KeyError:
            self._cost_long = None
        try:
            self._cost_short = kwargs["cost_short"][self.name]
        except KeyError:
            self._cost_short = None

        self.data["coupon"] = 0.0
        self.data["holding_cost"] = 0.0
        self._coupon_income = self.data["coupon"]
        self._holding_costs = self.data["holding_cost"]

    @cy.locals(coupon=cy.double, cost=cy.double)
    def update(self, date, data=None, inow=None):
        """
        Update security with a given date and optionally, some data.
        This will update price, value, weight, etc.
        """
        if inow is None:
            if date == 0:
                inow = 0
            else:
                inow = self.data.index.get_loc(date)

        if self._coupons is None:
            raise Exception("coupons have not been set for security %s" % self.name)

        # Standard update
        super(CouponPayingSecurity, self).update(date, data, inow)

        coupon = self._coupons.values[inow]
        # If we were to call self.parent.adjust, then all the child weights would
        # need to be updated. If each security pays a coupon, then this happens for
        # each child. Instead, we store the coupon on self._capital, and it gets
        # swept up as part of the strategy update

        if np.isnan(coupon):
            if is_zero(self._position):
                self._coupon = 0.0
            else:
                raise Exception(
                    "Position is open (non-zero) and latest coupon is NaN "
                    "for security %s on %s. Cannot update node value."
                    % (self.name, date)
                )
        else:
            self._coupon = self._position * coupon

        if self._position > 0 and self._cost_long is not None:
            cost = self._cost_long.values[inow]
            self._holding_cost = self._position * cost
        elif self._position < 0 and self._cost_short is not None:
            cost = self._cost_short.values[inow]
            self._holding_cost = -self._position * cost
        else:
            self._holding_cost = 0.0

        self._capital = self._coupon - self._holding_cost
        self._coupon_income.values[inow] = self._coupon
        self._holding_costs.values[inow] = self._holding_cost

    @property
    def coupon(self):
        """
        Current coupon payment (scaled by position)
        """
        if (
            self.root.stale
        ):  # Stale check needed because coupon paid depends on position
            self.root.update(self.root.now, None)
        return self._coupon

    @property
    def coupons(self):
        """
        TimeSeries of coupons paid (scaled by position)
        """
        if (
            self.root.stale
        ):  # Stale check needed because coupon paid depends on position
            self.root.update(self.root.now, None)
        return self._coupon_income.loc[: self.now]

    @property
    def holding_cost(self):
        """
        Current holding cost (scaled by position)
        """
        if (
            self.root.stale
        ):  # Stale check needed because coupon paid depends on position
            self.root.update(self.root.now, None)
        return self._holding_cost

    @property
    def holding_costs(self):
        """
        TimeSeries of coupons paid (scaled by position)
        """
        if (
            self.root.stale
        ):  # Stale check needed because coupon paid depends on position
            self.root.update(self.root.now, None)
        return self._holding_costs.loc[: self.now]


class HedgeSecurity(SecurityBase):
    """
    HedgeSecurity is a SecurityBase where the notional value is set to zero, and thus
    does not count towards the notional value of the strategy. It is intended for use
    in fixed income strategies.

    For example in a corporate bond strategy, the notional value might refer to the size
    of the corporate bond portfolio, and exclude the notional of treasury bonds or interest
    rate swaps used as hedges.
    """

    def update(self, date, data=None, inow=None):
        """
        Update security with a given date and optionally, some data.
        This will update price, value, weight, etc.
        """
        super(HedgeSecurity, self).update(date, data, inow)
        self._notl_value = 0.0
        self._notl_values.values.fill(0.0)


class CouponPayingHedgeSecurity(CouponPayingSecurity):
    """
    CouponPayingHedgeSecurity is a CouponPayingSecurity where the notional value is set to zero, and thus
    does not count towards the notional value of the strategy. It is intended for use
    in fixed income strategies.

    For example in a corporate bond strategy, the notional value might refer to the size
    of the corporate bond portfolio, and exclude the notional of treasury bonds or interest
    rate swaps used as hedges.
    """

    def update(self, date, data=None, inow=None):
        """
        Update security with a given date and optionally, some data.
        This will update price, value, weight, etc.
        """
        super(CouponPayingHedgeSecurity, self).update(date, data, inow)
        self._notl_value = 0.0
        self._notl_values.values.fill(0.0)


class Algo(object):

    """
    Algos are used to modularize strategy logic so that strategy logic becomes
    modular, composable, more testable and less error prone. Basically, the
    Algo should follow the unix philosophy - do one thing well.

    In practice, algos are simply a function that receives one argument, the
    Strategy (referred to as target) and are expected to return a bool.

    When some state preservation is necessary between calls, the Algo
    object can be used (this object). The __call___ method should be
    implemented and logic defined therein to mimic a function call. A
    simple function may also be used if no state preservation is necessary.

    Args:
        * name (str): Algo name

    """

    def __init__(self, name=None):
        self._name = name

    @property
    def name(self):
        """
        Algo name.
        """
        if self._name is None:
            self._name = self.__class__.__name__
        return self._name

    def __call__(self, target):
        raise NotImplementedError("%s not implemented!" % self.name)


class AlgoStack(Algo):

    """
    An AlgoStack derives from Algo runs multiple Algos until a
    failure is encountered.

    The purpose of an AlgoStack is to group a logic set of Algos together. Each
    Algo in the stack is run. Execution stops if one Algo returns False.

    Args:
        * algos (list): List of algos.

    """

    def __init__(self, *algos):
        super(AlgoStack, self).__init__()
        self.algos = algos
        self.check_run_always = any(hasattr(x, "run_always") for x in self.algos)

    def __call__(self, target):
        # normal running mode
        if not self.check_run_always:
            for algo in self.algos:
                if not algo(target):
                    return False
            return True
        # run mode when at least one algo has a run_always attribute
        else:
            # store result in res
            # allows continuation to check for and run
            # algos that have run_always set to True
            res = True
            for algo in self.algos:
                if res:
                    res = algo(target)
                elif hasattr(algo, "run_always"):
                    if algo.run_always:
                        algo(target)
            return res


class Strategy(StrategyBase):

    """
    Strategy expands on the StrategyBase and incorporates Algos.

    Basically, a Strategy is built by passing in a set of algos. These algos
    will be placed in an Algo stack and the run function will call the stack.

    Furthermore, two class attributes are created to pass data between algos.
    perm for permanent data, temp for temporary data.

    Args:
        * name (str): Strategy name
        * algos (list): List of Algos to be passed into an AlgoStack
        * children (dict, list): Children - useful when you want to create
            strategies of strategies
            Children can be any type of Node or str.
            String values correspond to children which will be lazily created
            with that name when needed.
        * parent (Node): The parent Node

    Attributes:
        * stack (AlgoStack): The stack
        * temp (dict): A dict containing temporary data - cleared on each call
            to run. This can be used to pass info to other algos.
        * perm (dict): Permanent data used to pass info from one algo to
            another. Not cleared on each pass.

    """

    def __init__(self, name, algos=None, children=None, parent=None):
        super(Strategy, self).__init__(name, children=children, parent=parent)
        if algos is None:
            algos = []
        self.stack = AlgoStack(*algos)
        self.temp = {}
        self.perm = {}

    def run(self):
        # clear out temp data
        self.temp = {}

        # run algo stack
        self.stack(self)

        # run children
        for c in self._childrenv:
            c.run()


class FixedIncomeStrategy(Strategy):
    """
    FixedIncomeStrategy is an alias for Strategy where the fixed_income flag
    is set to True.

    For this type of strategy:
        - capital allocations are not necessary, and initial capital is not used
        - bankruptcy is disabled
        - weights are based off notional_value rather than value
        - strategy price is computed from additive PNL returns
            per unit of notional_value, with a reference price of PAR
        - "transact" assumes the role of "allocate", in order to buy/sell
            children on a weighted notional basis
        - "rebalance" adjusts notionals rather than capital allocations based
            on weights
    """

    def __init__(self, name, algos=None, children=None):
        super(FixedIncomeStrategy, self).__init__(name, algos=algos, children=children)
        self._fixed_income = True
