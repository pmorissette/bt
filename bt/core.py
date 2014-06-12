import math

import pandas as pd
import numpy as np
import cython as cy


class Node(object):

    _price = cy.declare(cy.double)
    _value = cy.declare(cy.double)
    _weight = cy.declare(cy.double)
    _issec = cy.declare(cy.bint)

    def __init__(self, name, parent=None, children=None):

        self.name = name

        if children is not None:
            if isinstance(children, list):
                # if all strings - just save as universe_filter
                if all(isinstance(x, str) for x in children):
                    self._universe_tickers = children
                    # empty dict - don't want to uselessly create
                    # tons of children when they might not be needed
                    children = {}
                else:
                    # this will be case if we pass in children
                    # (say a bunch of sub-strategies)
                    tmp = {}
                    for c in children:
                        if type(c) == str:
                            tmp[c] = SecurityBase(c)
                        else:
                            tmp[c.name] = c
                    children = tmp
                    # we want to keep whole universe in this case
                    # so set to None
                    self._universe_tickers = None

        if parent is None:
            self.parent = self
            self.root = self
        else:
            self.parent = parent
            self.root = parent.root
            parent._add_child(self)

        # default children
        if children is None:
            children = {}
            self._universe_tickers = None
        self.children = children

        self._childrenv = children.values()
        for c in self._childrenv:
            c.parent = self
            c.root = self.root

        # set default value for now
        self.now = 0
        # make sure root has stale flag
        # used to avoid unncessary update
        # sometimes we change values in the tree and we know that we will need
        # to update if another node tries to access a given value (say weight).
        # This avoid calling the update until it is actually needed.
        self.root.stale = False

        # helper vars
        self._price = 0
        self._value = 0
        self._weight = 0

        # is security flag - used to avoid updating 0 pos securities
        self._issec = False

    def __getitem__(self, key):
        return self.children[key]

    @property
    def prices(self):
        # can optimize depending on type -
        # securities don't need to check stale to
        # return latest prices, whereas strategies do...
        raise NotImplementedError()

    @property
    def price(self):
        # can optimize depending on type -
        # securities don't need to check stale to
        # return latest prices, whereas strategies do...
        raise NotImplementedError()

    @property
    def value(self):
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._value

    @property
    def weight(self):
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._weight

    def setup(self, dates):
        raise NotImplementedError()

    def _add_child(self, child):
        child.parent = self
        child.root = self.root
        if self.children is None:
            self.children = {child.name: child}
        else:
            self.children[child.name] = child

        self._childrenv = self.children.values()

    def update(self, date, data=None):
        raise NotImplementedError()

    def adjust(self, amount, update=True, isflow=True):
        raise NotImplementedError()

    def allocate(self, amount, update=True):
        raise NotImplementedError()

    @property
    def members(self):
        res = [self]
        for c in self.children.values():
            res.extend(c.members)
        return res

    @property
    def full_name(self):
        if self.parent == self:
            return self.name
        else:
            return '%s>%s' % (self.parent.full_name, self.name)


class StrategyBase(Node):

    _capital = cy.declare(cy.double)
    _net_flows = cy.declare(cy.double)
    _last_value = cy.declare(cy.double)
    _last_price = cy.declare(cy.double)

    def __init__(self, name, children=None, parent=None):
        Node.__init__(self, name, children=children, parent=parent)
        self._capital = 0
        self._weight = 1
        self._value = 0
        self._price = 100

        # helper vars
        self._net_flows = 0
        self._last_value = 0
        self._last_price = 100

        # default commission function
        self.commission_fn = self._dflt_comm_fn

    @property
    def price(self):
        if self.root.stale:
            self.root.update(self.now, None)
        return self._price

    @property
    def prices(self):
        if self.root.stale:
            self.root.update(self.now, None)
        return self._prices.ix[:self.now]

    @property
    def values(self):
        if self.root.stale:
            self.root.update(self.now, None)
        return self._values.ix[:self.now]

    @property
    def capital(self):
        # no stale check needed
        return self._capital

    @property
    def universe(self):
        # avoid windowing every time
        # if calling and on same date return
        # cached value
        if self.now == self._last_chk:
            return self._funiverse
        else:
            self._last_chk = self.now
            self._funiverse = self._universe.ix[:self.now]
            return self._funiverse

    def setup(self, universe):
        # setup universe
        if self._universe_tickers is not None:
            # if we have universe_tickers defined, limit universe to
            # those tickers
            valid_filter = list(set(universe.columns)
                                .intersection(self._universe_tickers))
            universe = universe[valid_filter]
        self._universe = universe
        self._funiverse = universe
        self._last_chk = None

        # setup internal data
        self.data = pd.DataFrame(index=universe.index,
                                 columns=['price', 'value'],
                                 data=0.0)
        self._prices = self.data['price']
        self._values = self.data['value']

        # setup children as well
        if self.children is not None:
            [c.setup(universe) for c in self._childrenv]

    @cy.locals(newpt=cy.bint, val=cy.double, ret=cy.double)
    def update(self, date, data=None):
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
            newpt = True

        # update now
        self.now = date

        # update children if any and calculate value
        val = self._capital  # default if no children

        if self.children is not None:
            for c in self._childrenv:
                # avoid useless update call
                if c._issec and not c._needupdate:
                    continue
                c.update(date, data)
                val += c.value

        if self.root == self:
            if val < 0:
                raise ValueError('negative root node value!')

        # update data if this value is different or
        # if now has changed - avoid all this if not since it
        # won't change
        if newpt or self._value != val:
            self._value = val
            self._values[date] = val

            try:
                ret = self._value / (self._last_value
                                     + self._net_flows) - 1
            except ZeroDivisionError:
                if self._value == 0:
                    ret = 0
                else:
                    raise ZeroDivisionError(
                        'Could not update %s. Last value '
                        'was %s and net flows were %s. Current'
                        'value is %s. Therefore, '
                        'we are dividing by zero to obtain the return '
                        'for the period.' % (self.name,
                                             self._last_value,
                                             self._net_flows,
                                             self._value))

            self._price = self._last_price * (1 + ret)
            self._prices[date] = self._price

        # update children weights
        if self.children is not None:
            for c in self._childrenv:
                # avoid useless update call
                if c._issec and not c._needupdate:
                    continue
                try:
                    c._weight = c.value / val
                except ZeroDivisionError:
                    c._weight = 0

    @cy.locals(amount=cy.double, update=cy.bint, flow=cy.bint)
    def adjust(self, amount, update=True, flow=True):
        """
        adjust captial - used to inject capital
        """
        # adjust capital
        self._capital += amount

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
        # allocate to child
        if child is not None:
            if child not in self.children:
                c = SecurityBase(child)
                c.setup(self._universe)
                # update to bring up to speed
                c.update(self.now)
                # add child to tree
                self._add_child(c)

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
                [c.allocate(amount * c._weight, update=False)
                 for c in self._childrenv]

            # mark as stale if update requested
            if update:
                self.root.stale = True

    @cy.locals(delta=cy.double, weight=cy.double, base=cy.double)
    def rebalance(self, weight, child, base=np.nan, update=True):
        # if weight is 0 - we want to close child
        if weight == 0:
            if child in self.children:
                return self.close(child)
            else:
                return

        # if no base specified use self's value
        if np.isnan(base):
            base = self.value

        # else make sure we have child
        if child not in self.children:
            c = SecurityBase(child)
            c.setup(self._universe)
            # update child to bring up to speed
            c.update(self.now)
            self._add_child(c)

        # allocate to child
        # figure out weight delta
        c = self.children[child]
        delta = weight - c.weight
        c.allocate(delta * base)

    def close(self, child):
        c = self.children[child]
        # flatten if children not None
        if c.children is not None and len(c.children) != 0:
            c.flatten()
        c.allocate(-c.value)

    def flatten(self):
        # go right to base alloc
        [c.allocate(-c.value) for c in self._childrenv if c.value != 0]

    def run(self):
        pass

    def set_commissions(self, fn):
        self.commission_fn = fn

    @cy.locals(q=cy.double)
    def _dflt_comm_fn(self, q):
        return max(1, abs(q) * 0.01)


class SecurityBase(Node):

    _last_pos = cy.declare(cy.double)
    _position = cy.declare(cy.double)
    multiplier = cy.declare(cy.double)
    _prices_set = cy.declare(cy.bint)
    _needupdate = cy.declare(cy.bint)

    @cy.locals(multiplier=cy.double)
    def __init__(self, name, multiplier=1):
        Node.__init__(self, name, parent=None, children=None)
        self._value = 0
        self._price = 0
        self._weight = 0
        self._position = 0
        self.multiplier = multiplier

        # opt
        self._last_pos = 0
        self._issec = True
        self._needupdate = True

    @property
    def price(self):
        # if accessing and stale - update first
        if not self._needupdate:
            self.update(self.root.now)
        return self._price

    @property
    def prices(self):
        # if accessing and stale - update first
        if not self._needupdate:
            self.update(self.root.now)
        return self._prices.ix[:self.now]

    @property
    def values(self):
        # if accessing and stale - update first
        if not self._needupdate:
            self.update(self.root.now)
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._values.ix[:self.now]

    @property
    def position(self):
        # no stale check needed
        return self._position

    def setup(self, universe):
        # if we already have all the prices, we will store them to speed up
        # future udpates
        try:
            prices = universe[self.name]
        except KeyError:
            prices = None

        # setup internal data
        if prices is not None:
            self._prices = prices
            self.data = pd.DataFrame(index=universe.index,
                                     columns=['value', 'position'],
                                     data=0.0)
            self._prices_set = True
        else:
            self.data = pd.DataFrame(index=universe.index,
                                     columns=['price', 'value', 'position'])
            self._prices = self.data['price']
            self._prices_set = False

        self._values = self.data['value']
        self._positions = self.data['position']

    @cy.locals(prc=cy.double)
    def update(self, date, data=None):
        # filter for internal calls when position has not changed - nothing to
        # do. Internal calls (stale root calls) have None data. Also want to
        # make sure date has not changed, because then we do indeed want to
        # update.
        if date == self.now and self._last_pos == self._position:
            return

        # date change - update price
        if date != self.now:
            # update now
            self.now = date

            if self._prices_set:
                self._price = self._prices[self.now]
            # traditional data update
            elif data is not None:
                prc = data[self.name]
                self._price = prc
                self._prices[date] = prc

        self._positions[date] = self._position
        self._last_pos = self._position

        self._value = self._position * self._price * self.multiplier
        self._values[date] = self._value

        if self._weight == 0 and self._position == 0:
            self._needupdate = False

    @cy.locals(amount=cy.double, update=cy.bint, q=cy.double, outlay=cy.double)
    def allocate(self, amount, update=True):
        # buy/sell appropriate # of shares and pass
        # remaining capital back up to parent as
        # adjustment

        # will need to update if this has been idle for a while...
        # update if needupdate or if now is stale
        # fetch parent's now since our now is stale
        if self._needupdate or self.now != self.parent.now:
            self.update(self.parent.now)

        # ignore 0 alloc
        if amount == 0:
            return

        if self.parent is self or self.parent is None:
            raise Exception(
                'Cannot allocate capital to a parentless security')

        if self._price == 0 or np.isnan(self._price):
            raise Exception(
                'Cannot allocate capital to '
                '%s because price is 0 or nan as of %s'
                % (self.name, self.parent.now))

        # buy/sell
        # determine quantity - must also factor in commission
        # closing out?
        if amount == -self._value:
            q = -self._position
        else:
            if amount > 0:
                q = math.floor(amount / (self._price * self.multiplier))
            else:
                q = math.ceil(amount / (self._price * self.multiplier))

        # if q is 0 nothing to do
        if q == 0 or np.isnan(q):
            return

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
        outlay = self.outlay(q)

        # call parent
        self.parent.adjust(-outlay, update=update, flow=False)

    @cy.locals(q=cy.double)
    def commission(self, q):
        return self.parent.commission_fn(q)

    @cy.locals(q=cy.double)
    def outlay(self, q):
        return q * self._price * self.multiplier + self.commission(q)

    def run(self):
        pass


class Algo(object):

    def __init__(self, name=None):
        self._name = name

    @property
    def name(self):
        if self._name is None:
            self._name = self.__class__.__name__
        return self._name

    def __call__(self, target):
        raise NotImplementedError("%s not implemented!" % self.name)


class AlgoStack(Algo):

    def __init__(self, *algos):
        super(AlgoStack, self).__init__()
        self.algos = algos
        self.check_run_always = any(hasattr(x, 'run_always')
                                    for x in self.algos)

    def __call__(self, target):
        # normal runing mode
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
                elif hasattr(algo, 'run_always'):
                    if algo.run_always:
                        algo(target)
            return res


class Strategy(StrategyBase):

    def __init__(self, name, algos=[], children=None):
        super(Strategy, self).__init__(name, children=children)
        self.stack = AlgoStack(*algos)
        self.temp = {}
        self.perm = {}

    def run(self):
        # clear out temp data
        self.temp = {}

        # run algo stack
        self.stack(self)

        # run children
        for c in self.children.values():
            c.run()
