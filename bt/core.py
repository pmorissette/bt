import math
import copy

import pandas as pd
import numpy as np
import cython as cy


class Node(object):

    _price = cy.declare(cy.double)
    _value = cy.declare(cy.double)
    _weight = cy.declare(cy.double)

    def __init__(self, name, parent=None, children=None):

        self.name = name

        if children is not None:
            if isinstance(children, list):
                children = {c.name: copy.deepcopy(c)
                            for c in children}

        if parent is None:
            self.parent = self
            self.root = self
        else:
            self.parent = parent
            self.root = parent.root
            parent._add_child(self)

        self.children = children

        if children is not None:
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
            self.root.update(self.now, None)
        return self._value

    @property
    def weight(self):
        if self.root.stale:
            self.root.update(self.now, None)
        return self._weight

    def setup(self, dates):
        raise NotImplementedError()

    def _add_child(self, child):
        if self.children is None:
            self.children = {child.name: child}
        else:
            self.children[child.name] = child

        self._childrenv = self.children.values()

    def update(self, date, data):
        raise NotImplementedError()

    def adjust(self, amount, update=True, isflow=True):
        raise NotImplementedError()

    def allocate(self, amount, update=True):
        raise NotImplementedError()


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

    @property
    def price(self):
        if self.root.stale:
            self.root.update(self.now, None)
        return self._price

    @property
    def prices(self):
        if self.root.stale:
            self.root.update(self.now, None)
        return self._prices[:self.now]

    @property
    def values(self):
        if self.root.stale:
            self.root.update(self.now, None)
        return self._values[:self.now]

    @property
    def capital(self):
        # no stale check needed
        return self._capital

    def setup(self, dates):
        # setup internal data
        self.data = pd.DataFrame(index=dates,
                                 columns=['price', 'value'])
        self._prices = self.data['price']
        self._values = self.data['value']

        # setup children as well
        if self.children is not None:
            [c.setup(dates) for c in self._childrenv]

    @cy.locals(newpt=cy.bint, val=cy.double, ret=cy.double)
    def update(self, date, data):
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
                c.update(date, data)
                val += c.value

        # update data if this value is different or
        # if now has changed - avoid all this if not since it
        # won't change
        if newpt or self._value != val:
            self._value = val
            self._values[date] = val

            # calc price - artificial index representing strategy return
            try:
                ret = float(self._value) / \
                    (self._last_value + self._net_flows) - 1
            except ZeroDivisionError, e:
                # if denom is 0 as well - just have 0 return
                if self._value == 0:
                    ret = 0
                else:
                    raise e

            self._price = self._last_price * (1 + ret)
            self._prices[date] = self._price

        # update children weights
        if self.children is not None:
            for c in self._childrenv:
                try:
                    c._weight = float(c.value) / val
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
    def allocate(self, amount, update=True):
        # adjust parent's capital
        # no need to update now - avoids repetition
        self.parent.adjust(-amount, update=False, flow=True)

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

    def run(self):
        pass


class SecurityBase(Node):

    _last_pos = cy.declare(cy.int)
    _position = cy.declare(cy.int)
    multiplier = cy.declare(cy.double)

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

    @property
    def price(self):
        return self._price

    @property
    def prices(self):
        return self._prices[:self.now]

    @property
    def values(self):
        if self.root.stale:
            self.root.update(self.now, None)
        return self._values[:self.now]

    @property
    def position(self):
        # no stale check needed
        return self._position

    def setup(self, dates):
        # setup internal data
        self.data = pd.DataFrame(index=dates,
                                 columns=['price', 'value', 'position'])
        self._prices = self.data['price']
        self._values = self.data['value']
        self._positions = self.data['position']

    @cy.locals(prc=cy.double)
    def update(self, date, data):
        # filter for internal calls when position has not changed - nothing to
        # do. Internal calls (stale root calls) have None data. Also want to
        # make sure date has not changed, because then we do indeed want to
        # update.
        if date == self.now and self._last_pos == self._position:
            return

        # update now
        self.now = date

        if data is not None:
            prc = data[self.name]
            self._price = prc
            self._prices[date] = prc

        self._positions[date] = self._position
        self._last_pos = self._position

        self._value = self.position * self._price * self.multiplier
        self._values[date] = self._value

    @cy.locals(amount=cy.double, update=cy.bint, q=cy.int, outlay=cy.double)
    def allocate(self, amount, update=True):
        # buy/sell appropriate # of shares and pass
        # remaining capital back up to parent as
        # adjustment

        # ignore 0 alloc
        if amount == 0:
            return

        if self.parent is self or self.parent is None:
            raise Exception(
                'Cannot allocate capital to a parentless security')

        if self._price == 0 or np.isnan(self._price):
            return

        # buy/sell
        # determine quantity - must also factor in commission
        if amount > 0:
            q = math.floor(float(amount) / (self._price * self.multiplier))
        else:
            q = math.ceil(float(amount) / (self._price * self.multiplier))

        # if q is 0 nothing to do
        if q == 0 or np.isnan(q):
            return

        # calculate proper adjustment for parent
        # parent passed down amount so we want to pass
        # -outlay back up to parent to adjust for capital
        # used
        outlay = self.outlay(q)
        self.parent.adjust(-outlay, update=update, flow=False)

        # adjust position & value
        self._position += q

    @cy.locals(q=cy.int)
    def commission(self, q):
        return max(1, abs(q) * 0.01)

    @cy.locals(q=cy.int)
    def outlay(self, q):
        return q * self._price * self.multiplier + self.commission(q)
