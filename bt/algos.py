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

    def __call__(self, target):
        for algo in self.algos:
            if not algo(target):
                return False


class DatePrintAlgo(Algo):

    def __call__(self, target):
        print target.now
        return True
