from . import core
from . import algos
from . import backtest

from .backtest import Backtest, run
from .core import Strategy, Algo, AlgoStack

import ffn
from ffn import utils, data, get, merge

__version__ = (0, 2, 6)
