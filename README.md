![](http://pmorissette.github.io/bt/_static/logo.png)

[![Build Status](https://github.com/pmorissette/bt/workflows/Build%20Status/badge.svg)](https://github.com/pmorissette/bt/actions/)
[![PyPI Version](https://img.shields.io/pypi/v/bt)](https://pypi.org/project/bt/)
[![PyPI License](https://img.shields.io/pypi/l/bt)](https://pypi.org/project/bt/)

# bt - Flexible Backtesting for Python

bt is currently in alpha stage - if you find a bug, please submit an issue.

Read the docs here: http://pmorissette.github.io/bt.

## What is bt?

**bt** is a flexible backtesting framework for Python used to test quantitative
trading strategies. **Backtesting** is the process of testing a strategy over a given
data set. This framework allows you to easily create strategies that mix and match
different [Algos](http://pmorissette.github.io/bt/bt.html#bt.core.Algo). It aims to foster the creation of easily testable, re-usable and
flexible blocks of strategy logic to facilitate the rapid development of complex
trading strategies.

The goal: to save **quants** from re-inventing the wheel and let them focus on the
important part of the job - strategy development.

**bt** is coded in **Python** and joins a vibrant and rich ecosystem for data analysis.
Numerous libraries exist for machine learning, signal processing and statistics and can be leveraged to avoid
re-inventing the wheel - something that happens all too often when using other
languages that don't have the same wealth of high-quality, open-source projects.

bt is built atop [ffn](https://github.com/pmorissette/ffn) - a financial function library for Python. Check it out!

## Features

* **Tree Structure**
    [The tree structure](http://pmorissette.github.io/bt/tree.html) facilitates the construction and composition of complex algorithmic trading
    strategies that are modular and re-usable. Furthermore, each tree [Node](http://pmorissette.github.io/bt/bt.html#bt.core.Node) has its own
    price index that can be used by Algos to determine a Node's allocation.

* **Algorithm Stacks**
    [Algos](http://pmorissette.github.io/bt/bt.html#bt.core.Algo) and [AlgoStacks](http://pmorissette.github.io/bt/bt.html#bt.core.AlgoStack) are
    another core feature that facilitate the creation of modular and re-usable strategy
    logic. Due to their modularity, these logic blocks are also easier to test -
    an important step in building robust financial solutions.

* **Charting and Reporting**
    bt also provides many useful charting functions that help visualize backtest
    results. We also plan to add more charts, tables and report formats in the future,
    such as automatically generated PDF reports.

* **Detailed Statistics**
    Furthermore, bt calculates a bunch of stats relating to a backtest and offers a quick way to compare
    these various statistics across many different backtests via [Results](http://pmorissette.github.io/bt/bt.html#bt.backtest.Result) display methods.


## Roadmap

Future development efforts will focus on:

* **Speed**
    Due to the flexible nature of bt, a trade-off had to be made between
    usability and performance. Usability will always be the priority, but we do
    wish to enhance the performance as much as possible.

* **Algos**
    We will also be developing more algorithms as time goes on. We also
    encourage anyone to contribute their own algos as well.

* **Charting and Reporting**
    This is another area we wish to constantly improve on
    as reporting is an important aspect of the job. Charting and reporting also
    facilitate finding bugs in strategy logic.

## Installing bt

The easiest way to install `bt` is from the [Python Package Index](https://pypi.python.org/pypi/bt/)
using `pip`:


```bash
pip install bt
```


Since bt has many dependencies, we strongly recommend installing the [Anaconda Scientific Python
Distribution](https://store.continuum.io/cshop/anaconda/), especially on Windows. This distribution
comes with many of the required packages pre-installed, including pip. Once Anaconda is installed, the above
command should complete the installation.

## Recommended Setup

We believe the best environment to develop with bt is the [IPython Notebook](http://ipython.org/notebook.html).
From their homepage, the IPython Notebook is:

    "[...] a web-based interactive computational environment
    where you can combine code execution, text, mathematics, plots and rich
    media into a single document [...]"

This environment allows you to plot your charts in-line and also allows you to
easily add surrounding text with Markdown. You can easily create Notebooks that
you can share with colleagues and you can also save them as PDFs. If you are not
yet convinced, head over to their website.

## Local development

The following steps can be used to make an editable local copy of the repository in order to make changes and contribute to the `bt` framework.

1. Fork the project repository by clicking on the 'Fork' button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

2. Clone your fork of the `bt` repository from your GitHub account to your local disk:

```bash
git clone git@github.com:<your GitHub handle>/bt.git
cd bt
```

3. Create a new environment (e.g. `bt-dev`) using `venv` and activate it with the appropriate script for your system:

```bash
python -m venv bt-dev
bt-dev\Scripts\activate.ps1
```

4. Install the local copy of `bt` via `pip` in editable/development mode:

```bash
pip install -e .
```

5. Create a feature branch (e.g. `my-feature`) to hold your development changes:

```bash
git checkout -b my-feature
```

Always use a feature branch. It's good practice to never routinely work on the main branch of any repository.

6. Make your changes, commit locally and add tests as required. Run the linter (`ruff`) and the tests (`pytest`) often:

```bash
ruff check bt
pytest -vvv tests
```

7. Push the changes to your fork on GitHub:

```bash
git push
```

8. Create a Pull Request on your forked repository page, selecting your feature branch (e.g. `my-feature`) as head and pointing to the original `pmorissette/bt` repository as base.
