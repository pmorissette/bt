"""Load FXMacroData forex history into a bt price DataFrame.

The live fetch helper uses FXMacroData's forex endpoint, while the runnable
example below uses static rows so it can run without network access.
"""

from __future__ import annotations

import json
import os
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


FXMACRODATA_API_BASE_URL = "https://fxmacrodata.com/api/v1"


def split_currency_pair(pair: str) -> tuple[str, str]:
    """Return normalized base and quote currency codes from EURUSD or EUR/USD."""
    normalized = pair.replace("/", "").replace("-", "").replace("_", "").upper()
    if len(normalized) != 6:
        raise ValueError("currency pair must look like 'EURUSD' or 'EUR/USD'")
    return normalized[:3], normalized[3:]


def fxmacrodata_rows_to_prices(rows: Iterable[dict], pair: str) -> pd.DataFrame:
    """Convert FXMacroData forex rows into a single-column bt price frame."""
    split_currency_pair(pair)
    column_name = pair.replace("/", "").replace("-", "").replace("_", "").upper()
    records = []

    for row in rows:
        date_value = (
            row.get("date")
            or row.get("time")
            or row.get("timestamp")
            or row.get("datetime")
        )
        rate = (
            row.get("value")
            or row.get("val")
            or row.get("rate")
            or row.get("close")
            or row.get("fx_rate")
        )
        if date_value is None or rate is None:
            continue
        records.append({"Date": pd.to_datetime(date_value), column_name: float(rate)})

    if not records:
        raise ValueError("FXMacroData response did not include dated rate rows")

    prices = pd.DataFrame.from_records(records)
    prices = prices.drop_duplicates(subset="Date").sort_values("Date").set_index("Date")
    prices.index.name = None
    return prices[[column_name]]


def fetch_fxmacrodata_prices(
    pair: str,
    start_date: str,
    end_date: str,
    *,
    api_key: str | None = None,
    base_url: str = FXMACRODATA_API_BASE_URL,
    timeout: float = 30,
) -> pd.DataFrame:
    """Fetch FXMacroData daily spot rates and return a bt price frame."""
    base_currency, quote_currency = split_currency_pair(pair)
    params = {"start_date": start_date, "end_date": end_date}
    headers = {}
    api_key = api_key or os.getenv("FXMACRODATA_API_KEY") or os.getenv("FXMD_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key

    request = Request(
        f"{base_url.rstrip('/')}/forex/{base_currency}/{quote_currency}?{urlencode(params)}",
        headers=headers,
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)

    rows = payload.get("data") if isinstance(payload, dict) else payload
    return fxmacrodata_rows_to_prices(rows, pair)


def make_buy_and_hold_strategy(name: str = "fxmacrodata_buy_and_hold"):
    """Build a simple one-asset buy-and-hold strategy."""
    import bt

    return bt.Strategy(
        name,
        [
            bt.algos.RunOnce(),
            bt.algos.SelectAll(),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
    )


def run():
    """Run the offline example with FXMacroData-shaped sample rows."""
    import bt

    sample_rows = [
        {"date": "2024-01-01", "value": 1.1038},
        {"date": "2024-01-02", "value": 1.0943},
        {"date": "2024-01-03", "value": 1.0920},
        {"date": "2024-01-04", "value": 1.0950},
        {"date": "2024-01-05", "value": 1.0944},
        {"date": "2024-01-08", "value": 1.0951},
        {"date": "2024-01-09", "value": 1.0932},
        {"date": "2024-01-10", "value": 1.0970},
        {"date": "2024-01-11", "value": 1.0998},
        {"date": "2024-01-12", "value": 1.0950},
        {"date": "2024-01-15", "value": 1.0946},
        {"date": "2024-01-16", "value": 1.0875},
        {"date": "2024-01-17", "value": 1.0848},
        {"date": "2024-01-18", "value": 1.0873},
        {"date": "2024-01-19", "value": 1.0896},
        {"date": "2024-01-22", "value": 1.0887},
        {"date": "2024-01-23", "value": 1.0854},
        {"date": "2024-01-24", "value": 1.0885},
        {"date": "2024-01-25", "value": 1.0841},
        {"date": "2024-01-26", "value": 1.0853},
    ]

    prices = fxmacrodata_rows_to_prices(sample_rows, "EURUSD")
    strategy = make_buy_and_hold_strategy()
    backtest = bt.Backtest(
        strategy,
        prices,
        integer_positions=False,
        progress_bar=False,
    )
    result = bt.run(backtest)
    print(result.stats)
    return result


if __name__ == "__main__":
    run()
