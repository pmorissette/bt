import io
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest


def load_example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "fxmacrodata.py"
    spec = spec_from_file_location("fxmacrodata_example", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fxmacrodata_rows_to_prices_sorts_and_normalizes_pair():
    module = load_example_module()
    prices = module.fxmacrodata_rows_to_prices(
        [
            {"date": "2024-01-03", "value": "1.0920"},
            {"date": "2024-01-01", "rate": 1.1038},
            {"date": "2024-01-02", "close": 1.0943},
        ],
        "eur/usd",
    )

    expected = pd.DataFrame(
        {"EURUSD": [1.1038, 1.0943, 1.0920]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    expected.index.name = None
    pd.testing.assert_frame_equal(prices, expected)


def test_fxmacrodata_rows_to_prices_rejects_missing_rates():
    module = load_example_module()
    with pytest.raises(ValueError, match="dated rate rows"):
        module.fxmacrodata_rows_to_prices([{"date": "2024-01-01"}], "EURUSD")


def test_fetch_fxmacrodata_prices_builds_current_api_request():
    module = load_example_module()
    payload = {"data": [{"date": "2024-01-01", "value": 1.1038}]}
    captured = {}

    class FakeResponse(io.StringIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["api_key"] = request.get_header("X-api-key")
        captured["timeout"] = timeout
        return FakeResponse(json.dumps(payload))

    with mock.patch.object(module, "urlopen", side_effect=fake_urlopen):
        prices = module.fetch_fxmacrodata_prices(
            "EURUSD",
            "2024-01-01",
            "2024-01-31",
            api_key="test-key",
            timeout=12,
        )

    assert captured == {
        "url": (
            "https://fxmacrodata.com/api/v1/forex/EUR/USD"
            "?start_date=2024-01-01&end_date=2024-01-31"
        ),
        "api_key": "test-key",
        "timeout": 12,
    }
    assert prices.loc[pd.Timestamp("2024-01-01"), "EURUSD"] == 1.1038
