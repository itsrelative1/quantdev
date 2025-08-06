"""Unit-test voor sharpe()."""
import pandas as pd
from scripts.ma_backtest import sharpe


def test_sharpe_infinite_if_zero_vol():
    # 10 dagen +1 % rendement → σ = 0
    r = pd.Series([0.01] * 10, name="r")
    assert sharpe(r) == float("inf")
