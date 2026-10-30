import pandas as pd

from stock_predictor import data
from stock_predictor.universe import BENCHMARK_SYMBOL, VIX_SYMBOL


def _yahoo_frame(dates, tickers):
    """Same shape as yf.download: columns are (Close|Volume, ticker)."""
    return pd.concat({"Close": pd.DataFrame(100.0, index=dates, columns=tickers),
                      "Volume": pd.DataFrame(1000.0, index=dates, columns=tickers)}, axis=1)


def test_today_only_counts_once_the_market_has_closed():
    before_close = pd.Timestamp("2026-10-28 15:28", tz="Asia/Kolkata")
    after_close = pd.Timestamp("2026-10-28 16:05", tz="Asia/Kolkata")
    assert data._last_complete_day(before_close) == pd.Timestamp("2026-10-27")
    assert data._last_complete_day(after_close) == pd.Timestamp("2026-10-28")


def test_unfinished_day_is_left_out_of_prices(monkeypatch, tmp_path):
    dates = pd.bdate_range("2025-01-01", periods=300)
    raw = _yahoo_frame(dates, ["AAA.NS", BENCHMARK_SYMBOL, VIX_SYMBOL])
    monkeypatch.setattr(data, "_download", lambda symbols, start: raw)
    monkeypatch.setattr(data, "_last_complete_day", lambda: dates[-2])
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)

    close, volume, vix, _ = data.load_prices(["AAA"])

    assert close.index[-1] == volume.index[-1] == vix.index[-1] == dates[-2]


def test_extra_closes_use_plain_symbols_and_skip_unknown_ones(monkeypatch, tmp_path):
    dates = pd.bdate_range("2026-01-01", periods=5)
    raw = _yahoo_frame(dates, ["GONE.NS", "UNKNOWN.NS"])
    raw[("Close", "UNKNOWN.NS")] = float("nan")
    monkeypatch.setattr(data, "_download", lambda symbols, start: raw)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)

    close = data.load_extra_closes(["GONE", "UNKNOWN"], dates[0])

    assert list(close.columns) == ["GONE"]


def test_extra_closes_survive_a_failed_download(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "_download", lambda symbols, start: pd.DataFrame())
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    assert data.load_extra_closes(["GONE"], pd.Timestamp("2026-01-01")).empty
