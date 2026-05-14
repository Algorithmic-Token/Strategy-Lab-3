"""
Strategy Lab #3 — Falsification Framework for Intraday OHLCV Signals
=====================================================================
Algorithmic Token · ENTER Invest

Implements a five-criterion institutional falsification harness for
intraday OHLCV-based trading signals, based on:

    Mesfin, M. (2026) — "Structural Limits of OHLCV-Based Intraday
    Signals in MNQ Futures: A Systematic Falsification Study"
    arXiv:2605.04004 · q-fin.TR · May 2026

Corroborating reference:
    Garg (2025) — "Interpretable Hypothesis-Driven Trading: A Rigorous
    Walk-Forward Validation Framework for Market Microstructure Signals"
    arXiv:2512.12924 · q-fin.TR

The harness tests any intraday signal function against five institutional
criteria simultaneously: walk-forward validation, minimum T-statistic of
2.0, at least 30 trades per window, positive net return after transaction
costs, and multi-year stability across at least 75% of windows.

This is an experimental algorithm implementation. See risk disclosure at
the bottom of this file and in the accompanying Strategy Lab #3 article:
https://algorithmictoken.substack.com/p/strategy-lab-3-the-signals-that-dont

Dependencies: numpy, pandas, yfinance, scipy
"""

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf


# ---------------------------------------------------------------------------
# Data acquisition — 5-minute OHLCV
# ---------------------------------------------------------------------------

def get_intraday_data(ticker: str = "MNQ=F",
                      period: str = "60d",
                      interval: str = "5m") -> pd.DataFrame:
    """
    Download intraday OHLCV data via yfinance.

    NOTE: yfinance provides a maximum of 60 days of 5-minute data.
    The full 947-day dataset used in Mesfin (2026) requires a paid
    intraday data vendor — Interactive Brokers historical API, Norgate,
    or Refinitiv are the practical options. This function is the
    data layer entry point; substitute your vendor's loader here.

    Parameters
    ----------
    ticker   : str — Yahoo Finance ticker (MNQ=F = Micro E-mini Nasdaq 100)
    period   : str — lookback period string (max '60d' for 5m data)
    interval : str — bar interval ('5m', '15m', '1h')

    Returns
    -------
    pd.DataFrame — OHLCV with DatetimeIndex
    """
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# Regime filter — volatility and volume conditioning
# ---------------------------------------------------------------------------

def compute_regime_filter(df: pd.DataFrame,
                           vol_lookback: int = 20,
                           vol_threshold: float = 0.60,
                           volume_lookback: int = 60,
                           volume_ratio_threshold: float = 1.10) -> pd.Series:
    """
    Identify high-information regimes where OHLCV signals are most likely
    to carry predictive content.

    Based on the conditional applicability finding in Garg (2025,
    arXiv:2512.12924): OHLCV-based microstructure signals require elevated
    information arrival and trading activity to function effectively —
    generating positive returns during high-volatility periods while
    underperforming in stable markets.

    Parameters
    ----------
    df                       : pd.DataFrame — OHLCV data with DatetimeIndex
    vol_lookback             : int   — rolling window for realised vol (days)
    vol_threshold            : float — vol percentile threshold
                                       (0.60 = active only in top 40% vol days)
    volume_lookback          : int   — rolling window for volume baseline (days)
    volume_ratio_threshold   : float — relative volume minimum
                                       (1.10 = at least 10% above rolling avg)

    Returns
    -------
    pd.Series — boolean mask; True = active trading regime
    """
    bars_per_day = 78  # approximate 5-min bars per regular US session

    returns  = df["Close"].pct_change()
    daily_rv = (returns
                .rolling(vol_lookback * bars_per_day)
                .std() * np.sqrt(252 * bars_per_day))

    vol_condition = (daily_rv >
                     daily_rv.rolling(252 * bars_per_day)
                     .quantile(vol_threshold))

    relative_volume  = (df["Volume"] /
                        df["Volume"].rolling(volume_lookback * bars_per_day).mean())
    volume_condition = relative_volume > volume_ratio_threshold

    return vol_condition & volume_condition


# ---------------------------------------------------------------------------
# Signal family — Opening Range Breakout
# ---------------------------------------------------------------------------

def opening_range_breakout_signal(df: pd.DataFrame,
                                   orb_minutes: int = 30) -> pd.Series:
    """
    Opening Range Breakout (ORB) signal on 5-minute bars.

    Enters long when price breaks above the high of the first
    orb_minutes of the session; enters short below the low.
    Returns to flat at session close.

    This is Signal Family #1 in Mesfin (2026) — the most widely
    discussed intraday signal, and the first to fail the harness.
    Documented failure mode: T-statistic < 2.0 after walk-forward
    validation; edge reverses in low-volatility regimes.

    Parameters
    ----------
    df          : pd.DataFrame — OHLCV data with DatetimeIndex
    orb_minutes : int — formation window in minutes (default 30)

    Returns
    -------
    pd.Series — signal values (-1, 0, +1) aligned to df.index
    """
    orb_bars    = orb_minutes // 5
    signal      = pd.Series(0, index=df.index, dtype=int)
    session_day = df.index.normalize()

    for day in session_day.unique():
        day_mask = session_day == day
        day_data = df[day_mask]

        if len(day_data) < orb_bars + 1:
            continue

        orb_high = day_data["High"].iloc[:orb_bars].max()
        orb_low  = day_data["Low"].iloc[:orb_bars].min()
        post_orb = day_data.iloc[orb_bars:]

        for idx, row in post_orb.iterrows():
            if row["Close"] > orb_high:
                signal[idx] = 1
            elif row["Close"] < orb_low:
                signal[idx] = -1
            else:
                signal[idx] = 0

    return signal


# ---------------------------------------------------------------------------
# The Falsification Harness — five institutional criteria
# ---------------------------------------------------------------------------

def run_falsification_harness(df: pd.DataFrame,
                               signal: pd.Series,
                               regime_filter: pd.Series,
                               round_trip_cost_points: float = 2.0,
                               point_value: float = 2.0,
                               formation_days: int = 126,
                               test_days: int = 63,
                               min_trades: int = 30,
                               min_tstat: float = 2.0,
                               stability_threshold: float = 0.75,
                               verbose: bool = True) -> dict:
    """
    Apply Mesfin's (2026) five institutional criteria to any intraday signal.

    The five criteria applied simultaneously in each walk-forward window:

        Criterion 1 — Walk-forward validation
            Signal estimated on formation window, tested on out-of-sample
            window. Minimum 4 non-overlapping windows required.

        Criterion 2 — Statistical significance
            T-statistic of mean trade P&L must exceed min_tstat (default 2.0)

        Criterion 3 — Minimum trade count
            At least min_trades trades per window (default 30).
            Below 30, T-statistic is unreliable regardless of value.

        Criterion 4 — Positive net return after costs
            Net P&L after round_trip_cost_points must be positive.
            For MNQ: 2.0 points = $40 per contract round-trip.

        Criterion 5 — Multi-year stability
            Signal must pass Criteria 2–4 in at least stability_threshold
            fraction of all windows (default 75%).

    Parameters
    ----------
    df                      : pd.DataFrame — OHLCV data
    signal                  : pd.Series   — trade signal (-1, 0, +1)
    regime_filter           : pd.Series   — boolean active-regime mask
    round_trip_cost_points  : float — total round-trip cost in index points
    point_value             : float — dollar value per index point per contract
    formation_days          : int   — training window length (trading days)
    test_days               : int   — test window length (trading days)
    min_trades              : int   — Criterion 3 threshold
    min_tstat               : float — Criterion 2 threshold
    stability_threshold     : float — Criterion 5 pass-rate threshold
    verbose                 : bool  — print per-window diagnostics

    Returns
    -------
    dict with keys:
        window_results  : list of per-window dicts
        overall_verdict : str  — 'PASS' or 'FAIL'
        pass_rate       : float
        n_windows       : int
    """
    bars_per_day    = 78
    cost_per_trade  = round_trip_cost_points * point_value
    prices          = df["Close"]

    # Apply regime filter — zero out signal in inactive regimes
    filtered_signal = signal.copy()
    filtered_signal[~regime_filter] = 0

    # Build non-overlapping walk-forward windows
    unique_days  = pd.Series(df.index.normalize().unique())
    n_days       = len(unique_days)
    window_start = 0
    windows      = []

    while window_start + formation_days + test_days <= n_days:
        form_end = window_start + formation_days
        test_end = form_end + test_days
        windows.append({
            "form_days": unique_days.iloc[window_start:form_end],
            "test_days": unique_days.iloc[form_end:test_end],
        })
        window_start += test_days  # non-overlapping advance

    if len(windows) < 4:
        if verbose:
            print(f"INSUFFICIENT DATA: only {len(windows)} windows available. "
                  f"Need at least 4. Provide more historical data.")
        return {
            "overall_verdict": "INSUFFICIENT DATA",
            "window_results":  [],
            "pass_rate":       0.0,
            "n_windows":       len(windows),
        }

    window_results = []

    for w_idx, window in enumerate(windows):
        test_mask   = df.index.normalize().isin(window["test_days"])
        test_signal = filtered_signal[test_mask]
        test_price  = prices[test_mask]

        # Position and bar-level P&L
        position      = test_signal.shift(1).fillna(0)
        bar_pnl       = position * test_price.diff() * point_value
        trade_changes = test_signal.diff().abs() > 0
        bar_pnl      -= trade_changes.astype(float) * (cost_per_trade / 2)

        # Criterion 3 — minimum trade count
        n_trades = int(trade_changes.sum())
        c3_pass  = n_trades >= min_trades

        # Criterion 2 — T-statistic
        if n_trades >= 2:
            trade_pnls = bar_pnl[trade_changes].dropna().values
            if len(trade_pnls) >= 2 and trade_pnls.std() > 0:
                tstat, _ = stats.ttest_1samp(trade_pnls, 0)
            else:
                tstat = 0.0
            c2_pass = float(tstat) > min_tstat
        else:
            tstat   = 0.0
            c2_pass = False

        # Criterion 4 — positive net return
        net_return = float(bar_pnl.sum())
        c4_pass    = net_return > 0

        window_pass = c2_pass and c3_pass and c4_pass

        result = {
            "window":     w_idx + 1,
            "n_trades":   n_trades,
            "t_stat":     round(float(tstat), 3),
            "net_return": round(net_return, 2),
            "c2_tstat":   c2_pass,
            "c3_trades":  c3_pass,
            "c4_net_ret": c4_pass,
            "pass":       window_pass,
        }
        window_results.append(result)

        if verbose:
            status = "PASS" if window_pass else "FAIL"
            reasons = []
            if not c2_pass:
                reasons.append(f"T={tstat:.2f}<{min_tstat}")
            if not c3_pass:
                reasons.append(f"Trades={n_trades}<{min_trades}")
            if not c4_pass:
                reasons.append(f"Net={net_return:+.1f}")
            reason_str = " | ".join(reasons) if reasons else "all criteria met"
            print(f"  Window {w_idx+1:02d} [{status}]  "
                  f"Trades={n_trades:3d} | T={tstat:5.2f} | "
                  f"Net=${net_return:+8.1f}  —  {reason_str}")

    # Criterion 5 — multi-year stability
    pass_rate       = sum(r["pass"] for r in window_results) / len(window_results)
    c5_pass         = pass_rate >= stability_threshold
    overall_verdict = "PASS" if c5_pass else "FAIL"

    if verbose:
        print()
        print(f"  Windows evaluated  : {len(window_results)}")
        print(f"  Windows passing    : {sum(r['pass'] for r in window_results)}")
        print(f"  Pass rate          : {pass_rate:.1%}  "
              f"(threshold: {stability_threshold:.0%})")
        print(f"  Criterion 5        : {'PASS' if c5_pass else 'FAIL'}")
        print(f"  ── OVERALL VERDICT : {overall_verdict} ──")

    return {
        "window_results":  window_results,
        "overall_verdict": overall_verdict,
        "pass_rate":       pass_rate,
        "n_windows":       len(window_results),
    }


# ---------------------------------------------------------------------------
# Entry point — demo run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Strategy Lab #3 — Falsification Harness")
    print("Algorithmic Token · ENTER Invest")
    print("=" * 60)
    print()
    print("Ticker  : MNQ=F (Micro E-mini Nasdaq 100 Futures)")
    print("Signal  : Opening Range Breakout (30-min formation)")
    print("Costs   : 2.0 points round-trip ($40/contract)")
    print()
    print("NOTE: yfinance provides max 60 days of 5-min data.")
    print("Full 947-day replication requires a paid data vendor.")
    print("With 60 days, walk-forward windows may be insufficient")
    print("for a statistically meaningful verdict.")
    print()

    df = get_intraday_data("MNQ=F", period="60d", interval="5m")

    if df.empty:
        print("No data returned. Check ticker and internet connection.")
    else:
        print(f"Data loaded    : {len(df)} bars over "
              f"{df.index.normalize().nunique()} trading days")

        regime = compute_regime_filter(df)
        signal = opening_range_breakout_signal(df)

        active_pct = regime.mean() * 100
        print(f"Regime active  : {regime.sum()} bars ({active_pct:.1f}%)")
        print()
        print("Running falsification harness...")
        print()

        results = run_falsification_harness(
            df, signal, regime,
            round_trip_cost_points = 2.0,
            point_value            = 2.0,
            formation_days         = 15,   # reduced for 60-day demo dataset
            test_days              = 10,
            min_trades             = 10,   # reduced for demo; use 30 on full data
            verbose                = True,
        )


# ---------------------------------------------------------------------------
# Risk Disclosure
# ---------------------------------------------------------------------------
# The experimental algorithms and implementations in this file are provided
# for educational and research purposes only. Past performance is not
# indicative of future results. All algorithmic trading carries significant
# financial risk, including the potential total loss of capital. Nothing
# here constitutes financial advice. ENTER Invest does not manage client
# funds based on strategies described here unless explicitly contracted.
# ---------------------------------------------------------------------------
