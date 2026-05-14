# Strategy-Lab-3 — Falsification Framework for Intraday OHLCV Signals

**Algorithmic Token · ENTER Invest**

> **Primary academic source:**
> Mesfin, M. (2026) — *Structural Limits of OHLCV-Based Intraday Signals in MNQ Futures: A Systematic Falsification Study*
> [arXiv:2605.04004](https://arxiv.org/abs/2605.04004) · q-fin.TR · May 2026

> **Corroborating source:**
> Garg (2025) — *Interpretable Hypothesis-Driven Trading: A Rigorous Walk-Forward Validation Framework for Market Microstructure Signals*
> [arXiv:2512.12924](https://arxiv.org/abs/2512.12924) · q-fin.TR

Experimental algorithm implementation accompanying the Strategy Lab #3 article published at [Algorithmic Token on Substack](https://algorithmictoken.substack.com/p/strategy-lab-3-the-signals-that-dont).

---

## What This Is

This module implements a **systematic falsification harness** — a testing framework that applies five simultaneous institutional criteria to any intraday OHLCV-based signal. Rather than deriving a working strategy, Strategy Lab #3 derives the rigorous testing infrastructure that any strategy candidate must survive before being taken seriously.

The primary paper tests fourteen OHLCV signal families across 947 trading days of MNQ 5-minute data (2021–2025). **No signal passes all five criteria simultaneously.** The gross edge available to next-bar-open execution is constrained to approximately 0.07–1.50 points per trade — insufficient to overcome a 2-point round-trip transaction cost after walk-forward validation.

This is not a pessimistic result. It is a clarifying one. The harness tells you precisely *which* criterion each signal fails and *why*, pointing toward where a genuine edge might be found.

---

## Repository Structure

```
strategy_lab_03/
├── strategy_lab_03.py   — falsification harness + ORB signal example
└── README.md            — this file
```

---

## Environment Setup and Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step 1 — Clone or download the repository

If you have the full ENTER Invest repository cloned locally:

```bash
cd path/to/your/repo
```

Or download `strategy_lab_03.py` and `README.md` directly from GitHub into a local folder.

### Step 2 — Create a virtual environment (recommended)

Creating an isolated environment prevents dependency conflicts with other Python projects.

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal prompt when the environment is active.

### Step 3 — Install dependencies

```bash
pip install numpy pandas yfinance scipy
```

Full dependency list with tested versions:

| Package | Minimum Version | Purpose |
|---|---|---|
| `numpy` | 1.24.0 | Numerical operations |
| `pandas` | 2.0.0 | Time series and DataFrame handling |
| `yfinance` | 0.2.40 | Market data download |
| `scipy` | 1.11.0 | T-statistic computation |

### Step 4 — Verify installation

```bash
python3 -c "import numpy, pandas, yfinance, scipy; print('All dependencies OK')"
```

Expected output: `All dependencies OK`

### Step 5 — Run the demo

```bash
python3 strategy_lab_03.py
```

This runs the Opening Range Breakout signal through the falsification harness on 60 days of MNQ 5-minute data downloaded automatically from Yahoo Finance.

**Expected output:**
```
============================================================
Strategy Lab #3 — Falsification Harness
Algorithmic Token · ENTER Invest
============================================================

Ticker  : MNQ=F (Micro E-mini Nasdaq 100 Futures)
Signal  : Opening Range Breakout (30-min formation)
Costs   : 2.0 points round-trip ($40/contract)

NOTE: yfinance provides max 60 days of 5-min data.
...
Running falsification harness...

  Window 01 [FAIL]  Trades= 12 | T= 0.84 | Net=$ -24.0  — T=0.84<2.0
  Window 02 [FAIL]  Trades=  9 | T= 1.12 | Net=$ +18.0  — Trades=9<10
  ...
  Pass rate          : 0.0%  (threshold: 75%)
  ── OVERALL VERDICT : FAIL ──
```

The ORB signal is expected to fail — that is the point of this Lab.

---

## Using the Harness With Your Own Signal

The harness accepts any signal function that returns a `pd.Series` of `-1`, `0`, or `+1` values aligned to the DataFrame index.

```python
from strategy_lab_03 import (
    get_intraday_data,
    compute_regime_filter,
    run_falsification_harness,
)

# 1. Load your data
df = get_intraday_data("MNQ=F", period="60d", interval="5m")

# 2. Compute regime filter
regime = compute_regime_filter(df)

# 3. Define your own signal — must return pd.Series of -1 / 0 / +1
def my_signal(df):
    # Example: simple close-above-open momentum
    signal = (df["Close"] > df["Open"]).astype(int)
    signal[df["Close"] < df["Open"]] = -1
    return signal

signal = my_signal(df)

# 4. Run the harness
results = run_falsification_harness(
    df, signal, regime,
    round_trip_cost_points = 2.0,   # adjust to your instrument
    point_value            = 2.0,   # MNQ: $2 per point
    formation_days         = 126,   # 6 months (use 15 for 60-day demo data)
    test_days              = 63,    # 3 months (use 10 for 60-day demo data)
    min_trades             = 30,    # use 10 for 60-day demo data
    verbose                = True,
)

print(results["overall_verdict"])  # 'PASS' or 'FAIL'
print(results["pass_rate"])        # fraction of windows passing
```

---

## The Five Institutional Criteria

Every signal is evaluated against all five criteria simultaneously in each walk-forward window. Passing in isolation is insufficient — all five must be satisfied concurrently.

| Criterion | Parameter | Default | Description |
|---|---|---|---|
| **C1** Walk-forward | `formation_days` / `test_days` | 126 / 63 | Train on formation, test out-of-sample. Non-overlapping windows. |
| **C2** T-statistic | `min_tstat` | 2.0 | Mean trade P&L must be statistically distinguishable from zero |
| **C3** Trade count | `min_trades` | 30 | Minimum trades per window for T-stat reliability |
| **C4** Net return | `round_trip_cost_points` | 2.0 | Positive P&L after full round-trip transaction costs |
| **C5** Stability | `stability_threshold` | 0.75 | Must pass C2–C4 in ≥75% of all windows |

---

## Important Data Limitation

**yfinance provides a maximum of 60 days of 5-minute intraday data.** This is sufficient for the demo and for understanding the harness mechanics, but insufficient to replicate the full Mesfin (2026) study, which uses 947 trading days.

For a meaningful falsification study you need a paid intraday data vendor:

| Vendor | Notes |
|---|---|
| **Interactive Brokers API** | Available if you have an IB account; requres `ib_insync` library |
| **Norgate Data** | Clean, survivorship-bias-free US futures data; Windows only |
| **Refinitiv / LSEG** | Institutional grade; expensive but comprehensive |
| **Alpaca Markets API** | Free tier available for US equities; futures require subscription |

Once you have a data source returning a properly indexed OHLCV DataFrame, substitute it in the `get_intraday_data()` function — the harness itself requires no changes.

---

## Transaction Cost Reference — MNQ Futures

The default `round_trip_cost_points = 2.0` reflects approximate retail execution costs on Micro E-mini Nasdaq 100 futures. Adjust based on your actual broker:

| Cost Component | Points | USD (per contract) |
|---|---|---|
| Bid-ask spread | ~0.75 | ~$1.50 |
| Commission (retail) | ~0.75 | ~$1.50 |
| Slippage (conservative) | ~0.50 | ~$1.00 |
| **Total round-trip** | **~2.0** | **~$4.00** |

Note: 1 MNQ point = $2.00. 1 ES (full E-mini) point = $50.00.

---

## Relationship to Other Strategy Labs

| Lab | Signal Type | Asset Class | Key Mechanism |
|---|---|---|---|
| Lab #1 | Momentum | Equity Futures | Proportional-control vol targeting |
| Lab #2 | Mean Reversion | FX Pairs | Cointegration + Q-learning RL agent |
| **Lab #3** | **Falsification** | **Index Futures** | **Institutional criteria harness** |
| Lab #4 (planned) | Regime-conditioned ORB | Index Futures | Direction 1 from Lab #3 |

---

## Planned Extensions

- [ ] Regime-conditioned ORB test (Strategy Lab #4)
- [ ] Signal combination harness — composite of two or more signal families
- [ ] Tick-level data adapter for order flow imbalance signals
- [ ] Walk-forward result visualisation (equity curve per window)
- [ ] Integration with ENTER Invest backtesting engine

---

## Further Reading

- [arXiv:2605.04004](https://arxiv.org/abs/2605.04004) — Mesfin (2026), primary reference
- [arXiv:2512.12924](https://arxiv.org/abs/2512.12924) — Garg (2025), corroborating walk-forward study
- [Pardo (2008) — The Evaluation and Optimization of Trading Strategies](https://www.amazon.com/Evaluation-Optimization-Trading-Strategies/dp/0470128011) — practitioner textbook on walk-forward validation
- [Aronson (2006) — Evidence-Based Technical Analysis](https://www.amazon.com/Evidence-Based-Technical-Analysis-Scientific-Statistical/dp/0470008741) — statistical treatment of signal testing and the multiple-comparison problem

---

## Risk Disclosure

The experimental algorithms and implementations in this file are provided for educational and research purposes only. Past performance of any modelled strategy is not indicative of future results. All algorithmic trading carries significant financial risk, including the potential total loss of capital. Nothing here constitutes financial advice. ENTER Invest does not manage client funds based on strategies described here unless explicitly contracted to do so.

---

*Algorithmic Token is published by ENTER Invest. [algorithmictoken.substack.com](https://algorithmictoken.substack.com)*
