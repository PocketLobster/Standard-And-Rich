# Standard-And-Rich

**"Standard and Rich"** is an end-to-end quantitative trading system for semiconductor stocks. It pulls minute-level data via the Alpaca API, normalizes relative strength to SPY, engineers 100+ lagged features (percentiles, volatility, RSI, earnings proximity, session/time-of-day interactions), trains gradient-boosted models to predict the 85th and 15th percentiles of future relative price moves, and runs a live inference + execution engine.

## Core Idea
Identify high-conviction alpha in semiconductor tickers (AMD, NVDA, TSM, etc.) by forecasting **high-confidence upside/downside ranges** relative to the broader market (SPY) over the next N market days.

## Repository Structure
- `standard_and_rich_500.ipynb` — Full research pipeline: Alpaca data ingestion, 15-min aggregation, feature engineering (rolling percentiles, Z-scores, RSI divergence, earnings interactions), model training, and daily feature table generation (`ggp_rel_tables/` + `daily_feature_pres/`).
- `real_time_inference_for_SANDR_LIVE.ipynb` — Production live engine: loads trained models + static baseline CSV, runs async Alpaca WebSocket stream, real-time feature updates, and a complete trading bot (entry/exit logic, stop-loss/take-profit, penalty box, rebalancing).
- `models/` — Trained gradient-boosted model pickles.
- `ggp_rel_tables/` & `daily_feature_pres/` — Generated relative-price tables and daily "feature presentations".
- `trading_config.json` — Auto-generated watchlist and model paths.
- `requirements.txt` — All dependencies (Alpaca, pandas, joblib, yfinance, etc.).

## Quick Start (Research Pipeline)
1. `pip install -r requirements.txt`
2. Add your Alpaca keys to a `.env` file (`ALPACA_API_KEY` / `ALPACA_SECRET_KEY`).
3. Open `standard_and_rich_500.ipynb` and run top-to-bottom to regenerate feature tables and config.
4. (Optional) Run the live notebook for real-time inference + paper trading.

**Note**: The live trading loop is fully functional in the notebook (paper trading by default). It includes position tracking, risk limits, and automatic hedging.

**Work in progress** — Updated March 2026. Code, data files, and README refreshed for clarity. Repo now matches the current implementation (research + live execution).

---

**About the project**  
This repo demonstrates time-series feature engineering, percentile forecasting, and real-time inference with risk controls.
