# Standard-And-Rich

"Standard and Rich" is a quantitative finance project that implements an end-to-end machine learning pipeline for semiconductor price prediction. The system utilizes **Asynchronous Python** to bridge the gap between historical "Relative Strength" research and live, sub-second market data execution.

## 🚀 Executive Summary
* **Core Strategy:** Benchmarks specific semiconductor tickers (DIOD, CRUS, etc.) against the market (SPY) to identify alpha-generating deviations.
* **Infrastructure:** Built with `asyncio` to handle high-frequency WebSocket streams from Alpaca Markets.
* **Feature Engineering:** Dynamically calculates Z-scores and percentile-based price lags by normalizing live quotes against a historical baseline library.

## 🧠 Technical Architecture
The system is divided into a research pipeline and a production engine to demonstrate a full-cycle engineering approach:

### 1. The Research Pipeline (`Pipeline_and_Research.ipynb`)
Handles heavy data lifting and model validation:
* **Data Ingestion:** Fetches 2+ years of minute-level bar data via Alpaca Historical API.
* **Relative Normalization:** Transforms absolute prices into relative pricing ratios ($Price_{Ticker} / Price_{SPY}$).
* **Feature Library:** Generates a "Baseline" CSV at the 15-minute level containing rolling means and standard deviations ($\mu, \sigma$) for every ticker in the semiconductor sector.
* **Unique Targets:** Creates target variables as high-confidence upsides over the next 15 market days, reoresented as the 85th- and 15th- percentiles over the next 15 market days
* **ML Training:** Trains Gradient Boosting classifiers to predict high-probability price movements based on lagging relative price percentiles, volatility. RSI and other metrics

### 2. The Production Engine (`Realtime_Inference.py`)
A standalone Python script designed for 24/7 reliability:
* **Async Stream:** Manages a persistent WebSocket connection with automatic exponential backoff for reconnection.
* **Live Normalization:** Uses the formula: 
    $\text{Z}_{\text{live}} = \frac{(Price_{ticker} / Price_{SPY}) - \mu_{baseline}}{\sigma_{baseline}}$$
* **Feature Breadth** In addition to live features, those that are static are stored to be retrieved if needed
* **Signal Generation:** Evaluates model probabilities every 10 seconds and triggers high-conviction alerts (>85% probability).


## 📁 Repository Structure
* `standard_and_rich_500.ipynb`: Detailed notebook covering feature selection, training, and the baseline "GGP" table logic.
* `real_time_inference_SANDR.py`: Clean, modular Python script for live deployment.
* `models/`: Serialized `.pkl` files containing trained model weights and feature requirements.
* `data/`: The feature baseline CSV serving as the "historical anchor" for live calculations.

## 🛠 Setup
1.  **Environment:** `pip install alpaca-py pandas joblib nest-asyncio`
2.  **API Keys:** Configure `API_KEY` and `SECRET_KEY` in `Realtime_Inference.py`.
3.  **Run:** Execute the inference script to begin monitoring the semiconductor sector.
