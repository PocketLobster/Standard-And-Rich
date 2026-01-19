#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
from dotenv import load_dotenv
load_dotenv()
# --- Secure Key Loading ---
# os.getenv(KEY_NAME, DEFAULT_VALUE)
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("API Keys not found! Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your environment.")
else:
    print('time to focus')


# In[ ]:





# In[ ]:


#REAL-TIME 1: MONITOR PRICES
#REAL_TIME 2: CONFIGURE STATIC/DYNAMIC SPLIT AND PREDICT FXN
# Add this to the top of your real-time cell (before importing asyncio or alpaca)
#!pip install nest-asyncio
import nest_asyncio
nest_asyncio.apply()
import asyncio
import datetime as dt
import pandas as pd
import numpy as np
import joblib
import re
import time
from alpaca.data.live.stock import StockDataStream
import os
# ==========================================
# 1. THE REAL-TIME ENGINE CLASS
# ==========================================
class TradingModelRealTime:
    def __init__(self, model_pkl_path, static_features_csv_path):
        self.data = joblib.load(model_pkl_path)
        self.model = self.data['model']
        self.feature_order = self.data['features']
        self.symbol = self.data['symbol']
        self.dynamic_map = self._parse_feature_requirements()
        
        # Load the "Baseline" (The last row of your weekly CSV)
        full_df = pd.read_csv(static_features_csv_path)
        self.baseline_row = full_df.iloc[-1].to_dict()

    def _parse_feature_requirements(self):
        dynamic_map = []
        cl_pattern = r"change_lag_(\d+)(\d+)th_ptile_(.*)"
        zs_pattern = r"z_score_(\d+)_(.*)"
        for feat in self.feature_order:
            m_cl = re.match(cl_pattern, feat)
            m_zs = re.match(zs_pattern, feat)
            if m_cl:
                lag, ptile, ticker = m_cl.groups()
                dynamic_map.append({'feature': feat, 'type': 'cl', 'ticker': ticker,
                                  'ancillary': [f'{ptile}th_ptile_lag_{lag}days_{ticker}']})
            elif m_zs:
                lag, ticker = m_zs.groups()
                dynamic_map.append({'feature': feat, 'type': 'zs', 'ticker': ticker,
                                  'ancillary': [f'rolling_mean_{lag}_{ticker}', f'rolling_std_{lag}_{ticker}']})
        return dynamic_map

    def predict(self, live_prices):
        spy_price = live_prices.get('SPY')
        if not spy_price: return None
        
        input_vector = {}
        # Fill static features
        for feat in self.feature_order:
            input_vector[feat] = self.baseline_row.get(feat, 0)

        # Calculate dynamic features
        for item in self.dynamic_map:
            ticker = item['ticker']
            if ticker in live_prices:
                rel_price = live_prices[ticker] / spy_price
                if item['type'] == 'cl':
                    thresh = self.baseline_row[item['ancillary'][0]]
                    input_vector[item['feature']] = (rel_price - thresh) / thresh
                elif item['type'] == 'zs':
                    mu = self.baseline_row[item['ancillary'][0]]
                    sigma = self.baseline_row[item['ancillary'][1]]
                    input_vector[item['feature']] = (rel_price - mu) / sigma if sigma != 0 else 0

        X = pd.DataFrame([input_vector])[self.feature_order]
        return self.model.predict_proba(X)[0][1]

# ==========================================
# 2. GLOBAL CONFIG & INITIALIZATION
# ==========================================
import os

# --- Secure Key Loading ---
# os.getenv(KEY_NAME, DEFAULT_VALUE)
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
MODEL_PATH ='models/champion_model_DIOD_[0.88221709 0.5414673 ].pkl'
CSV_PATH = 'feature_presentation_semiconductors.csv'
PREDICTION_INTERVAL = 10  # Run prediction every 10 seconds

# Global dictionary to store incoming prices
live_prices = {}

# Initialize Engine
engine = TradingModelRealTime(MODEL_PATH, CSV_PATH)

# Auto-generate Watchlist from Model Features
base_name = os.path.basename(MODEL_PATH) # Gets 'champion_model_DIOD_85.pkl'
target_stock = base_name.split('_')[2]
watchlist = set(f.split('_')[-1] for f in engine.feature_order)
watchlist.update([target_stock, 'SPY'])
WATCHLIST = list(watchlist)

# ==========================================
# 3. ASYNC TASKS
# ==========================================
async def on_quote_update(data):
    """Updates the price dictionary as fast as quotes arrive."""
    if data.bid_price > 0 and data.ask_price > 0:
        live_prices[data.symbol] = (data.bid_price + data.ask_price) / 2

async def inference_loop():
    """Calculates predictions on a fixed timer (N seconds)."""
    print(f"[INFO] Inference loop started (Interval: {PREDICTION_INTERVAL}s)")
    while True:
        # 1. Ensure we have SPY and at least some data before predicting
        if 'SPY' in live_prices and len(live_prices) > 1:
            try:
                prob = engine.predict(live_prices)
                print(prob)
                timestamp = dt.datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp}] Model: {engine.symbol} | Signal Probability: {prob:.4f}")
                
                # Logic for trading:
                if prob > 0.85:
                    print(f"*** HIGH CONVICTION SIGNAL DETECTED FOR {engine.symbol} ***")
                    
            except Exception as e:
                print(f"[ERROR] Prediction failed: {e}")
        else:
            print(f"[WAIT] Waiting for more data... (Current count: {len(live_prices)}/{len(WATCHLIST)})")
            
        await asyncio.sleep(PREDICTION_INTERVAL)

async def main():
    while True:
        inference_task = None
        try:
            # 1. Start Fresh
            stream = StockDataStream(API_KEY, SECRET_KEY)

            # 2. Subscribe
            for symbol in WATCHLIST:
                stream.subscribe_quotes(on_quote_update, symbol)
            
            print(f"[START] Attempting connection at {dt.datetime.now()}")

            # 3. Start the Inference Loop as a background task
            inference_task = asyncio.create_task(inference_loop())
            
            # 4. Run the stream (This is the "main" blocking call)
            # We await this directly. If it fails, it jumps to 'except'
            await stream.run()

        except Exception as e:
            print(f"[RECONNECT] Error caught in main loop: {e}")
            
            # 5. Mandatory Cleanup
            if inference_task:
                inference_task.cancel()
                try:
                    await inference_task
                except asyncio.CancelledError:
                    pass
            
            print("Retrying in 5 seconds...")
            await asyncio.sleep(5)

# --- JUPYTER EXECUTION ---
# If you are in a notebook, just run this:
await main()

