import asyncio
import json
import time
import threading
from collections import deque
import psutil
import os
import numpy as np
import joblib
import paho.mqtt.client as mqtt
import onnxruntime as ort

# 檔案路徑
MODEL_ONNX_PATH   = "/share/edge_ai_gateway/ae_model.onnx"
MODEL_SCALER_PATH = "/share/edge_ai_gateway/scaler.joblib"
CONF_PATH         = "/share/edge_ai_gateway/runtime_config.json"

FEATURE_COLS = ["temperature", "humidity", "mq5", "dust_ratio"]
WINDOW_SIZE = 10 

STATE = {
    "buffer": deque(maxlen=WINDOW_SIZE),
    "alarm": False,
    "last_change": 0.0,
    "ae_threshold": 0.015,
    "total_count": 0
}

LATEST_SENSOR_DATA = {col: 0.0 for col in FEATURE_COLS}
models = {"session": None, "scaler": None}

def load_ai_models():
    print(f"[BOOT] 載入 ONNX 模型與 Scaler...", flush=True)
    models["scaler"] = joblib.load(MODEL_SCALER_PATH)
    models["session"] = ort.InferenceSession(MODEL_ONNX_PATH)
    print(f"[BOOT] 載入完成，模型預期輸入維度: {models['session'].get_inputs()[0].shape}", flush=True)

def extract_features_18(window_data):
    """
    將滑動視窗轉換為 18 維特徵
    請根據你訓練時的邏輯調整這裡的組合成 18 維
    """
    window = np.array(window_data, dtype=np.float64)
    curr_val = window[-1]        # 4 維
    win_mean = np.mean(window, axis=0) # 4 維
    win_std = np.std(window, axis=0)   # 4 維
    win_trend = window[-1] - window[0] # 4 維
    
    # 這裡目前共 16 維，請檢查剩下的 2 維是什麼 (例如最後一秒的斜率或特定比值)
    # 範例：補上兩個常數或額外計算以符合 18 維
    ext_1 = np.array([curr_val[0] / (curr_val[1] + 1e-6)]) # 假設特徵 17
    ext_2 = np.array([curr_val[2] - win_mean[2]])          # 假設特徵 18
    
    return np.concatenate([curr_val, win_mean, win_std, win_trend, ext_1, ext_2]).reshape(1, -1)

def infer_autoencoder(window_data):
    # 1. 提取 18 維特徵
    features_18 = extract_features_18(window_data)
    
    # 2. 縮放
    data_scaled = models["scaler"].transform(features_18).astype(np.float32)

    # 3. ONNX 推論
    input_name = models["session"].get_inputs()[0].name
    reconstructed = models["session"].run(None, {input_name: data_scaled})[0]

    # 4. 計算誤差
    loss = np.mean((data_scaled - reconstructed) ** 2)
    is_anomaly = loss > STATE["ae_threshold"]
    
    return is_anomaly, loss

# ... MQTT 與 Loop 邏輯保持不變 ...