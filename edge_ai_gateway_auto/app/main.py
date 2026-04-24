import asyncio
import json
import time
import threading
import traceback
from typing import Optional
from collections import deque
import psutil
import os

import numpy as np
import joblib
import paho.mqtt.client as mqtt
from aioesphomeapi import APIClient
# import torch  # 假設使用 PyTorch
import torch.nn as nn

# =========================================
# 1. 檔案路徑與全域設定
# =========================================
MODEL_AE_PATH    = "/share/edge_ai_gateway/pi_model_auto_ae_weights.pth"        # Autoencoder 模型權重
MODEL_SCALER_PATH = "/share/edge_ai_gateway/pi_model_auto_scaler.joblib"      # 訓練時的 MinMaxScaler 或 StandardScaler
CONF_PATH        = "/share/edge_ai_gateway/runtime_config.json"

# 感測器映射順序
FEATURE_COLS = ["temperature", "humidity", "mq5", "dust_ratio"]
WINDOW_SIZE = 1  # 如果你的 AE 是針對單一時間點設計，設為 1；若是 LSTM-AE 則視訓練長度而定

LATEST_SENSOR_DATA = {
    "temperature": 25.0,
    "humidity": 50.0,
    "mq5": 0.0,
    "dust_ratio": 0.0
}

STATE = {
    "buffer": deque(maxlen=WINDOW_SIZE),
    "alarm": False,
    "last_change": 0.0,
    "switch_key": None,
    "total_count": 0,
    "ae_threshold": 0.05  # 重構誤差門檻值，稍後可從 config 或模型載入
}

models = {"ae": None, "scaler": None}

# =========================================
# 2. 模型結構定義 (需與訓練時一致)
# =========================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# =========================================
# 3. 工具函式
# =========================================
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_ai_models():
    print(f"[BOOT] 載入 Autoencoder 模型中...", flush=True)
    # 載入 Scaler
    models["scaler"] = joblib.load(MODEL_SCALER_PATH)
    
    # 初始化並載入 AE 模型
    input_dim = len(FEATURE_COLS)
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load(MODEL_AE_PATH, map_location=torch.device('cpu')))
    model.eval() # 切換至推論模式
    models["ae"] = model
    
    # 這裡可以手動設定門檻，或從外部 json 讀取
    STATE["ae_threshold"] = 0.015 
    print(f"[BOOT] 模型載入完成！輸入維度: {input_dim}, 門檻值: {STATE['ae_threshold']}", flush=True)

def get_system_usage():
    process = psutil.Process(os.getpid())
    cpu_usage = psutil.cpu_percent(interval=None)
    ram_usage = process.memory_info().rss / (1024 * 1024)
    system_ram = psutil.virtual_memory().percent
    return cpu_usage, ram_usage, system_ram

# =========================================
# 4. 邊緣推論核心 (Autoencoder 版本)
# =========================================
def infer_autoencoder(current_vals):
    """
    計算重構誤差並判定異常
    """
    # 1. 資料預處理 (Scaling)
    data_raw = np.array(current_vals).reshape(1, -1)
    data_scaled = models["scaler"].transform(data_raw)
    input_tensor = torch.FloatTensor(data_scaled)

    # 2. 模型推論
    with torch.no_grad():
        reconstructed = models["ae"](input_tensor)
        # 計算 MSE (Mean Squared Error) 作為異常分數
        loss = torch.mean((input_tensor - reconstructed) ** 2).item()
    
    is_anomaly = loss > STATE["ae_threshold"]
    
    # 3. 根因分析 (找重構誤差最大的維度)
    if is_anomaly:
        errors = torch.abs(input_tensor - reconstructed).numpy().flatten()
        reason_sensor = FEATURE_COLS[np.argmax(errors)]
        return True, reason_sensor, loss
    
    return False, None, loss

async def handle_sensor_data(current_vals: list, conf: dict):
    try:
        # AE 推論
        is_anomaly, reason_sensor, score = infer_autoencoder(current_vals)
        print(f"[AI 推論] 分數: {score:.4f}, 異常: {is_anomaly}, 根因: {reason_sensor}", flush=True)

        # 防抖邏輯
        hold = float(conf.get("hold_seconds", 5))
        now = time.time()
        in_hold = (now - STATE["last_change"]) < hold
        
        # 狀態決策
        should_alarm = is_anomaly if not STATE["alarm"] else not (not is_anomaly and not in_hold)

        if should_alarm != STATE["alarm"]:
            target = "General Alarm"
            if reason_sensor == "mq5": target = "Gas Valve"
            elif reason_sensor in ["temperature", "humidity"]: target = "Fan Relay"

            # 實際部署時取消註解
            # await esphome_set_switch(conf["esphome"], target, should_alarm)
            
            STATE["alarm"] = should_alarm
            STATE["last_change"] = now
            print(f"[ACTION] 狀態改變: {should_alarm} (原因: {reason_sensor})", flush=True)

    except Exception as e:
        print(f"[ERROR] 推論崩潰: {repr(e)}", flush=True)

async def periodic_inference_loop(conf: dict):
    interval = float(conf.get("inference_interval_seconds", 5.0))
    while True:
        await asyncio.sleep(interval)
        cpu_usage, ram_usage, system_ram = get_system_usage()
        print(f"[PERF] CPU: {cpu_usage}% | AI RAM: {ram_usage:.2f}MB | Sys RAM: {system_ram}%", flush=True)
        # 僅在有資料時推論
        if STATE["total_count"] > 0:
            current_vals = [float(LATEST_SENSOR_DATA[col]) for col in FEATURE_COLS]
            await handle_sensor_data(current_vals, conf)

# =========================================
# 5. MQTT 與啟動 (保持與原模板一致)
# =========================================
def main():
    print("[BOOT] Autoencoder Edge AI Gateway starting...", flush=True)
    conf = load_json(CONF_PATH)
    load_ai_models()

    loop = asyncio.new_event_loop()
    def runner():
        asyncio.set_event_loop(loop)
        loop.run_forever()
    threading.Thread(target=runner, daemon=True).start()
    asyncio.run_coroutine_threadsafe(periodic_inference_loop(conf), loop)

    mqtt_conf = conf["mqtt"]
    topic = str(mqtt_conf["topic"]).strip()

    def on_connect(client, userdata, flags, rc):
        print(f"[MQTT] 已連線: {topic}", flush=True)
        client.subscribe(topic)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            updated = False
            for k, v in payload.items():
                if k in LATEST_SENSOR_DATA:
                    LATEST_SENSOR_DATA[k] = float(v)
                    updated = True
            if updated:
                STATE["total_count"] += 1
        except:
            pass

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    if mqtt_conf.get("username"):
        c.username_pw_set(mqtt_conf["username"], mqtt_conf.get("password"))
    c.on_connect = on_connect
    c.on_message = on_message
    c.connect(mqtt_conf["broker"], int(mqtt_conf.get("port", 1883)))
    c.loop_forever()

if __name__ == "__main__":
    main()