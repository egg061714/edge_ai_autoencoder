ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base-python:3.12-bookworm
# 如果你是用 Raspberry Pi 3 (32位元)，請改用: 
# ARG BUILD_FROM=ghcr.io/home-assistant/armv7-base-python:3.12-bookworm

FROM $BUILD_FROM

# 安裝系統依賴 (ONNX 運行所需的共用庫)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 庫 (移除所有編譯負擔重的庫，只留執行必要的)
RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    joblib \
    paho-mqtt \
    aioesphomeapi \
    psutil \
    onnxruntime

# 複製程式碼
COPY main.py /
WORKDIR /
chmod +x /main.py

CMD [ "python3", "app/main.py" ]