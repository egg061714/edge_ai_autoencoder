ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base-python:3.12-bookworm
FROM $BUILD_FROM

# 安裝必要系統套件
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 安裝輕量化依賴項
RUN pip3 install --no-cache-dir \
    numpy \
    joblib \
    paho-mqtt \
    aioesphomeapi \
    psutil \
    onnxruntime 

# 複製程式碼
COPY main.py /
WORKDIR /
chmod +x /main.py

CMD [ "python3", "/main.py" ]