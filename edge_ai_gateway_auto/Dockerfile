# 1. 確保基礎鏡像正確
ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base-python:3.12-bookworm
FROM $BUILD_FROM

# 2. 安裝執行 ONNX 必備的系統庫 (不裝這個執行會報錯)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. 關鍵修正：強迫 pip 只下載預編譯好的 wheel，絕不現場編譯 (避免記憶體崩潰)
# 同時指定較輕量的版本
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir --prefer-binary \
    numpy \
    joblib \
    paho-mqtt \
    aioesphomeapi \
    psutil \
    onnxruntime

# 4. 複製程式碼
COPY main.py /
WORKDIR /
RUN chmod +x /main.py

CMD [ "python3", "/main.py" ]