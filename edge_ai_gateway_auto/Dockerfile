ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base-python:3.12-bookworm
FROM $BUILD_FROM

# 合併更新與安裝，減少鏡像層並清理緩存
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir --prefer-binary \
    numpy joblib paho-mqtt aioesphomeapi psutil onnxruntime

COPY main.py /
WORKDIR /
RUN chmod +x /main.py

CMD [ "python3", "/main.py" ]