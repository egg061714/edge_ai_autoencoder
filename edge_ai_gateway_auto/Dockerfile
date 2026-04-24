ARG BUILD_FROM
FROM $BUILD_FROM

# 安裝系統依賴 (PyTorch 需要一些 C++ 庫)
RUN apk add --no-cache \
    python3 \
    py3-pip \
    python3-dev \
    gcc \
    g++ \
    make \
    libffi-dev \
    openssl-dev \
    openblas-dev

# 安裝 Python 庫
# 注意：在 Raspberry Pi 上直接 pip install torch 可能會很久甚至失敗
# 建議指定輕量化版本或確保有足夠的 Swap 空間
RUN pip3 install --no-cache-dir \
    numpy \
    joblib \
    paho-mqtt \
    aioesphomeapi \
    psutil \
    # torch --index-url https://download.pytorch.org/whl/cpu

COPY main.py /
WORKDIR /
CMD [ "python3", "/main.py" ]