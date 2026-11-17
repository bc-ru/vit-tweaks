FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /tmp

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace

RUN python3 -m pip install --no-cache-dir kaggle

RUN mkdir -p /root/.kaggle