FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ../src/ ./src/
COPY ../configs/ ./configs/
COPY ../scripts/ ./scripts/
COPY run.sh .

CMD ["bash", "run.sh"]
