
FROM python:3.10-slim

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential libgl1 libglib2.0-0 && \
#     rm -rf /var/lib/apt/lists/*

# 拿掉build-essential
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements.txt ./
COPY static /app/static
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
