# Base image nhẹ, đủ tương thích với scikit-learn/scikit-image/opencv
FROM python:3.12-slim

WORKDIR /app

# Thư viện hệ thống cần cho opencv-python-headless chạy ổn định trên image slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Cài Python dependencies trước để tận dụng cache layer của Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + model + template
COPY app.py .
COPY pipeline_catdog_model_final.pkl .
COPY templates/ templates/

# Render (và hầu hết nền tảng cloud) sẽ set biến môi trường PORT lúc runtime,
# container PHẢI lắng nghe đúng cổng đó thay vì hardcode 5000.
ENV PORT=5000
EXPOSE 5000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} app:app"]
