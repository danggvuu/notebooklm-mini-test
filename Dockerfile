# Sử dụng image Python chuẩn và nhẹ
FROM python:3.10-slim

# Ngăn Python tạo ra các file .pyc và bật stdout logging ngay lập tức
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cài đặt các thư viện hệ thống cần thiết (nếu có thư viện c++ compiling)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirement và cài đặt dependencies trước để tận dụng Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . .

# Tạo sẵn các thư mục data và storage
RUN mkdir -p data storage/qdrant storage/bm25

# Phân quyền cho user không root (Best practice về security)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports: 8000 cho FastAPI, 8501 cho Streamlit
EXPOSE 8000 8501

# Chạy bằng script start.sh hoặc chỉ định CMD mặc định
# CMD sẽ được override trong docker-compose.yml để chạy API hoặc UI
CMD ["uvicorn", "src.interfaces.api:app", "--host", "0.0.0.0", "--port", "8000"]
