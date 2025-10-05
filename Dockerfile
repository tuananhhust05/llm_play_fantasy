# Sử dụng base image Python
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các biến môi trường cần thiết cho việc build llama-cpp-python
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=OFF -DLLAMA_HIPBLAS=OFF -DLLAMA_CLBLAST=OFF"
ENV FORCE_CMAKE=1

# Sao chép file requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép code của ứng dụng vào image
COPY ./app /app

# Mở port 8000 để bên ngoài có thể truy cập
EXPOSE 8000

# Lệnh để chạy ứng dụng khi container khởi động
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]