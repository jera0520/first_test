# Python 런타임 이미지를 기반으로 합니다.
FROM python:3.9-slim

# 컨테이너의 작업 디렉토리를 설정합니다.
WORKDIR /app

# requirements.txt 파일을 복사하고 의존성을 설치합니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 애플리케이션 코드를 복사합니다.
COPY . .

# Gunicorn을 사용하여 애플리케이션을 실행합니다.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
