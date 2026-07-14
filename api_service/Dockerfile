# Build from ML_CJ_stage19: docker build -f api_service/Dockerfile -t rrai-ml-service .
FROM python:3.13-slim
WORKDIR /app
COPY api_service/requirements.runtime.txt ./requirements.runtime.txt
COPY api_service/requirements.txt ./requirements.api.txt
RUN pip install --no-cache-dir -r requirements.runtime.txt -r requirements.api.txt
COPY . .
ENV PORT=7860 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api_service.app:app --host 0.0.0.0 --port ${PORT}"]
