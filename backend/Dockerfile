FROM python:3.11-slim
WORKDIR /repo
COPY pyproject.toml .
COPY shared shared
COPY backend backend
RUN pip install --no-cache-dir '.[api]'
USER 10001
EXPOSE 8000
CMD ["python","-m","uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
