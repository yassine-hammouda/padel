FROM python:3.12-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY api.py .
COPY data/ ./data/
COPY models/ ./models/

RUN mkdir -p outputs

EXPOSE 5000

CMD ["python", "api.py"]