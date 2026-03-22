# Provider backend — GPT-4o orchestration only, no data storage
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared contracts
COPY shared/ shared/

# Copy server code
COPY server/ server/

EXPOSE 8000

CMD ["uvicorn", "server.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
