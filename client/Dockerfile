# Client container — all data stays here
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY client/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared contracts
COPY shared/ shared/

# Copy client code
COPY client/ client/

# Expose FastAPI + Streamlit ports
EXPOSE 8001 8501

# Start both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn client.api.main:app --host 0.0.0.0 --port 8001 & streamlit run client/app/dashboard.py --server.port 8501 --server.address 0.0.0.0"]
