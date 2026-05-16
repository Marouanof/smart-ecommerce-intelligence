FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas numpy scikit-learn matplotlib seaborn \
    streamlit plotly httpx python-dotenv

COPY TopKselection/ TopKselection/
COPY dashboard/ dashboard/
COPY agents/ agents/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
