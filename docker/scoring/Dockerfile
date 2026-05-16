FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas numpy scikit-learn

COPY TopKselection/ TopKselection/

CMD ["python", "-m", "TopKselection.scoring"]
