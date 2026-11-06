FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY phase1_data_windowing.py /app/
COPY phase2_lstm_forecaster.py /app/
COPY phase3_dynamic_thresholding.py /app/
COPY phase4_fleet_benchmark_and_api.py /app/

EXPOSE 8000
CMD ["uvicorn", "phase4_fleet_benchmark_and_api:app", "--host", "0.0.0.0", "--port", "8000"]
