FROM python:3.9-slim-buster

RUN mkdir /app  
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip cache purge

COPY model.keras ./ 
COPY inference_api.py . 

EXPOSE 8501

CMD ["python", "inference_api.py"]