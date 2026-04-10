FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev gcc zip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN touch dummy.txt && zip dummy_archive.zip dummy.txt && rm dummy.txt
RUN mkdir -p uploads

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]