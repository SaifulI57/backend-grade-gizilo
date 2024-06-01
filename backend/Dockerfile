FROM python:3.10

WORKDIR /app

COPY requirements.txt .


RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

RUN pip install uvicorn

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:base_api.app", "--host", "0.0.0.0",  "--port", "8080"]
