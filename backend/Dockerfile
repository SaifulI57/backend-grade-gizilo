FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install tensorflow
RUN pip install -r requirements.txt

RUN pip install uvicorn

COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:base_api.app", "--host", "0.0.0.0",  "--port", "8080"]