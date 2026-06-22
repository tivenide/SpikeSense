# Use the official Python slim image as a base
FROM python:3.13.9-slim

# Keep Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turn off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y gcc g++ make\
    && rm -rf /var/lib/apt/lists/*

# Install pip requirements
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY ./src /app/src

# copy custom probe design to folder
COPY SqMEA-8-200-60.yaml /usr/local/lib/python3.13/site-packages/MEAutility/electrodes/.

# Create a non-root user
# RUN adduser -u 1000 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser


# Set the command to run your application
CMD ["python", "src/allen_workflow.py"]

