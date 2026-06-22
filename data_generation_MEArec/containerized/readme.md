# mearecont
## Overview
containerized pipeline using MEArec and Allen Brain Atlas for generating insilico MEA data

## Build container
To build the Docker container, run the following command in your terminal:
```bash
docker build -t mearecont:latest .
```
## Run container
To run the Docker container, run the following command in your terminal:
```bash
docker run -v ./:/app/ mearecont:latest python src/allen_workflow.py
```
