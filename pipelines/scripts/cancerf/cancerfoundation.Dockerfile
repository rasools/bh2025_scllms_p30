# Use official Python 3.10 slim image
FROM docker.io/python:3.10-slim

# Set working directory
WORKDIR /workspace

# Upgrade pip and install uv
RUN pip install --upgrade pip

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN pip install jupyterlab ipykernel

# If running Jupyter server, can omit
# Expose Jupyter port
EXPOSE 8888

# Run Jupyter server
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
