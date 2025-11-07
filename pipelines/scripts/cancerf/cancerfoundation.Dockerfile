# Use official Python 3.10 slim image
FROM docker.io/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /workspace

# # Install system dependencies for building some packages
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     curl \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install uv
RUN pip install --upgrade pip

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN pip install jupyterlab ipykernel notebook

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
