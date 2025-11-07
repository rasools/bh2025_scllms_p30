# Change the cuda version depending on your GPU
FROM docker.io/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set environment variable to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list
RUN apt-get update -y

# Install git
RUN apt-get install -y git

# Install r-base and tzdata
# RUN apt-get install -y r-base tzdata

# Install Python packages using pip
# RUN pip install packaging
# RUN pip install scgpt "flash-attn<1.0.5"
# RUN pip install markupsafe==2.0.1
# RUN pip install wandb jupyterlab ipykernel notebook

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Optional: Install torch-geometric
# RUN pip install torch-geometric

# If running Jupyter server, can omit
# Expose Jupyter port
EXPOSE 8888

# Run Jupyter server
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
