FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel AS builder

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# Set pip defaults
RUN pip config set global.break-system-packages true

# Install Python, git and other necessary tools
RUN apt update
RUN apt upgrade --yes
RUN apt install --yes python3-virtualenv

# Create venv
RUN python -m venv --system-site-packages /venv
ENV PATH="/venv/bin:$PATH"

# Install core Python dependencies
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt xformers --index-url https://download.pytorch.org/whl/cu121
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt scikit-image scikit-learn opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt mmcv
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt safetensors pytorch_lightning accelerate
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt open-clip-torch

# Some have specific dependency versions and have to be installed early on, without constraints
RUN pip install mediapipe

# Install secondary Python dependencies
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt einops transformers kornia ultralytics segment_anything openmim mmdet mmengine fvcore
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt ftfy svglib piexif trimesh[easy] pillow-jxl-plugin pillow-avif-plugin torchsde numba spandrel
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt runpod aiohttp cachetools cmake PyGithub GitPython pyyaml psutil omegaconf simpleeval
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt matrix-client
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Set pip defaults
RUN pip config set global.break-system-packages true
RUN pip config set global.no-cache-dir true

# Install Python, git and other necessary tools
RUN apt update && apt upgrade --yes && apt install --yes git curl ffmpeg libsm6 libxext6 openssh-server htop python3-virtualenv && apt autoremove --yes && apt clean --yes && rm -rf /var/lib/apt/lists/*

# Copy venv from builder layers
COPY --from=builder /venv /venv

# Set environment variables
ENV COMFYUI_PATH=/workspace/ComfyUI
ENV COMFYUI_MODEL_PATH=$COMFYUI_PATH/models
ENV PATH="/venv/bin:$PATH"