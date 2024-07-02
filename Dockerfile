# Use Nvidia CUDA base image
FROM ubuntu:23.10

ARG SDXL_MODEL=https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt update
RUN apt upgrade --yes
RUN apt install --yes git curl python3 pip ffmpeg libsm6 libxext6 openssh-server htop

# Clean up to reduce image size
RUN apt autoremove --yes
RUN apt clean --yes
RUN rm -rf /var/lib/apt/lists/*

# Set pip defaults
RUN pip config set global.break-system-packages true
RUN pip config set global.no-cache-dir true

# Install core Python dependencies
RUN pip install torch torchvision torchaudio 
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt scikit-image scikit-learn opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
# Some are machine-code-based and might take a long time to compile so we put them on their own layer
RUN pip install -c constraints.txt xformers mmcv safetensors
# Some have specific dependency versions and have to be installed early on, without constraints
RUN pip install mediapipe
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt

# Install secondary Python dependencies
RUN pip install -c constraints.txt einops transformers kornia ultralytics segment_anything openmim mmdet mmengine fvcore
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt ftfy svglib piexif trimesh[easy] pillow-jxl-plugin pillow-avif-plugin torchsde numba spandrel
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt runpod aiohttp cachetools cmake PyGithub GitPython pyyaml psutil omegaconf
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt

# Set environment variables
ENV COMFYUI_PATH=/workspace/ComfyUI
ENV COMFYUI_MODEL_PATH=$COMFYUI_PATH/models
