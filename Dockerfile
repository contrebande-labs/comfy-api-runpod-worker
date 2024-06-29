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
RUN apt install --yes git curl python3 pip ffmpeg libsm6 libxext6

# Clean up to reduce image size
RUN apt autoremove --yes
RUN apt clean --yes
RUN rm -rf /var/lib/apt/lists/*

# Set pip defaults
RUN pip config set global.break-system-packages true
RUN pip config set global.no-cache-dir true

# Install Comfy UI
RUN git clone https://github.com/contrebande-labs/ComfyUI /comfyui

# Include base SDXL model
RUN cd /comfyui/models/checkpoints && curl -LO $SDXL_MODEL

# Include detailing lora models
RUN cd /comfyui/models/loras && curl -LO https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors

# Include upscaling models
RUN cd /comfyui/models/upscale_models && curl -LO https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth

# Include detaling preprocessor models
RUN mkdir -p /comfyui/models/sams
RUN cd /comfyui/models/sams && curl -L https://huggingface.co/facebook/sam-vit-huge/blob/main/pytorch_model.bin -o sam_vit_h_4b8939.pth
RUN mkdir -p /comfyui/models/ultralytics/segm
RUN cd /comfyui/models/ultralytics/segm && curl -LO https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt
RUN cd /comfyui/models/controlnet/ && curl -L https://huggingface.co/xinsir/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors -o controlnet-depth-sdxl-1.0-xinsir.safetensors
RUN mkdir -p /comfyui/models/onnx

# Install python dependencies
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
RUN pip install xformers einops transformers safetensors psutil kornia pillow-avif-plugin
RUN pip install scikit-image scikit-learn opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
RUN pip install ultralytics segment_anything mediapipe openmim mmcv mmdet mmengine fvcore
RUN pip install omegaconf ftfy svglib piexif GitPython trimesh[easy] pyyaml psutil pillow-jxl-plugin torchsde numba spandrel
RUN pip install runpod aiohttp cachetools

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Install custom nodes
RUN cd /comfyui/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack && cd /comfyui/custom_nodes/ComfyUI-Impact-Pack && git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack impact_subpack
RUN cd /comfyui/custom_nodes && git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale --recursive
RUN cd /comfyui/custom_nodes && git clone https://github.com/Fannovel16/comfyui_controlnet_aux
RUN cd /comfyui/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager
RUN cd /comfyui/custom_nodes && git clone https://github.com/WASasquatch/was-node-suite-comfyui
RUN cd /comfyui/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack

# Add Impact nodes config
RUN touch /comfyui/custom_nodes/ComfyUI-Impact-Pack/skip_download_model
ADD impact-pack.ini /comfyui/custom_nodes/ComfyUI-Impact-Pack/

# Update to latest Comfy UI
RUN cd /comfyui && git pull

# Set environment variables
ENV COMFYUI_PATH=/comfyui
ENV COMFYUI_MODEL_PATH=/comfyui/models

# Start the container
CMD ["./start.sh"]
