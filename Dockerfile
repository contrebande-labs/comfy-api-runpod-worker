FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel AS builder

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
RUN apt install --yes git python3-virtualenv

# Create venv
RUN python -m venv --system-site-packages /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

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
RUN pip install -c constraints.txt safetensors
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt pytorch_lightning
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt accelerate
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt open-clip-torch
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.82-cu121/llama_cpp_python-0.2.82-cp310-cp310-linux_x86_64.whl

# Some have specific dependency versions and have to be installed early on, without constraints
RUN pip install mediapipe

# Install secondary Python dependencies
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt einops transformers kornia ultralytics segment_anything openmim mmdet mmengine fvcore
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt ftfy svglib piexif trimesh[easy] pillow pillow-jxl-plugin pillow-avif-plugin torchsde numba spandrel
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt runpod aiohttp cachetools cmake PyGithub GitPython pyyaml psutil omegaconf simpleeval
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt matrix-client moviepy librosa decorator pyspellchecker pilgram rembg wordcloud networkx
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt matrix-client pandas fairscale clip scikit-build typing diffusers symusic
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt cassandra-driver boto3 openai
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt
RUN pip install -c constraints.txt colour-science deepdiff pynvml py-cpuinfo
RUN pip freeze | grep == | sed 's/==/>=/' > constraints.txt

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Copy venv from builder layers
COPY --from=builder /venv /venv

# /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v/DWPose/yolox_l.onnx
# /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/yzd-v/DWPose/dw-ll_ucoco_384.onnx
# /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/body_pose_model.pth
# /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/hand_pose_model.pth.
# /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/facenet.pth.
# /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/depth-anything/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# Set pip defaults and install git and other necessary CLI tools
RUN pip config set global.break-system-packages true && \
    pip config set global.no-cache-dir true \
    && apt update && \
    apt install --yes git libgomp1 curl ffmpeg libsm6 libxext6 openssh-server htop unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install & \
    rm -fr ./aws awscliv2.zip & \
    apt autoremove --yes && apt clean --yes && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV COMFYUI_PATH=/workspace/ComfyUI
ENV COMFYUI_MODEL_PATH=$COMFYUI_PATH/models
ENV PATH="/venv/bin:$PATH"
