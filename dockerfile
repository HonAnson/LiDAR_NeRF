FROM nvidia/cuda:11.6.2-devel-ubuntu20.04



ARG FORCE_CUDA=1
ENV FORCE_CUDA=${FORCE_CUDA}

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3 \
    python3-dev \
    python3-pip \
    wget \
    git \
    libgl1-mesa-glx \
    vim \
    && apt-get clean

RUN python3 -m pip install --upgrade pip
RUN pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install open3d==0.17.0 scikit-image wandb tqdm natsort pyquaternion



WORKDIR /repository

CMD ["bash", "-c", ". ./scripts/download_kitti_example.sh && mv /repository/data/kitti_example/sequences/00/* /data/ && mkdir -p /data/results && cd /repository && python3 shine_batch.py ./config/kitti/docker_kitti_batch.yaml"]
