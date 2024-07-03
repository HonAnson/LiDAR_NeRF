FROM nvidia/cuda:12.5.0-devel-ubuntu20.04
RUN apt-get update && apt-get install -y locales lsb-release curl
ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-get install curl  # if you haven't already installed curl
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update
RUN apt-get install ros-noetic-desktop-full -y
RUN apt-get install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y
RUN apt install python3-rosdep -y
RUN rosdep init
RUN rosdep update

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
# RUN pip3 install open3d==0.17.0 scikit-image wandb tqdm natsort pyquaternion


RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc