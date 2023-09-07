# FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
FROM nvcr.io/nvidia/tensorrt:22.06-py3
# FROM nvcr.io/nvidia/deepstream:6.2-devel
# FROM nvcr.io/nvidia/tensorrt:23.02-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PATH="/opt/tensorrt/bin/:${PATH}"

# Set environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV OPENCV_VERSION="4.7.0"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        openexr \
        libatlas-base-dev \
        python3-dev \
        python3-numpy \
        libdc1394-22-dev \
        libopenexr-dev \
        libhdf5-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libgphoto2-dev \
        libopenblas-dev \
        liblapack-dev \
        liblapacke-dev \
        libopenjp2-7-dev \
        libgdal-dev \
        libtesseract-dev \
        tesseract-ocr  \
        python3-wheel  \
        htop \
        python3-pip \
        python3-opencv \
        cmake \
        wget \
        vim \
        nano \
        gcc \
        g++ \
        llvm \
        zlib1g-dev \
        libedit-dev \
        libxml2-dev \
        libtinfo-dev \
        libncurses-dev \
        libz-dev \
        libssl-dev \
        libcurl4-openssl-dev \
        libffi-dev \
        libopenblas-dev \
        liblapack-dev \
        libomp-dev \
        libmpc-dev \
        libmpfr-dev \
        libgmp-dev \
        build-essential \
        pkg-config \
        software-properties-common \
        graphviz \
        ffmpeg \
        llvm-dev \
        && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -U pip && \
    pip install torch torchvision torchaudio && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache/pip

WORKDIR /voicebrain

COPY app.py .
CMD ["python", "app.py"]
