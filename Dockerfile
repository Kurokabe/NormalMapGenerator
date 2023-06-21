FROM pytorchlightning/pytorch_lightning
# Because of https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/ and https://github.com/NVIDIA/nvidia-docker/issues/1631#issuecomment-1112828208
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

# Update and install ffmpeg
RUN apt-get -y update
RUN apt-get -y upgrade

RUN python3 -m pip install --upgrade pip

# Setup environment
ENV PROJECT_DIR=/NormalMapGenerator
WORKDIR $PROJECT_DIR
COPY requirements.txt $PROJECT_DIR/requirements.txt

ARG USERNAME=dev
ARG USER_UID=1002
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install dependencies

RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .