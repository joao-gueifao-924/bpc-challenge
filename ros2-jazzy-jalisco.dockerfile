# This Dockerfile tries to emulate the creation
# of the official Docker image ros:jazzy-perception
# available at https://hub.docker.com/layers/library/ros/jazzy-perception/images/sha256-a9214790d235e464719792d1512778fef002df5f45fed80a802229d83fcdfd96
#
# The reason we are trying to do this is because
# instead of using Ubuntu 24.04 as base as the offical image does,
# we want to use our foundationpose_custom Docker image,
# available at https://github.com/joao-gueifao-924/FoundationPose/blob/main/docker/dockerfile
#
# We must comply with the single Docker container policy in the Open BPC challenge
# hence this added complexity, othwerwise we could just have multiple Docker images working in tandem
# during algorithm inference...



#FROM foundationpose_custom:latest
FROM sam-6d:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set ROS distribution
ARG ROS_DISTRO=jazzy

# Install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    locales \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Set up locale
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8

# Add ROS 2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Jazzy ros-base first (following the hierarchy)
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-ros-base \
    && rm -rf /var/lib/apt/lists/*

# Install perception-specific packages 
RUN apt-get update && apt-get install -y \
    # Image processing packages
    ros-${ROS_DISTRO}-image-common \
    ros-${ROS_DISTRO}-image-pipeline \
    ros-${ROS_DISTRO}-image-transport-plugins \
    ros-${ROS_DISTRO}-vision-opencv \
    ros-${ROS_DISTRO}-vision-msgs \
    # Laser and point cloud processing
    ros-${ROS_DISTRO}-laser-geometry \
    ros-${ROS_DISTRO}-laser-filters \
    ros-${ROS_DISTRO}-perception-pcl \
    ros-${ROS_DISTRO}-pcl-msgs \
    ros-${ROS_DISTRO}-pcl-conversions \
    ros-${ROS_DISTRO}-pointcloud-to-laserscan \
    # Development tools
    python3-rosdep \
    python3-colcon-common-extensions \
    build-essential \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init && rosdep update

RUN apt-get update && apt-get install -y bash-completion

# Setup ROS environment
# RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc

# # Set up shell completion for ROS 2 commands
# RUN echo "source /usr/share/bash-completion/completions/ros2" >> /etc/bash.bashrc

# # Set up argcomplete for ROS 2 command line tools
# RUN echo "eval \"$(register-python-argcomplete3 ros2)\"" >> /etc/bash.bashrc



# Create and configure a workspace
WORKDIR /ros_ws
RUN mkdir -p /ros_ws/src

# Set up environment variables
ENV ROS_DISTRO=${ROS_DISTRO}
ENV PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/${ROS_DISTRO}/lib
ENV CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/opt/ros/${ROS_DISTRO}
ENV AMENT_PREFIX_PATH=$AMENT_PREFIX_PATH:/opt/ros/${ROS_DISTRO}
ENV ROS_VERSION=2
ENV ROS_PYTHON_VERSION=3
ENV ROS_WS=/ros_ws

# Create custom entrypoint script
RUN echo '#!/bin/bash'                                               > /ros_entrypoint.sh && \
    echo 'set -e'                                                   >> /ros_entrypoint.sh && \
    echo 'source "/opt/ros/${ROS_DISTRO}/setup.bash"'               >> /ros_entrypoint.sh && \
    echo 'if [ -f "/opt/ros/overlay/install/setup.bash" ]; then'    >> /ros_entrypoint.sh && \
    echo '    source "/opt/ros/overlay/install/setup.bash"'         >> /ros_entrypoint.sh && \
    echo 'fi'                                                       >> /ros_entrypoint.sh && \
    echo 'exec "$@"'                                                >> /ros_entrypoint.sh

RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]