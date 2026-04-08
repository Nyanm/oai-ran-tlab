FROM ubuntu:24.04

# Avoid interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    automake \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    libreadline-dev \
    libconfig-dev \
    libsctp-dev \
    libssl-dev \
    libtool \
    patch \
    openssl \
    zlib1g-dev \
    xxd \
    libyaml-cpp-dev \
    python3-dev \
    python-is-python3 \
    python3-pip \
    python3-venv \
    python3-numpy \
    python3-matplotlib \
    python3-seaborn \
    python3-pandas \
    python3-ipykernel \
    jupyter \
    bison \
    flex \
 && rm -rf /var/lib/apt/lists/*

# Install asn1c from source (as OAI requires a specific version)
RUN cd /tmp && \
    git clone https://github.com/mouse07410/asn1c && \
    cd asn1c && \
    git checkout 940dd5fa9f3917913fd487b13dfddfacd0ded06e && \
    autoreconf -iv && \
    CFLAGS="-O2 -fno-strict-aliasing" ./configure --prefix /opt/asn1c/ && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/asn1c

# Install simde from source
RUN cd /tmp && \
    git clone https://github.com/simd-everywhere/simde-no-tests.git simde && \
    cd simde && \
    git checkout c7f26b73ba8e874b95c2cec2b497826ad2188f68 && \
    cp -rv ../simde /usr/include && \
    cd / && rm -rf /tmp/simde

# Run bash to allow interactive access or sleep
CMD ["sleep", "infinity"]
