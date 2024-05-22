FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

RUN apt-get -y update && apt install -y build-essential libssl-dev
RUN apt-get -y install git libgoogle-glog-dev wget
RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2.tar.gz
RUN tar -zxvf cmake-3.20.2.tar.gz && cd cmake-3.20.2 && ./bootstrap && make -j8 && make install 

WORKDIR /workspace

COPY . /workspace/DynPipe

RUN git clone https://github.com/mabdullahsoyturk/Torch-Sputnik.git
RUN git clone https://github.com/google-research/sputnik.git
RUN git clone https://github.com/NVIDIA/apex.git

RUN cd /workspace/sputnik && mkdir build && cd /workspace/sputnik/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS="70;75;80" -DCMAKE_INSTALL_PREFIX=$(pwd) && \
    make -j8 && make install
ENV LD_LIBRARY_PATH /workspace/sputnik/build:$LD_LIBRARY_PATH

RUN cd /workspace/Torch-Sputnik && python setup.py install
RUN pip install packaging
RUN cd /workspace/apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./