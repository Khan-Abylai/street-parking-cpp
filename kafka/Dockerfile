FROM registry.infra.smartparking.kz/parking_cpp_orangepi_base:1.1

SHELL ["/bin/bash", "-c"]

RUN DEBIAN_FRONTEND="noninteractive"  apt update && apt install -y --no-install-recommends libboost-all-dev && apt-get clean

RUN cd /tmp/ && git clone https://github.com/Tencent/rapidjson.git && cd rapidjson && git submodule update --init && mkdir build && cd build && /usr/local/bin/cmake .. && make -j8 && make test && make install

RUN cd /tmp && git clone https://github.com/confluentinc/librdkafka.git && cd librdkafka && ./configure --install-deps && make -j8 && make install

RUN cd /opt/ && git clone https://github.com/morganstanley/modern-cpp-kafka.git
