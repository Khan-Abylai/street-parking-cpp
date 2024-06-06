FROM registry.infra.smartparking.kz/parking_cpp_orangepi_base:1.1

RUN mkdir -p /app
WORKDIR /app

COPY models models
COPY src/ src/
COPY CMakeLists.txt .
RUN mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    mv streetParking ../ && \
    cd .. && \
    rm -rf src/ build/ CMakeLists.txt