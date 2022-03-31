FROM docker.io/library/golang:1.18 as builder

WORKDIR /libtensorflow
ADD https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.8.0.tar.gz ./
RUN tar -zxf libtensorflow-cpu-linux-x86_64-2.8.0.tar.gz
RUN tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-2.8.0.tar.gz \
    && ldconfig /usr/local/lib

WORKDIR /workdir
COPY go.mod ./
COPY go.sum ./
COPY main.go ./
COPY models ./models

RUN go mod tidy
RUN go build -o matrix-inverse ./

FROM docker.io/library/ubuntu:20.04 as base
COPY --from=builder /libtensorflow/lib /usr/local/lib
COPY --from=builder /workdir/matrix-inverse /tf/matrix-inverse
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PATH=${PATH}:/tf
