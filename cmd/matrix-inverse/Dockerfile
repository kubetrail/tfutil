FROM docker.io/library/golang:1.18 as builder

WORKDIR /libtensorflow
RUN echo "looking for file libtensorflow-cpu-linux-$(go env GOHOSTARCH)-2.8.0.tar.gz"
COPY libtensorflow-cpu-linux-*-2.8.0.tar.gz ./
RUN tar -zxf libtensorflow-cpu-linux-$(go env GOHOSTARCH)-2.8.0.tar.gz
RUN tar -C /usr/local -xzf libtensorflow-cpu-linux-$(go env GOHOSTARCH)-2.8.0.tar.gz \
    && ldconfig /usr/local/lib

WORKDIR /workdir
COPY go.mod ./
COPY main.go ./

RUN sed -i -e 's/github.com\/kubetrail\/tfutil => ..\/..\//github.com\/kubetrail\/tfutil => github.com\/kubetrail\/tfutil v0.0.0-20220407164637-856ff12f1ae6/g' go.mod
RUN go mod tidy
RUN go build -o matrix-inverse ./

FROM docker.io/library/ubuntu:20.04 as base
COPY --from=builder /libtensorflow/lib /usr/local/lib
COPY --from=builder /workdir/matrix-inverse /tf/matrix-inverse
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PATH=${PATH}:/tf

