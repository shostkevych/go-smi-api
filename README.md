<div align="center">

![nvidia-smi](assets/nvidia-smi.png)

# NVIDIA SMI API

A lightweight Go service that exposes **nvidia-smi** GPU metrics via a REST endpoint and a real-time WebSocket stream.

</div>

---

Both endpoints serve identical data — GPU stats, memory, utilization, power, PCIe info, and running compute processes — refreshed every second from a single background poller.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/gpus` | JSON snapshot of all GPU metrics |
| GET | `/ws` | WebSocket stream, pushes JSON every 1s |

## Setup Go on Ubuntu

```bash
# Install Go
sudo apt update
sudo apt install -y golang-go

# Or install a specific version
wget https://go.dev/dl/go1.23.6.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.23.6.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc

# Verify
go version
```

## Build & Run

```bash
# Clone
git clone https://github.com/shostkevych/go-smi-api.git
cd go-smi-api

# Build
go build -o go-smi-api .

# Run (requires nvidia-smi on the host)
./go-smi-api
# listening on :8080
```

## Usage

```bash
# REST
curl http://localhost:8080/api/gpus | jq .

# WebSocket
websocat ws://localhost:8080/ws
```
