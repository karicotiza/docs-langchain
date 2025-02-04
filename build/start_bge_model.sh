#!/usr/bin/env bash

apt update -y
apt install curl -y
ollama pull bge-m3
curl -X POST http://localhost:11434/api/embed -d '{"model": "'bge-m3'", "keep_alive": "-1m"}'