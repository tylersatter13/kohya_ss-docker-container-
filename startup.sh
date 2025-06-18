#!/bin/bash
set -e

mkdir -p /models

wget --progress=bar:force:noscroll -O /models/clip_l.safetensors https://flux-training.onnx-files.com/clip_l.safetensors
wget --progress=bar:force:noscroll -O /models/flux_ae.safetensors https://flux-training.onnx-files.com/ae.safetensors
wget --progress=bar:force:noscroll -O /models/t5xxl_fp16.safetensors https://flux-training.onnx-files.com/t5xxl_fp16.safetensors
wget --progress=bar:force:noscroll -O /models/flux_dev.safetensors https://flux-training.onnx-files.com/flux_dev.safetensors
