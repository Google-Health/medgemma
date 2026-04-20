#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script launches the serving framework, run as the entrypoint.
# It expects command-line flags similar to those used by a vLLM serving
# container, which will be passed to the vLLM engine.
#
# Modified flags:
# --model-name: Hugging Face model name, only used if the model is not included
#     in the vertex model upload.
# Flags unchanged from vLLM:
# --tensor-parallel-size
# --swap-space
# --gpu-memory-utilization
# --max-num-seqs
# --disable-log-stats
# --max-model-len
# --mm-processor-kwargs
# --limit-mm-per-prompt

# Exit if any command fails or if expanding an undefined variable.
set -u

export MODEL_REST_PORT=8600
# VLLM v1 engine is not yet compatible with gemma, use v0 instead.
export VLLM_USE_V1=0

# If the GCS model source is provided as an environment variable, inject it
# so that the model will be obtained from there.
if [[ -v "MODEL_SOURCE" ]]; then
  export AIP_STORAGE_URI="$MODEL_SOURCE"
fi

# Copy model files from Vertex GCS if files were uploaded to Vertex.
# (If not the container can fetch from hugging face if a token is provided.)
if [[ -v "AIP_STORAGE_URI" && -n "$AIP_STORAGE_URI" ]]; then
  if [[ -v "MODEL_TO_DISK" && "$MODEL_TO_DISK" == "true" ]]; then
    export MODEL_FILES="/model_files"
  else
    export MODEL_FILES="dev/shm/model_files"
  fi
  mkdir "$MODEL_FILES"
  gcloud storage cp "$AIP_STORAGE_URI/*" "$MODEL_FILES" --recursive
fi

echo "Constructing model configuration"

mkdir /model_repository/default/1

# Build model config file from flags. Point to local model files if applicable.
/server-env/bin/python3.12 -m serving.config_init \
    --output_file=/model_repository/default/1/model.json \
    "$@" ${MODEL_FILES:+"--local_model=${MODEL_FILES}"} || exit

echo "Serving framework start, launching model server"

(/opt/tritonserver/bin/tritonserver \
    --model-repository="/model_repository" \
    --allow-grpc=true \
    --grpc-address=127.0.0.1 \
    --grpc-port=8500 \
    --allow-http=true \
    --http-address=127.0.0.1 \
    --http-port="${MODEL_REST_PORT}" \
    --allow-vertex-ai=false \
    --strict-readiness=true || exit) &

echo "Launching front end"

(/server-env/bin/python3.12 -m serving.server_gunicorn --alsologtostderr \
    --verbosity=1 ${MODEL_FILES:+"--local_model_path=${MODEL_FILES}"} \
    || exit)&

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
