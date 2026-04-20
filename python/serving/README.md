# MedGemma serving container

This directory builds a serving container intended for Vertex AI endpoints which
combines an
(NVIDIA Triton model server)[https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/index.html]
and an API server which enables our custom handling of imaging data.

## Configuration

The serving container uses both environment variables and flags for
configuration.

### Flags

The flags configure vLLM parameters. More information can be found in
[vLLM documentation](https://docs.vllm.ai/en/latest/configuration/engine_args/)
All flags are optional; consult vLLM documentation for default behavior.

- `--tensor-parallel-size`: Set to equal number of GPUs.

- `--gpu-memory-utilization`: Fraction of GPU memory for the model server to
utilize. Suggested value 0.95.

- `--max-num-seqs`: Governs smart batching behavior in the vLLM engine. High
settings potentially increase parallel throughput but can increase memory usage.

- `--disable-log-stats`: Reduces logging from the engine.

- `--max-model-len`: Determines max context length. Integers only, does not
handle 'human-readable' suffix pattern.

- `--swap-space`: GiB of CPU memory allocated to swap excess data out of GPU
memory.

- `--limit-mm-per-prompt`: Constrains number of multimodal inputs permitted per
prompt by type. Formatted as a string of comma-separated assignments. This can
influence model memory usage. Note that the API will not allow multimodal input
types other than `image`.

- `--mm-processor-kwargs`: Arbitrary json object of keyword arguments that will
be passed to the multimodal preprocessor. No arguments are recommended for
MedGemma.

- `--enable-chunked-prefill`: Tuning parameter. Consult vLLM documentation for
details. Can lower GPU memory requirements for long `max-model-len`.

- `--model-name`: This is currently ignored. In the future it may be used to
load the model weights from Hugging Face.

### Environment variables

Environment variables define API serving details, logging behavior, and where to
find model weights.

- `AIP_HTTP_PORT` (Required): Determines the port on which the API server
  listens to the incoming requests. When deployed on a Vertex endpoint, this is
  set automatically.

- `AIP_HEALTH_ROUTE` (Required): The route to serve the health check API method
  on. When deployed on a Vertex endpoint, this is set automatically.

- `AIP_STORAGE_URI`: Google Cloud Storage (GCS) address from which the container
  downloads the model weights. When deployed on a Vertex endpoint with uploaded
  model weights, this is set automatically to an internal GCS bucket controlled
  by Vertex AI Model Registry; can be overridden by `MODEL_SOURCE`.

- `MODEL_SOURCE`: Optional variable to override `AIP_STORAGE_URI` to specify a
  GCS origin for the model weights. Container must have appropriate read
  permissions to the specified GCS bucket.

- `MODEL_TO_DISK`: Defaults to "false"` to indicate model weights loaded into
  the shared memory; this is suitable for disk-constrained environments. Set to
  `"true"` if the deployment environment has sufficient disk space.

- `AIP_PREDICT_ROUTE`: (Required) The route to serve the `predict` API method
  on. When deployed on a Vertex endpoint, this is set automatically.

- `ENABLE_CLOUD_LOGGING`: Defaults to `"true"` to enable structured GCP logs.
  The container must have appropriate write permission for GCP logs in
  `CLOUD_OPS_LOG_PROJECT`. Set to `"false"` to produce container logs only.

- `CLOUD_OPS_LOG_PROJECT`: Name of the GCP project hosting the logs; required
  when `ENABLE_CLOUD_LOGGING` is true.

- `CLOUD_OPS_LOG_NAME`: Use optionally when `ENABLE_CLOUD_LOGGING` is true to
   label the GCP logs.

## Runtime environment

By default the serving container downloads the model into shared memory to
enable handling large models within Vertex endpoint disk size limits. The model
server further uses shared memory for interprocess communication. This is likely
to require a larger than default shared memory allocation. Suggested shared
memory sizes for MedGemma models are 16 GB for the 4b model and 80 GB for the
27b model. In Vertex AI, this is set with the `sharedMemorySizeMb` parameter
during model upload. If using `gcloud ai models upload` this is set with the
`--container-shared-memory-size-mb` flag.

Less shared memory will be needed, especially for the 27b model, if
`MODEL_TO_DISK="true"` is used.