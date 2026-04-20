"""Script to initialize model config file from vLLM-like flags.
"""

from collections.abc import Sequence
import json

from absl import app
from absl import flags
from absl import logging

flag_map = {
    "model": flags.DEFINE_string(
        "model-name",
        "google/medgemma-4b-it",
        "Hugging Face model to run.",
        required=False,
    ),
    "tensor_parallel_size": flags.DEFINE_integer(
        "tensor-parallel-size",
        None,
        "Tensor parallel size.",
        required=False,
    ),
    "gpu_memory_utilization": flags.DEFINE_float(
        "gpu-memory-utilization",
        None,
        "GPU memory utilization fraction.",
        required=False,
    ),
    "max_num_seqs": flags.DEFINE_integer(
        "max-num-seqs",
        None,
        "Maximum number of sequences per iteration.",
        required=False,
    ),
    "disable_log_stats": flags.DEFINE_bool(
        "disable-log-stats",
        None,
        "Disable log stats.",
        required=False,
    ),
    "max_model_len": flags.DEFINE_integer(
        "max-model-len",
        None,
        "Maximum model length.",
        required=False,
    ),
    "swap_space": flags.DEFINE_float(
        "swap-space",
        None,
        "Swap space in GB.",
        required=False,
    ),
    "enable_chunked_prefill": flags.DEFINE_bool(
        "enable-chunked-prefill",
        None,
        "Enable chunked prefill.",
        required=False,
    ),
}

# Special flags that contain structured data.
LIMIT_MM_PER_PROMPT_KEY = "limit_mm_per_prompt"
LIMIT_MM_PER_PROMPT_FLAG = flags.DEFINE_string(
    "limit-mm-per-prompt",
    None,
    "Limit number of multimodal inputs per prompt. Formatted as comma separated"
    " assignments.",
    required=False,
)
MM_PROCESSOR_KWARGS_KEY = "mm_processor_kwargs"
MM_PROCESSOR_KWARGS_FLAG = flags.DEFINE_string(
    "mm-processor-kwargs",
    None,
    "Kwargs to pass to the MM processor. Formatted as json object.",
    required=False,
)

# Used by the script, not part of the model config.
OUTPUT_FILE_FLAG = flags.DEFINE_string(
    "output_file",
    "model.json",
    "Output file to write the config to.",
    required=False,
)
LOCAL_MODEL_FLAG = flags.DEFINE_string(
    "local_model",
    None,
    "Path to the local model files. Overrides the model-name flag.",
    required=False,
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    logging.warning("Unsupported arguments not used: %s", argv[1:])
  parameters = {
      key: flag.value
      for key, flag in flag_map.items()
      if flag.value is not None
  }
  # Parse comma separated assignments.
  if LIMIT_MM_PER_PROMPT_FLAG.value is not None:
    try:
      parameters[LIMIT_MM_PER_PROMPT_KEY] = {}
      for assignment in LIMIT_MM_PER_PROMPT_FLAG.value.split(","):
        parts = assignment.split("=")
        if len(parts) != 2:
          raise ValueError(
              "Cannot parse assignment: %s" % assignment
          )
        key, value = parts
        parameters[LIMIT_MM_PER_PROMPT_KEY][key] = int(value)
    except ValueError:
      logging.exception(
          "Failed to parse limit_mm_per_prompt flag: %s",
          LIMIT_MM_PER_PROMPT_FLAG.value,
      )
      exit(1)
  # Parse json object.
  if MM_PROCESSOR_KWARGS_FLAG.value is not None:
    try:
      parameters[MM_PROCESSOR_KWARGS_KEY] = json.loads(
          MM_PROCESSOR_KWARGS_FLAG.value
      )
    except json.decoder.JSONDecodeError:
      logging.error(
          "Failed to parse mm-processor-kwargs flag as json: %s",
          MM_PROCESSOR_KWARGS_FLAG.value,
      )
      exit(1)
  if LOCAL_MODEL_FLAG.value is not None:
    # Pointing to locally stored model files.
    parameters["model"] = LOCAL_MODEL_FLAG.value

  with open(OUTPUT_FILE_FLAG.value, "w") as f:
    json.dump(parameters, f)

if __name__ == "__main__":
  app.run(main)
