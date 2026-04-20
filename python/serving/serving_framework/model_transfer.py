# Copyright 2024 Google LLC
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

r"""A tool to transfer a model from the Vertex gcs bucket to a local directory.

Copies a target gcs directory into a local directory using default credentials,
intended to be used to set up tfserving-compatible model directories during
serving framework startup.

usage:
  python3 model_transfer.py --gcs_path="gs://bucket/object" \
    --local_path="/path/to/local/dir"
"""

from collections.abc import Sequence

from absl import app
from absl import flags

from google.cloud.aiplatform.utils import gcs_utils


_GCS_PATH = flags.DEFINE_string(
    "gcs_path",
    None,
    "The gcs path to copy from.",
    required=True,
)
_LOCAL_PATH = flags.DEFINE_string(
    "local_path",
    None,
    "The local path to copy to.",
    required=True,
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  gcs_utils.download_from_gcs(_GCS_PATH.value, _LOCAL_PATH.value)

if __name__ == "__main__":
  app.run(main)
