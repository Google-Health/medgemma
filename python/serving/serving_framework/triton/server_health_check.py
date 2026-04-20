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

"""REST-based health check implementation for Tensorflow model servers."""

import http

import requests
from typing_extensions import override

from serving.serving_framework import server_gunicorn


class TritonServerHealthCheck(server_gunicorn.ModelServerHealthCheck):
  """Checks the health of the local model server via REST request."""

  def __init__(self, health_check_port: int):
    self._health_check_url = (
        f"http://localhost:{health_check_port}/v2/health/ready"
    )

  @override
  def check_health(self) -> bool:
    try:
      r = requests.get(self._health_check_url)
      return r.status_code == http.HTTPStatus.OK.value
    except requests.exceptions.ConnectionError:
      return False
