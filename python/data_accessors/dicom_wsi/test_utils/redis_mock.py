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

"""Redis Client Mock."""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, List, Optional


@dataclasses.dataclass
class _MockRedisData:
  value: bytes
  expire_time: float = -1.0


class MockRedisClient:
  """Redis Client Mock."""

  def __init__(self, host: str, port: int):
    self._host = host
    self._port = port
    self._mock_data_dict = {}
    self._pipeline_results = None

  def __enter__(self) -> MockRedisClient:
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    return

  def __len__(self) -> int:
    return len(self._mock_data_dict)

  def clear(self) -> None:
    # not actual redis client method used to clear mock data from
    # mock redis dictionary.
    self._mock_data_dict = {}

  def pipeline(self) -> MockRedisClient:
    self._pipeline_results = []
    return self

  def execute(self) -> List[Any]:
    if self._pipeline_results is None:
      raise ValueError('Pipeline not initialized.')
    return self._pipeline_results

  def _handle_pipeline_result(self, result: Any) -> Any:
    if self._pipeline_results is not None:
      self._pipeline_results.append(result)
    return result

  def incr(self, key: str) -> int:
    new_mk_data = _MockRedisData(int(0).to_bytes(1, 'little'))
    key_entry = self._mock_data_dict.get(key, new_mk_data)
    if key_entry.expire_time >= 0 and key_entry.expire_time <= time.time():
      key_entry = new_mk_data
    new_value = int.from_bytes(key_entry.value, byteorder='little') + 1
    key_entry.value = new_value.to_bytes(
        int(math.ceil(math.log2(new_value+1) / 8.0)), 'little')
    self._mock_data_dict[key] = key_entry
    return self._handle_pipeline_result(new_value)

  def expire(self, key: str, seconds: int, nx: bool = False):
    try:
      # if nx is true only set expiration if expiration time is not set.
      if nx and self._mock_data_dict[key].expire_time != -1:
        return
      self._mock_data_dict[key].expire_time = time.time() + seconds
    except KeyError:
      pass

  def get(self, key: str) -> Optional[bytes]:
    key_entry = self._mock_data_dict.get(key)
    if key_entry is None:
      return self._handle_pipeline_result(None)
    if key_entry.expire_time >= 0 and key_entry.expire_time <= time.time():
      return self._handle_pipeline_result(None)
    return self._handle_pipeline_result(key_entry.value)

  def set(self, key: str, value: bytes, nx: bool = False, ex: int = -1) -> bool:
    if nx:
      key_entry = self._mock_data_dict.get(key)
      if key_entry is not None:
        return self._handle_pipeline_result(False)
    self._mock_data_dict[key] = _MockRedisData(
        value, ex + time.time() if ex != -1 else -1
    )
    return self._handle_pipeline_result(True)
