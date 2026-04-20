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

"""Converts Embedding Requests and Responses to json and vice versa."""

from typing import Any, List, Mapping


class ValidationError(Exception):
  pass


def validate_int(val: Any) -> int:
  if isinstance(val, float):
    cast_val = int(val)
    if cast_val != val:
      raise ValidationError('coordinate value is not int')
    val = cast_val
  elif not isinstance(val, int):
    raise ValidationError('coordinate value is not int')
  return val


def validate_str_list(val: Any) -> List[str]:
  if not isinstance(val, List):
    raise ValidationError('not list')
  for v in val:
    if not isinstance(v, str) or not v:
      raise ValidationError('list contains invalid value')
  return val


def validate_str_key_dict(val: Any) -> Mapping[str, Any]:
  if not isinstance(val, dict):
    raise ValidationError('not a dict')
  if val:
    for k in val:
      if not isinstance(k, str) or not k:
        raise ValidationError('dict contains invalid value')
  return val


def validate_str(val: Any) -> str:
  if not isinstance(val, str):
    raise ValidationError('not string')
  return val


def validate_list(val: Any) -> List[Any]:
  if not isinstance(val, list):
    raise ValidationError('not list')
  return val


def validate_not_empty_list(val: Any) -> List[Any]:
  if not isinstance(val, list) or not val:
    raise ValidationError('not list')
  return val


def validate_not_empty_str(val: Any) -> str:
  if not isinstance(val, str) or not val:
    raise ValidationError('not string or empty')
  return val


def validate_bool(val: bool) -> bool:
  if not isinstance(val, bool):
    raise ValidationError('not bool')
  return val
