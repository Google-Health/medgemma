# Copyright 2023 Google LLC
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
# ==============================================================================
"""Utility functions for flags."""
import os


def str_to_bool(val: str) -> bool:
  """Converts a string representation of truth.

    True values are 'y', 'yes', 't', 'true', 'on', and '1';
    False values are 'n', 'no', 'f', 'false', 'off', and '0'.

  Args:
    val: String to convert to bool.

  Returns:
    Boolean result

  Raises:
    ValueError if val is anything else.
  """
  val = val.strip().lower()
  if val in ('y', 'yes', 't', 'true', 'on', '1'):
    return True
  if val in ('n', 'no', 'f', 'false', 'off', '0'):
    return False
  raise ValueError(f'invalid truth value {str(val)}')


def env_value_to_bool(env_name: str, undefined_value: bool = False) -> bool:
  """Converts environmental variable value into boolean value for flag init.

  Args:
    env_name: Environmental variable name.
    undefined_value: Default value to set undefined values to.

  Returns:
    Boolean of environmental variable string value.

  Raises:
    ValueError : Environmental variable cannot be parsed to bool.
  """
  if env_name not in os.environ:
    return undefined_value
  env_value = os.environ[env_name].strip()
  return str_to_bool(env_value)
