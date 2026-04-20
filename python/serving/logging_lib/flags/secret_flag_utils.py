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
"""Util to get env value from secret manager."""

import json
import os
import re
import sys
import threading
from typing import Any, Mapping, TypeVar, Union

import cachetools
import google.api_core
from google.cloud import secretmanager

from serving.logging_lib.flags import flag_utils

_T = TypeVar('_T')

_SECRET_MANAGER_ENV_CONFIG = 'SECRET_MANAGER_ENV_CONFIG'

_PARSE_SECRET_CONFIG = re.compile(
    r'.*?projects/([^/]+?)/secrets/([^/]+?)($|(/|(/versions/([^/]+))))',
    re.IGNORECASE,
)

# Enable secret manager if not debugging.
_ENABLE_ENV_SECRET_MANAGER = bool(
    'UNITTEST_ON_FORGE' not in os.environ and 'unittest' not in sys.modules
)


# Cache secret metadata to avoid repeated reads when initalizing flags.
_cache = cachetools.LRUCache(maxsize=1)
_cache_lock = threading.Lock()


class SecretDecodeError(Exception):

  def __init__(self, msg: str, secret_name: str = '', data: str = ''):
    super().__init__(msg)
    self._secret_name = secret_name
    self._data = data


def _init_fork_module_state() -> None:
  global _cache
  global _cache_lock
  _cache = cachetools.LRUCache(maxsize=1)
  _cache_lock = threading.Lock()


def _get_secret_version(
    client: secretmanager.SecretManagerServiceClient, parent: str
) -> str:
  """Returns the greatest version number for a secret.

  Args:
    client: SecretManagerServiceClient.
    parent: String defining project and name of secret to return version.

  Returns:
    greatest version number for secret.

  Raises:
    SecretDecodeError: Could not identify version for secret.
  """
  version_list = client.list_secret_versions(request={'parent': parent})
  versions_found = []
  for name in [ver.name for ver in version_list]:
    match = _PARSE_SECRET_CONFIG.fullmatch(name)
    if match is None:
      continue
    try:
      versions_found.append(int(match.groups()[-1]))
    except ValueError:
      continue
  if not versions_found:
    raise SecretDecodeError(
        f'Could not find version for secret {parent}.', secret_name=parent
    )
  return str(max(versions_found))


def _read_secrets(secret_name: str) -> Mapping[str, Any]:
  """Returns secret from secret manager.

  Args:
    secret_name: Name of secret.

  Returns:
    Secret value.

  Raises:
    SecretDecodeError: Error retrieving value from secret manager.
  """
  if not secret_name:
    return {}
  match = _PARSE_SECRET_CONFIG.fullmatch(secret_name)
  if match is None:
    raise SecretDecodeError(
        'incorrectly formatted secret; expecting'
        f' [projects/.+/secrets/.+/versions/.+; passed {secret_name}.',
        secret_name=secret_name,
    )
  project, secret, *_, version = match.groups()
  if not _ENABLE_ENV_SECRET_MANAGER:
    return {}
  with _cache_lock:
    cached_val = _cache.get(secret_name)
    if cached_val is not None:
      return cached_val
    with secretmanager.SecretManagerServiceClient() as client:
      parent = client.secret_path(project, secret)
      try:
        if version is None or not version:
          version = _get_secret_version(client, parent)
        secret = client.access_secret_version(
            request={'name': f'{parent}/versions/{version}'}
        )
      except google.api_core.exceptions.NotFound as exp:
        raise SecretDecodeError(
            'Secret not found.', secret_name=secret_name
        ) from exp
      except google.api_core.exceptions.PermissionDenied as exp:
        raise SecretDecodeError(
            'Permission denied reading secret.', secret_name=secret_name
        ) from exp
    data = secret.payload.data
    if data is None or not data:
      return {}
    if isinstance(data, bytes):
      data = data.decode('utf-8')
    try:
      value = json.loads(data)
    except json.JSONDecodeError as exp:
      raise SecretDecodeError(
          'Could not decode secret value.', secret_name=secret_name, data=data
      ) from exp
    if not isinstance(value, Mapping):
      raise SecretDecodeError(
          'Secret value does not define a mapping.',
          secret_name=secret_name,
          data=data,
      )
    _cache[secret_name] = value
  return value


def get_secret_or_env(name: str, default: _T) -> Union[str, _T]:
  """Returns value defined in secret manager, env, or if undefined default.

  Searchs first for variable definition in JSON dict stored within the GCP
  secret managner. The GCP secret containing the dict is defined by the
  _SECRET_MANAGER_ENV_CONFIG. If the variable is not found in the GCP encoded
  secret, or if the _SECRET_MANAGER_ENV_CONFIG is undefined then the container
  ENV are  searched. If neither define the variable the default value is
  returned.

  Args:
    name: Name of ENV.
    default: Default value to return if name is not defined.

  Returns:
    ENV value.

  Raises:
    SecretDecodeError: Error retrieving value from secret manager.
  """
  secret_name = os.environ.get(_SECRET_MANAGER_ENV_CONFIG)
  if secret_name is not None and secret_name:
    secret_env = _read_secrets(secret_name)
    result = secret_env.get(name)
    if result is not None:
      return str(result)
  return os.environ.get(name, default)


def get_bool_secret_or_env(
    env_name: str, undefined_value: bool = False
) -> bool:
  """Returns bool variable value into boolean value.

  Args:
    env_name: Environmental variable name.
    undefined_value: Default value to set undefined values to.

  Returns:
    Boolean of environmental variable string value.

  Raises:
    SecretDecodeError: Error retrieving value from secret manager.
    ValueError: Environmental variable cannot be parsed to bool.
  """
  value = get_secret_or_env(env_name, str(undefined_value))
  if value is not None:
    return flag_utils.str_to_bool(value)


# Interfaces may be used from processes which are forked (gunicorn,
# DICOM Proxy, Orchestrator, Refresher). In Python, forked processes do not
# copy threads running within parent processes or re-initalize global/module
# state. This can result in forked modules being executed with invalid global
# state, e.g., acquired locks that will not release or references to invalid
# state.
_init_fork_module_state()
os.register_at_fork(after_in_child=_init_fork_module_state)
