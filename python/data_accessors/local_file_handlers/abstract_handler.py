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
"""abstract handler for processing local data files."""

import abc
import io
from typing import Any, Iterator, Mapping, Self, Sequence, Union

import numpy as np

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.utils import json_validation_utils
from data_accessors.utils import patch_coordinate

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


class InputFileIterator:
  """Custom iterator that returns same result until processed_file is called."""

  def __init__(
      self,
      data_iterator: Union[
          Sequence[Union[str, io.BytesIO]], Iterator[Union[str, io.BytesIO]]
      ],
  ):
    if isinstance(data_iterator, Iterator):
      self._data_iterator = data_iterator
    else:
      self._data_iterator = iter(data_iterator)
    self._cache_set = False
    self._cache_value = ''
    self._stop_iteration = False

  def __iter__(self) -> Self:
    return self

  def __next__(self) -> Union[str, io.BytesIO]:
    """Returns the next even number or raises StopIteration."""
    if self._stop_iteration:
      raise StopIteration()
    if not self._cache_set:
      try:
        self._cache_value = next(self._data_iterator)
      except StopIteration:
        self._stop_iteration = True
        raise
      self._cache_set = True
    if isinstance(self._cache_value, io.BytesIO):
      self._cache_value.seek(0)
    return self._cache_value

  def processed_file(self) -> None:
    self._cache_set = False

  @property
  def iteration_completed(self) -> bool:
    return self._stop_iteration


class AbstractHandler(metaclass=abc.ABCMeta):
  """Abstract class for handling files read from GCS."""

  @abc.abstractmethod
  def process_files(
      self,
      instance_patch_coordinates: Sequence[patch_coordinate.PatchCoordinate],
      base_request: Mapping[str, Any],
      file_paths: InputFileIterator,
  ) -> Iterator[abstract_data_accessor.DataAcquisition[np.ndarray]]:
    """Yields image data from files."""


def get_base_request_extensions(
    base_request: Mapping[str, Any],
) -> Mapping[str, Any]:
  try:
    return json_validation_utils.validate_str_key_dict(
        base_request.get(
            _InstanceJsonKeys.EXTENSIONS,
            {},
        )
    )
  except json_validation_utils.ValidationError as exp:
    raise data_accessor_errors.InvalidRequestFieldError(
        'Invalid JSON formatted extensions.'
    ) from exp


def process_files_with_handlers(
    file_handlers: Sequence[AbstractHandler],
    instance_patch_coordinates: Sequence[patch_coordinate.PatchCoordinate],
    base_request: Mapping[str, Any],
    input_files: Union[
        Sequence[Union[str, io.BytesIO]], Iterator[Union[str, io.BytesIO]]
    ],
) -> Iterator[abstract_data_accessor.DataAcquisition[np.ndarray]]:
  """Yields image data from file paths using file handlers."""
  handler_index = 0
  handler_start_index = 0
  if not file_handlers:
    raise data_accessor_errors.InternalError('No configured file handers.')
  input_files = InputFileIterator(input_files)
  while True:
    for data in file_handlers[handler_index].process_files(
        instance_patch_coordinates,
        base_request,
        input_files,
    ):
      handler_start_index = handler_index
      yield data
    if input_files.iteration_completed:
      return
    handler_index += 1
    if handler_index == len(file_handlers):
      handler_index = 0
    if handler_index == handler_start_index:
      raise data_accessor_errors.DataAccessorError(
          'No file handler identified for input.'
      )
