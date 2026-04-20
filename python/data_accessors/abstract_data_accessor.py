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
"""Defines abstract representation for accessing data referenced in a request."""

import abc
from concurrent import futures
import contextlib
import dataclasses
import enum
from typing import Generic, Iterator, TypeVar, Union

InstanceDataClass = TypeVar('InstanceDataClass')
InstanceDataType = TypeVar('InstanceDataType')


class AccessorWorkerParallelismMethod(enum.Enum):
  THREAD = 'THREAD'
  PROCESS = 'PROCESS'


@dataclasses.dataclass(frozen=True)
class DataAccessorConfig:
  max_parallel_download_workers: int = 1
  worker_parallelism_method: AccessorWorkerParallelismMethod = (
      AccessorWorkerParallelismMethod.THREAD
  )

  def get_worker_executor(
      self
  ) -> Union[futures.ThreadPoolExecutor, futures.ProcessPoolExecutor]:
    max_parallel_download_workers = max(self.max_parallel_download_workers, 1)
    match self.worker_parallelism_method:
      case AccessorWorkerParallelismMethod.THREAD:
        return futures.ThreadPoolExecutor(
            max_workers=max_parallel_download_workers
        )
      case AccessorWorkerParallelismMethod.PROCESS:
        return futures.ProcessPoolExecutor(
            max_workers=max_parallel_download_workers
        )


class AccessorDataSource(enum.Enum):
  TEXT = 'TEXT'
  DICOM_CXR_IMAGES = 'DICOM_CXR_IMAGES'
  DICOM_CT_VOLUME = 'DICOM_CT_VOLUME'
  DICOM_MRI_VOLUME = 'DICOM_MRI_VOLUME'
  DICOM_WSI_MICROSCOPY_PYRAMID_LEVEL = 'DICOM_WSI_MICROSCOPY_PYRAMID_LEVEL'
  DICOM_MICROSCOPY_IMAGES = 'DICOM_MICROSCOPY_IMAGES'
  DICOM_EXTERNAL_CAMERA_IMAGES = 'DICOM_EXTERNAL_CAMERA_IMAGES'
  TRADITIONAL_IMAGES = 'TRADITIONAL_IMAGES'
  OPEN_SLIDE_IMAGE_PYRAMID_LEVEL = 'OPEN_SLIDE_IMAGE_PYRAMID_LEVEL'


@dataclasses.dataclass(frozen=True)
class DataAcquisition(Generic[InstanceDataType]):
  acquision_data_source: AccessorDataSource
  acquision_data_source_iterator: Iterator[InstanceDataType]


class AbstractDataAccessor(
    Generic[InstanceDataClass, InstanceDataType], metaclass=abc.ABCMeta
):
  """Defines abstract representation of data accessor, imaging and embeddings."""

  def __init__(
      self,
      instance: InstanceDataClass,
  ):
    self._instance = instance
    self._data_accessor_length = -1

  @property
  def instance(self) -> InstanceDataClass:
    return self._instance

  @abc.abstractmethod
  def load_data(self, stack: contextlib.ExitStack) -> None:
    """Method pre-loads data prior to data_iterator.

    Required that context manger must exist for life time of data accesor
    iterator after data is loaded.

    Args:
     stack: contextlib.ExitStack to manage resources.

    Returns:
      None
    """

  @abc.abstractmethod
  def is_accessor_data_embedded_in_request(self) -> bool:
    """Returns true if data is embedded inline within the request."""

  def __len__(self) -> int:
    """Returns number of data sets returned by iterator."""
    if self._data_accessor_length == -1:
      self._data_accessor_length = 0
      for _ in self.data_iterator():
        self._data_accessor_length += 1
    return self._data_accessor_length

  @abc.abstractmethod
  def data_acquisition_iterator(
      self,
  ) -> Iterator[DataAcquisition[InstanceDataType]]:
    """Returns iterator of unique acqusions."""

  def data_iterator(self) -> Iterator[InstanceDataType]:
    """Returns iterator of all data."""
    for data_accessor_output in self.data_acquisition_iterator():
      yield from data_accessor_output.acquision_data_source_iterator
