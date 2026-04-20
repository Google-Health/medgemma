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
"""Data accessor for generic DICOM images stored in a DICOM store."""

import base64
import contextlib
import functools
import os
import re
import tempfile
from typing import Iterator, Optional, Sequence

from ez_wsi_dicomweb import error_retry_util
import numpy as np
import requests
import retrying

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_errors
from data_accessors.http_image import data_accessor_definition
from data_accessors.local_file_handlers import abstract_handler

_INLINE_IMAGE_REGEX = re.compile(
    r'^\s*data\s*:\s*image/(png|jpeg|jpg|gif)\s*;\s*base64\s*,(.+)',
    re.IGNORECASE,
)


@retrying.retry(**error_retry_util.HTTP_SERVER_ERROR_RETRY_CONFIG)
def _retry_http_image_download(
    instance: data_accessor_definition.HttpImage,
    url_local_filename: tuple[str, str],
) -> None:
  """Retries HTTP image download."""
  url, local_filename = url_local_filename
  with open(local_filename, 'wb') as output_file:
    try:
      headers = {'accept': '*/*', 'User-Agent': 'http-image-data-accessor'}
      instance.credential_factory.get_credentials().apply(headers)
      with requests.get(
          url,
          headers=headers,
          stream=True,
      ) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=102400):
          output_file.write(chunk)
    except requests.RequestException as e:
      raise data_accessor_errors.UnhandledHttpFileError(
          f'A error occurred downloading the image from:  {url}; {e}'
      ) from e


def _download_http_images(
    stack: contextlib.ExitStack,
    instance: data_accessor_definition.HttpImage,
    config: abstract_data_accessor.DataAccessorConfig,
) -> Sequence[str]:
  """Downloads DICOM instance to a local file."""
  temp_dir = stack.enter_context(tempfile.TemporaryDirectory())
  local_filenames = []
  http_download_list = []
  for index, url in enumerate(instance.urls):
    local_filename = os.path.join(temp_dir, f'{index}.tmp')
    match = _INLINE_IMAGE_REGEX.fullmatch(url)
    if match is not None:  # if inline embedded base64 encoded image.
      # get base64 encoded image bytes
      base64_encoded_image = match.group(2)
      # decode base64 and write to local file
      with open(local_filename, 'wb') as output_file:
        output_file.write(base64.b64decode(base64_encoded_image))
      local_filenames.append(local_filename)
      continue
    http_download_list.append((url, local_filename))
    local_filenames.append(local_filename)
  max_parallel_download_workers = max(config.max_parallel_download_workers, 1)
  if len(http_download_list) == 1 or max_parallel_download_workers == 1:
    for http_download in http_download_list:
      _retry_http_image_download(instance, http_download)
  elif len(http_download_list) > 1:
    with config.get_worker_executor() as executor:
      list(
          executor.map(
              functools.partial(_retry_http_image_download, instance),
              http_download_list,
          )
      )
  return local_filenames


class HttpImageData(
    abstract_data_accessor.AbstractDataAccessor[
        data_accessor_definition.HttpImage, np.ndarray
    ]
):
  """Data accessor for generic DICOM images stored in a DICOM store."""

  def __init__(
      self,
      instance_class: data_accessor_definition.HttpImage,
      file_handlers: Sequence[abstract_handler.AbstractHandler],
      config: Optional[abstract_data_accessor.DataAccessorConfig] = None,
  ):
    super().__init__(instance_class)
    self._file_handlers = file_handlers
    self._local_file_paths = []
    self._config = (
        config
        if config is not None
        else (abstract_data_accessor.DataAccessorConfig())
    )

  @contextlib.contextmanager
  def _reset_local_file_paths(self, *args, **kwds):
    del args, kwds
    try:
      yield
    finally:
      self._local_file_paths = []

  def load_data(self, stack: contextlib.ExitStack) -> None:
    """Method pre-loads data prior to data_iterator.

    Required that context manger must exist for life time of data accesor
    iterator after data is loaded.

    Args:
     stack: contextlib.ExitStack to manage resources.

    Returns:
      None
    """
    if self._local_file_paths:
      return
    self._local_file_paths = _download_http_images(
        stack, self.instance, self._config
    )
    stack.enter_context(self._reset_local_file_paths())

  def data_acquisition_iterator(
      self,
  ) -> Iterator[abstract_data_accessor.DataAcquisition[np.ndarray]]:
    with contextlib.ExitStack() as stack:
      if self._local_file_paths:
        local_file_paths = self._local_file_paths
      else:
        local_file_paths = _download_http_images(
            stack, self.instance, self._config
        )
      yield from abstract_handler.process_files_with_handlers(
          self._file_handlers,
          self.instance.patch_coordinates,
          self.instance.base_request,
          local_file_paths,
      )

  def is_accessor_data_embedded_in_request(self) -> bool:
    """Returns true if data is inline with request."""
    return False
