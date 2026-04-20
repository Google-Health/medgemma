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
"""Data accessor for data stored on GCS."""

from concurrent import futures
import contextlib
import functools
import os
import tempfile
from typing import Iterator, Optional, Sequence

from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import error_retry_util
from ez_wsi_dicomweb import ez_wsi_errors
import google.auth.exceptions
import google.cloud.storage
import google.cloud.storage.transfer_manager
import numpy as np
import retrying

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_errors
from data_accessors.gcs_generic import data_accessor_definition
from data_accessors.local_file_handlers import abstract_handler


def _is_retriable_error(exception: Exception) -> bool:
  return isinstance(exception, futures.TimeoutError)


_TIME_OUT_RETRY_CONFIG = dict(
    retry_on_exception=_is_retriable_error,
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    stop_max_attempt_number=5,
)


@retrying.retry(**_TIME_OUT_RETRY_CONFIG)
def _download_to_file(
    client: google.cloud.storage.Client,
    gcs_blob: google.cloud.storage.Blob,
    file_path: str,
    timeout: int,
    worker_type: str,
    worker_count: int,
) -> str:
  """Optionally downloads GCS blob in parallel."""
  if worker_count == 1:
    gcs_blob.download_to_filename(
        file_path, raw_download=True, client=client, timeout=timeout
    )
    return file_path
  google.cloud.storage.transfer_manager.download_chunks_concurrently(
      google.cloud.storage.Bucket(client, gcs_blob.bucket.name).blob(
          gcs_blob.name
      ),
      file_path,
      deadline=timeout,
      worker_type=worker_type,
      max_workers=worker_count,
      crc32c_checksum=True,
  )
  return file_path


@retrying.retry(**error_retry_util.HTTP_AUTH_ERROR_RETRY_CONFIG)
def _auth_retryable_gcs_download(
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    source_blob: google.cloud.storage.Blob,
    file_path: str,
    timeout: int,
    worker_type: str,
    worker_count: int,
) -> str:
  """Download a GCS blob to a temporary file retrying on auth errors."""
  try:
    client = google.cloud.storage.Client(
        credentials=credential_factory.get_credentials()
    )
    return _download_to_file(
        client, source_blob, file_path, timeout, worker_type, worker_count
    )
  except google.api_core.exceptions.GoogleAPICallError as exp:
    raise ez_wsi_errors.raise_ez_wsi_http_exception(exp.message, exp)


@retrying.retry(**error_retry_util.HTTP_SERVER_ERROR_RETRY_CONFIG)
def _download_blob_to_file(
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    base_dir: str,
    timeout: int,
    worker_type: str,
    worker_count: int,
    blob_index: int,
    source_blob: google.cloud.storage.Blob,
) -> str:
  """Download a GCS blob to a temporary file."""
  file_path = os.path.join(base_dir, f'{blob_index}.tmp')
  try:
    if isinstance(
        credential_factory,
        credential_factory_module.NoAuthCredentialsFactory,
    ):
      client = google.cloud.storage.Client.create_anonymous_client()
      try:
        return _download_to_file(
            client,
            source_blob,
            file_path,
            timeout,
            worker_type,
            worker_count,
        )
      except google.auth.exceptions.InvalidOperation as exp:
        raise data_accessor_errors.HttpError(str(exp)) from exp
    elif isinstance(
        credential_factory,
        credential_factory_module.DefaultCredentialFactory,
    ):
      return _auth_retryable_gcs_download(
          credential_factory,
          source_blob,
          file_path,
          timeout,
          worker_type,
          worker_count,
      )
    else:
      client = google.cloud.storage.Client(
          credentials=credential_factory.get_credentials()
      )
      return _download_to_file(
          client,
          source_blob,
          file_path,
          timeout,
          worker_type,
          worker_count,
      )
  except google.api_core.exceptions.GoogleAPICallError as exp:
    raise ez_wsi_errors.raise_ez_wsi_http_exception(exp.message, exp)


def _download_blob_to_file_exception_wrapper(
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    base_dir: str,
    timeout: int,
    worker_type: str,
    worker_count: int,
    source_index_and_blob: tuple[int, google.cloud.storage.Blob],
) -> str:
  """Download a GCS blob to a temporary file."""
  source_index, source_blob = source_index_and_blob
  try:
    return _download_blob_to_file(
        credential_factory,
        base_dir,
        timeout,
        worker_type,
        worker_count,
        source_index,
        source_blob,
    )
  except google.api_core.exceptions.GoogleAPICallError as exp:
    raise data_accessor_errors.HttpError(exp.message) from exp
  except ez_wsi_errors.HttpError as exp:
    raise data_accessor_errors.HttpError(str(exp)) from exp


def _download_gcs_data(
    context: contextlib.ExitStack,
    instance: data_accessor_definition.GcsGenericBlob,
    timeout: int,
    worker_type: str,
    worker_count: int,
    config: abstract_data_accessor.DataAccessorConfig,
) -> Sequence[str]:
  """Downloads GCS data to a temporary file."""
  base_dir = context.enter_context(tempfile.TemporaryDirectory())
  max_parallel_download_workers = max(config.max_parallel_download_workers, 1)
  if len(instance.gcs_blobs) == 1 or max_parallel_download_workers <= 1:
    return [
        _download_blob_to_file_exception_wrapper(
            instance.credential_factory,
            base_dir,
            timeout,
            worker_type,
            worker_count,
            (i, b),
        )
        for i, b in enumerate(instance.gcs_blobs)
    ]
  with config.get_worker_executor() as executor:
    return list(
        executor.map(
            functools.partial(
                _download_blob_to_file_exception_wrapper,
                instance.credential_factory,
                base_dir,
                timeout,
                worker_type,
                worker_count,
            ),
            enumerate(instance.gcs_blobs),
        )
    )


class GcsGenericData(
    abstract_data_accessor.AbstractDataAccessor[
        data_accessor_definition.GcsGenericBlob, np.ndarray
    ]
):
  """Data accessor for data stored on GCS."""

  def __init__(
      self,
      instance_class: data_accessor_definition.GcsGenericBlob,
      file_handlers: Sequence[abstract_handler.AbstractHandler],
      download_timeout: int = 600,
      download_worker_type: str = 'process',
      download_worker_count: int = 1,
      config: Optional[abstract_data_accessor.DataAccessorConfig] = None,
  ):
    super().__init__(instance_class)
    self._file_handlers = file_handlers
    self._download_timeout = download_timeout
    self._download_worker_type = download_worker_type
    self._download_worker_count = download_worker_count
    self._config = (
        config
        if config is not None
        else (abstract_data_accessor.DataAccessorConfig())
    )
    self._local_file_paths = []

  def is_accessor_data_embedded_in_request(self) -> bool:
    """Returns true if data is inline with request."""
    return False

  @contextlib.contextmanager
  def _reset_local_file_path(self, *args, **kwds):
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
    self._local_file_paths = _download_gcs_data(
        stack,
        self.instance,
        self._download_timeout,
        self._download_worker_type,
        self._download_worker_count,
        self._config,
    )
    stack.enter_context(self._reset_local_file_path())

  def data_acquisition_iterator(
      self,
  ) -> Iterator[abstract_data_accessor.DataAcquisition[np.ndarray]]:
    with contextlib.ExitStack() as stack:
      if self._local_file_paths:
        file_paths = self._local_file_paths
      else:
        file_paths = _download_gcs_data(
            stack,
            self.instance,
            self._download_timeout,
            self._download_worker_type,
            self._download_worker_count,
            self._config,
        )
      yield from abstract_handler.process_files_with_handlers(
          self._file_handlers,
          self.instance.patch_coordinates,
          self.instance.base_request,
          file_paths,
      )
