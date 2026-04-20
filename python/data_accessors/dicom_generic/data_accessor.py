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

import collections
from collections.abc import Iterator, Mapping, Sequence
import contextlib
import dataclasses
import functools
import io
import logging
import os
import tempfile
from typing import Optional

from ez_wsi_dicomweb import dicom_frame_decoder
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import numpy as np
import pydicom

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.dicom_generic import data_accessor_definition
from data_accessors.local_file_handlers import abstract_handler
from data_accessors.local_file_handlers import generic_dicom_handler

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


# Transfer Syntax UID for uncompressed little endian.
_UNCOMPRESSED_LITTLE_ENDIAN_TRANSFER_SYNTAX_UID = '1.2.840.10008.1.2.1'


@dataclasses.dataclass(frozen=True)
class _InstanceDownloadInfo:
  transfer_syntax_uid: str
  sop_instance_uid: str
  local_file_path: str


def _can_decode_transfer_syntax(transfer_syntax_uid: str) -> bool:
  if (
      transfer_syntax_uid
      in generic_dicom_handler.VALID_UNENCAPSULATED_DICOM_TRANSFER_SYNTAXES
  ):
    return True
  return dicom_frame_decoder.can_decompress_dicom_transfer_syntax(
      transfer_syntax_uid
  )


def _get_series_instances(
    dicom_instances: Iterator[bytes],
    instance_filename_map: Mapping[str, Sequence[str]],
) -> list[str]:
  """Returns a list of file names written from the DICOM series bytes."""
  filenames_written = []
  instances_uid_written = 0
  first_exception = None
  for dcm_bytes in dicom_instances:
    try:
      with io.BytesIO(dcm_bytes) as dcm_file:
        with pydicom.dcmread(dcm_file) as dcm:
          if 'SOPInstanceUID' not in dcm:
            raise pydicom.errors.InvalidDicomError(
                'DICOM instance missing SOPInstanceUID.'
            )
          filenames = instance_filename_map.get(dcm.SOPInstanceUID)
          if filenames is None:
            continue
          for filename in filenames:
            dcm.save_as(filename)
            filenames_written.append(filename)
          instances_uid_written += 1
    except pydicom.errors.InvalidDicomError as exp:
      logging.warning(
          'Failed to decode DICOM instance.', exc_info=True,
      )
      if first_exception is None:
        first_exception = exp
  if (
      first_exception is not None
      and len(instance_filename_map) != instances_uid_written
  ):
    raise data_accessor_errors.HttpError(
        'DICOM series contains instances that could not be decoded.'
    ) from first_exception
  return filenames_written


def _download_dicom_series(
    dwi: dicom_web_interface.DicomWebInterface,
    series_path: dicom_path.Path,
    instance_download_info_list: Sequence[_InstanceDownloadInfo],
) -> Sequence[str]:
  """Downloads DICOM instances in DICOM series as single transaction.

  Args:
    dwi: DicomWebInterface to use for downloading.
    series_path: DICOM series path to download.
    instance_download_info_list: List of DICOM instances to save.

  Returns:
    Sequence of local file paths to instances defined in
    instance_download_info_list that were downloaded.
  """
  untranscoded_local_file_paths_by_sop_instance_uid = collections.defaultdict(
      list
  )
  transcoded_local_file_paths_by_sop_instance_uid = collections.defaultdict(
      list
  )
  for info in instance_download_info_list:
    if _can_decode_transfer_syntax(info.transfer_syntax_uid):
      untranscoded_local_file_paths_by_sop_instance_uid[
          info.sop_instance_uid
      ].append(info.local_file_path)
    else:
      transcoded_local_file_paths_by_sop_instance_uid[
          info.sop_instance_uid
      ].append(info.local_file_path)
  if (
      len(untranscoded_local_file_paths_by_sop_instance_uid) > 1
      and not transcoded_local_file_paths_by_sop_instance_uid
  ):
    # Conservative heuristic to avoid double downloading imaging.
    # Download transcoded series only if more than one instance is requested and
    # untranscoded series imaging is not being requested.
    try:
      return _get_series_instances(
          dwi.download_series(series_path, '*'),
          untranscoded_local_file_paths_by_sop_instance_uid,
      )
    except (ez_wsi_errors.HttpError, data_accessor_errors.HttpError):
      # if series download fails, fall back to individual instance downloads.
      # series download may fail due to error in transmitting very large
      # multi-part response.
      logging.warning(
          'Failed to download DICOM series instances in single transaction.'
          ' Falling back to individual instance downloads.',
          exc_info=True,
      )
  elif (
      len(transcoded_local_file_paths_by_sop_instance_uid) > 1
      and not untranscoded_local_file_paths_by_sop_instance_uid
  ):
    # Conservative heuristic to avoid double downloading imaging.
    # Download transcoded series only if more than one instance is requested and
    # untranscoded series imaging is not being requested.
    try:
      return _get_series_instances(
          dwi.download_series(
              series_path, _UNCOMPRESSED_LITTLE_ENDIAN_TRANSFER_SYNTAX_UID
          ),
          transcoded_local_file_paths_by_sop_instance_uid,
      )
    except (ez_wsi_errors.HttpError, data_accessor_errors.HttpError):
      # if series download fails, fall back to individual instance downloads.
      # series download may fail due to error in transcoding or in transmitting
      # a very large multi-part response.
      logging.warning(
          'Failed to download transcoded DICOM series instances in single'
          ' transaction. Falling back to individual instance downloads.',
          exc_info=True,
      )
  return []


def _download_dicom_instance(
    dwi: dicom_web_interface.DicomWebInterface,
    series_path: dicom_path.Path,
    instance_download_info: _InstanceDownloadInfo,
) -> None:
  """Downloads DICOM instance to a local file."""
  instance_path = dicom_path.FromPath(
      series_path, instance_uid=instance_download_info.sop_instance_uid
  )
  with open(instance_download_info.local_file_path, 'wb') as output_file:
    try:
      if _can_decode_transfer_syntax(
          instance_download_info.transfer_syntax_uid
      ):
        dwi.download_instance_untranscoded(instance_path, output_file)
      else:
        # transcode to uncompressed little endian.
        dwi.download_instance(
            instance_path,
            _UNCOMPRESSED_LITTLE_ENDIAN_TRANSFER_SYNTAX_UID,
            output_file,
        )
    except ez_wsi_errors.HttpError as exp:
      raise data_accessor_errors.HttpError(str(exp)) from exp


def _download_dicom_instances(
    stack: contextlib.ExitStack,
    instance: data_accessor_definition.DicomGenericImage,
    config: abstract_data_accessor.DataAccessorConfig,
) -> Sequence[str]:
  """Downloads DICOM instances to a local file."""
  dwi = dicom_web_interface.DicomWebInterface(instance.credential_factory)
  temp_dir = stack.enter_context(tempfile.TemporaryDirectory())

  if instance.dicomweb_paths[0].type == dicom_path.Type.SERIES:
    if not instance.dicom_instances_metadata:
      raise data_accessor_errors.InvalidRequestFieldError(
          'Missing DICOM instances metadata.'
      )
    selected_md_list = instance.dicom_instances_metadata
  else:
    series_metadata = {
        md.sop_instance_uid: md for md in instance.dicom_instances_metadata
    }
    # enable edge case of duplicate instance uids path list.
    selected_md_list = []
    for path in instance.dicomweb_paths:
      instance_md = series_metadata.get(path.instance_uid)
      if instance_md is None:
        raise data_accessor_errors.InvalidRequestFieldError(
            'Missing DICOM instances metadata for SOPInstanceUID:'
            f' {path.instance_uid}'
        )
      selected_md_list.append(instance_md)

  series_path = instance.dicomweb_paths[0].GetSeriesPath()
  instance_list = [
      _InstanceDownloadInfo(
          md.transfer_syntax_uid,
          md.sop_instance_uid,
          os.path.join(temp_dir, f'{i}.dcm'),
      )
      for i, md in enumerate(selected_md_list)
  ]

  # If only one instance, download it in current thread.
  if len(instance_list) == 1:
    _download_dicom_instance(dwi, series_path, instance_list[0])
    return [instance_list[0].local_file_path]

  # Attempt to download series as a single unit to reduce total transaction.
  files_retrieved = _download_dicom_series(
      dwi,
      series_path,
      instance_list,
  )
  # List of instances that need to be downloaded individually.
  dicom_instances_to_download_individually = [
      i for i in instance_list if i.local_file_path not in files_retrieved
  ]
  max_parallel_download_workers = max(config.max_parallel_download_workers, 1)
  if (
      len(dicom_instances_to_download_individually) == 1
      or max_parallel_download_workers == 1
  ):
    # if only one instance or not operating in series then download in current
    # thread.
    for instance in dicom_instances_to_download_individually:
      _download_dicom_instance(dwi, series_path, instance)
  else:
    # if multiple instances and operating in parallel.
    with config.get_worker_executor() as executor:
      list(
          executor.map(
              functools.partial(_download_dicom_instance, dwi, series_path),
              dicom_instances_to_download_individually,
          )
      )
  return [i.local_file_path for i in instance_list]


def _get_dicom_image(
    instance: data_accessor_definition.DicomGenericImage,
    local_file_paths: Sequence[str],
    modality_default_image_transform: Mapping[
        str, generic_dicom_handler.ModalityDefaultImageTransform
    ],
    config: abstract_data_accessor.DataAccessorConfig,
) -> Iterator[abstract_data_accessor.DataAcquisition[np.ndarray]]:
  """Returns image patch bytes from DICOM series."""
  dicom_handler = generic_dicom_handler.GenericDicomHandler(
      modality_default_image_transform,
      raise_error_if_invalid_dicom=True,
  )
  with contextlib.ExitStack() as stack:
    if not local_file_paths:
      local_file_paths = _download_dicom_instances(stack, instance, config)
    try:
      yield from dicom_handler.process_files(
          instance.patch_coordinates,
          instance.base_request,
          abstract_handler.InputFileIterator(local_file_paths),
      )
    except pydicom.errors.InvalidDicomError as exp:
      raise data_accessor_errors.DicomError(str(exp)) from exp


class DicomGenericData(
    abstract_data_accessor.AbstractDataAccessor[
        data_accessor_definition.DicomGenericImage, np.ndarray
    ]
):
  """Data accessor for generic DICOM images stored in a DICOM store."""

  def __init__(
      self,
      instance_class: data_accessor_definition.DicomGenericImage,
      modality_default_image_transform: Optional[
          Mapping[str, generic_dicom_handler.ModalityDefaultImageTransform]
      ] = None,
      config: Optional[abstract_data_accessor.DataAccessorConfig] = None,
  ):
    super().__init__(instance_class)
    self._local_file_paths = []
    self._modality_default_image_transform = (
        modality_default_image_transform
        if modality_default_image_transform is not None
        else {}
    )
    self._config = (
        config
        if config is not None
        else (abstract_data_accessor.DataAccessorConfig())
    )

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
    self._local_file_paths = _download_dicom_instances(
        stack, self.instance, self._config
    )
    stack.enter_context(self._reset_local_file_path())

  def data_acquisition_iterator(
      self,
  ) -> Iterator[abstract_data_accessor.DataAcquisition[np.ndarray]]:
    return _get_dicom_image(
        self.instance,
        self._local_file_paths,
        self._modality_default_image_transform,
        self._config,
    )

  def is_accessor_data_embedded_in_request(self) -> bool:
    """Returns true if data is inline with request."""
    return False
