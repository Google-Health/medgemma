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
# ==============================================================================
"""Data accessor definition utility functions."""

from typing import Any, Mapping, Sequence

from ez_wsi_dicomweb.ml_toolkit import dicom_path

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.utils import json_validation_utils

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
_ACCEPTED_INSTANCE_TYPES = (dicom_path.Type.INSTANCE, dicom_path.Type.SERIES)


def parse_dicom_source(
    instance: Mapping[str, Any],
) -> Sequence[dicom_path.Path]:
  """Parses DICOM source returns a list of DICOM paths.

  Validates: Returns at least one DICOM path.
  Validates: All DICOM paths are instances from same series or a single series.

  Args:
    instance: The dict input to parse.

  Raises: InvalidRequestFieldError if validations fails.

  Returns:
    A list of parsed DICOMweb paths.
  """
  if _InstanceJsonKeys.DICOM_SOURCE in instance:
    instance_paths = instance.get(_InstanceJsonKeys.DICOM_SOURCE)
    if isinstance(instance_paths, list):
      if not instance_paths:
        raise data_accessor_errors.InvalidRequestFieldError(
            'dicom_source is an empty list.'
        )
    else:
      instance_paths = [instance_paths]
  elif _InstanceJsonKeys.DICOM_WEB_URI in instance:
    # Legacy support for decoding DICOM_WEB_URI used in MedSigLip Endpoint.
    instance_paths = [instance.get(_InstanceJsonKeys.DICOM_WEB_URI)]
  else:
    raise data_accessor_errors.InvalidRequestFieldError(
        'DICOM path not defined.'
    )
  dicom_paths = []
  test_series = None
  for instance_path in instance_paths:
    try:
      json_validation_utils.validate_not_empty_str(instance_path)
    except json_validation_utils.ValidationError as exp:
      raise data_accessor_errors.InvalidRequestFieldError(
          'Invalid DICOM path.'
      ) from exp
    try:
      dicom_web_path = dicom_path.FromString(instance_path)
    except ValueError as exp:
      raise data_accessor_errors.InvalidRequestFieldError(
          f'Invalid DICOMWEB uri "{instance_path}".'
      ) from exp
    if dicom_web_path.type not in _ACCEPTED_INSTANCE_TYPES:
      raise data_accessor_errors.InvalidRequestFieldError(
          f'DICOM path "{instance_path}" does not define a instance or series.'
      )
    if (
        dicom_web_path.type == dicom_path.Type.SERIES
        and len(instance_paths) > 1
    ):
      raise data_accessor_errors.InvalidRequestFieldError(
          'DICOM path cannot define multiple URI when defining path to a'
          ' series.'
      )
    series_url = dicom_web_path.GetSeriesPath().complete_url
    if test_series is None:
      test_series = series_url
    elif test_series != series_url:
      raise data_accessor_errors.InvalidRequestFieldError(
          'DICOMweb path defines instances from multiple series.'
      )
    dicom_paths.append(dicom_web_path)
  return dicom_paths
