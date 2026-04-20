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

"""Request dataclasses for DICOM generic data accessor."""

import dataclasses
import json
from typing import Any, Mapping, Never, Optional, Sequence

from ez_wsi_dicomweb import credential_factory as credential_factory_module
import google.cloud.storage

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.utils import json_validation_utils
from data_accessors.utils import patch_coordinate as patch_coordinate_module

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
_PRESENT = 'PRESENT'


@dataclasses.dataclass(frozen=True)
class GcsGenericBlob:
  credential_factory: credential_factory_module.AbstractCredentialFactory
  gcs_blobs: Sequence[google.cloud.storage.Blob]
  base_request: Mapping[str, Any]
  patch_coordinates: Sequence[patch_coordinate_module.PatchCoordinate]


def _generate_instance_metadata_error_string(
    metadata: Mapping[str, Any], *keys: str
) -> str:
  """returns instance metadata as a error string."""
  result = {}
  for key in keys:
    if key not in metadata:
      continue
    if key == _InstanceJsonKeys.BEARER_TOKEN:
      value = metadata[key]
      # If bearer token is present, and defined strip
      if isinstance(value, str) and value:
        result[key] = _PRESENT
        continue
    # otherwise just associate key and value.
    result[key] = metadata[key]
  return json.dumps(result, sort_keys=True)


def _raise_exception(
    msg: str,
    exp: Optional[Exception],
    instance: Mapping[str, Any],
) -> Never:
  """Raises an exception with a consistent error message."""
  error_msg = _generate_instance_metadata_error_string(
      instance,
      _InstanceJsonKeys.GCS_URI,
      _InstanceJsonKeys.GCS_SOURCE,
      _InstanceJsonKeys.BEARER_TOKEN,
      _InstanceJsonKeys.EXTENSIONS,
  )
  if exp is not None:
    raise data_accessor_errors.InvalidRequestFieldError(
        f'DICOM instance JSON formatting is invalid; {msg}; {error_msg}'
    ) from exp
  raise data_accessor_errors.InvalidRequestFieldError(
      f'DICOM instance JSON formatting is invalid; {msg}; {error_msg}'
  )


def json_to_generic_gcs_image(
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    instance: Mapping[str, Any],
    default_patch_width: int,
    default_patch_height: int,
    require_patch_dim_match_default_dim: bool,
) -> GcsGenericBlob:
  """Converts json to DicomGenericImage."""
  try:
    patch_coordinates = patch_coordinate_module.parse_patch_coordinates(
        instance.get(_InstanceJsonKeys.PATCH_COORDINATES, []),
        default_patch_width,
        default_patch_height,
        require_patch_dim_match_default_dim,
    )
  except patch_coordinate_module.InvalidCoordinateError as exp:
    instance_error_msg = _generate_instance_metadata_error_string(
        instance,
        _InstanceJsonKeys.PATCH_COORDINATES,
    )
    raise data_accessor_errors.InvalidRequestFieldError(
        f'Invalid patch coordinate; {exp}; {instance_error_msg}'
    ) from exp

  if _InstanceJsonKeys.GCS_SOURCE in instance:
    gcs_uris = instance.get(_InstanceJsonKeys.GCS_SOURCE, '')
    if isinstance(gcs_uris, list):
      if not gcs_uris:
        _raise_exception('gcs_source is an empty list', None, instance)
    else:
      gcs_uris = [gcs_uris]
  elif _InstanceJsonKeys.GCS_URI in instance:
    # Legacy support for decoding GCS_URI used in MedSigLip Endpoint.
    gcs_uris = [instance.get(_InstanceJsonKeys.GCS_URI, '')]
  else:
    _raise_exception('GCS URI not defined', None, instance)
  gcs_blobs = []
  for uri in gcs_uris:
    try:
      json_validation_utils.validate_not_empty_str(uri)
    except (ValueError, json_validation_utils.ValidationError) as exp:
      _raise_exception('invalid GCS URI', exp, instance)
    try:
      gcs_blobs.append(google.cloud.storage.Blob.from_string(uri))
    except ValueError as exp:
      _raise_exception('invalid GCS URI', exp, instance)
  return GcsGenericBlob(
      credential_factory=credential_factory,
      gcs_blobs=gcs_blobs,
      base_request=instance,
      patch_coordinates=patch_coordinates,
  )
