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
import base64
import binascii
import dataclasses
import json
from typing import Any, Mapping, Sequence

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.utils import json_validation_utils
from data_accessors.utils import patch_coordinate as patch_coordinate_module

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
_PRESENT = 'PRESENT'


@dataclasses.dataclass(frozen=True)
class InlineBytes:
  input_bytes: bytes
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


def json_to_generic_bytes(
    instance: Mapping[str, Any],
    default_patch_width: int,
    default_patch_height: int,
    require_patch_dim_match_default_dim: bool,
) -> InlineBytes:
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

  input_bytes = instance.get(_InstanceJsonKeys.INPUT_BYTES, '')
  try:
    input_bytes = json_validation_utils.validate_not_empty_str(input_bytes)
  except json_validation_utils.ValidationError as exp:
    raise data_accessor_errors.InvalidRequestFieldError(
        'Invalid input bytes.'
    ) from exp
  try:
    input_bytes = base64.b64decode(input_bytes, validate=True)
  except (binascii.Error, ValueError) as exp:
    raise data_accessor_errors.InvalidRequestFieldError(
        'Invalid input bytes.'
    ) from exp
  try:
    return InlineBytes(
        input_bytes=input_bytes,
        base_request=instance,
        patch_coordinates=patch_coordinates,
    )
  except json_validation_utils.ValidationError as exp:
    error_msg = _generate_instance_metadata_error_string(
        instance,
        _InstanceJsonKeys.INPUT_BYTES,
        _InstanceJsonKeys.BEARER_TOKEN,
        _InstanceJsonKeys.EXTENSIONS,
    )
    raise data_accessor_errors.InvalidRequestFieldError(
        f'DICOM instance JSON formatting is invalid; {error_msg}'
    ) from exp
