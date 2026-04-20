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
from typing import Any, Mapping, Sequence, Union

from ez_wsi_dicomweb import credential_factory as credential_factory_module

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.utils import patch_coordinate as patch_coordinate_module

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
_PRESENT = 'PRESENT'
_IMAGE_URLS_JSON_KEYS = (
    _InstanceJsonKeys.IMAGE_URL,
    _InstanceJsonKeys.URL,
    _InstanceJsonKeys.IMAGE,
)


@dataclasses.dataclass(frozen=True)
class HttpImage:
  credential_factory: credential_factory_module.AbstractCredentialFactory
  urls: Sequence[str]
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


def _parse_url(image_url_dict: Union[Mapping[str, Any], str]) -> str:
  """Parses URL from image_url dict."""
  if isinstance(image_url_dict, Mapping):
    image_url_dict = image_url_dict.get(_InstanceJsonKeys.URL)
    if image_url_dict is None:
      raise data_accessor_errors.InvalidRequestFieldError(
          'Missing URL in image_url dict.'
      )
  if isinstance(image_url_dict, str):
    url = image_url_dict
  else:
    raise data_accessor_errors.InvalidRequestFieldError(
        'Invalid URL, must be a string.'
    )
  if not url:
    raise data_accessor_errors.InvalidRequestFieldError(
        'Invalid URL, must be not emptyt.'
    )
  return url


def json_to_http_image(
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    instance: Mapping[str, Any],
    default_patch_width: int,
    default_patch_height: int,
    require_patch_dim_match_default_dim: bool,
) -> HttpImage:
  """Converts json to HttpImage."""
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
  urls = []
  image_urls = None
  for url_key in _IMAGE_URLS_JSON_KEYS:
    image_urls = instance.get(url_key)
    if image_urls is not None:
      break
  if image_urls is None:
    raise data_accessor_errors.InvalidRequestFieldError(
        'Missing URL in image_url dict.'
    )
  if isinstance(image_urls, list):
    if not image_urls:
      raise data_accessor_errors.InvalidRequestFieldError('Empty URL list.')
    for url in image_urls:
      urls.append(_parse_url(url))
  else:
    urls.append(_parse_url(image_urls))
  return HttpImage(credential_factory, urls, instance, patch_coordinates)
