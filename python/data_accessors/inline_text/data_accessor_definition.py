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
from typing import Any, Mapping

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


@dataclasses.dataclass(frozen=True)
class InlineText:
  text: str
  base_request: Mapping[str, Any]


def json_to_text(instance: Mapping[str, Any]) -> InlineText:
  """Converts json to DicomGenericImage."""
  text = instance.get(_InstanceJsonKeys.TEXT, '')
  if not isinstance(text, str):
    raise data_accessor_errors.InvalidRequestFieldError('Text is not a string.')
  return InlineText(text=text, base_request=instance)
