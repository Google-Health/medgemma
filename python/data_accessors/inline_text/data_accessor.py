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
"""Data accessor for text passed inline in a request."""

import contextlib
from typing import Iterator

from data_accessors import abstract_data_accessor
from data_accessors.inline_text import data_accessor_definition


class InlineText(
    abstract_data_accessor.AbstractDataAccessor[
        data_accessor_definition.InlineText, str
    ]
):
  """Data accessor for text passed inline in a request."""

  def data_acquisition_iterator(
      self,
  ) -> Iterator[abstract_data_accessor.DataAcquisition[str]]:
    return iter([
        abstract_data_accessor.DataAcquisition(
            abstract_data_accessor.AccessorDataSource.TEXT,
            iter([self.instance.text]),
        )
    ])

  def is_accessor_data_embedded_in_request(self) -> bool:
    return True

  def load_data(self, stack: contextlib.ExitStack) -> None:
    """Method pre-loads data prior to data_iterator."""
    return
