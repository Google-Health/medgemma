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
"""Tests for image dimension type."""

from absl.testing import absltest

from data_accessors.utils import image_dimension_type


class ImageDimensionTypeTest(absltest.TestCase):

  def test_image_dimension_type_copy(self):
    test = image_dimension_type.ImageDimensions(100, 200).copy()
    self.assertEqual(test.width, 100)
    self.assertEqual(test.height, 200)


if __name__ == "__main__":
  absltest.main()
