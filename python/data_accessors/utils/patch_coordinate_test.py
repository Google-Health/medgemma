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

"""Tests for patch coordinate."""

from absl.testing import absltest
from absl.testing import parameterized

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.utils import patch_coordinate

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


class PatchCoordinateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self._patch_coordinate_zero_dimensions = (
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 1, 'y_origin': 3}, default_width=224,
            default_height=224,
        )
    )
    self._patch_coordinate_zero_dimensions_dict = {
        'x_origin': 1,
        'y_origin': 3,
        'width': 224,
        'height': 224,
    }

  def test_dicom_embedding_patch_coordinate_invalid_dimensions(self):
    with self.assertRaises(data_accessor_errors.PatchCoordinateError):
      _ = patch_coordinate.create_patch_coordinate(
          dict(
              x_origin=1,
              y_origin=3,
              width=11,
              height=10,
          ),
          224,
          224,
          True,
      )

  def test_dicom_embedding_patch_coordinate_alternative_dimensions(self):
    pc = patch_coordinate.create_patch_coordinate(
        dict(
            x_origin=1,
            y_origin=3,
            width=11,
            height=10,
        ),
        224,
        224,
    )
    self.assertEqual(pc.x_origin, 1)
    self.assertEqual(pc.y_origin, 3)
    self.assertEqual(pc.width, 11)
    self.assertEqual(pc.height, 10)

  def test_dicom_embedding_patch_coordinate_default_dimensions(self):
    parameters = self._patch_coordinate_zero_dimensions

    self.assertEqual(
        parameters.__dict__, self._patch_coordinate_zero_dimensions_dict
    )

  def test_get_required_fully_in_source_image_extension_default(self):
    self.assertTrue(
        patch_coordinate.patch_required_to_be_fully_in_source_image({}),
    )

  @parameterized.parameters([True, 'TRUE', 'True', 'true', 'TrUe'])
  def test_get_required_fully_in_source_image_extension_true(self, val):
    self.assertTrue(
        patch_coordinate.patch_required_to_be_fully_in_source_image(
            {_InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: val}
        ),
    )

  @parameterized.parameters([False, 'FALSE', 'False', 'false', 'FaLsE'])
  def test_get_required_fully_in_source_image_extension_false(self, val):
    self.assertFalse(
        patch_coordinate.patch_required_to_be_fully_in_source_image(
            {_InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: val}
        ),
    )

  @parameterized.parameters(['', 'BadString', 1, 2, ({},), ([],), 0, 0.0])
  def test_get_required_fully_in_source_image_extension_raises(self, val):
    with self.assertRaises(data_accessor_errors.InvalidRequestFieldError):
      patch_coordinate.patch_required_to_be_fully_in_source_image(
          {_InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: val}
      )

if __name__ == '__main__':
  absltest.main()
