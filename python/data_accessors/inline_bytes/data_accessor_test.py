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
"""Tests for gcs_generic data accessor."""
import base64
from typing import Any, Mapping, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import PIL.Image

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.inline_bytes import data_accessor
from data_accessors.inline_bytes import data_accessor_definition
from data_accessors.local_file_handlers import generic_dicom_handler
from data_accessors.local_file_handlers import traditional_image_handler
from data_accessors.utils import test_utils

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


def _test_load_from_request(
    file_handlers,
    json_instance: Mapping[str, Any],
) -> Sequence[np.ndarray]:
  instance = data_accessor_definition.json_to_generic_bytes(
      json_instance,
      default_patch_width=256,
      default_patch_height=256,
      require_patch_dim_match_default_dim=False,
  )
  local_data_accessor = data_accessor.InlineBytesData(
      instance,
      file_handlers=file_handlers,
  )
  return list(local_data_accessor.data_iterator())


def _test_load_image(json_instance: Mapping[str, Any]) -> Sequence[np.ndarray]:
  return _test_load_from_request(
      [
          generic_dicom_handler.GenericDicomHandler(),
          traditional_image_handler.TraditionalImageHandler(),
      ],
      json_instance,
  )


def _test_load_generic_dicom(
    json_instance: Mapping[str, Any],
) -> Sequence[np.ndarray]:
  return _test_load_from_request(
      [
          traditional_image_handler.TraditionalImageHandler(),
          generic_dicom_handler.GenericDicomHandler(),
      ],
      json_instance,
  )


def _test_base64_encoded_bytes(*filename: str) -> str:
  with open(test_utils.testdata_path(*filename), 'rb') as f:
    return base64.b64encode(f.read()).decode('utf-8')


def _test_jpeg_image() -> str:
  return _test_base64_encoded_bytes('image.jpeg')


def _test_jpeg_bw_image() -> str:
  return _test_base64_encoded_bytes('image_bw.jpeg')


def _test_dicom() -> str:
  return _test_base64_encoded_bytes('cxr', 'encapsulated_cxr.dcm')


class DataAccessorTest(parameterized.TestCase):

  def test_traditional_image_handler_color_image(self):
    json_instance = {_InstanceJsonKeys.INPUT_BYTES: _test_jpeg_image()}
    source_image_path = test_utils.testdata_path('image.jpeg')
    img = _test_load_image(json_instance)
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, (67, 100, 3))
    with PIL.Image.open(source_image_path) as source_img:
      expected_img = np.asarray(source_img)
    np.testing.assert_array_equal(img[0], expected_img)

  def test_traditional_image_handler_bw_image(self):
    json_instance = {_InstanceJsonKeys.INPUT_BYTES: _test_jpeg_bw_image()}
    source_image_path = test_utils.testdata_path('image_bw.jpeg')
    img = _test_load_image(json_instance)
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, (67, 100, 1))
    with PIL.Image.open(source_image_path) as source_img:
      expected_img = np.asarray(source_img)
    np.testing.assert_array_equal(img[0][..., 0], expected_img)

  @parameterized.named_parameters(
      dict(
          testcase_name='patch_list',
          patch_coordinates=[
              {'x_origin': 0, 'y_origin': 0, 'width': 256, 'height': 256}
          ],
      ),
  )
  def test_traditional_image_patch_coordinates_outside_of_image_raises(
      self, patch_coordinates
  ):
    json_instance = {
        _InstanceJsonKeys.INPUT_BYTES: _test_jpeg_image(),
        _InstanceJsonKeys.PATCH_COORDINATES: patch_coordinates,
    }
    with self.assertRaises(
        data_accessor_errors.PatchOutsideOfImageDimensionsError
    ):
      _test_load_image(json_instance)

  @parameterized.named_parameters(
      dict(
          testcase_name='patch_list',
          patch_coordinates=[
              {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10}
          ],
          expected_shape=(10, 10, 3),
      ),
      dict(
          testcase_name='empty_patch_list',
          patch_coordinates=[],
          expected_shape=(67, 100, 3),
      ),
  )
  def test_traditional_image_patch_coordinates(
      self, patch_coordinates, expected_shape
  ):
    json_instance = {
        _InstanceJsonKeys.INPUT_BYTES: _test_jpeg_image(),
        _InstanceJsonKeys.PATCH_COORDINATES: patch_coordinates,
    }
    img = _test_load_image(json_instance)
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, expected_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='patch_list',
          patch_coordinates=[
              {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10}
          ],
          expected_shape=(10, 10, 1),
      ),
      dict(
          testcase_name='empty_patch_list',
          patch_coordinates=[],
          expected_shape=(1024, 1024, 1),
      ),
  )
  def test_dicom_image_patch_coordinates(
      self, patch_coordinates, expected_shape
  ):
    json_instance = {
        _InstanceJsonKeys.INPUT_BYTES: _test_dicom(),
        _InstanceJsonKeys.PATCH_COORDINATES: patch_coordinates,
    }
    img = _test_load_generic_dicom(json_instance)
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, expected_shape)

  def test_dicom_image_patch_coordinates_outside_of_image_raises(self):
    json_instance = {
        _InstanceJsonKeys.INPUT_BYTES: _test_dicom(),
        _InstanceJsonKeys.PATCH_COORDINATES: [
            {'x_origin': 0, 'y_origin': 0, 'width': 5000, 'height': 10}
        ],
    }
    with self.assertRaises(
        data_accessor_errors.PatchOutsideOfImageDimensionsError
    ):
      _test_load_generic_dicom(json_instance)

  def test_is_accessor_data_embedded_in_request(self):
    json_instance = {
        _InstanceJsonKeys.INPUT_BYTES: _test_dicom(),
    }
    instance = data_accessor_definition.json_to_generic_bytes(
        json_instance,
        default_patch_width=256,
        default_patch_height=256,
        require_patch_dim_match_default_dim=False,
    )
    local_data_accessor = data_accessor.InlineBytesData(
        instance,
        file_handlers=[
            traditional_image_handler.TraditionalImageHandler(),
            generic_dicom_handler.GenericDicomHandler(),
        ],
    )
    self.assertTrue(local_data_accessor.is_accessor_data_embedded_in_request())

  @parameterized.named_parameters(
      dict(
          testcase_name='no_patch_coordinates',
          metadata={},
          expected=1,
      ),
      dict(
          testcase_name='one_patch',
          metadata={
              _InstanceJsonKeys.PATCH_COORDINATES: [
                  {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10},
              ]
          },
          expected=1,
      ),
      dict(
          testcase_name='two_patches',
          metadata={
              _InstanceJsonKeys.PATCH_COORDINATES: [
                  {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10},
                  {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10},
              ]
          },
          expected=2,
      ),
  )
  def test_accessor_length(self, metadata, expected):
    json_instance = {
        _InstanceJsonKeys.INPUT_BYTES: _test_dicom(),
    }
    json_instance.update(metadata)
    instance = data_accessor_definition.json_to_generic_bytes(
        json_instance,
        default_patch_width=256,
        default_patch_height=256,
        require_patch_dim_match_default_dim=False,
    )
    local_data_accessor = data_accessor.InlineBytesData(
        instance,
        file_handlers=[
            traditional_image_handler.TraditionalImageHandler(),
            generic_dicom_handler.GenericDicomHandler(),
        ],
    )
    self.assertLen(local_data_accessor, expected)


if __name__ == '__main__':
  absltest.main()
