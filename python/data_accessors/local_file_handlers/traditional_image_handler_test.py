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
"""Unit tests for traditional image handler."""

import io
import os
from typing import Any, Mapping

from absl.testing import absltest
from absl.testing import parameterized
import cv2
from ez_wsi_dicomweb import dicom_slide
import numpy as np
import PIL.Image

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.local_file_handlers import abstract_handler
from data_accessors.local_file_handlers import traditional_image_handler
from data_accessors.utils import patch_coordinate
from data_accessors.utils import test_utils

_traditonal_image_handler = traditional_image_handler.TraditionalImageHandler()
_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


def _mock_instance_extension_metadata(
    extensions: Mapping[str, Any],
) -> Mapping[str, Any]:
  return {_InstanceJsonKeys.EXTENSIONS: extensions}


_MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM = (
    _mock_instance_extension_metadata(
        {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: "ADOBERGB"}
    )
)


class TraditionalImageHandlerTest(parameterized.TestCase):

  def test_load_image(self):
    images = test_utils.flatten_data_acquisition(
        _traditonal_image_handler.process_files(
            [],
            {},
            abstract_handler.InputFileIterator(
                [test_utils.testdata_path("image.jpeg")]
            ),
        )
    )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, (67, 100, 3))
    with PIL.Image.open(test_utils.testdata_path("image.jpeg")) as source_img:
      expected_img = np.asarray(source_img)
    np.testing.assert_array_equal(images[0], expected_img)

  def test_load_bw_image(self):
    images = test_utils.flatten_data_acquisition(
        _traditonal_image_handler.process_files(
            [],
            {},
            abstract_handler.InputFileIterator(
                [test_utils.testdata_path("image_bw.jpeg")]
            ),
        )
    )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, (67, 100, 1))
    with PIL.Image.open(
        test_utils.testdata_path("image_bw.jpeg")
    ) as source_img:
      expected_img = np.asarray(source_img)
    np.testing.assert_array_equal(images[0][..., 0], expected_img)

  def test_load_image_from_bytes_io(self):
    with open(test_utils.testdata_path("image.jpeg"), "rb") as f:
      with io.BytesIO(f.read()) as binary_file:
        images = test_utils.flatten_data_acquisition(
            _traditonal_image_handler.process_files(
                [], {}, abstract_handler.InputFileIterator([binary_file])
            )
        )
        self.assertLen(images, 1)
        self.assertEqual(images[0].shape, (67, 100, 3))
        with PIL.Image.open(
            test_utils.testdata_path("image.jpeg")
        ) as source_img:
          expected_img = np.asarray(source_img)
        np.testing.assert_array_equal(images[0], expected_img)

  def test_load_image_patches_coordinates(self):
    images = test_utils.flatten_data_acquisition(
        _traditonal_image_handler.process_files(
            [
                patch_coordinate.PatchCoordinate(0, 0, 10, 10),
                patch_coordinate.PatchCoordinate(10, 10, 10, 10),
            ],
            {},
            abstract_handler.InputFileIterator(
                [test_utils.testdata_path("image.jpeg")]
            ),
        )
    )
    self.assertLen(images, 2)
    for img in images:
      self.assertEqual(img.shape, (10, 10, 3))
    with PIL.Image.open(test_utils.testdata_path("image.jpeg")) as source_img:
      expected_img = np.asarray(source_img)
    np.testing.assert_array_equal(images[0], expected_img[:10, :10, :])
    np.testing.assert_array_equal(images[1], expected_img[10:20, 10:20, :])

  def test_load_image_bw_patches_coordinates(self):
    images = test_utils.flatten_data_acquisition(
        _traditonal_image_handler.process_files(
            [
                patch_coordinate.PatchCoordinate(0, 0, 10, 10),
                patch_coordinate.PatchCoordinate(10, 10, 10, 10),
            ],
            {},
            abstract_handler.InputFileIterator(
                [test_utils.testdata_path("image_bw.jpeg")]
            ),
        )
    )
    self.assertLen(images, 2)
    for img in images:
      self.assertEqual(img.shape, (10, 10, 1))
    with PIL.Image.open(
        test_utils.testdata_path("image_bw.jpeg")
    ) as source_img:
      expected_img = np.expand_dims(np.asarray(source_img), axis=2)
    np.testing.assert_array_equal(images[0], expected_img[:10, :10, :])
    np.testing.assert_array_equal(images[1], expected_img[10:20, 10:20, :])

  def test_load_image_patches_coordinates_outside_of_dim_raises(self):
    with self.assertRaises(
        data_accessor_errors.PatchOutsideOfImageDimensionsError
    ):
      test_utils.flatten_data_acquisition(
          _traditonal_image_handler.process_files(
              [
                  patch_coordinate.PatchCoordinate(-1, 0, 10, 10),
              ],
              {},
              abstract_handler.InputFileIterator(
                  [test_utils.testdata_path("image.jpeg")]
              ),
          )
      )

  def test_disable_patches_coordinates_outside_of_dim_raises(self):
    images = test_utils.flatten_data_acquisition(
        _traditonal_image_handler.process_files(
            [
                patch_coordinate.PatchCoordinate(5, 0, 4000, 10),
            ],
            _mock_instance_extension_metadata(
                {_InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False}
            ),
            abstract_handler.InputFileIterator(
                [test_utils.testdata_path("image.jpeg")]
            ),
        )
    )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, (10, 4000, 3))

  def test_transform_to_unsupported_icc_profile_raises(self):
    with self.assertRaises(
        data_accessor_errors.InvalidIccProfileTransformError
    ):
      test_utils.flatten_data_acquisition(
          _traditonal_image_handler.process_files(
              [
                  patch_coordinate.PatchCoordinate(0, 0, 10, 10),
              ],
              _mock_instance_extension_metadata({
                  _InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: (
                      "bad_value"
                  )
              }),
              abstract_handler.InputFileIterator(
                  [test_utils.testdata_path("image.jpeg")]
              ),
          )
      )

  def test_transform_to_unsupported_icc_profile_nop_if_no_embedded_profile(
      self,
  ):
    images = test_utils.flatten_data_acquisition(
        _traditonal_image_handler.process_files(
            [],
            _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
            abstract_handler.InputFileIterator(
                [test_utils.testdata_path("image.jpeg")]
            ),
        )
    )
    with PIL.Image.open(test_utils.testdata_path("image.jpeg")) as source_img:
      expected_img = np.asarray(source_img)
    np.testing.assert_array_equal(images[0], expected_img)

  def test_transform_to_icc_profile(self):
    romm_profile = dicom_slide.get_rommrgb_icc_profile_bytes()
    temp_image_path = os.path.join(self.create_tempdir(), "test_image.png")
    with PIL.Image.open(test_utils.testdata_path("image.jpeg")) as source_img:
      source_img.save(temp_image_path, icc_profile=romm_profile)
    images = test_utils.flatten_data_acquisition(
        _traditonal_image_handler.process_files(
            [],
            _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
            abstract_handler.InputFileIterator([temp_image_path]),
        )
    )
    with PIL.Image.open(test_utils.testdata_path("image.jpeg")) as source_img:
      expected_img = np.asarray(source_img)
      self.assertFalse(np.array_equal(images[0], expected_img))

  def test_transform_bw_image_to_icc_profile_no_change(self):
    romm_profile = dicom_slide.get_rommrgb_icc_profile_bytes()
    temp_image_path = os.path.join(self.create_tempdir(), "test_image.png")
    with PIL.Image.open(
        test_utils.testdata_path("image_bw.jpeg")
    ) as source_img:
      source_img.save(temp_image_path, icc_profile=romm_profile)
    images = test_utils.flatten_data_acquisition(
        _traditonal_image_handler.process_files(
            [],
            _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
            abstract_handler.InputFileIterator([temp_image_path]),
        )
    )
    with PIL.Image.open(
        test_utils.testdata_path("image_bw.jpeg")
    ) as source_img:
      expected_img = np.expand_dims(np.asarray(source_img), axis=2)
      np.testing.assert_array_equal(images[0], expected_img)

  @parameterized.named_parameters(
      dict(
          testcase_name="downsample_2x",
          scale_factor=1 / 2,
          interpolation=cv2.INTER_AREA,
      ),
      dict(
          testcase_name="upsample_2x",
          scale_factor=2,
          interpolation=cv2.INTER_CUBIC,
      ),
  )
  def test_load_whole_image_resize(self, scale_factor, interpolation):
    img_path = test_utils.testdata_path("image.jpeg")
    with PIL.Image.open(img_path) as source_img:
      pixel_data = np.asarray(source_img)
      width, height = source_img.width, source_img.height
    img = test_utils.flatten_data_acquisition(
        _traditonal_image_handler.process_files(
            [],
            _mock_instance_extension_metadata({
                _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                    _InstanceJsonKeys.WIDTH: int(width * scale_factor),
                    _InstanceJsonKeys.HEIGHT: int(height * scale_factor),
                }
            }),
            abstract_handler.InputFileIterator([img_path]),
        )
    )
    self.assertLen(img, 1)
    np.testing.assert_array_equal(
        img[0],
        cv2.resize(
            pixel_data,
            (int(width * scale_factor), int(height * scale_factor)),
            interpolation=interpolation,
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="downsample_2x",
          scale_factor=1 / 2,
          interpolation=cv2.INTER_AREA,
      ),
      dict(
          testcase_name="upsample_2x",
          scale_factor=2,
          interpolation=cv2.INTER_CUBIC,
      ),
  )
  def test_load_whole_image_resize_patchs(self, scale_factor, interpolation):
    patch_coordinates = [
        patch_coordinate.PatchCoordinate(0, 0, 10, 10),
        patch_coordinate.PatchCoordinate(10, 10, 10, 10),
    ]
    img_path = test_utils.testdata_path("image.jpeg")
    with PIL.Image.open(img_path) as source_img:
      pixel_data = np.asarray(source_img)
      width, height = source_img.width, source_img.height
    images = test_utils.flatten_data_acquisition(
        _traditonal_image_handler.process_files(
            patch_coordinates,
            _mock_instance_extension_metadata({
                _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                    _InstanceJsonKeys.WIDTH: int(width * scale_factor),
                    _InstanceJsonKeys.HEIGHT: int(height * scale_factor),
                }
            }),
            abstract_handler.InputFileIterator([img_path]),
        )
    )
    expected_img = cv2.resize(
        pixel_data,
        (int(width * scale_factor), int(height * scale_factor)),
        interpolation=interpolation,
    )
    self.assertLen(images, 2)
    for pc, img in zip(patch_coordinates, images):
      np.testing.assert_array_equal(
          img,
          expected_img[
              pc.y_origin : pc.y_origin + pc.height,
              pc.x_origin : pc.x_origin + pc.width,
              ...,
          ],
      )

  def test_patch_outside_of_image_upper_left(self):
    img_path = test_utils.testdata_path("image.jpeg")
    with PIL.Image.open(img_path) as source_img:
      pixel_data = np.asarray(source_img)
    img = test_utils.flatten_data_acquisition(
        _traditonal_image_handler.process_files(
            [patch_coordinate.PatchCoordinate(-10, -10, 20, 20)],
            _mock_instance_extension_metadata(
                {_InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False}
            ),
            abstract_handler.InputFileIterator([img_path]),
        )
    )
    self.assertLen(img, 1)
    expected_image = np.zeros((20, 20, 3), dtype=np.uint8)
    expected_image[10:, 10:, :] = pixel_data[0:10, 0:10, :]
    np.testing.assert_array_equal(
        img[0],
        expected_image,
    )

  def test_patch_extends_over_lower_right(self):
    img_path = test_utils.testdata_path("image.jpeg")
    with PIL.Image.open(img_path) as source_img:
      pixel_data = np.asarray(source_img)

    img = test_utils.flatten_data_acquisition(
        _traditonal_image_handler.process_files(
            [patch_coordinate.PatchCoordinate(10, 10, 100, 67)],
            _mock_instance_extension_metadata(
                {_InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False}
            ),
            abstract_handler.InputFileIterator([img_path]),
        )
    )
    self.assertLen(img, 1)
    expected_image = np.zeros((67, 100, 3), dtype=np.uint8)
    expected_image[0:57, 0:90, :] = pixel_data[10:67, 10:100, :]
    np.testing.assert_array_equal(
        img[0],
        expected_image,
    )

  def test_patch_both_sides(self):
    img_path = test_utils.testdata_path("image.jpeg")
    with PIL.Image.open(img_path) as source_img:
      pixel_data = np.asarray(source_img)

    img = test_utils.flatten_data_acquisition(
        _traditonal_image_handler.process_files(
            [patch_coordinate.PatchCoordinate(-1, -1, 102, 69)],
            _mock_instance_extension_metadata(
                {_InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False}
            ),
            abstract_handler.InputFileIterator([img_path]),
        )
    )
    self.assertLen(img, 1)
    expected_image = np.zeros((69, 102, 3), dtype=np.uint8)
    expected_image[1:68, 1:101, :] = pixel_data[...]
    np.testing.assert_array_equal(
        img[0],
        expected_image,
    )

  def test_patch_inside(self):
    img_path = test_utils.testdata_path("image.jpeg")
    with PIL.Image.open(img_path) as source_img:
      pixel_data = np.asarray(source_img)

    results = _traditonal_image_handler.process_files(
        [patch_coordinate.PatchCoordinate(10, 11, 20, 21)],
        _mock_instance_extension_metadata(
            {_InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: True}
        ),
        abstract_handler.InputFileIterator([img_path]),
    )
    ds = next(results)
    self.assertEqual(
        ds.acquision_data_source,
        abstract_data_accessor.AccessorDataSource.TRADITIONAL_IMAGES,
    )
    image_data = list(ds.acquision_data_source_iterator)
    self.assertLen(image_data, 1)
    np.testing.assert_array_equal(
        image_data[0], pixel_data[11:32, 10:30, :]
    )
    with self.assertRaises(StopIteration):
      next(results)


if __name__ == "__main__":
  absltest.main()
