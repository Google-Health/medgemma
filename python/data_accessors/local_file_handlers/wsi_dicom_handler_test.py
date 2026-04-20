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
"""Tests for WsiDicomHandler."""

import io
import os
from typing import Any, Mapping

from absl.testing import absltest
from absl.testing import parameterized
import cv2
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_frame_decoder
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import numpy as np
import PIL.Image
import pydicom

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.local_file_handlers import abstract_handler
from data_accessors.local_file_handlers import wsi_dicom_handler
from data_accessors.utils import dicom_source_utils
from data_accessors.utils import patch_coordinate
from data_accessors.utils import test_utils
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock


_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
_MOCK_DICOM_STORE_PATH = 'https://www.mock_dicom_store.com'


def _mock_instance_extension_metadata(
    extensions: Mapping[str, Any],
) -> Mapping[str, Any]:
  return {_InstanceJsonKeys.EXTENSIONS: extensions}


_MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM = (
    _mock_instance_extension_metadata(
        {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'ADOBERGB'}
    )
)


def _wsi_dicom_gt_path() -> str:
  return test_utils.testdata_path(
      'wsi', 'multiframe_camelyon_challenge_image.png'
  )


def _wsi_dicom_file_path() -> str:
  return test_utils.testdata_path(
      'wsi', 'multiframe_camelyon_challenge_image.dcm'
  )


class WsiDicomHandlerTest(parameterized.TestCase):

  def test_wsi_dicom_handler_load_whole_image(self):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    img = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [], {}, abstract_handler.InputFileIterator([_wsi_dicom_file_path()])
        )
    )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, (700, 1152, 3))
    with PIL.Image.open(_wsi_dicom_gt_path()) as expected_img:
      np.testing.assert_array_equal(img[0], np.asarray(expected_img))

  def test_wsi_dicom_handler_and_ez_wsi_decode_same_result_for_whole_image(
      self,
  ):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    img = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [], {}, abstract_handler.InputFileIterator([_wsi_dicom_file_path()])
        )
    )
    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      with dicom_store_mock.MockDicomStores(
          _MOCK_DICOM_STORE_PATH
      ) as dicom_store:
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        dwi = dicom_web_interface.DicomWebInterface(
            credential_factory.NoAuthCredentialsFactory()
        )
        ds = dicom_slide.DicomSlide(
            dwi,
            dicom_path.FromString(
                f'{_MOCK_DICOM_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}'
            ),
        )
        expected_image_bytes = ds.get_image(ds.native_level).image_bytes()
        np.testing.assert_array_equal(img[0], expected_image_bytes)

  @parameterized.named_parameters([
      dict(
          testcase_name='single_patch',
          patches=[patch_coordinate.PatchCoordinate(1, 2, 9, 10)],
      ),
      dict(
          testcase_name='multiple_patches',
          patches=[
              patch_coordinate.PatchCoordinate(1, 2, 11, 13),
              patch_coordinate.PatchCoordinate(2, 3, 20, 30),
          ],
      ),
  ])
  def test_wsi_dicom_handler_get_patches(self, patches):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            patches,
            {},
            abstract_handler.InputFileIterator([_wsi_dicom_file_path()]),
        )
    )
    self.assertLen(imgs, len(patches))
    with PIL.Image.open(_wsi_dicom_gt_path()) as expected_img:
      expected_img = np.asarray(expected_img)
      expected = [
          expected_img[
              pc.y_origin : pc.y_origin + pc.height,
              pc.x_origin : pc.x_origin + pc.width,
              ...,
          ]
          for pc in patches
      ]
      for i, img in enumerate(imgs):
        np.testing.assert_array_equal(img, expected[i])

  @parameterized.named_parameters([
      dict(
          testcase_name='single_patch',
          patches=[patch_coordinate.PatchCoordinate(1, 2, 9, 10)],
      ),
      dict(
          testcase_name='multiple_patches',
          patches=[
              patch_coordinate.PatchCoordinate(1, 2, 11, 13),
              patch_coordinate.PatchCoordinate(2, 3, 20, 30),
          ],
      ),
  ])
  def test_wsi_dicom_handler_get_patches_matches_ez_wsi_result(self, patches):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            patches,
            {},
            abstract_handler.InputFileIterator([_wsi_dicom_file_path()]),
        )
    )
    self.assertLen(imgs, len(patches))
    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      with dicom_store_mock.MockDicomStores(
          _MOCK_DICOM_STORE_PATH
      ) as dicom_store:
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        dwi = dicom_web_interface.DicomWebInterface(
            credential_factory.NoAuthCredentialsFactory()
        )
        ds = dicom_slide.DicomSlide(
            dwi,
            dicom_path.FromString(
                f'{_MOCK_DICOM_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}'
            ),
        )
        for i, pc in enumerate(patches):
          expected_image_bytes = ds.get_patch(
              ds.native_level, pc.x_origin, pc.y_origin, pc.width, pc.height
          ).image_bytes()
          np.testing.assert_array_equal(imgs[i], expected_image_bytes)

  def test_wsi_dicom_handler_icc_profile_correction_nop_if_no_embedded_profile(
      self,
  ):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [],
            _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
            abstract_handler.InputFileIterator([_wsi_dicom_file_path()]),
        )
    )
    self.assertLen(imgs, 1)
    with PIL.Image.open(_wsi_dicom_gt_path()) as expected_img:
      expected_img = np.asarray(expected_img)
    np.testing.assert_array_equal(imgs[0], expected_img)

  def test_wsi_dicom_handler_icc_profile_correction_if_optical_path_embedded_profile(
      self,
  ):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      dcm.OpticalPathSequence = [pydicom.Dataset()]
      dcm.OpticalPathSequence[0].ICCProfile = (
          dicom_slide.get_srgb_icc_profile_bytes()
      )
      dcm.save_as(dcm_path)

    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [],
            _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
            abstract_handler.InputFileIterator([dcm_path]),
        )
    )
    self.assertLen(imgs, 1)
    with PIL.Image.open(_wsi_dicom_gt_path()) as expected_img:
      expected_img = np.asarray(expected_img)
    self.assertFalse(np.array_equal(imgs[0], expected_img))

  def test_wsi_dicom_handler_icc_profile_correction_if_embedded_profile(self):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      dcm.ICCProfile = dicom_slide.get_srgb_icc_profile_bytes()
      dcm.save_as(dcm_path)

    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [],
            _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
            abstract_handler.InputFileIterator([dcm_path]),
        )
    )
    self.assertLen(imgs, 1)
    with PIL.Image.open(_wsi_dicom_gt_path()) as expected_img:
      expected_img = np.asarray(expected_img)
    self.assertFalse(np.array_equal(imgs[0], expected_img))

  def test_wsi_dicom_handler_icc_profile_noop_if_no_icc_profile_transform(self):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      dcm.ICCProfile = dicom_slide.get_srgb_icc_profile_bytes()
      dcm.save_as(dcm_path)

    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [],
            {},
            abstract_handler.InputFileIterator([dcm_path]),
        )
    )
    self.assertLen(imgs, 1)
    with PIL.Image.open(_wsi_dicom_gt_path()) as expected_img:
      expected_img = np.asarray(expected_img)
    np.testing.assert_array_equal(imgs[0], expected_img)

  def test_read_wsi_from_buffer(self):
    with io.BytesIO() as io_bytes:
      with open(_wsi_dicom_file_path(), 'rb') as infile:
        io_bytes.write(infile.read())
      dicom_handler = wsi_dicom_handler.WsiDicomHandler()
      imgs = test_utils.flatten_data_acquisition(
          dicom_handler.process_files(
              [],
              {},
              abstract_handler.InputFileIterator([io_bytes]),
          )
      )
      self.assertLen(imgs, 1)
      with PIL.Image.open(_wsi_dicom_gt_path()) as expected_img:
        expected_img = np.asarray(expected_img)
      np.testing.assert_array_equal(imgs[0], expected_img)

  def test_read_cannot_read_wsi_from_return_empty(self):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [],
            {},
            abstract_handler.InputFileIterator(
                [test_utils.testdata_path('image.jpeg')]
            ),
        )
    )
    self.assertEmpty(imgs)

  def test_read_patch_outof_bounds_raises(self):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    with self.assertRaises(
        data_accessor_errors.PatchOutsideOfImageDimensionsError
    ):
      test_utils.flatten_data_acquisition(
          dicom_handler.process_files(
              [patch_coordinate.PatchCoordinate(1, 2, 900, 1000)],
              {},
              abstract_handler.InputFileIterator([_wsi_dicom_file_path()]),
          )
      )

  def test_read_patch_outof_bounds_disabled_does_not_raise(self):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    img = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [patch_coordinate.PatchCoordinate(1, 2, 900, 1000)],
            _mock_instance_extension_metadata(
                {_InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False}
            ),
            abstract_handler.InputFileIterator([_wsi_dicom_file_path()]),
        )
    )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, (1000, 900, 3))

  def test_wsi_dicom_handler_decode_uncompressed_dicom(self):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')

    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      frames = wsi_dicom_handler._get_compressed_dicom_frame_bytes(dcm)
      decoded_frames = []
      for frame in frames:
        result = dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
            frame, dcm.file_meta.TransferSyntaxUID
        )
        decoded_frames.append(result)
      pixel_data = np.asarray(decoded_frames).tobytes()
      if len(pixel_data) % 2 != 0:
        pixel_data = b'{pixeldata}\x00'
      dcm.PixelData = pixel_data
      dcm.PhotometricInterpretation = 'RGB'
      dcm.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
      dcm.save_as(dcm_path)

    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [],
            {},
            abstract_handler.InputFileIterator([dcm_path]),
        )
    )
    self.assertLen(imgs, 1)
    with PIL.Image.open(_wsi_dicom_gt_path()) as expected_img:
      expected_img = np.asarray(expected_img)
    np.testing.assert_array_equal(imgs[0], expected_img)

  def test_wsi_dicom_handler_decode_uncompressed_dicom_monochrome2(self):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')

    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      frames = wsi_dicom_handler._get_compressed_dicom_frame_bytes(dcm)
      decoded_frames = []
      for frame in frames:
        result = dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
            frame, dcm.file_meta.TransferSyntaxUID
        )
        decoded_frames.append(result)
      pixel_data = np.asarray(decoded_frames)[..., 0:1].tobytes()
      if len(pixel_data) % 2 != 0:
        pixel_data = b'{pixeldata}\x00'
      dcm.PixelData = pixel_data
      dcm.SamplesPerPixel = 1
      dcm.PhotometricInterpretation = 'MONOCHROME2'
      dcm.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
      dcm.save_as(dcm_path)

    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [],
            {},
            abstract_handler.InputFileIterator([dcm_path]),
        )
    )
    self.assertLen(imgs, 1)
    with PIL.Image.open(_wsi_dicom_gt_path()) as expected_img:
      expected_img = np.asarray(expected_img)
    np.testing.assert_array_equal(imgs[0], expected_img[..., 0:1])

  def test_wsi_dicom_handler_decode_uncompressed_dicom_monochrome1(self):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')

    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      frames = wsi_dicom_handler._get_compressed_dicom_frame_bytes(dcm)
      decoded_frames = []
      for frame in frames:
        result = dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
            frame, dcm.file_meta.TransferSyntaxUID
        )
        decoded_frames.append(result)
      pixel_data = np.asarray(decoded_frames)[..., 0:1].tobytes()
      if len(pixel_data) % 2 != 0:
        pixel_data = b'{pixeldata}\x00'
      dcm.PixelData = pixel_data
      dcm.SamplesPerPixel = 1
      dcm.PhotometricInterpretation = 'MONOCHROME1'
      dcm.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
      dcm.save_as(dcm_path)

      dicom_handler = wsi_dicom_handler.WsiDicomHandler()
      imgs = test_utils.flatten_data_acquisition(
          dicom_handler.process_files(
              [],
              {},
              abstract_handler.InputFileIterator([dcm_path]),
          )
      )
      self.assertLen(imgs, 1)
      with PIL.Image.open(_wsi_dicom_gt_path()) as expected_img:
        expected_img = np.asarray(expected_img)
      np.testing.assert_array_equal(imgs[0], 255 - expected_img[..., 0:1])

  def test_resized_whole_wsi(self):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [],
            _mock_instance_extension_metadata({
                _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                    _InstanceJsonKeys.WIDTH: 200,
                    _InstanceJsonKeys.HEIGHT: 100,
                }
            }),
            abstract_handler.InputFileIterator([_wsi_dicom_file_path()]),
        )
    )
    self.assertLen(imgs, 1)
    with PIL.Image.open(_wsi_dicom_gt_path()) as expected_img:
      expected_img = np.asarray(expected_img)
      expected_img = cv2.resize(
          expected_img, (200, 100), interpolation=cv2.INTER_AREA
      )
    np.testing.assert_array_equal(imgs[0], expected_img)

  def test_wsi_dicom_handler_decode_whole_image_match_ez_wsi(
      self,
  ):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    img = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [],
            _mock_instance_extension_metadata({
                _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                    _InstanceJsonKeys.WIDTH: 200,
                    _InstanceJsonKeys.HEIGHT: 100,
                }
            }),
            abstract_handler.InputFileIterator([_wsi_dicom_file_path()]),
        )
    )
    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      with dicom_store_mock.MockDicomStores(
          _MOCK_DICOM_STORE_PATH
      ) as dicom_store:
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        dwi = dicom_web_interface.DicomWebInterface(
            credential_factory.NoAuthCredentialsFactory()
        )
        ds = dicom_slide.DicomSlide(
            dwi,
            dicom_path.FromString(
                f'{_MOCK_DICOM_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}'
            ),
        )
        resized = ds.native_level.resize(dicom_slide.ImageDimensions(200, 100))
        expected_image_bytes = ds.get_image(resized).image_bytes()
        np.testing.assert_array_equal(img[0], expected_image_bytes)

  @parameterized.named_parameters([
      dict(
          testcase_name='test_patch1',
          patch=patch_coordinate.PatchCoordinate(1, 2, 20, 20),
          expected_diff=0.08,
      ),
      dict(
          testcase_name='test_patch2',
          patch=patch_coordinate.PatchCoordinate(1, 2, 11, 13),
          expected_diff=0,
      ),
      dict(
          testcase_name='test_patch3',
          patch=patch_coordinate.PatchCoordinate(2, 3, 20, 30),
          expected_diff=0.4,
      ),
  ])
  def test_resized_wsi_patches_smaller_than_original(
      self, patch, expected_diff
  ):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [patch],
            _mock_instance_extension_metadata({
                _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                    _InstanceJsonKeys.WIDTH: 200,
                    _InstanceJsonKeys.HEIGHT: 100,
                }
            }),
            abstract_handler.InputFileIterator([_wsi_dicom_file_path()]),
        )
    )
    self.assertLen(imgs, 1)
    with PIL.Image.open(_wsi_dicom_gt_path()) as expected_img:
      expected_img = np.asarray(expected_img)
    expected_img = cv2.resize(
        expected_img, (200, 100), interpolation=cv2.INTER_AREA
    )
    expected_img = expected_img[
        patch.y_origin : patch.y_origin + patch.height,
        patch.x_origin : patch.x_origin + patch.width,
        ...,
    ]
    diff = np.abs(
        imgs[0].astype(np.int32) - expected_img.astype(np.int32)
    ).astype(np.uint8)
    self.assertLessEqual(np.mean(diff), expected_diff)

  @parameterized.named_parameters([
      dict(
          testcase_name='test_patch1',
          patch=patch_coordinate.PatchCoordinate(1, 2, 20, 20),
      ),
      dict(
          testcase_name='test_patch2',
          patch=patch_coordinate.PatchCoordinate(1, 2, 11, 13),
      ),
      dict(
          testcase_name='test_patch3',
          patch=patch_coordinate.PatchCoordinate(2, 3, 20, 30),
      ),
  ])
  def test_resized_wsi_patches_smaller_than_original_match_ez_wsi(self, patch):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [patch],
            _mock_instance_extension_metadata({
                _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                    _InstanceJsonKeys.WIDTH: 200,
                    _InstanceJsonKeys.HEIGHT: 100,
                }
            }),
            abstract_handler.InputFileIterator([_wsi_dicom_file_path()]),
        )
    )
    self.assertLen(imgs, 1)
    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      with dicom_store_mock.MockDicomStores(
          _MOCK_DICOM_STORE_PATH
      ) as dicom_store:
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        dwi = dicom_web_interface.DicomWebInterface(
            credential_factory.NoAuthCredentialsFactory()
        )
        ds = dicom_slide.DicomSlide(
            dwi,
            dicom_path.FromString(
                f'{_MOCK_DICOM_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}'
            ),
        )
        resized = ds.native_level.resize(dicom_slide.ImageDimensions(200, 100))
        expected_image_bytes = ds.get_patch(
            resized, patch.x_origin, patch.y_origin, patch.width, patch.height
        ).image_bytes()
    np.testing.assert_array_equal(imgs[0], expected_image_bytes)

  @parameterized.named_parameters([
      dict(
          testcase_name='test_patch1',
          patch=patch_coordinate.PatchCoordinate(1, 2, 200, 200),
          expected_diff=0.5,
      ),
      dict(
          testcase_name='test_patch2',
          patch=patch_coordinate.PatchCoordinate(1, 2, 110, 130),
          expected_diff=0.1,
      ),
      dict(
          testcase_name='test_patch3',
          patch=patch_coordinate.PatchCoordinate(2, 3, 200, 300),
          expected_diff=0.3,
      ),
  ])
  def test_resized_wsi_patches_larger_than_original(self, patch, expected_diff):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [patch],
            _mock_instance_extension_metadata({
                _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                    _InstanceJsonKeys.WIDTH: 2000,
                    _InstanceJsonKeys.HEIGHT: 1000,
                }
            }),
            abstract_handler.InputFileIterator([_wsi_dicom_file_path()]),
        )
    )
    self.assertLen(imgs, 1)
    with PIL.Image.open(_wsi_dicom_gt_path()) as expected_img:
      expected_img = np.asarray(expected_img)
    expected_img = cv2.resize(
        expected_img, (2000, 1000), interpolation=cv2.INTER_CUBIC
    )
    expected_img = expected_img[
        patch.y_origin : patch.y_origin + patch.height,
        patch.x_origin : patch.x_origin + patch.width,
        ...,
    ]
    diff = np.abs(
        imgs[0].astype(np.int32) - expected_img.astype(np.int32)
    ).astype(np.uint8)
    self.assertLessEqual(np.mean(diff), expected_diff)

  @parameterized.named_parameters([
      dict(
          testcase_name='test_patch1',
          patch=patch_coordinate.PatchCoordinate(1, 2, 200, 200),
      ),
      dict(
          testcase_name='test_patch2',
          patch=patch_coordinate.PatchCoordinate(1, 2, 110, 130),
      ),
      dict(
          testcase_name='test_patch3',
          patch=patch_coordinate.PatchCoordinate(2, 3, 200, 300),
      ),
  ])
  def test_resized_wsi_patches_larger_than_original_match_ez_wsi(self, patch):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [patch],
            _mock_instance_extension_metadata({
                _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                    _InstanceJsonKeys.WIDTH: 2000,
                    _InstanceJsonKeys.HEIGHT: 1000,
                }
            }),
            abstract_handler.InputFileIterator([_wsi_dicom_file_path()]),
        )
    )
    self.assertLen(imgs, 1)
    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      with dicom_store_mock.MockDicomStores(
          _MOCK_DICOM_STORE_PATH
      ) as dicom_store:
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        dwi = dicom_web_interface.DicomWebInterface(
            credential_factory.NoAuthCredentialsFactory()
        )
        ds = dicom_slide.DicomSlide(
            dwi,
            dicom_path.FromString(
                f'{_MOCK_DICOM_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}'
            ),
        )
        resized = ds.native_level.resize(
            dicom_slide.ImageDimensions(2000, 1000)
        )
        expected_image_bytes = ds.get_patch(
            resized, patch.x_origin, patch.y_origin, patch.width, patch.height
        ).image_bytes()
    np.testing.assert_array_equal(imgs[0], expected_image_bytes)

  def test_resized_wsi_patches_larger_than_original_whole_image(self):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [],
            _mock_instance_extension_metadata({
                _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                    _InstanceJsonKeys.WIDTH: 2000,
                    _InstanceJsonKeys.HEIGHT: 1000,
                }
            }),
            abstract_handler.InputFileIterator([_wsi_dicom_file_path()]),
        )
    )
    self.assertLen(imgs, 1)
    with PIL.Image.open(_wsi_dicom_gt_path()) as expected_img:
      expected_img = np.asarray(expected_img)
    expected_img = cv2.resize(
        expected_img, (2000, 1000), interpolation=cv2.INTER_CUBIC
    )
    np.testing.assert_array_equal(imgs[0], expected_img)

  def test_resized_wsi_larger_than_original_whole_image_match_ez_wsi(self):
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    imgs = test_utils.flatten_data_acquisition(
        dicom_handler.process_files(
            [],
            _mock_instance_extension_metadata({
                _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                    _InstanceJsonKeys.WIDTH: 2000,
                    _InstanceJsonKeys.HEIGHT: 1000,
                }
            }),
            abstract_handler.InputFileIterator([_wsi_dicom_file_path()]),
        )
    )
    self.assertLen(imgs, 1)
    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      with dicom_store_mock.MockDicomStores(
          _MOCK_DICOM_STORE_PATH
      ) as dicom_store:
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        dwi = dicom_web_interface.DicomWebInterface(
            credential_factory.NoAuthCredentialsFactory()
        )
        ds = dicom_slide.DicomSlide(
            dwi,
            dicom_path.FromString(
                f'{_MOCK_DICOM_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}'
            ),
        )
        resized = ds.native_level.resize(
            dicom_slide.ImageDimensions(2000, 1000)
        )
        expected_image_bytes = ds.get_image(resized).image_bytes()
    np.testing.assert_array_equal(imgs[0], expected_image_bytes)

  def test_wsi_dicom_handler_dicom_invalid_photometric_interpretation_raises(
      self,
  ):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')

    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      frames = wsi_dicom_handler._get_compressed_dicom_frame_bytes(dcm)
      decoded_frames = []
      for frame in frames:
        result = dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
            frame, dcm.file_meta.TransferSyntaxUID
        )
        decoded_frames.append(result)
      pixel_data = np.asarray(decoded_frames)[..., 0:1].tobytes()
      if len(pixel_data) % 2 != 0:
        pixel_data = b'{pixeldata}\x00'
      dcm.PixelData = pixel_data
      dcm.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
      dcm.save_as(dcm_path)
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    with self.assertRaisesRegex(
        data_accessor_errors.DicomError,
        '.*unsupported PhotometricInterpretation.*',
    ):
      test_utils.flatten_data_acquisition(
          dicom_handler.process_files(
              [], {}, abstract_handler.InputFileIterator([dcm_path])
          )
      )

  def test_wsi_dicom_handler_dicom_missing_pixel_data_raises(self):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')

    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      del dcm['PixelData']
      dcm.save_as(dcm_path)
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    with self.assertRaisesRegex(
        data_accessor_errors.DicomError, '.*DICOM missing PixelData.*'
    ):
      test_utils.flatten_data_acquisition(
          dicom_handler.process_files(
              [], {}, abstract_handler.InputFileIterator([dcm_path])
          )
      )

  def test_wsi_dicom_handler_not_vl_whole_slide_micropscopy_image_skips(self):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')

    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      dcm.SOPClassUID = dicom_source_utils._VL_MICROSCOPIC_IMAGE_SOP_CLASS_UID
      del dcm['PixelData']
      dcm.save_as(dcm_path)
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    self.assertEmpty(
        test_utils.flatten_data_acquisition(
            dicom_handler.process_files(
                [], {}, abstract_handler.InputFileIterator([dcm_path])
            )
        )
    )

  def test_uncompressed_wsi_missing_pixel_data_raises(
      self,
  ):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')

    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      del dcm['PixelData']
      dcm.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
      dcm.PhotometricInterpretation = 'RGB'
      dcm.SamplesPerPixel = 3
      dcm.save_as(dcm_path)
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    with self.assertRaisesRegex(
        data_accessor_errors.DicomError,
        '.*Cannot decode pixel data.*',
    ):
      test_utils.flatten_data_acquisition(
          dicom_handler.process_files(
              [], {}, abstract_handler.InputFileIterator([dcm_path])
          )
      )

  def test_concatenated_dicom_raises(self):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')

    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      dcm.ConcatenationUID = '1.2.3'
      dcm.save_as(dcm_path)
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    with self.assertRaisesRegex(
        data_accessor_errors.DicomError,
        '.*Reading concatenated WSI DICOM from sources other than a DICOM store'
        ' is not supported.*',
    ):
      test_utils.flatten_data_acquisition(
          dicom_handler.process_files(
              [], {}, abstract_handler.InputFileIterator([dcm_path])
          )
      )

  def test_tiled_sparse_dicom_raises(self):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')

    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      dcm.DimensionOrganizationType = 'TILED_SPARSE'
      dcm.save_as(dcm_path)
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    with self.assertRaisesRegex(
        data_accessor_errors.DicomTiledFullError,
        '.*DICOM DimensionOrganizationType is not TILED_FULL.*',
    ):
      test_utils.flatten_data_acquisition(
          dicom_handler.process_files(
              [], {}, abstract_handler.InputFileIterator([dcm_path])
          )
      )

  def test_tiled_sparse_dicom_implicit_number_of_frames_one(self):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      del dcm['NumberOfFrames']
      dcm.DimensionOrganizationType = 'TILED_SPARSE'
      dcm.save_as(dcm_path)
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    self.assertLen(
        test_utils.flatten_data_acquisition(
            dicom_handler.process_files(
                [], {}, abstract_handler.InputFileIterator([dcm_path])
            )
        ),
        1,
    )

  def test_tiled_sparse_dicom_explicit_number_of_frames_one(self):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
    with pydicom.dcmread(_wsi_dicom_file_path()) as dcm:
      dcm.NumberOfFrames = 1
      dcm.DimensionOrganizationType = 'TILED_SPARSE'
      dcm.save_as(dcm_path)
    dicom_handler = wsi_dicom_handler.WsiDicomHandler()
    results = dicom_handler.process_files(
        [], {}, abstract_handler.InputFileIterator([dcm_path])
    )

    r = next(results)
    self.assertEqual(
        r.acquision_data_source,
        abstract_data_accessor.AccessorDataSource.DICOM_WSI_MICROSCOPY_PYRAMID_LEVEL,
    )
    self.assertLen(list(r.acquision_data_source_iterator), 1)
    with self.assertRaises(StopIteration):
      next(results)


if __name__ == '__main__':
  absltest.main()
